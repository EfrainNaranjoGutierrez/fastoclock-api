import os
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from app.config import settings

router = APIRouter()


def _load_models() -> dict:
    """Carga modelos globales (sin job_id)."""
    models = {}
    files = {
        "scaler": "scaler.pkl",
        "kmeans": "kmeans.pkl",
        "7d": "xgb_clf_7d.pkl",
        "14d": "xgb_clf_14d.pkl",
        "30d": "xgb_clf_30d.pkl",
        "60d": "xgb_clf_60d.pkl",
        "reg": "xgb_reg_days.pkl",
    }
    for key, fname in files.items():
        path = os.path.join(settings.MODELS_DIR, fname)
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models


def _find_features(job_id: str) -> pd.DataFrame:
    """Busca features del job, o usa el más reciente disponible."""
    # Try job-specific first
    path = os.path.join(settings.OUTPUTS_DIR, f"{job_id}_features.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    # Fall back to any available features file
    outputs_dir = settings.OUTPUTS_DIR
    if os.path.exists(outputs_dir):
        files = [f for f in os.listdir(outputs_dir) if f.endswith("_features.parquet")]
        if files:
            # Use most recent
            files.sort(key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)), reverse=True)
            return pd.read_parquet(os.path.join(outputs_dir, files[0]))

    return None


def _build_interpretation(name, prob30, days, segment, product, interval):
    if prob30 >= settings.UMBRAL_ALTA:
        return (
            f"Con base en nuestro pronóstico, {name} tiene una probabilidad del {prob30:.0%} "
            f"de realizar una nueva compra en los próximos 30 días. Su patrón de compra cada "
            f"~{interval:.0f} días y su historial con {product} indican que es momento de "
            f"programar el pedido de forma anticipada. Estimamos que comprará en aproximadamente "
            f"{days:.0f} días. Segmento: {segment}."
        )
    elif prob30 >= settings.UMBRAL_MEDIA:
        return (
            f"Con base en nuestro pronóstico, {name} muestra una probabilidad moderada ({prob30:.0%}) "
            f"de recompra en los próximos 30 días. Su producto principal es {product} con un ciclo "
            f"de ~{interval:.0f} días. Recomendamos contactar al cliente para confirmar sus necesidades. "
            f"Segmento: {segment}."
        )
    else:
        return (
            f"Con base en nuestro pronóstico, {name} tiene baja probabilidad de recompra ({prob30:.0%}) "
            f"en los próximos 30 días. Ha pasado más tiempo del habitual desde su última compra. "
            f"Se recomienda una campaña de reactivación enfocada en {product}. "
            f"Segmento: {segment}."
        )


def _generate_predictions(features: pd.DataFrame, models: dict) -> list:
    feature_cols = [c for c in features.columns if c not in (
        "cliente", "primera_compra", "ultima_compra", "cluster",
        "segmento", "producto_top"
    ) and features[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    predictions = []
    for _, row in features.iterrows():
        try:
            X = row[feature_cols].fillna(0).values.reshape(1, -1)
            if "scaler" in models:
                X = models["scaler"].transform(X)

            probs = {}
            for h in [7, 14, 30, 60]:
                key = f"{h}d"
                if key in models:
                    probs[key] = round(float(models[key].predict_proba(X)[0][1]), 4)
                else:
                    probs[key] = 0.0

            dias = round(float(models["reg"].predict(X)[0]), 1) if "reg" in models else 30.0
            p30 = probs["30d"]

            if p30 >= settings.UMBRAL_ALTA:
                accion, prioridad = "PROGRAMAR_PEDIDO", "ALTA"
            elif p30 >= settings.UMBRAL_MEDIA:
                accion, prioridad = "CONTACTAR_CLIENTE", "MEDIA"
            else:
                accion, prioridad = "REACTIVAR_CAMPANA", "BAJA"

            segment = str(row.get("segmento", f"Cluster {int(row.get('cluster', 0))}"))
            product = str(row.get("producto_top", "N/A"))
            interval = float(row.get("intervalo_promedio_dias", 0) or 0)

            antiguedad = float(row.get("antiguedad_dias", 1) or 1)
            total_compras = float(row.get("total_compras", 0) or 0)
            fill = round(min(total_compras / max(antiguedad / max(interval, 1), 1), 1.0) * 100, 1) if interval > 0 else 0

            predictions.append({
                "cliente": str(row["cliente"]),
                "segmento": segment,
                "cluster": int(row.get("cluster", 0) or 0),
                "prob_7d": probs["7d"],
                "prob_14d": probs["14d"],
                "prob_30d": probs["30d"],
                "prob_60d": probs["60d"],
                "dias_estimados": dias,
                "ultima_compra": str(row.get("ultima_compra", ""))[:10],
                "total_ordenes": int(row.get("total_compras", 0) or 0),
                "revenue_total": round(float(row.get("total_revenue", 0) or 0), 2),
                "ticket_promedio": round(float(row.get("avg_ticket", 0) or 0), 2),
                "intervalo_dias": round(interval, 1),
                "fill_rate": fill,
                "producto_top": product,
                "accion": accion,
                "prioridad": prioridad,
                "interpretacion": _build_interpretation(
                    str(row["cliente"]), p30, dias, segment, product, interval
                ),
            })
        except Exception:
            continue

    return predictions


@router.get("/predict/{job_id}")
async def predict_all(
    job_id: str,
    top_n: int = Query(50, ge=1, le=500),
    sort_by: str = Query("prob_30d"),
):
    features = _find_features(job_id)
    if features is None:
        raise HTTPException(404, "Features no encontradas. Sube y entrena de nuevo.")

    models = _load_models()
    if not models:
        raise HTTPException(400, "Modelos no entrenados.")

    predictions = _generate_predictions(features, models)

    if not predictions:
        raise HTTPException(500, "No se pudieron generar predicciones.")

    reverse = sort_by != "dias_estimados"
    predictions.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
    predictions = predictions[:top_n]

    alta = sum(1 for p in predictions if p["prioridad"] == "ALTA")
    media = sum(1 for p in predictions if p["prioridad"] == "MEDIA")
    baja = sum(1 for p in predictions if p["prioridad"] == "BAJA")

    return {
        "status": "success",
        "job_id": job_id,
        "total_clients": len(features),
        "modelo": "XGBoost",
        "predictions": predictions,
        "resumen": {
            "programar_pedido": alta,
            "contactar_cliente": media,
            "reactivar_campana": baja,
            "fill_rate_promedio": round(np.mean([p["fill_rate"] for p in predictions]), 1),
        },
    }


@router.get("/predict/{job_id}/download")
async def download_predictions(job_id: str):
    features = _find_features(job_id)
    if features is None:
        raise HTTPException(404, "Features no encontradas.")

    models = _load_models()
    predictions = _generate_predictions(features, models)
    df = pd.DataFrame(predictions)

    os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
    csv_path = os.path.join(settings.OUTPUTS_DIR, f"{job_id}_predicciones.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return StreamingResponse(
        open(csv_path, "rb"),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predicciones_{job_id}.csv"},
    )


@router.get("/thresholds/{job_id}")
async def threshold_analysis(job_id: str, horizonte: int = 30):
    features = _find_features(job_id)
    if features is None:
        raise HTTPException(404, "Features no encontradas.")

    models = _load_models()
    if f"{horizonte}d" not in models:
        raise HTTPException(400, f"Modelo {horizonte}d no disponible.")

    feature_cols = [c for c in features.columns if c not in (
        "cliente", "primera_compra", "ultima_compra", "cluster",
        "segmento", "producto_top"
    ) and features[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

    X = features[feature_cols].fillna(0)
    if "scaler" in models:
        X = models["scaler"].transform(X)

    y_prob = models[f"{horizonte}d"].predict_proba(X)[:, 1]

    from sklearn.metrics import precision_score, recall_score
    analysis = []
    dummy_y = (y_prob >= 0.5).astype(int)

    for u in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_prob >= u).astype(int)
        prec = precision_score(dummy_y, y_pred, zero_division=0)
        rec = recall_score(dummy_y, y_pred, zero_division=0)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        act = int(y_pred.sum())
        analysis.append({
            "umbral": round(u, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "clientes_activados": act,
            "pct_activados": round(act / len(y_prob) * 100, 1),
        })

    best = max(analysis, key=lambda x: x["f1"])
    return {
        "horizonte": f"{horizonte}d",
        "umbral_optimo": best["umbral"],
        "mejor_f1": best["f1"],
        "analisis": analysis,
    }
