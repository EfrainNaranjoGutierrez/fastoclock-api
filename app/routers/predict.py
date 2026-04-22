import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from app.config import settings

router = APIRouter()


def _build_interpretation(name, prob30, days, segment, product, interval):
    """Genera interpretación en lenguaje natural de la predicción."""
    if prob30 >= settings.UMBRAL_ALTA:
        return (
            f"{name} tiene una probabilidad del {prob30:.0%} de realizar una nueva compra "
            f"en los próximos 30 días. Basado en su patrón de compra cada ~{interval:.0f} días "
            f"y su historial con {product}, se recomienda programar el pedido de forma anticipada. "
            f"Segmento: {segment}. Estimamos que comprará en aproximadamente {days:.0f} días."
        )
    elif prob30 >= settings.UMBRAL_MEDIA:
        return (
            f"{name} muestra una probabilidad moderada ({prob30:.0%}) de recompra en 30 días. "
            f"Su producto principal es {product} con un ciclo de ~{interval:.0f} días. "
            f"Recomendamos contactar al cliente para confirmar necesidades. "
            f"Segmento: {segment}."
        )
    else:
        return (
            f"{name} tiene baja probabilidad de recompra ({prob30:.0%}) en los próximos 30 días. "
            f"Han pasado más tiempo del habitual desde su última compra. "
            f"Se recomienda una campaña de reactivación enfocada en {product}. "
            f"Segmento: {segment}."
        )


@router.get("/predict/{job_id}")
async def predict_all(
    job_id: str,
    top_n: int = Query(50, ge=1, le=500),
    sort_by: str = Query("prob_30d", regex="^(prob_7d|prob_14d|prob_30d|prob_60d|dias_estimados|revenue_total)$"),
):
    """Genera predicciones para todos los clientes del dataset."""
    features_path = os.path.join(settings.OUTPUTS_DIR, f"{job_id}_features.parquet")
    if not os.path.exists(features_path):
        raise HTTPException(404, "Features no encontradas. Sube y entrena primero.")

    features = pd.read_parquet(features_path)
    models = _load_models(job_id)
    if not models:
        raise HTTPException(400, "Modelos no entrenados. Ejecuta POST /api/train")

    feature_cols = [c for c in features.columns if c not in
                    ("cliente", "primera_compra", "ultima_compra", "cluster",
                     "segmento", "producto_top")]

    predictions = []
    for _, row in features.iterrows():
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

        dias = round(float(models["reg"].predict(X)[0]), 1) if "reg" in models else 0
        p30 = probs["30d"]

        if p30 >= settings.UMBRAL_ALTA:
            accion, prioridad = "PROGRAMAR_PEDIDO", "ALTA"
        elif p30 >= settings.UMBRAL_MEDIA:
            accion, prioridad = "CONTACTAR_CLIENTE", "MEDIA"
        else:
            accion, prioridad = "REACTIVAR_CAMPANA", "BAJA"

        segment = row.get("segmento", f"Cluster {int(row.get('cluster', 0))}")
        product = row.get("producto_top", "N/A")
        interval = row.get("intervalo_promedio_dias", 0)

        # Fill rate: ratio de compras cumplidas vs esperadas
        expected = row.get("antiguedad_dias", 1) / max(interval, 1) if interval > 0 else 0
        actual = row.get("total_compras", 0)
        fill = round(min(actual / max(expected, 1), 1.0) * 100, 1) if expected > 0 else 0

        predictions.append({
            "cliente": row["cliente"],
            "segmento": segment,
            "cluster": int(row.get("cluster", 0)),
            "prob_7d": probs["7d"],
            "prob_14d": probs["14d"],
            "prob_30d": probs["30d"],
            "prob_60d": probs["60d"],
            "dias_estimados": dias,
            "ultima_compra": str(row.get("ultima_compra", ""))[:10],
            "total_ordenes": int(row.get("total_compras", 0)),
            "revenue_total": round(float(row.get("total_revenue", 0)), 2),
            "ticket_promedio": round(float(row.get("avg_ticket", 0)), 2),
            "intervalo_dias": round(float(interval), 1),
            "fill_rate": fill,
            "producto_top": product,
            "accion": accion,
            "prioridad": prioridad,
            "interpretacion": _build_interpretation(
                row["cliente"], p30, dias, segment, product, interval
            ),
        })

    reverse = sort_by != "dias_estimados"
    predictions.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
    predictions = predictions[:top_n]

    # Resumen
    total = len(features)
    alta = sum(1 for p in predictions if p["prioridad"] == "ALTA")
    media = sum(1 for p in predictions if p["prioridad"] == "MEDIA")
    baja = sum(1 for p in predictions if p["prioridad"] == "BAJA")

    return {
        "status": "success",
        "job_id": job_id,
        "total_clients": total,
        "modelo": "XGBoost Ensemble",
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
    """Descarga predicciones como CSV."""
    features_path = os.path.join(settings.OUTPUTS_DIR, f"{job_id}_features.parquet")
    if not os.path.exists(features_path):
        raise HTTPException(404, "Features no encontradas")

    resp = await predict_all(job_id, top_n=500, sort_by="prob_30d")
    df = pd.DataFrame(resp["predictions"])

    csv_path = os.path.join(settings.OUTPUTS_DIR, f"{job_id}_predicciones.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return StreamingResponse(
        open(csv_path, "rb"),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predicciones_{job_id}.csv"},
    )


@router.get("/thresholds/{job_id}")
async def threshold_analysis(job_id: str, horizonte: int = 30):
    """Analiza diferentes umbrales de decisión."""
    dataset_path = os.path.join(settings.OUTPUTS_DIR, f"{job_id}_dataset.parquet")
    if not os.path.exists(dataset_path):
        raise HTTPException(404, "Dataset no encontrado")

    models = _load_models(job_id)
    dataset = pd.read_parquet(dataset_path)
    feature_cols = [c for c in dataset.columns if c not in
                    ("cliente", "primera_compra", "ultima_compra", "cluster",
                     "segmento", "producto_top", "folio", "fecha", "producto",
                     "next_purchase_date", "days_to_next",
                     "recompra_7d", "recompra_14d", "recompra_30d", "recompra_60d")]

    X = dataset[feature_cols].fillna(0)
    if "scaler" in models:
        X = models["scaler"].transform(X)

    target = f"recompra_{horizonte}d"
    if target not in dataset.columns:
        raise HTTPException(400, f"Horizonte {horizonte}d no disponible")

    y = dataset[target].values
    key = f"{horizonte}d"
    if key not in models:
        raise HTTPException(400, "Modelo no entrenado para este horizonte")

    y_prob = models[key].predict_proba(X)[:, 1]

    from sklearn.metrics import precision_score, recall_score
    analysis = []
    for u in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_prob >= u).astype(int)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        act = int(y_pred.sum())
        analysis.append({
            "umbral": round(u, 2), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4),
            "clientes_activados": act,
            "pct_activados": round(act / len(y) * 100, 1),
        })

    best = max(analysis, key=lambda x: x["f1"])
    return {
        "horizonte": f"{horizonte}d",
        "umbral_optimo": best["umbral"],
        "mejor_f1": best["f1"],
        "analisis": analysis,
    }


def _load_models(job_id: str) -> dict:
    models = {}
    files = {
        "scaler": "scaler.pkl", "kmeans": "kmeans.pkl",
        "7d": "xgb_clf_7d.pkl", "14d": "xgb_clf_14d.pkl",
        "30d": "xgb_clf_30d.pkl", "60d": "xgb_clf_60d.pkl",
        "reg": "xgb_reg_days.pkl",
    }
    for key, fname in files.items():
        path = os.path.join(settings.MODELS_DIR, f"{job_id}_{fname}")
        if not os.path.exists(path):
            path = os.path.join(settings.MODELS_DIR, fname)
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models
