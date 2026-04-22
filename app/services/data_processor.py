import os
import pandas as pd
import numpy as np
from app.config import settings


class DataProcessor:
    """Procesa cualquier archivo de historial de ventas con detección automática de columnas."""

    COLUMN_MAPPINGS = {
        "fecha": ["createdat", "created_at", "fecha", "date", "order_date", "fecha_pedido",
                  "fecha_orden", "orderdate", "invoice_date", "invoicedate"],
        "cliente": ["customer", "cliente", "client", "nombre_cliente", "customer_name",
                    "buyer", "comprador", "razon_social"],
        "total": ["total", "monto", "amount", "revenue", "venta", "importe",
                  "grand_total", "order_total", "sale_amount"],
        "folio": ["folio", "order_id", "id", "pedido", "invoice", "factura",
                  "order_number", "numero_pedido"],
        "producto": ["producto", "product", "descripcion", "descripción", "sku",
                     "item", "product_name", "nombre_producto", "comment"],
        "unidades": ["unidades", "units", "quantity", "cantidad", "qty", "pieces"],
        "precio": ["precio", "price", "preciounitario", "precio_unitario",
                   "unit_price", "precio_unit"],
        "sucursal": ["branch", "sucursal", "tienda", "store", "location", "sede"],
        "telefono": ["phone", "telefono", "tel", "celular", "mobile"],
        "metodo_pago": ["paymentmethod", "payment_method", "pago", "metodo_pago",
                       "forma_pago", "payment"],
    }

    def process(self, filepath: str, job_id: str) -> dict:
        """Carga, detecta columnas, limpia y genera features + dataset."""
        df = self._load_file(filepath)
        col_map = self._detect_columns(df)
        df = df.rename(columns=col_map)
        df = self._clean(df)

        # Feature Engineering
        features = self._build_features(df)

        # Build training dataset
        dataset = self._build_dataset(df, features)

        # Save
        os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
        df.to_parquet(os.path.join(settings.OUTPUTS_DIR, f"{job_id}_clean.parquet"), index=False)
        features.to_parquet(os.path.join(settings.OUTPUTS_DIR, f"{job_id}_features.parquet"), index=False)
        dataset.to_parquet(os.path.join(settings.OUTPUTS_DIR, f"{job_id}_dataset.parquet"), index=False)

        n_clients = df["cliente"].nunique() if "cliente" in df.columns else 0
        d_min = df["fecha"].min().strftime("%Y-%m-%d") if "fecha" in df.columns else "N/A"
        d_max = df["fecha"].max().strftime("%Y-%m-%d") if "fecha" in df.columns else "N/A"

        return {
            "records": len(df),
            "clients": n_clients,
            "date_range": f"{d_min} / {d_max}",
            "columns_detected": {v: k for k, v in col_map.items()},
            "training_samples": len(dataset),
        }

    def _load_file(self, filepath: str) -> pd.DataFrame:
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        xls = pd.ExcelFile(filepath)
        frames = [pd.read_excel(filepath, sheet_name=s) for s in xls.sheet_names]
        return pd.concat(frames, ignore_index=True)

    def _detect_columns(self, df: pd.DataFrame) -> dict:
        col_map = {}
        for orig in df.columns:
            low = orig.lower().strip().replace(" ", "_")
            for standard, variants in self.COLUMN_MAPPINGS.items():
                if low in variants:
                    col_map[orig] = standard
                    break
        return col_map

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
            df = df.dropna(subset=["fecha"])
        if "cliente" in df.columns:
            df["cliente"] = df["cliente"].astype(str).str.strip().str.upper()
            df = df[df["cliente"].str.len() > 1]
        if "total" in df.columns:
            df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0)
        if "unidades" in df.columns:
            df["unidades"] = pd.to_numeric(df["unidades"], errors="coerce").fillna(1)
        if "folio" in df.columns:
            df = df.drop_duplicates(subset=["folio"], keep="first")

        df = df.sort_values("fecha").reset_index(drop=True) if "fecha" in df.columns else df
        return df

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "cliente" not in df.columns or "fecha" not in df.columns:
            raise ValueError("Se requieren al menos columnas de fecha y cliente")

        today = df["fecha"].max()

        agg_dict = {
            "total_compras": ("fecha", "count"),
            "primera_compra": ("fecha", "min"),
            "ultima_compra": ("fecha", "max"),
            "total_revenue": ("total", "sum") if "total" in df.columns else ("fecha", "count"),
            "avg_ticket": ("total", "mean") if "total" in df.columns else ("fecha", "count"),
            "std_ticket": ("total", "std") if "total" in df.columns else ("fecha", "count"),
            "mediana_ticket": ("total", "median") if "total" in df.columns else ("fecha", "count"),
            "max_ticket": ("total", "max") if "total" in df.columns else ("fecha", "count"),
        }
        features = df.groupby("cliente").agg(**agg_dict).reset_index()

        if "unidades" in df.columns:
            u = df.groupby("cliente")["unidades"].agg(["sum", "mean"]).reset_index()
            u.columns = ["cliente", "total_unidades", "avg_unidades"]
            features = features.merge(u, on="cliente", how="left")
        else:
            features["total_unidades"] = features["total_compras"]
            features["avg_unidades"] = 1

        if "producto" in df.columns:
            sku_count = df.groupby("cliente")["producto"].nunique().reset_index()
            sku_count.columns = ["cliente", "skus_distintos"]
            features = features.merge(sku_count, on="cliente", how="left")
            top_prod = df.groupby("cliente")["producto"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A").reset_index()
            top_prod.columns = ["cliente", "producto_top"]
            features = features.merge(top_prod, on="cliente", how="left")
        else:
            features["skus_distintos"] = 1
            features["producto_top"] = "N/A"

        features["recencia_dias"] = (today - features["ultima_compra"]).dt.days
        features["antiguedad_dias"] = (features["ultima_compra"] - features["primera_compra"]).dt.days

        def _avg_interval(g):
            f = g.sort_values()
            return f.diff().dt.days.dropna().mean() if len(f) >= 2 else np.nan

        def _std_interval(g):
            f = g.sort_values()
            return f.diff().dt.days.dropna().std() if len(f) >= 3 else np.nan

        features["intervalo_promedio_dias"] = features["cliente"].map(
            df.groupby("cliente")["fecha"].apply(_avg_interval))
        features["std_intervalo_dias"] = features["cliente"].map(
            df.groupby("cliente")["fecha"].apply(_std_interval))
        features["cv_intervalo"] = (
            features["std_intervalo_dias"] / features["intervalo_promedio_dias"]).fillna(0)

        def _slope(g):
            g = g.sort_values("fecha")
            g["m"] = (g["fecha"].dt.year - g["fecha"].dt.year.min()) * 12 + g["fecha"].dt.month
            if g["m"].nunique() < 2:
                return 0.0
            col = "total" if "total" in g.columns else "fecha"
            monthly = g.groupby("m")[col].sum() if col == "total" else g.groupby("m").size()
            return np.polyfit(np.arange(len(monthly)), monthly.values, 1)[0]

        features["tendencia_volumen"] = features["cliente"].map(
            df.groupby("cliente").apply(_slope))
        features["ratio_diversificacion"] = (
            features["skus_distintos"] / features["total_compras"]).clip(0, 1)
        features["engagement_score"] = (
            features["total_compras"] * 0.3
            + (1 / (features["recencia_dias"] + 1)) * 100 * 0.3
            + features["total_revenue"] / features["total_revenue"].max() * 100 * 0.2
            + features["skus_distintos"] / features["skus_distintos"].max() * 100 * 0.2
        )

        features = features.fillna(0)
        return features

    def _build_dataset(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["cliente", "fecha"])
    d = d.copy()
    d["next_purchase_date"] = d.groupby("cliente")["fecha"].shift(-1)
    d["days_to_next"] = (d["next_purchase_date"] - d["fecha"]).dt.days
    d = d.dropna(subset=["days_to_next"]).copy()

    for h in [7, 14, 30, 60]:
        d[f"recompra_{h}d"] = (d["days_to_next"] <= h).astype(int)

    numeric_feat_cols = [c for c in features.select_dtypes(include=["number"]).columns]
    merge_cols = ["cliente"] + [c for c in numeric_feat_cols if c not in d.columns]

    return d.merge(features[merge_cols], on="cliente", how="left")