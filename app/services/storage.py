# FastOclock Storage Service v2
import os
import io
import logging
import pandas as pd
from app.config import settings

logger = logging.getLogger("api")


def get_supabase_client():
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    except Exception as e:
        logger.warning(f"Supabase no disponible: {e}")
        return None


def save_parquet(df: pd.DataFrame, filename: str) -> bool:
    os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
    local_path = os.path.join(settings.OUTPUTS_DIR, filename)
    df.to_parquet(local_path, index=False)
    logger.info(f"Guardado en disco: {local_path}")

    sb = get_supabase_client()
    if sb:
        try:
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            data = buffer.read()
            try:
                sb.storage.from_(settings.SUPABASE_BUCKET).upload(
                    path=filename, file=data,
                    file_options={"content-type": "application/octet-stream"})
            except Exception:
                sb.storage.from_(settings.SUPABASE_BUCKET).update(
                    path=filename, file=data,
                    file_options={"content-type": "application/octet-stream"})
            logger.info(f"Guardado en Supabase: {filename}")
        except Exception as e:
            logger.warning(f"No se pudo guardar en Supabase: {e}")
    return True


def load_parquet(filename: str) -> pd.DataFrame:
    local_path = os.path.join(settings.OUTPUTS_DIR, filename)
    if os.path.exists(local_path):
        logger.info(f"Cargando desde disco: {local_path}")
        return pd.read_parquet(local_path)

    sb = get_supabase_client()
    if sb:
        try:
            data = sb.storage.from_(settings.SUPABASE_BUCKET).download(filename)
            df = pd.read_parquet(io.BytesIO(data))
            os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
            df.to_parquet(local_path, index=False)
            logger.info(f"Descargado de Supabase: {filename}")
            return df
        except Exception as e:
            logger.warning(f"No encontrado en Supabase: {e}")

    raise FileNotFoundError(f"Archivo no encontrado: {filename}")


def find_latest_features(job_id: str) -> pd.DataFrame:
    outputs_dir = settings.OUTPUTS_DIR
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Buscar el job específico en disco
    specific = os.path.join(outputs_dir, f"{job_id}_features.parquet")
    if os.path.exists(specific):
        logger.info(f"Features encontradas: {specific}")
        return pd.read_parquet(specific)

    # 2. Buscar cualquier features en disco (el más reciente)
    files = [f for f in os.listdir(outputs_dir)
             if f.endswith("_features.parquet")]
    if files:
        files.sort(
            key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)),
            reverse=True
        )
        latest = os.path.join(outputs_dir, files[0])
        logger.info(f"Usando features más recientes del disco: {files[0]}")
        return pd.read_parquet(latest)

    # 3. Buscar en Supabase
    sb = get_supabase_client()
    if sb:
        try:
            all_files = sb.storage.from_(settings.SUPABASE_BUCKET).list()
            feature_files = sorted(
                [f["name"] for f in all_files
                 if f["name"].endswith("_features.parquet")],
                reverse=True
            )
            if feature_files:
                data = sb.storage.from_(settings.SUPABASE_BUCKET).download(
                    feature_files[0])
                df = pd.read_parquet(io.BytesIO(data))
                save_path = os.path.join(outputs_dir, feature_files[0])
                df.to_parquet(save_path, index=False)
                logger.info(f"Features de Supabase: {feature_files[0]}")
                return df
        except Exception as e:
            logger.error(f"Error Supabase: {e}")

    raise FileNotFoundError("No se encontraron features disponibles")