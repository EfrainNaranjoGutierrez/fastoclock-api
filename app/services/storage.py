import os
import io
import logging
import pandas as pd
from app.config import settings
import app.cache as cache

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
    cache.set(filename, df.copy())
    logger.info(f"Guardado en cache: {filename}")

    os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
    local_path = os.path.join(settings.OUTPUTS_DIR, filename)
    df.to_parquet(local_path, index=False)

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
    cached = cache.get(filename)
    if cached is not None:
        return cached

    local_path = os.path.join(settings.OUTPUTS_DIR, filename)
    if os.path.exists(local_path):
        df = pd.read_parquet(local_path)
        cache.set(filename, df)
        return df

    sb = get_supabase_client()
    if sb:
        try:
            data = sb.storage.from_(settings.SUPABASE_BUCKET).download(filename)
            df = pd.read_parquet(io.BytesIO(data))
            cache.set(filename, df)
            return df
        except Exception as e:
            logger.warning(f"No encontrado en Supabase: {e}")

    raise FileNotFoundError(f"Archivo no encontrado: {filename}")


def find_latest_features(job_id: str) -> pd.DataFrame:
    filename = f"{job_id}_features.parquet"
    try:
        return load_parquet(filename)
    except FileNotFoundError:
        pass

    cached_keys = [k for k in cache.keys() if k.endswith("_features.parquet")]
    if cached_keys:
        return cache.get(sorted(cached_keys, reverse=True)[0])

    outputs_dir = settings.OUTPUTS_DIR
    if os.path.exists(outputs_dir):
        files = [f for f in os.listdir(outputs_dir) if f.endswith("_features.parquet")]
        if files:
            files.sort(key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)), reverse=True)
            df = pd.read_parquet(os.path.join(outputs_dir, files[0]))
            cache.set(files[0], df)
            return df

    sb = get_supabase_client()
    if sb:
        try:
            files = sb.storage.from_(settings.SUPABASE_BUCKET).list()
            feature_files = sorted([f["name"] for f in files if f["name"].endswith("_features.parquet")], reverse=True)
            if feature_files:
                data = sb.storage.from_(settings.SUPABASE_BUCKET).download(feature_files[0])
                df = pd.read_parquet(io.BytesIO(data))
                cache.set(feature_files[0], df)
                return df
        except Exception as e:
            logger.error(f"Error Supabase: {e}")

    raise FileNotFoundError("No se encontraron features disponibles")