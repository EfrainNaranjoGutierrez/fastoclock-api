import os
import io
import logging
import pandas as pd
from app.config import settings
from supabase import create_client

logger = logging.getLogger("api")


def get_supabase_client():
    """Retorna cliente de Supabase si está configurado."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        return None
    try:
        return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    except Exception as e:
        logger.warning(f"Supabase no disponible: {e}")
        return None


def save_parquet(df: pd.DataFrame, filename: str) -> bool:
    """Guarda un DataFrame como parquet en Supabase Storage y en disco local."""
    # Siempre guardar en disco local también
    os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
    local_path = os.path.join(settings.OUTPUTS_DIR, filename)
    df.to_parquet(local_path, index=False)
    logger.info(f"Guardado local: {local_path}")

    # Intentar guardar en Supabase
    sb = get_supabase_client()
    if sb is None:
        logger.warning("Supabase no configurado, solo guardado local")
        return True

    try:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        data = buffer.read()

        # Intentar subir, si ya existe sobreescribir
        try:
            sb.storage.from_(settings.SUPABASE_BUCKET).upload(
                path=filename,
                file=data,
                file_options={"content-type": "application/octet-stream"}
            )
        except Exception:
            sb.storage.from_(settings.SUPABASE_BUCKET).update(
                path=filename,
                file=data,
                file_options={"content-type": "application/octet-stream"}
            )

        logger.info(f"Guardado en Supabase: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error guardando en Supabase: {e}")
        return True  # El local ya fue guardado


def load_parquet(filename: str) -> pd.DataFrame:
    """Carga un parquet desde disco local o Supabase Storage."""
    # Primero intentar disco local
    local_path = os.path.join(settings.OUTPUTS_DIR, filename)
    if os.path.exists(local_path):
        logger.info(f"Cargando desde disco: {local_path}")
        return pd.read_parquet(local_path)

    # Si no está en disco, buscar en Supabase
    sb = get_supabase_client()
    if sb is None:
        raise FileNotFoundError(f"Archivo no encontrado: {filename}")

    try:
        data = sb.storage.from_(settings.SUPABASE_BUCKET).download(filename)
        df = pd.read_parquet(io.BytesIO(data))

        # Guardar en disco para acceso rápido posterior
        os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
        df.to_parquet(local_path, index=False)
        logger.info(f"Descargado de Supabase y cacheado: {filename}")
        return df
    except Exception as e:
        raise FileNotFoundError(f"No se pudo cargar {filename}: {e}")


def find_latest_features(job_id: str) -> pd.DataFrame:
    """Busca features del job, local o en Supabase."""
    filename = f"{job_id}_features.parquet"

    # Intentar cargar el archivo específico del job
    try:
        return load_parquet(filename)
    except FileNotFoundError:
        pass

    # Buscar el más reciente en disco local
    outputs_dir = settings.OUTPUTS_DIR
    if os.path.exists(outputs_dir):
        files = [f for f in os.listdir(outputs_dir) if f.endswith("_features.parquet")]
        if files:
            files.sort(
                key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)),
                reverse=True
            )
            logger.info(f"Usando features más recientes: {files[0]}")
            return pd.read_parquet(os.path.join(outputs_dir, files[0]))

    # Buscar en Supabase cualquier features disponible
    sb = get_supabase_client()
    if sb:
        try:
            files = sb.storage.from_(settings.SUPABASE_BUCKET).list()
            feature_files = [f["name"] for f in files if f["name"].endswith("_features.parquet")]
            if feature_files:
                feature_files.sort(reverse=True)
                logger.info(f"Usando features de Supabase: {feature_files[0]}")
                data = sb.storage.from_(settings.SUPABASE_BUCKET).download(feature_files[0])
                return pd.read_parquet(io.BytesIO(data))
        except Exception as e:
            logger.error(f"Error buscando en Supabase: {e}")

    raise FileNotFoundError("No se encontraron features disponibles")