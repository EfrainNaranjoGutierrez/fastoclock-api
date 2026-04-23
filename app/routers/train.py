import os
import json
from fastapi import APIRouter, HTTPException
from app.schemas.models import TrainRequest
from app.config import settings
from app.ml.pipeline import TrainPipeline
from app.services.storage import load_parquet

router = APIRouter()

# Cache en memoria mientras el proceso está vivo
_features_cache = {}

def get_cached_features(job_id: str):
    return _features_cache.get(job_id)

def set_cached_features(job_id: str, features):
    _features_cache[job_id] = features

@router.post("/train")
async def train_models(request: TrainRequest):
    dataset_path = os.path.join(settings.OUTPUTS_DIR, f"{request.job_id}_dataset.parquet")
    if not os.path.exists(dataset_path):
        raise HTTPException(404, f"Dataset no encontrado para job_id={request.job_id}")

    try:
        pipeline = TrainPipeline(job_id=request.job_id)
        results = pipeline.run(
            dataset_path=dataset_path,
            horizontes=request.horizontes,
            incluir_neural=request.incluir_neural,
            grid_search=request.grid_search,
        )

        # Cache features en memoria
        features_path = os.path.join(settings.OUTPUTS_DIR, f"{request.job_id}_features.parquet")
        if os.path.exists(features_path):
            features = load_parquet(f"{request.job_id}_features.parquet")
            set_cached_features(request.job_id, features)

    except Exception as e:
        raise HTTPException(500, f"Error en entrenamiento: {str(e)}")

    return {
        "status": "success",
        "job_id": request.job_id,
        "models_trained": results["models_trained"],
        "metrics": results["metrics"],
        "accuracy_summary": results["accuracy_summary"],
    }