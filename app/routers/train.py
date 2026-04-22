import os
from fastapi import APIRouter, HTTPException
from app.schemas.models import TrainRequest
from app.config import settings
from app.ml.pipeline import TrainPipeline

router = APIRouter()


@router.post("/train")
async def train_models(request: TrainRequest):
    """Entrena XGBoost + Red Neuronal con el dataset subido."""
    dataset_path = os.path.join(settings.OUTPUTS_DIR, f"{request.job_id}_dataset.parquet")

    if not os.path.exists(dataset_path):
        raise HTTPException(404, f"Dataset no encontrado para job_id={request.job_id}. Sube datos primero.")

    try:
        pipeline = TrainPipeline(job_id=request.job_id)
        results = pipeline.run(
            dataset_path=dataset_path,
            horizontes=request.horizontes,
            incluir_neural=request.incluir_neural,
            grid_search=request.grid_search,
        )
    except Exception as e:
        raise HTTPException(500, f"Error en entrenamiento: {str(e)}")

    return {
        "status": "success",
        "job_id": request.job_id,
        "models_trained": results["models_trained"],
        "metrics": results["metrics"],
        "accuracy_summary": results["accuracy_summary"],
    }
