import os
from datetime import datetime
from fastapi import APIRouter
from app.config import settings

router = APIRouter()


@router.get("/health")
def health_check():
    model_files = ["scaler.pkl", "kmeans.pkl", "xgb_clf_7d.pkl",
                   "xgb_clf_14d.pkl", "xgb_clf_30d.pkl", "xgb_clf_60d.pkl",
                   "xgb_reg_days.pkl", "neural_model.keras"]

    models = {}
    for f in model_files:
        path = os.path.join(settings.MODELS_DIR, f)
        exists = os.path.exists(path)
        models[f] = {
            "loaded": exists,
            "size_kb": round(os.path.getsize(path) / 1024, 1) if exists else 0,
        }

    return {
        "service": settings.APP_NAME,
        "status": "ready" if any(m["loaded"] for m in models.values()) else "awaiting_data",
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "umbrales": {
            "alta": settings.UMBRAL_ALTA,
            "media": settings.UMBRAL_MEDIA,
            "baja": settings.UMBRAL_BAJA,
        },
    }
