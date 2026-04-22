import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import settings
from app.services.data_processor import DataProcessor

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Sube cualquier archivo Excel o CSV con historial de ventas."""
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(400, "Formato no soportado. Usa .xlsx, .xls o .csv")

    job_id = uuid.uuid4().hex[:8]
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        processor = DataProcessor()
        result = processor.process(filepath, job_id)
    except Exception as e:
        raise HTTPException(400, f"Error procesando archivo: {str(e)}")

    return {
        "status": "success",
        "job_id": job_id,
        "filename": file.filename,
        "records": result["records"],
        "clients": result["clients"],
        "date_range": result["date_range"],
        "columns_detected": result["columns_detected"],
        "training_samples": result["training_samples"],
        "message": f"Listo. {result['records']} registros de {result['clients']} clientes procesados. Entrena con POST /api/train",
    }
