from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    records: int
    clients: int
    date_range: str
    columns_detected: Dict[str, str]
    training_samples: int


class TrainRequest(BaseModel):
    job_id: str
    horizontes: List[int] = [7, 14, 30, 60]
    incluir_neural: bool = True
    grid_search: bool = False


class TrainResponse(BaseModel):
    job_id: str
    status: str
    models_trained: List[str]
    metrics: Dict[str, Any]
    accuracy_summary: Dict[str, Any]


class ClientPrediction(BaseModel):
    cliente: str
    segmento: str
    cluster: int
    prob_7d: float
    prob_14d: float
    prob_30d: float
    prob_60d: float
    dias_estimados: float
    ultima_compra: str
    total_ordenes: int
    revenue_total: float
    ticket_promedio: float
    intervalo_dias: float
    fill_rate: float
    producto_top: str
    accion: str
    prioridad: str
    interpretacion: str


class PredictResponse(BaseModel):
    job_id: str
    total_clients: int
    modelo: str
    predictions: List[ClientPrediction]
    resumen: Dict[str, Any]


class ThresholdAnalysis(BaseModel):
    umbral: float
    precision: float
    recall: float
    f1: float
    clientes_activados: int
    pct_activados: float
