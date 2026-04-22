from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, train, predict, health
from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Motor predictivo de recompra. Sube tu historial de ventas y obtén pronósticos por cliente.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(train.router, prefix="/api", tags=["Training"])
app.include_router(predict.router, prefix="/api", tags=["Predictions"])
