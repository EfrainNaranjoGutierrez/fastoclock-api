from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "FastOclock"
    APP_ENV: str = "development"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "*"]

    UPLOAD_DIR: str = "uploads"
    MODELS_DIR: str = "models"
    OUTPUTS_DIR: str = "outputs"

    UMBRAL_ALTA: float = 0.75
    UMBRAL_MEDIA: float = 0.40
    UMBRAL_BAJA: float = 0.15

    HORIZONTES: List[int] = [7, 14, 30, 60]
    NEURAL_EPOCHS: int = 50
    TEST_SIZE: float = 0.2

    class Config:
        env_file = ".env"


settings = Settings()
