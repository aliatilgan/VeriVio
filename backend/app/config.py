from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_TITLE: str = "VeriVio Backend API"
    API_DESCRIPTION: str = "Veri Analizi Platformu"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    LOG_FILE: str = "logs/verivio.log"
    LOG_LEVEL: str = "INFO"

settings = Settings()