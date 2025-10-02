"""
VeriVio Backend Konfigürasyon Ayarları
"""

import os
from typing import List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Uygulama ayarları"""
    
    # API Ayarları
    API_TITLE: str = "VeriVio Analytics API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Modüler veri analizi ve görselleştirme sistemi"
    
    # Server Ayarları
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS Ayarları
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    
    # Dosya Ayarları
    UPLOAD_DIR: str = "uploads"
    STATIC_DIR: str = "static"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".xlsx", ".xls", ".json"]
    
    # Görselleştirme Ayarları
    VISUALIZATION_DIR: str = "static/visualizations"
    PLOT_DPI: int = 300
    PLOT_FORMAT: str = "png"
    PLOT_WIDTH: int = 12
    PLOT_HEIGHT: int = 8
    
    # Logging Ayarları
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_FILE: str = "verivio.log"
    
    # Cache Ayarları
    CACHE_TTL: int = 3600  # 1 saat
    MAX_CACHE_SIZE: int = 100
    
    # Analiz Ayarları
    DEFAULT_CONFIDENCE_LEVEL: float = 0.95
    MAX_ROWS_FOR_ANALYSIS: int = 100000
    
    # Database Ayarları (gelecekte kullanım için)
    DATABASE_URL: str = "sqlite:///./verivio.db"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Klasör oluşturma fonksiyonu
def create_directories():
    """Gerekli klasörleri oluştur"""
    directories = [
        settings.UPLOAD_DIR,
        settings.STATIC_DIR,
        settings.VISUALIZATION_DIR,
        settings.LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Uygulama başlangıcında klasörleri oluştur
create_directories()