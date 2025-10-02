"""
VeriVio Backend - Ana FastAPI Uygulaması
Modüler veri analizi sistemi için REST API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

# Konfigürasyon ve yardımcı modüller
from .config import settings
from .models import AnalysisRequest, AnalysisResponse, ErrorResponse
from .utils import setup_logging, validate_file_type, save_uploaded_file

# Analiz modülleri
from modules.data_processing.cleaner import DataCleaner
from modules.descriptive_stats.calculator import DescriptiveStatsCalculator
from modules.visualization.plotter import DataPlotter
from modules.hypothesis_tests.tester import HypothesisTester
from modules.regression.analyzer import RegressionAnalyzer

# Domain-specific modüller
from modules.domain_specific.finance import FinanceAnalyzer
from modules.domain_specific.marketing import MarketingAnalyzer
from modules.domain_specific.healthcare import HealthcareAnalyzer

# Logging kurulumu
setup_logging()
logger = logging.getLogger(__name__)

# FastAPI uygulaması
app = FastAPI(
    title="VeriVio Analytics API",
    description="Modüler veri analizi ve görselleştirme sistemi",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statik dosyalar
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global değişkenler
uploaded_files = {}  # Geçici dosya depolama
analysis_results = {}  # Analiz sonuçları cache


@app.get("/")
async def root():
    """Ana endpoint - API durumu"""
    return {
        "message": "VeriVio Analytics API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Sağlık kontrolü endpoint'i"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "data_processing": True,
            "descriptive_stats": True,
            "visualization": True,
            "hypothesis_tests": True,
            "regression": True
        }
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = Form(default="auto")
):
    """
    Dosya yükleme endpoint'i
    Desteklenen formatlar: CSV, Excel, JSON
    """
    try:
        # Dosya türü validasyonu
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Desteklenmeyen dosya türü. CSV, Excel veya JSON dosyası yükleyin."
            )
        
        # Dosyayı kaydet
        file_id = save_uploaded_file(file)
        
        # Dosya bilgilerini sakla
        uploaded_files[file_id] = {
            "filename": file.filename,
            "file_type": file_type,
            "upload_time": datetime.now().isoformat(),
            "size": file.size
        }
        
        logger.info(f"Dosya yüklendi: {file.filename} (ID: {file_id})")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "message": "Dosya başarıyla yüklendi",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Dosya yükleme hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dosya yükleme hatası: {str(e)}")


@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """
    Ana analiz endpoint'i
    Seçilen analiz türüne göre veri analizi yapar
    """
    try:
        # Dosya kontrolü
        if request.file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="Dosya bulunamadı")
        
        # Analiz türüne göre işlem
        result = await perform_analysis(request)
        
        # Sonucu cache'le
        analysis_id = f"{request.file_id}_{request.analysis_type}_{datetime.now().timestamp()}"
        analysis_results[analysis_id] = result
        
        logger.info(f"Analiz tamamlandı: {request.analysis_type} (ID: {analysis_id})")
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="success",
            results=result,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Analiz hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")


@app.get("/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Analiz sonuçlarını getir"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analiz sonucu bulunamadı")
    
    return analysis_results[analysis_id]


@app.get("/visualizations/{analysis_id}")
async def get_visualization(analysis_id: str):
    """Görselleştirme dosyasını getir"""
    try:
        # Görselleştirme dosyası yolu
        viz_path = f"static/visualizations/{analysis_id}.png"
        
        if not os.path.exists(viz_path):
            raise HTTPException(status_code=404, detail="Görselleştirme bulunamadı")
        
        return FileResponse(viz_path, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Görselleştirme hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Görselleştirme hatası: {str(e)}")


async def perform_analysis(request: AnalysisRequest) -> Dict[str, Any]:
    """Analiz türüne göre uygun modülü çağır"""
    
    # Veri yükleme ve temizleme
    cleaner = DataCleaner()
    data = cleaner.load_data(request.file_id)
    
    if request.clean_data:
        data = cleaner.clean_data(data, request.cleaning_options or {})
    
    # Analiz türüne göre işlem
    if request.analysis_type == "descriptive":
        calculator = DescriptiveStatsCalculator()
        return calculator.calculate(data, request.parameters or {})
    
    elif request.analysis_type == "visualization":
        plotter = DataPlotter()
        return plotter.create_plot(data, request.parameters or {})
    
    elif request.analysis_type == "hypothesis_test":
        tester = HypothesisTester()
        return tester.run_test(data, request.parameters or {})
    
    elif request.analysis_type == "regression":
        analyzer = RegressionAnalyzer()
        return analyzer.analyze(data, request.parameters or {})
    
    elif request.analysis_type == "finance":
        analyzer = FinanceAnalyzer()
        return analyzer.get_finance_summary(data, request.parameters or {})
    
    elif request.analysis_type == "marketing":
        analyzer = MarketingAnalyzer()
        return analyzer.get_marketing_summary(data, request.parameters or {})
    
    elif request.analysis_type == "healthcare":
        analyzer = HealthcareAnalyzer()
        return analyzer.get_healthcare_summary(data, request.parameters or {})
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen analiz türü: {request.analysis_type}"
        )


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Yüklenen dosyayı sil"""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="Dosya bulunamadı")
        
        # Dosyayı sil
        file_path = f"uploads/{file_id}"
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Cache'den kaldır
        del uploaded_files[file_id]
        
        logger.info(f"Dosya silindi: {file_id}")
        
        return {"message": "Dosya başarıyla silindi", "status": "success"}
        
    except Exception as e:
        logger.error(f"Dosya silme hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dosya silme hatası: {str(e)}")


@app.get("/files")
async def list_files():
    """Yüklenen dosyaları listele"""
    return {
        "files": uploaded_files,
        "count": len(uploaded_files)
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )