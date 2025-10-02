"""
VeriVio Backend Yardımcı Fonksiyonlar
"""

import os
import logging
import uuid
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import UploadFile, HTTPException
import pandas as pd
import json
from pathlib import Path

from .config import settings


def setup_logging():
    """Logging sistemini kur"""
    
    # Log klasörünü oluştur
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    
    # Log formatı
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Logging konfigürasyonu
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=log_format,
        handlers=[
            logging.FileHandler(
                os.path.join(settings.LOG_DIR, settings.LOG_FILE),
                encoding='utf-8'
            ),
            logging.StreamHandler()
        ]
    )


def validate_file_type(filename: str) -> bool:
    """Dosya türünü doğrula"""
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower()
    return file_extension in settings.ALLOWED_FILE_TYPES


def generate_file_id() -> str:
    """Benzersiz dosya ID'si oluştur"""
    return str(uuid.uuid4())


def get_file_hash(file_content: bytes) -> str:
    """Dosya hash'i hesapla"""
    return hashlib.md5(file_content).hexdigest()


async def save_uploaded_file(file: UploadFile) -> str:
    """Yüklenen dosyayı kaydet"""
    
    # Dosya boyutu kontrolü
    file_content = await file.read()
    if len(file_content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Dosya boyutu çok büyük. Maksimum {settings.MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Dosya ID'si oluştur
    file_id = generate_file_id()
    
    # Dosya uzantısını koru
    file_extension = Path(file.filename).suffix
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{file_extension}")
    
    # Dosyayı kaydet
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return file_id


def load_data_file(file_id: str) -> pd.DataFrame:
    """Dosyayı pandas DataFrame olarak yükle"""
    
    # Dosya yolunu bul
    upload_dir = Path(settings.UPLOAD_DIR)
    file_path = None
    
    for ext in settings.ALLOWED_FILE_TYPES:
        potential_path = upload_dir / f"{file_id}{ext}"
        if potential_path.exists():
            file_path = potential_path
            break
    
    if not file_path:
        raise FileNotFoundError(f"Dosya bulunamadı: {file_id}")
    
    # Dosya türüne göre yükle
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.csv':
            # CSV dosyası için encoding tespiti
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("CSV dosyası okunamadı - encoding sorunu")
                
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            
        elif file_extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.json_normalize(data)
            
        else:
            raise ValueError(f"Desteklenmeyen dosya türü: {file_extension}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Dosya yükleme hatası: {str(e)}")


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """DataFrame hakkında bilgi al"""
    
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist()
    }
    
    return info


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun isimlerini temizle"""
    
    # Boşlukları alt çizgi ile değiştir
    df.columns = df.columns.str.replace(' ', '_')
    
    # Özel karakterleri kaldır
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
    
    # Büyük harfleri küçük harfe çevir
    df.columns = df.columns.str.lower()
    
    # Başında ve sonunda alt çizgi varsa kaldır
    df.columns = df.columns.str.strip('_')
    
    # Çift alt çizgileri tek alt çizgi yap
    df.columns = df.columns.str.replace('__+', '_', regex=True)
    
    return df


def detect_outliers(series: pd.Series, method: str = 'iqr') -> pd.Series:
    """Aykırı değerleri tespit et"""
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = abs((series - series.mean()) / series.std())
        return z_scores > 3
    
    else:
        return pd.Series([False] * len(series), index=series.index)


def format_number(value: float, decimals: int = 4) -> str:
    """Sayıyı formatla"""
    if pd.isna(value):
        return "N/A"
    
    if abs(value) < 0.001:
        return f"{value:.2e}"
    else:
        return f"{value:.{decimals}f}"


def generate_interpretation(analysis_type: str, results: Dict[str, Any]) -> List[str]:
    """Analiz sonuçları için otomatik yorum oluştur"""
    
    interpretations = []
    
    if analysis_type == "descriptive":
        # Betimsel istatistikler yorumu
        for column, stats in results.get("descriptive_stats", {}).items():
            if stats.get("std") and stats.get("mean"):
                cv = stats["std"] / stats["mean"]
                if cv > 0.3:
                    interpretations.append(f"{column} değişkeni yüksek değişkenlik gösteriyor (CV: {cv:.2f})")
                
            if stats.get("skewness"):
                if abs(stats["skewness"]) > 1:
                    direction = "sağa" if stats["skewness"] > 0 else "sola"
                    interpretations.append(f"{column} değişkeni {direction} çarpık dağılım gösteriyor")
    
    elif analysis_type == "hypothesis_test":
        # Hipotez testi yorumu
        test_result = results.get("statistical_test", {})
        p_value = test_result.get("p_value")
        alpha = 0.05
        
        if p_value is not None:
            if p_value < alpha:
                interpretations.append(f"H0 hipotezi reddedildi (p={p_value:.4f} < α={alpha})")
                interpretations.append("Gruplar arasında istatistiksel olarak anlamlı fark vardır")
            else:
                interpretations.append(f"H0 hipotezi kabul edildi (p={p_value:.4f} ≥ α={alpha})")
                interpretations.append("Gruplar arasında istatistiksel olarak anlamlı fark yoktur")
    
    elif analysis_type == "regression":
        # Regresyon analizi yorumu
        reg_results = results.get("regression", {})
        r_squared = reg_results.get("r_squared")
        
        if r_squared is not None:
            if r_squared > 0.7:
                interpretations.append(f"Model güçlü açıklayıcılığa sahip (R² = {r_squared:.3f})")
            elif r_squared > 0.3:
                interpretations.append(f"Model orta düzeyde açıklayıcılığa sahip (R² = {r_squared:.3f})")
            else:
                interpretations.append(f"Model zayıf açıklayıcılığa sahip (R² = {r_squared:.3f})")
    
    return interpretations


def create_visualization_path(analysis_id: str, plot_type: str) -> str:
    """Görselleştirme dosya yolu oluştur"""
    
    # Görselleştirme klasörünü oluştur
    os.makedirs(settings.VISUALIZATION_DIR, exist_ok=True)
    
    # Dosya yolu
    filename = f"{analysis_id}_{plot_type}.{settings.PLOT_FORMAT}"
    return os.path.join(settings.VISUALIZATION_DIR, filename)


def cleanup_old_files(max_age_hours: int = 24):
    """Eski dosyaları temizle"""
    
    current_time = datetime.now()
    
    # Upload klasörünü temizle
    for file_path in Path(settings.UPLOAD_DIR).glob("*"):
        if file_path.is_file():
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age.total_seconds() > max_age_hours * 3600:
                file_path.unlink()
    
    # Görselleştirme klasörünü temizle
    for file_path in Path(settings.VISUALIZATION_DIR).glob("*"):
        if file_path.is_file():
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age.total_seconds() > max_age_hours * 3600:
                file_path.unlink()


def validate_analysis_parameters(analysis_type: str, parameters: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Analiz parametrelerini doğrula ve düzelt"""
    
    validated_params = parameters.copy()
    
    # Sütun kontrolü
    if "columns" in validated_params:
        valid_columns = [col for col in validated_params["columns"] if col in df.columns]
        validated_params["columns"] = valid_columns
    
    # Hedef sütun kontrolü
    if "target_column" in validated_params:
        if validated_params["target_column"] not in df.columns:
            validated_params["target_column"] = None
    
    # Grup sütunu kontrolü
    if "group_column" in validated_params:
        if validated_params["group_column"] not in df.columns:
            validated_params["group_column"] = None
    
    return validated_params