import pandas as pd
import logging
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile
from .config import settings
from .models import AnalysisRequest

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(settings.LOG_FILE), logging.StreamHandler()]
    )

def generate_file_id() -> str:
    return str(uuid.uuid4())

async def save_uploaded_file(file: UploadFile, file_id: str) -> str:
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{file_id}{Path(file.filename).suffix}"
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(await file.read())
    return str(file_path)

def load_data_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    raise ValueError("Desteklenmeyen dosya formatı.")

def validate_analysis_parameters(request: AnalysisRequest, df: pd.DataFrame) -> dict:
    """Analiz parametrelerini doğrula"""
    params = request.parameters
    
    # Genel sütun kontrolü
    if params.columns:
        missing = [col for col in params.columns if col not in df.columns]
        if missing:
            return {'valid': False, 'error': f"Sütunlar bulunamadı: {missing}"}
    
    # Hipotez testleri için özel validasyonlar
    if request.analysis_type == "hypothesis" and params.test_type:
        return validate_hypothesis_parameters(params, df)
    
    return {'valid': True}

def validate_hypothesis_parameters(params, df: pd.DataFrame) -> dict:
    """Hipotez testi parametrelerini doğrula"""
    test_type = params.test_type
    
    # Paired t-test validasyonu
    if test_type == "t_test_paired":
        if not params.paired_col_1 or not params.paired_col_2:
            return {'valid': False, 'error': "Paired t-test için iki sütun gerekli"}
        
        missing_cols = []
        if params.paired_col_1 not in df.columns:
            missing_cols.append(params.paired_col_1)
        if params.paired_col_2 not in df.columns:
            missing_cols.append(params.paired_col_2)
        
        if missing_cols:
            return {'valid': False, 'error': f"Sütunlar bulunamadı: {missing_cols}"}
        
        # Sayısal veri kontrolü
        if not pd.api.types.is_numeric_dtype(df[params.paired_col_1]):
            return {'valid': False, 'error': f"'{params.paired_col_1}' sayısal bir sütun değil"}
        if not pd.api.types.is_numeric_dtype(df[params.paired_col_2]):
            return {'valid': False, 'error': f"'{params.paired_col_2}' sayısal bir sütun değil"}
        
        # Minimum veri kontrolü
        valid_pairs = df[[params.paired_col_1, params.paired_col_2]].dropna()
        if len(valid_pairs) < 3:
            return {'valid': False, 'error': "Paired t-test için en az 3 geçerli çift gerekli"}
    
    # One-way ANOVA validasyonu
    elif test_type == "anova_one_way":
        if not params.dependent_var:
            return {'valid': False, 'error': "One-way ANOVA için bağımlı değişken gerekli"}
        
        if not params.independent_var:
            return {'valid': False, 'error': "One-way ANOVA için bağımsız değişken gerekli"}
        
        # Sütunları kontrol et
        missing_cols = []
        if params.dependent_var not in df.columns:
            missing_cols.append(params.dependent_var)
        if params.independent_var not in df.columns:
            missing_cols.append(params.independent_var)
        
        if missing_cols:
            return {'valid': False, 'error': f"Sütunlar bulunamadı: {missing_cols}"}
        
        # Bağımlı değişken sayısal olmalı
        if not pd.api.types.is_numeric_dtype(df[params.dependent_var]):
            return {'valid': False, 'error': f"'{params.dependent_var}' sayısal bir sütun değil"}
        
        # Minimum veri kontrolü
        valid_data = df[[params.dependent_var, params.independent_var]].dropna()
        if len(valid_data) < 3:
            return {'valid': False, 'error': "One-way ANOVA için en az 3 geçerli gözlem gerekli"}
        
        # Grup sayısı kontrolü
        groups = valid_data[params.independent_var].nunique()
        if groups < 2:
            return {'valid': False, 'error': "One-way ANOVA için en az 2 grup gerekli"}
    
    # MANOVA validasyonu
    elif test_type == "manova":
        if not params.dependent_columns or len(params.dependent_columns) < 2:
            return {'valid': False, 'error': "MANOVA için en az 2 bağımlı değişken gerekli"}
        
        if not params.independent_formula:
            return {'valid': False, 'error': "MANOVA için bağımsız değişken formülü gerekli"}
        
        # Bağımlı değişkenleri kontrol et
        missing_deps = [col for col in params.dependent_columns if col not in df.columns]
        if missing_deps:
            return {'valid': False, 'error': f"Bağımlı değişkenler bulunamadı: {missing_deps}"}
        
        # Sayısal veri kontrolü
        for col in params.dependent_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return {'valid': False, 'error': f"'{col}' sayısal bir sütun değil"}
        
        # Bağımsız değişkenleri kontrol et
        formula_vars = params.independent_formula.replace('+', ' ').replace('*', ' ').replace(':', ' ').split()
        missing_indeps = [var.strip() for var in formula_vars if var.strip() and var.strip() not in df.columns]
        if missing_indeps:
            return {'valid': False, 'error': f"Bağımsız değişkenler bulunamadı: {missing_indeps}"}
        
        # Minimum veri kontrolü
        all_vars = params.dependent_columns + [var.strip() for var in formula_vars if var.strip()]
        valid_data = df[all_vars].dropna()
        if len(valid_data) < 10:
            return {'valid': False, 'error': "MANOVA için en az 10 geçerli gözlem gerekli"}
    
    # Mixed ANOVA validasyonu
    elif test_type == "mixed_anova":
        required_params = ['dv_column', 'within_column', 'between_column', 'subject_column']
        missing_params = [param for param in required_params if not getattr(params, param, None)]
        if missing_params:
            return {'valid': False, 'error': f"Mixed ANOVA için gerekli parametreler eksik: {missing_params}"}
        
        # Sütunları kontrol et
        required_cols = [params.dv_column, params.within_column, params.between_column, params.subject_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {'valid': False, 'error': f"Sütunlar bulunamadı: {missing_cols}"}
        
        # Bağımlı değişken sayısal olmalı
        if not pd.api.types.is_numeric_dtype(df[params.dv_column]):
            return {'valid': False, 'error': f"'{params.dv_column}' sayısal bir sütun değil"}
        
        # Minimum veri kontrolü
        valid_data = df[[params.dv_column, params.within_column, params.between_column, params.subject_column]].dropna()
        if len(valid_data) < 4:
            return {'valid': False, 'error': "Mixed ANOVA için en az 4 geçerli gözlem gerekli"}
    
    # Wilcoxon signed-rank test validasyonu
    elif test_type == "wilcoxon_signed_rank":
        if not params.paired_col_1 or not params.paired_col_2:
            return {'valid': False, 'error': "Wilcoxon signed-rank test için iki sütun gerekli"}
        
        missing_cols = []
        if params.paired_col_1 not in df.columns:
            missing_cols.append(params.paired_col_1)
        if params.paired_col_2 not in df.columns:
            missing_cols.append(params.paired_col_2)
        
        if missing_cols:
            return {'valid': False, 'error': f"Sütunlar bulunamadı: {missing_cols}"}
        
        # Sayısal veri kontrolü
        if not pd.api.types.is_numeric_dtype(df[params.paired_col_1]):
            return {'valid': False, 'error': f"'{params.paired_col_1}' sayısal bir sütun değil"}
        if not pd.api.types.is_numeric_dtype(df[params.paired_col_2]):
            return {'valid': False, 'error': f"'{params.paired_col_2}' sayısal bir sütun değil"}
        
        # Minimum veri kontrolü
        valid_pairs = df[[params.paired_col_1, params.paired_col_2]].dropna()
        if len(valid_pairs) < 3:
            return {'valid': False, 'error': "Wilcoxon signed-rank test için en az 3 geçerli çift gerekli"}
    
    return {'valid': True}