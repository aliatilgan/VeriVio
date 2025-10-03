from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any
from .config import settings
from .models import AnalysisRequest, AnalysisResponse, FileUploadResponse, ErrorResponse
from .utils import setup_logging, save_uploaded_file, load_data_file, generate_file_id, validate_analysis_parameters
from modules.descriptive_stats.calculator import DescriptiveStatsCalculator
from modules.data_processing.cleaner import DataCleaner
from modules.visualization.plotter import DataPlotter

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8081"],  # Vite ve React portları
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_files: Dict[str, pd.DataFrame] = {}

@app.get("/")
async def root():
    return {"message": "VeriVio API is running", "status": "healthy"}

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(400, detail="Desteklenmeyen dosya formatı: CSV, XLSX veya XLS bekleniyor.")
        
        file_id = generate_file_id()
        file_path = await save_uploaded_file(file, file_id)
        df = load_data_file(file_path)
        
        uploaded_files[file_id] = df
        logger.info(f"Dosya yüklendi: {file.filename}, ID: {file_id}")
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            size=len(df),
            columns=list(df.columns),
            data_info={"rows": len(df), "columns": len(df.columns)},
            message="Dosya başarıyla yüklendi."
        )
    except Exception as e:
        logger.error(f"Yükleme hatası: {str(e)}")
        raise HTTPException(500, detail=f"Dosya yükleme hatası: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    try:
        # Handle both file_id and direct data requests
        if request.data is not None:
            # Direct data request
            df = pd.DataFrame(request.data)
            file_id = "direct_data"
        elif request.file_id and request.file_id in uploaded_files:
            # File ID request
            df = uploaded_files[request.file_id]
            file_id = request.file_id
        else:
            raise HTTPException(400, detail="Either 'data' or valid 'file_id' must be provided.")
        
        validation = validate_analysis_parameters(request, df)
        if not validation['valid']:
            raise HTTPException(400, detail=validation['error'])
        
        result = await perform_analysis(df, request)
        response_data = AnalysisResponse(
            analysis_type=request.analysis_type,
            file_id=file_id,
            timestamp=datetime.now(),
            data=result,
            success=True,
            message="Analiz başarıyla tamamlandı."
        )
        # Convert to dict and handle datetime serialization
        response_dict = response_data.dict()
        response_dict['timestamp'] = response_data.timestamp.isoformat()
        
        return JSONResponse(content=response_dict)
    except Exception as e:
        logger.error(f"Analiz hatası: {str(e)}")
        error_response = ErrorResponse(error=str(e), timestamp=datetime.now())
        error_dict = error_response.dict()
        error_dict['timestamp'] = error_response.timestamp.isoformat()
        return JSONResponse(
            status_code=500,
            content=error_dict
        )

async def perform_analysis(df: pd.DataFrame, request: AnalysisRequest) -> Dict[str, Any]:
    analysis_type = request.analysis_type
    params = request.parameters
    
    if analysis_type == "descriptive":
        calc = DescriptiveStatsCalculator(df)
        columns = params.columns if params.columns else df.columns.tolist()
        return calc.calculate(columns)
    elif analysis_type == "cleaning":
        cleaner = DataCleaner(df)
        strategy = params.missing_strategy if params.missing_strategy else 'mean'
        return cleaner.clean(strategy)
    elif analysis_type == "visualization":
        plotter = DataPlotter()
        columns = params.columns if params.columns else df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        return plotter.create_histogram(df, columns)
    elif analysis_type == "hypothesis":
        # Kapsamlı hipotez testleri
        from modules.hypothesis_testing.tester import ComprehensiveHypothesisTester
        
        tester = ComprehensiveHypothesisTester(df)
        
        # AnalysisParameters nesnesini dictionary'ye çevir
        params_dict = {
            'test_type': params.test_type,
            'paired_col_1': params.paired_col_1,
            'paired_col_2': params.paired_col_2,
            'dependent_columns': params.dependent_columns,
            'independent_formula': params.independent_formula,
            'subject_column': params.subject_column,
            'within_column': params.within_column,
            'between_column': params.between_column,
            'dv_column': params.dv_column,
            'alpha': params.alpha or 0.05,
            'alternative': params.alternative or 'two-sided',
            'columns': params.columns
        }
        
        # None değerleri temizle
        params_dict = {k: v for k, v in params_dict.items() if v is not None}
        
        return tester.run_comprehensive_test(df, params_dict)
    elif analysis_type == "regression":
        # Basit linear regression
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_cols) >= 2:
            X = df[numeric_cols[:-1]].dropna()
            y = df[numeric_cols[-1]].dropna()
            # Ortak indeksleri al
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            if len(X) > 1:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                return {
                    "regression_results": {
                        "target_variable": numeric_cols[-1],
                        "features": numeric_cols[:-1],
                        "r2_score": float(r2),
                        "coefficients": {col: float(coef) for col, coef in zip(numeric_cols[:-1], model.coef_)},
                        "intercept": float(model.intercept_),
                        "interpretation": f"Model açıklama gücü: %{r2*100:.1f}"
                    }
                }
        return {"error": "Regresyon için yeterli sayısal veri bulunamadı"}
    elif analysis_type == "advanced":
        # Basit korelasyon analizi
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if len(numeric_df.columns) >= 2:
            correlation_matrix = numeric_df.corr()
            return {
                "advanced_analysis": {
                    "correlation_matrix": correlation_matrix.to_dict(),
                    "strong_correlations": [
                        {"variables": f"{col1} - {col2}", "correlation": float(corr)}
                        for col1 in correlation_matrix.columns
                        for col2 in correlation_matrix.columns
                        if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.7
                    ][:5],  # İlk 5 güçlü korelasyon
                    "message": "Korelasyon analizi tamamlandı"
                }
            }
        return {"error": "Gelişmiş analiz için yeterli sayısal veri bulunamadı"}
    elif analysis_type == "domain":
        # Basit domain analizi - veri özeti
        return {
            "domain_analysis": {
                "data_summary": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "numeric_columns": len(df.select_dtypes(include=['float64', 'int64']).columns),
                    "categorical_columns": len(df.select_dtypes(include=['object']).columns),
                    "missing_values": df.isnull().sum().sum(),
                    "data_types": df.dtypes.astype(str).to_dict()
                },
                "message": "Domain analizi tamamlandı"
            }
        }
    else:
        raise ValueError(f"Desteklenmeyen analiz türü: {analysis_type}")