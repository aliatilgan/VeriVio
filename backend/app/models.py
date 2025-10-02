"""
VeriVio Backend Veri Modelleri
Pydantic modelleri ile API request/response şemaları
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum


class AnalysisType(str, Enum):
    """Desteklenen analiz türleri"""
    DESCRIPTIVE = "descriptive"
    VISUALIZATION = "visualization"
    HYPOTHESIS_TEST = "hypothesis_test"
    REGRESSION = "regression"
    CORRELATION = "correlation"
    CLUSTERING = "clustering"
    FACTOR_ANALYSIS = "factor_analysis"
    TIME_SERIES = "time_series"
    SURVIVAL_ANALYSIS = "survival_analysis"
    FINANCE = "finance"
    MARKETING = "marketing"
    HEALTHCARE = "healthcare"


class VisualizationType(str, Enum):
    """Görselleştirme türleri"""
    HISTOGRAM = "histogram"
    BOXPLOT = "boxplot"
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HEATMAP = "heatmap"
    VIOLIN = "violin"
    PAIR_PLOT = "pair_plot"


class HypothesisTestType(str, Enum):
    """Hipotez testi türleri"""
    T_TEST_ONE_SAMPLE = "t_test_one_sample"
    T_TEST_TWO_SAMPLE = "t_test_two_sample"
    T_TEST_PAIRED = "t_test_paired"
    CHI_SQUARE = "chi_square"
    ANOVA_ONE_WAY = "anova_one_way"
    ANOVA_TWO_WAY = "anova_two_way"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KRUSKAL_WALLIS = "kruskal_wallis"


class RegressionType(str, Enum):
    """Regresyon analizi türleri"""
    LINEAR = "linear"
    LOGISTIC = "logistic"
    POLYNOMIAL = "polynomial"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class FileUploadResponse(BaseModel):
    """Dosya yükleme yanıtı"""
    file_id: str
    filename: str
    message: str
    status: str
    upload_time: Optional[str] = None
    size: Optional[int] = None


class CleaningOptions(BaseModel):
    """Veri temizleme seçenekleri"""
    remove_duplicates: bool = True
    handle_missing: str = Field(default="drop", pattern="^(drop|fill_mean|fill_median|fill_mode|forward_fill|backward_fill)$")
    outlier_method: str = Field(default="iqr", pattern="^(iqr|zscore|isolation_forest|none)$")
    normalize: bool = False
    standardize: bool = False
    encoding: str = Field(default="utf-8")


class AnalysisParameters(BaseModel):
    """Analiz parametreleri"""
    columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    group_column: Optional[str] = None
    confidence_level: float = Field(default=0.95, ge=0.01, le=0.99)
    test_type: Optional[str] = None
    visualization_type: Optional[str] = None
    regression_type: Optional[str] = None
    include_plots: bool = True
    plot_style: str = Field(default="seaborn")
    color_palette: str = Field(default="viridis")


class AnalysisRequest(BaseModel):
    """Analiz isteği modeli"""
    file_id: str
    analysis_type: AnalysisType
    parameters: Optional[AnalysisParameters] = None
    clean_data: bool = True
    cleaning_options: Optional[CleaningOptions] = None
    
    @validator('file_id')
    def validate_file_id(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Geçersiz dosya ID')
        return v


class StatisticalResult(BaseModel):
    """İstatistiksel sonuç modeli"""
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    degrees_of_freedom: Optional[int] = None
    effect_size: Optional[float] = None
    interpretation: Optional[str] = None


class DescriptiveStats(BaseModel):
    """Betimsel istatistikler"""
    count: int
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Union[float, str]] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    missing_count: int = 0
    unique_count: Optional[int] = None


class RegressionResults(BaseModel):
    """Regresyon analizi sonuçları"""
    r_squared: Optional[float] = None
    adjusted_r_squared: Optional[float] = None
    coefficients: Optional[Dict[str, float]] = None
    p_values: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    residuals_stats: Optional[Dict[str, float]] = None
    model_summary: Optional[str] = None
    predictions: Optional[List[float]] = None


class VisualizationResult(BaseModel):
    """Görselleştirme sonucu"""
    plot_path: str
    plot_type: str
    title: str
    description: Optional[str] = None
    insights: Optional[List[str]] = None


class AnalysisResults(BaseModel):
    """Analiz sonuçları"""
    descriptive_stats: Optional[Dict[str, DescriptiveStats]] = None
    statistical_test: Optional[StatisticalResult] = None
    regression: Optional[RegressionResults] = None
    visualizations: Optional[List[VisualizationResult]] = None
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    insights: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    data_quality_report: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    """Analiz yanıtı"""
    analysis_id: str
    status: str
    results: AnalysisResults
    timestamp: str
    execution_time: Optional[float] = None
    warnings: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    """Hata yanıtı"""
    error: str
    detail: str
    timestamp: str
    status_code: int


class HealthCheckResponse(BaseModel):
    """Sağlık kontrolü yanıtı"""
    status: str
    timestamp: str
    modules: Dict[str, bool]
    version: str = "1.0.0"


class FileListResponse(BaseModel):
    """Dosya listesi yanıtı"""
    files: Dict[str, Dict[str, Any]]
    count: int
    total_size: Optional[int] = None