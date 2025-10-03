from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

class AnalysisType(str, Enum):
    DESCRIPTIVE = "descriptive"
    CLEANING = "cleaning"
    VISUALIZATION = "visualization"
    HYPOTHESIS = "hypothesis"
    REGRESSION = "regression"
    ADVANCED = "advanced"
    DOMAIN = "domain"

class HypothesisTestType(str, Enum):
    T_TEST_ONE_SAMPLE = "t_test_one_sample"
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    ANOVA_ONE_WAY = "anova_one_way"
    ANOVA_TWO_WAY = "anova_two_way"
    MIXED_ANOVA = "mixed_anova"
    MANOVA = "manova"
    CHI_SQUARE = "chi_square"
    CORRELATION = "correlation"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    MANN_WHITNEY_U = "mann_whitney_u"
    KRUSKAL_WALLIS = "kruskal_wallis"

class AnalysisParameters(BaseModel):
    columns: Optional[List[str]] = None
    missing_strategy: Optional[str] = "mean"
    # Hypothesis testing parameters
    test_type: Optional[HypothesisTestType] = None
    # ANOVA parameters
    dependent_var: Optional[str] = None
    independent_var: Optional[str] = None
    group_column: Optional[str] = None
    # Paired t-test parameters
    paired_col_1: Optional[str] = None
    paired_col_2: Optional[str] = None
    # MANOVA parameters
    dependent_columns: Optional[List[str]] = None
    independent_formula: Optional[str] = None
    # Mixed ANOVA parameters
    subject_column: Optional[str] = None
    within_column: Optional[str] = None
    between_column: Optional[str] = None
    dv_column: Optional[str] = None
    # General test parameters
    alpha: Optional[float] = 0.05
    alternative: Optional[str] = "two-sided"

class AnalysisRequest(BaseModel):
    file_id: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    analysis_type: AnalysisType
    parameters: AnalysisParameters = AnalysisParameters()
    clean_data: Optional[bool] = True
    cleaning_options: Optional[Dict[str, Any]] = None
    confidence_level: Optional[float] = 0.95

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    columns: List[str]
    data_info: Dict[str, Any]
    message: str

class AnalysisResponse(BaseModel):
    analysis_type: str
    file_id: str
    timestamp: datetime
    data: Dict[str, Any]
    success: bool
    message: str

class ErrorResponse(BaseModel):
    error: str
    timestamp: datetime