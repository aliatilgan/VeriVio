"""
VeriVio Gelişmiş Veri Temizleme Modülü
Eksik veriler, aykırı değerler ve veri kalitesi sorunlarını çözer
Gelişmiş interpolasyon ve outlier detection yöntemleri
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from scipy import stats
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Gelişmiş veri temizleme ve ön işleme sınıfı"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df.copy() if df is not None else None
        self.scaler = None
        self.imputer = None
        self.original_shape = None
        self.cleaning_report = {}
        self.outlier_indices = {}
        
    def load_and_analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Veriyi yükle ve kapsamlı analiz yap"""
        self.original_shape = df.shape
        
        analysis = {
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'missing_summary': self._analyze_missing_data(df),
            'outlier_summary': self._analyze_outliers(df),
            'data_quality_score': self._calculate_quality_score(df),
            'column_types': self._classify_columns(df)
        }
        
        logger.info(f"Veri analizi tamamlandı: {df.shape[0]} satır, {df.shape[1]} sütun")
        return analysis
    
    def clean_data_comprehensive(self, df: pd.DataFrame, options: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Kapsamlı veri temizleme pipeline"""
        
        logger.info("Kapsamlı veri temizleme başlatıldı")
        cleaned_df = df.copy()
        
        self.cleaning_report = {
            "original_shape": df.shape,
            "operations": [],
            "removed_rows": 0,
            "modified_columns": [],
            "quality_improvements": {}
        }
        
        # 1. Temel temizlik
        cleaned_df = self._basic_cleaning(cleaned_df, options)
        
        # 2. Gelişmiş eksik veri işleme
        cleaned_df = self._advanced_missing_handling(cleaned_df, options)
        
        # 3. Gelişmiş aykırı değer işleme
        cleaned_df = self._advanced_outlier_handling(cleaned_df, options)
        
        # 4. Veri tipi optimizasyonu
        cleaned_df = self._optimize_dtypes_advanced(cleaned_df)
        
        # 5. Veri dönüşümleri
        cleaned_df = self._apply_transformations(cleaned_df, options)
        
        # 6. Veri kalitesi değerlendirmesi
        self._assess_final_quality(cleaned_df)
        
        logger.info(f"Veri temizleme tamamlandı: {cleaned_df.shape[0]} satır, {cleaned_df.shape[1]} sütun")
        return cleaned_df, self.cleaning_report
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Eksik veri analizi"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            'total_missing': int(missing_counts.sum()),
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist(),
            'missing_patterns': self._identify_missing_patterns(df)
        }
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aykırı değer analizi"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:  # Minimum veri gereksinimi
                continue
                
            # IQR yöntemi
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            # Z-score yöntemi
            z_scores = np.abs(stats.zscore(series))
            z_outliers = series[z_scores > 3]
            
            # Isolation Forest
            if len(series) > 10:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
                iso_outliers = series[outlier_labels == -1]
            else:
                iso_outliers = pd.Series([], dtype=series.dtype)
            
            outlier_summary[col] = {
                'iqr_outliers_count': len(iqr_outliers),
                'z_score_outliers_count': len(z_outliers),
                'isolation_forest_outliers_count': len(iso_outliers),
                'outlier_percentage': (len(iqr_outliers) / len(series)) * 100
            }
        
        return outlier_summary
    
    def _basic_cleaning(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Temel veri temizleme işlemleri"""
        cleaned_df = df.copy()
        
        # Duplicate satırları kaldır
        if options.get("remove_duplicates", True):
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_df)
            if removed_duplicates > 0:
                self.cleaning_report["operations"].append(f"{removed_duplicates} duplicate satır kaldırıldı")
                self.cleaning_report["removed_rows"] += removed_duplicates
        
        # Tamamen boş satırları kaldır
        if options.get("remove_empty_rows", True):
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(how='all')
            removed_empty = initial_rows - len(cleaned_df)
            if removed_empty > 0:
                self.cleaning_report["operations"].append(f"{removed_empty} tamamen boş satır kaldırıldı")
                self.cleaning_report["removed_rows"] += removed_empty
        
        # Tamamen boş sütunları kaldır
        if options.get("remove_empty_columns", True):
            initial_cols = len(cleaned_df.columns)
            cleaned_df = cleaned_df.dropna(axis=1, how='all')
            removed_cols = initial_cols - len(cleaned_df.columns)
            if removed_cols > 0:
                self.cleaning_report["operations"].append(f"{removed_cols} tamamen boş sütun kaldırıldı")
        
        # String sütunlardaki whitespace'leri temizle
        if options.get("strip_whitespace", True):
            string_cols = cleaned_df.select_dtypes(include=['object']).columns
            for col in string_cols:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                self.cleaning_report["operations"].append(f"{col}: Whitespace temizlendi")
        
        return cleaned_df
    
    def _classify_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Sütunları tipine göre sınıflandır"""
        return {
            'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'boolean': df.select_dtypes(include=['bool']).columns.tolist()
        }
    
    def _identify_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Eksik veri desenlerini tanımla"""
        missing_matrix = df.isnull()
        patterns = {}
        
        # Tamamen eksik satırlar
        completely_missing_rows = missing_matrix.all(axis=1).sum()
        patterns['completely_missing_rows'] = int(completely_missing_rows)
        
        # Hiç eksik verisi olmayan satırlar
        no_missing_rows = (~missing_matrix.any(axis=1)).sum()
        patterns['no_missing_rows'] = int(no_missing_rows)
        
        return patterns
    
    def _optimize_dtypes_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Veri tiplerini optimize et"""
        optimized_df = df.copy()
        
        # Numeric sütunları optimize et
        for col in optimized_df.select_dtypes(include=[np.number]).columns:
            if optimized_df[col].dtype == 'float64':
                if optimized_df[col].isnull().sum() == 0:
                    # Integer'a dönüştürülebilir mi kontrol et
                    if (optimized_df[col] % 1 == 0).all():
                        optimized_df[col] = optimized_df[col].astype('int64')
                # Float32'ye dönüştür
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
            elif optimized_df[col].dtype == 'int64':
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
        
        self.cleaning_report["operations"].append("Veri tipleri optimize edildi")
        return optimized_df
    
    def _assess_final_quality(self, df: pd.DataFrame) -> None:
        """Final veri kalitesini değerlendir"""
        self.cleaning_report["final_shape"] = df.shape
        self.cleaning_report["data_quality_score"] = self._calculate_quality_score(df)
    
    def _basic_missing_handling(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Temel eksik veri işleme"""
        if method == "drop":
            return df.dropna()
        elif method == "forward_fill":
            return df.fillna(method='ffill')
        elif method == "backward_fill":
            return df.fillna(method='bfill')
        else:
            # Numeric için mean, categorical için mode
            df_filled = df.copy()
            for col in df.columns:
                if df[col].dtype in [np.number]:
                    df_filled[col].fillna(df[col].mean(), inplace=True)
                else:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df_filled[col].fillna(mode_val.iloc[0], inplace=True)
            return df_filled
    
    def _advanced_missing_handling(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Gelişmiş eksik veri işleme"""
        missing_method = options.get("missing_method", "smart")
        
        if missing_method == "smart":
            return self._smart_missing_imputation(df, options)
        elif missing_method == "knn":
            return self._knn_imputation(df, options)
        elif missing_method == "iterative":
            return self._iterative_imputation(df, options)
        elif missing_method == "interpolation":
            return self._interpolation_imputation(df, options)
        else:
            return self._basic_missing_handling(df, missing_method)
    
    def _smart_missing_imputation(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Akıllı eksik veri doldurma - sütun tipine göre en uygun yöntemi seç"""
        df_imputed = df.copy()
        
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct == 0:
                continue
            
            if missing_pct > 50:
                # %50'den fazla eksik veri varsa sütunu kaldırmayı öner
                self.cleaning_report["operations"].append(f"Uyarı: {col} sütununda %{missing_pct:.1f} eksik veri")
                continue
            
            if df[col].dtype in ['object', 'category']:
                # Kategorik veriler için mod
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df_imputed[col].fillna(mode_value.iloc[0], inplace=True)
                    self.cleaning_report["operations"].append(f"{col}: Kategorik eksik veriler mod ile dolduruldu")
            
            elif df[col].dtype in [np.number, 'int64', 'float64']:
                if missing_pct < 5:
                    # Az eksik veri için medyan
                    df_imputed[col].fillna(df[col].median(), inplace=True)
                    self.cleaning_report["operations"].append(f"{col}: Eksik veriler medyan ile dolduruldu")
                elif missing_pct < 20:
                    # Orta düzey eksik veri için KNN
                    df_imputed = self._knn_imputation_single_column(df_imputed, col)
                    self.cleaning_report["operations"].append(f"{col}: Eksik veriler KNN ile dolduruldu")
                else:
                    # Yüksek eksik veri için iterative imputation
                    df_imputed = self._iterative_imputation_single_column(df_imputed, col)
                    self.cleaning_report["operations"].append(f"{col}: Eksik veriler iterative imputation ile dolduruldu")
        
        return df_imputed
    
    def _advanced_outlier_handling(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Gelişmiş aykırı değer işleme"""
        outlier_method = options.get("outlier_method", "isolation_forest")
        outlier_action = options.get("outlier_action", "cap")  # cap, remove, transform
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_processed = df.copy()
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            outlier_indices = self._detect_outliers_advanced(series, outlier_method)
            
            if len(outlier_indices) > 0:
                if outlier_action == "remove":
                    df_processed = df_processed.drop(outlier_indices)
                    self.cleaning_report["removed_rows"] += len(outlier_indices)
                elif outlier_action == "cap":
                    df_processed = self._cap_outliers(df_processed, col, outlier_indices)
                elif outlier_action == "transform":
                    df_processed = self._transform_outliers(df_processed, col)
                
                self.cleaning_report["operations"].append(
                    f"{col}: {len(outlier_indices)} aykırı değer {outlier_action} yöntemi ile işlendi"
                )
        
        return df_processed
    
    def _detect_outliers_advanced(self, series: pd.Series, method: str) -> List[int]:
        """Gelişmiş aykırı değer tespiti"""
        if method == "isolation_forest":
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(series.values.reshape(-1, 1))
            return series.index[outlier_labels == -1].tolist()
        
        elif method == "modified_zscore":
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return series.index[np.abs(modified_z_scores) > 3.5].tolist()
        
        elif method == "iqr_strict":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Daha katı sınır
            upper_bound = Q3 + 3 * IQR
            return series.index[(series < lower_bound) | (series > upper_bound)].tolist()
        
        else:  # varsayılan IQR
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return series.index[(series < lower_bound) | (series > upper_bound)].tolist()
    
    def _apply_transformations(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Veri dönüşümleri uygula"""
        df_transformed = df.copy()
        
        # Normalizasyon
        if options.get("normalize", False):
            scaler_type = options.get("scaler_type", "standard")
            df_transformed = self._apply_scaling(df_transformed, scaler_type)
        
        # Log dönüşümü
        if options.get("log_transform", False):
            df_transformed = self._apply_log_transform(df_transformed, options.get("log_columns", []))
        
        # Box-Cox dönüşümü
        if options.get("boxcox_transform", False):
            df_transformed = self._apply_boxcox_transform(df_transformed, options.get("boxcox_columns", []))
        
        return df_transformed
    
    def clean(self, strategy: str = 'mean') -> Dict[str, Any]:
        """Basit veri temizleme (main.py uyumluluğu için)"""
        if not hasattr(self, 'df') or self.df is None:
            raise ValueError("DataFrame not set. Use clean_data_comprehensive instead.")
            
        original_missing = self.df.isna().sum().sum()
        
        for col in self.df.select_dtypes(include=['float64', 'int64']).columns:
            if strategy == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        final_missing = self.df.isna().sum().sum()
        filled_values = original_missing - final_missing
        
        return {
            'cleaned_rows': len(self.df), 
            'missing_values_filled': filled_values
        }
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Veri kalitesi skoru hesapla (0-100)"""
        score = 100.0
        
        # Eksik veri cezası
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        score -= missing_ratio * 30
        
        # Duplicate cezası
        duplicate_ratio = df.duplicated().sum() / len(df)
        score -= duplicate_ratio * 20
        
        # Veri tipi tutarlılığı
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype == 'object':
                score -= 5  # Sayısal olması gereken ama object olan sütunlar
        
        return max(0.0, score)
    
    def generate_cleaning_report(self) -> str:
        """Temizleme raporu oluştur"""
        report = f"""
        VeriVio Veri Temizleme Raporu
        ============================
        
        Orijinal Veri Boyutu: {self.cleaning_report['original_shape']}
        Final Veri Boyutu: {self.cleaning_report.get('final_shape', 'Hesaplanmadı')}
        Kaldırılan Satır Sayısı: {self.cleaning_report['removed_rows']}
        
        Uygulanan İşlemler:
        """
        
        for operation in self.cleaning_report['operations']:
            report += f"- {operation}\n"
        
        report += f"\nVeri Kalitesi Skoru: {self.cleaning_report.get('data_quality_score', 'Hesaplanmadı'):.1f}/100"
        
        return report
    
    def _smart_missing_imputation(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Akıllı eksik veri doldurma - sütun tipine göre en uygun yöntemi seç"""
        # Bu metod zaten yukarıda tanımlanmış, tekrar tanımlamaya gerek yok
        return df
    
    def _knn_imputation(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """KNN ile eksik veri doldurma"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        
        df_imputed = df.copy()
        imputer = KNNImputer(n_neighbors=options.get('n_neighbors', 5))
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
        
        return df_imputed
    
    def _iterative_imputation(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Iterative imputation ile eksik veri doldurma"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        
        df_imputed = df.copy()
        imputer = IterativeImputer(random_state=42)
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
        
        return df_imputed
    
    def _interpolation_imputation(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Interpolation ile eksik veri doldurma"""
        df_imputed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df_imputed[col] = df_imputed[col].interpolate(method='linear')
        
        return df_imputed
    
    def _knn_imputation_single_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Tek sütun için KNN imputation"""
        if col not in df.columns or df[col].dtype not in [np.number]:
            return df
        
        df_imputed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            imputer = KNNImputer(n_neighbors=5)
            df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
        
        return df_imputed
    
    def _iterative_imputation_single_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Tek sütun için iterative imputation"""
        if col not in df.columns or df[col].dtype not in [np.number]:
            return df
        
        df_imputed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            imputer = IterativeImputer(random_state=42)
            df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
        
        return df_imputed
    
    def _cap_outliers(self, df: pd.DataFrame, col: str, outlier_indices: List[int]) -> pd.DataFrame:
        """Aykırı değerleri sınırla"""
        df_capped = df.copy()
        series = df[col]
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_capped.loc[df_capped[col] < lower_bound, col] = lower_bound
        df_capped.loc[df_capped[col] > upper_bound, col] = upper_bound
        
        return df_capped
    
    def _transform_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Aykırı değerleri dönüştür (log transform)"""
        df_transformed = df.copy()
        
        if df_transformed[col].min() > 0:
            df_transformed[col] = np.log1p(df_transformed[col])
        
        return df_transformed
    
    def _apply_scaling(self, df: pd.DataFrame, scaler_type: str) -> pd.DataFrame:
        """Ölçeklendirme uygula"""
        df_scaled = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        else:
            return df_scaled
        
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        self.scaler = scaler
        
        return df_scaled
    
    def _apply_log_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Log dönüşümü uygula"""
        df_transformed = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in [np.number] and df[col].min() > 0:
                df_transformed[col] = np.log1p(df_transformed[col])
        
        return df_transformed
    
    def _apply_boxcox_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Box-Cox dönüşümü uygula"""
        df_transformed = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in [np.number] and df[col].min() > 0:
                try:
                    df_transformed[col], _ = stats.boxcox(df_transformed[col])
                except:
                    # Box-Cox başarısız olursa log transform kullan
                    df_transformed[col] = np.log1p(df_transformed[col])
        
        return df_transformed