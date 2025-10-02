"""
VeriVio Veri Temizleme Modülü
Eksik veriler, aykırı değerler ve veri kalitesi sorunlarını çözer
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import logging

from ...app.utils import load_data_file, detect_outliers, clean_column_names, get_data_info

logger = logging.getLogger(__name__)


class DataCleaner:
    """Veri temizleme ve ön işleme sınıfı"""
    
    def __init__(self):
        self.scaler = None
        self.original_shape = None
        self.cleaning_report = {}
    
    def load_data(self, file_id: str) -> pd.DataFrame:
        """Dosyayı yükle ve temel temizlik yap"""
        try:
            df = load_data_file(file_id)
            self.original_shape = df.shape
            
            # Sütun isimlerini temizle
            df = clean_column_names(df)
            
            logger.info(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
            return df
            
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> pd.DataFrame:
        """Kapsamlı veri temizleme"""
        
        logger.info("Veri temizleme başlatıldı")
        cleaned_df = df.copy()
        self.cleaning_report = {
            "original_shape": df.shape,
            "operations": [],
            "removed_rows": 0,
            "modified_columns": []
        }
        
        # 1. Duplicate satırları kaldır
        if options.get("remove_duplicates", True):
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed = before_count - len(cleaned_df)
            if removed > 0:
                self.cleaning_report["operations"].append(f"Duplicate satırlar kaldırıldı: {removed}")
                self.cleaning_report["removed_rows"] += removed
        
        # 2. Eksik verileri işle
        missing_method = options.get("handle_missing", "drop")
        cleaned_df = self._handle_missing_values(cleaned_df, missing_method)
        
        # 3. Aykırı değerleri işle
        outlier_method = options.get("outlier_method", "iqr")
        if outlier_method != "none":
            cleaned_df = self._handle_outliers(cleaned_df, outlier_method)
        
        # 4. Veri tiplerini optimize et
        cleaned_df = self._optimize_dtypes(cleaned_df)
        
        # 5. Normalizasyon/Standardizasyon
        if options.get("normalize", False):
            cleaned_df = self._normalize_data(cleaned_df)
        
        if options.get("standardize", False):
            cleaned_df = self._standardize_data(cleaned_df)
        
        self.cleaning_report["final_shape"] = cleaned_df.shape
        self.cleaning_report["data_quality_score"] = self._calculate_quality_score(cleaned_df)
        
        logger.info(f"Veri temizleme tamamlandı: {cleaned_df.shape[0]} satır, {cleaned_df.shape[1]} sütun")
        return cleaned_df
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Eksik değerleri işle"""
        
        missing_before = df.isnull().sum().sum()
        
        if method == "drop":
            # Eksik değeri olan satırları kaldır
            df_cleaned = df.dropna()
            removed_rows = len(df) - len(df_cleaned)
            if removed_rows > 0:
                self.cleaning_report["operations"].append(f"Eksik değerli satırlar kaldırıldı: {removed_rows}")
                self.cleaning_report["removed_rows"] += removed_rows
        
        elif method == "fill_mean":
            # Sayısal sütunları ortalama ile doldur
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_cleaned = df.copy()
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    mean_val = df_cleaned[col].mean()
                    df_cleaned[col].fillna(mean_val, inplace=True)
                    self.cleaning_report["modified_columns"].append(f"{col} (ortalama ile dolduruldu)")
        
        elif method == "fill_median":
            # Sayısal sütunları medyan ile doldur
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_cleaned = df.copy()
            for col in numeric_cols:
                if df_cleaned[col].isnull().any():
                    median_val = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
                    self.cleaning_report["modified_columns"].append(f"{col} (medyan ile dolduruldu)")
        
        elif method == "fill_mode":
            # Tüm sütunları mod ile doldur
            df_cleaned = df.copy()
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().any():
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col].fillna(mode_val[0], inplace=True)
                        self.cleaning_report["modified_columns"].append(f"{col} (mod ile dolduruldu)")
        
        elif method == "forward_fill":
            df_cleaned = df.fillna(method='ffill')
            self.cleaning_report["operations"].append("Forward fill uygulandı")
        
        elif method == "backward_fill":
            df_cleaned = df.fillna(method='bfill')
            self.cleaning_report["operations"].append("Backward fill uygulandı")
        
        else:
            df_cleaned = df.copy()
        
        missing_after = df_cleaned.isnull().sum().sum()
        if missing_before > missing_after:
            self.cleaning_report["operations"].append(
                f"Eksik değerler azaltıldı: {missing_before} → {missing_after}"
            )
        
        return df_cleaned
    
    def _handle_outliers(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Aykırı değerleri işle"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_cleaned = df.copy()
        total_outliers = 0
        
        for col in numeric_cols:
            if method == "iqr":
                outliers = detect_outliers(df_cleaned[col], method="iqr")
            elif method == "zscore":
                outliers = detect_outliers(df_cleaned[col], method="zscore")
            elif method == "isolation_forest":
                outliers = self._detect_outliers_isolation_forest(df_cleaned[[col]])
            else:
                continue
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                # Aykırı değerleri medyan ile değiştir
                median_val = df_cleaned[col].median()
                df_cleaned.loc[outliers, col] = median_val
                total_outliers += outlier_count
                self.cleaning_report["modified_columns"].append(
                    f"{col} ({outlier_count} aykırı değer düzeltildi)"
                )
        
        if total_outliers > 0:
            self.cleaning_report["operations"].append(
                f"Toplam {total_outliers} aykırı değer düzeltildi ({method} yöntemi)"
            )
        
        return df_cleaned
    
    def _detect_outliers_isolation_forest(self, data: pd.DataFrame) -> pd.Series:
        """Isolation Forest ile aykırı değer tespiti"""
        
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data)
            return pd.Series(outliers == -1, index=data.index)
        except:
            # Hata durumunda boş Series döndür
            return pd.Series([False] * len(data), index=data.index)
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Veri tiplerini optimize et"""
        
        df_optimized = df.copy()
        
        # Integer sütunları optimize et
        int_cols = df_optimized.select_dtypes(include=['int64']).columns
        for col in int_cols:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif col_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        # Float sütunları optimize et
        float_cols = df_optimized.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Kategorik sütunları optimize et
        object_cols = df_optimized.select_dtypes(include=['object']).columns
        for col in object_cols:
            unique_count = df_optimized[col].nunique()
            total_count = len(df_optimized[col])
            
            # Eğer benzersiz değer sayısı toplam sayının %50'sinden azsa kategorik yap
            if unique_count / total_count < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        self.cleaning_report["operations"].append("Veri tipleri optimize edildi")
        return df_optimized
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Veriyi normalize et (0-1 arası)"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_normalized = df.copy()
        
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler()
            df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
            self.cleaning_report["operations"].append("Normalizasyon uygulandı")
            self.cleaning_report["modified_columns"].extend([f"{col} (normalize)" for col in numeric_cols])
        
        return df_normalized
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Veriyi standardize et (z-score)"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_standardized = df.copy()
        
        if len(numeric_cols) > 0:
            self.scaler = StandardScaler()
            df_standardized[numeric_cols] = self.scaler.fit_transform(df_standardized[numeric_cols])
            self.cleaning_report["operations"].append("Standardizasyon uygulandı")
            self.cleaning_report["modified_columns"].extend([f"{col} (standardize)" for col in numeric_cols])
        
        return df_standardized
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Veri kalitesi skoru hesapla (0-100)"""
        
        score = 100.0
        
        # Eksik değer cezası
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        score -= missing_ratio * 30
        
        # Duplicate cezası
        duplicate_ratio = df.duplicated().sum() / len(df)
        score -= duplicate_ratio * 20
        
        # Veri tipi çeşitliliği bonusu
        dtype_variety = len(df.dtypes.unique()) / 5  # Maksimum 5 farklı tip
        score += min(dtype_variety * 10, 10)
        
        return max(0, min(100, score))
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Temizleme raporunu döndür"""
        return self.cleaning_report
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detaylı veri kalitesi raporu"""
        
        report = {
            "basic_info": get_data_info(df),
            "missing_values": {
                "total": df.isnull().sum().sum(),
                "by_column": df.isnull().sum().to_dict(),
                "percentage": (df.isnull().sum() / len(df) * 100).to_dict()
            },
            "duplicates": {
                "count": df.duplicated().sum(),
                "percentage": df.duplicated().sum() / len(df) * 100
            },
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage": df.memory_usage(deep=True).to_dict(),
            "quality_score": self._calculate_quality_score(df)
        }
        
        # Sayısal sütunlar için istatistikler
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        # Kategorik sütunlar için istatistikler
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            report["categorical_summary"] = {}
            for col in categorical_cols:
                report["categorical_summary"][col] = {
                    "unique_count": df[col].nunique(),
                    "most_frequent": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    "frequency": df[col].value_counts().head().to_dict()
                }
        
        return report