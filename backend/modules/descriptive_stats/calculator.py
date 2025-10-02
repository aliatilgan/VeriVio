"""
VeriVio Betimsel İstatistikler Hesaplayıcısı
Temel ve gelişmiş betimsel istatistikleri hesaplama
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DescriptiveStatsCalculator:
    """Betimsel istatistikler hesaplama sınıfı"""
    
    def __init__(self):
        self.results = {}
        self.data_info = {}
    
    def calculate_basic_stats(self, df: pd.DataFrame, 
                             columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Temel betimsel istatistikleri hesapla"""
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.columns and 
                           df[col].dtype in [np.number, 'int64', 'float64']]
        
        basic_stats = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            stats_dict = {
                'count': len(series),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'mode': float(series.mode().iloc[0]) if not series.mode().empty else None,
                'std': float(series.std()),
                'var': float(series.var()),
                'min': float(series.min()),
                'max': float(series.max()),
                'range': float(series.max() - series.min()),
                'q1': float(series.quantile(0.25)),
                'q3': float(series.quantile(0.75)),
                'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'missing_count': int(df[col].isna().sum()),
                'missing_percentage': float((df[col].isna().sum() / len(df)) * 100)
            }
            
            # Güven aralıkları
            confidence_interval = stats.t.interval(
                0.95, len(series)-1, 
                loc=series.mean(), 
                scale=stats.sem(series)
            )
            stats_dict['confidence_interval_95'] = {
                'lower': float(confidence_interval[0]),
                'upper': float(confidence_interval[1])
            }
            
            basic_stats[col] = stats_dict
        
        self.results['basic_stats'] = basic_stats
        logger.info(f"{len(numeric_cols)} sütun için temel istatistikler hesaplandı")
        
        return basic_stats
    
    def calculate_categorical_stats(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Kategorik değişkenler için istatistikler"""
        
        if columns is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            categorical_cols = [col for col in columns if col in df.columns and 
                              df[col].dtype in ['object', 'category']]
        
        categorical_stats = {}
        
        for col in categorical_cols:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            value_counts = series.value_counts()
            
            stats_dict = {
                'count': len(series),
                'unique_count': series.nunique(),
                'most_frequent': str(value_counts.index[0]) if not value_counts.empty else None,
                'most_frequent_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                'least_frequent': str(value_counts.index[-1]) if not value_counts.empty else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if not value_counts.empty else 0,
                'missing_count': int(df[col].isna().sum()),
                'missing_percentage': float((df[col].isna().sum() / len(df)) * 100),
                'value_counts': value_counts.head(10).to_dict(),
                'entropy': float(-np.sum((value_counts / len(series)) * np.log2(value_counts / len(series))))
            }
            
            categorical_stats[col] = stats_dict
        
        self.results['categorical_stats'] = categorical_stats
        logger.info(f"{len(categorical_cols)} kategorik sütun için istatistikler hesaplandı")
        
        return categorical_stats
    
    def calculate_correlation_matrix(self, df: pd.DataFrame, 
                                    method: str = 'pearson') -> Dict[str, Any]:
        """Korelasyon matrisi hesapla"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            logger.warning("Korelasyon analizi için en az 2 sayısal sütun gerekli")
            return {}
        
        # Korelasyon matrisi
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # En yüksek korelasyonları bul
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if not np.isnan(corr_val):
                    corr_pairs.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': float(corr_val),
                        'abs_correlation': float(abs(corr_val))
                    })
        
        # Korelasyona göre sırala
        corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        correlation_results = {
            'method': method,
            'correlation_matrix': corr_matrix.to_dict(),
            'highest_correlations': corr_pairs[:10],
            'strong_positive_correlations': [p for p in corr_pairs if p['correlation'] > 0.7],
            'strong_negative_correlations': [p for p in corr_pairs if p['correlation'] < -0.7],
            'weak_correlations': [p for p in corr_pairs if abs(p['correlation']) < 0.3]
        }
        
        self.results['correlation'] = correlation_results
        logger.info(f"Korelasyon analizi tamamlandı ({method} yöntemi)")
        
        return correlation_results
    
    def calculate_distribution_stats(self, df: pd.DataFrame, 
                                    columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Dağılım istatistikleri hesapla"""
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.columns and 
                           df[col].dtype in [np.number, 'int64', 'float64']]
        
        distribution_stats = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 3:
                continue
            
            # Normallik testleri
            shapiro_stat, shapiro_p = stats.shapiro(series) if len(series) <= 5000 else (None, None)
            
            # Anderson-Darling testi
            try:
                anderson_result = stats.anderson(series, dist='norm')
                anderson_stat = float(anderson_result.statistic)
                anderson_critical = anderson_result.critical_values[2]  # %5 seviyesi
                anderson_normal = anderson_stat < anderson_critical
            except:
                anderson_stat, anderson_normal = None, None
            
            # Jarque-Bera testi
            try:
                jb_stat, jb_p = stats.jarque_bera(series)
            except:
                jb_stat, jb_p = None, None
            
            # Percentile'lar
            percentiles = {}
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                percentiles[f'p{p}'] = float(series.quantile(p/100))
            
            # Outlier tespiti (IQR yöntemi)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            stats_dict = {
                'normality_tests': {
                    'shapiro_wilk': {
                        'statistic': float(shapiro_stat) if shapiro_stat else None,
                        'p_value': float(shapiro_p) if shapiro_p else None,
                        'is_normal': bool(shapiro_p > 0.05) if shapiro_p else None
                    },
                    'anderson_darling': {
                        'statistic': anderson_stat,
                        'is_normal': anderson_normal
                    },
                    'jarque_bera': {
                        'statistic': float(jb_stat) if jb_stat else None,
                        'p_value': float(jb_p) if jb_p else None,
                        'is_normal': bool(jb_p > 0.05) if jb_p else None
                    }
                },
                'percentiles': percentiles,
                'outliers': {
                    'count': len(outliers),
                    'percentage': float((len(outliers) / len(series)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'values': outliers.tolist()[:20]  # İlk 20 outlier
                },
                'distribution_shape': {
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis()),
                    'skewness_interpretation': self._interpret_skewness(series.skew()),
                    'kurtosis_interpretation': self._interpret_kurtosis(series.kurtosis())
                }
            }
            
            distribution_stats[col] = stats_dict
        
        self.results['distribution'] = distribution_stats
        logger.info(f"{len(numeric_cols)} sütun için dağılım analizi tamamlandı")
        
        return distribution_stats
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Çarpıklık yorumlama"""
        if abs(skewness) < 0.5:
            return "Yaklaşık simetrik"
        elif skewness > 0.5:
            return "Sağa çarpık (pozitif çarpıklık)"
        else:
            return "Sola çarpık (negatif çarpıklık)"
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Basıklık yorumlama"""
        if abs(kurtosis) < 0.5:
            return "Normal basıklık (mezokurtik)"
        elif kurtosis > 0.5:
            return "Yüksek basıklık (leptokurtik)"
        else:
            return "Düşük basıklık (platikurtik)"
    
    def calculate_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Veri kalitesi metrikleri"""
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        
        # Sütun bazında eksik veri analizi
        column_missing = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            column_missing[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float((missing_count / len(df)) * 100),
                'data_type': str(df[col].dtype)
            }
        
        # Satır bazında eksik veri analizi
        row_missing = df.isna().sum(axis=1)
        complete_rows = (row_missing == 0).sum()
        
        # Duplicate analizi
        duplicate_rows = df.duplicated().sum()
        
        # Veri tipi dağılımı
        dtype_counts = df.dtypes.value_counts().to_dict()
        dtype_counts = {str(k): int(v) for k, v in dtype_counts.items()}
        
        quality_metrics = {
            'overall_completeness': float((1 - missing_cells / total_cells) * 100),
            'total_rows': int(df.shape[0]),
            'total_columns': int(df.shape[1]),
            'complete_rows': int(complete_rows),
            'complete_rows_percentage': float((complete_rows / len(df)) * 100),
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': float((duplicate_rows / len(df)) * 100),
            'column_missing_analysis': column_missing,
            'data_type_distribution': dtype_counts,
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        self.results['data_quality'] = quality_metrics
        logger.info("Veri kalitesi analizi tamamlandı")
        
        return quality_metrics
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Kapsamlı özet rapor oluştur"""
        
        # Tüm analizleri çalıştır
        basic_stats = self.calculate_basic_stats(df)
        categorical_stats = self.calculate_categorical_stats(df)
        correlation = self.calculate_correlation_matrix(df)
        distribution = self.calculate_distribution_stats(df)
        quality = self.calculate_data_quality_metrics(df)
        
        # Özet istatistikler
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        summary = {
            'dataset_overview': {
                'total_rows': int(df.shape[0]),
                'total_columns': int(df.shape[1]),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols),
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
            },
            'basic_statistics': basic_stats,
            'categorical_statistics': categorical_stats,
            'correlation_analysis': correlation,
            'distribution_analysis': distribution,
            'data_quality_metrics': quality,
            'recommendations': self._generate_recommendations(df, basic_stats, categorical_stats, quality)
        }
        
        self.results['summary'] = summary
        logger.info("Kapsamlı özet rapor oluşturuldu")
        
        return summary
    
    def _generate_recommendations(self, df: pd.DataFrame, 
                                 basic_stats: Dict, 
                                 categorical_stats: Dict,
                                 quality_metrics: Dict) -> List[str]:
        """Veri analizi önerileri oluştur"""
        
        recommendations = []
        
        # Eksik veri önerileri
        if quality_metrics['overall_completeness'] < 90:
            recommendations.append(
                f"Veri setinde %{100-quality_metrics['overall_completeness']:.1f} eksik veri var. "
                "Eksik veri işleme stratejileri uygulanmalı."
            )
        
        # Duplicate önerileri
        if quality_metrics['duplicate_percentage'] > 5:
            recommendations.append(
                f"Veri setinde %{quality_metrics['duplicate_percentage']:.1f} tekrarlayan satır var. "
                "Duplicate kayıtlar temizlenmelidir."
            )
        
        # Outlier önerileri
        for col, stats in basic_stats.items():
            if 'outliers' in self.results.get('distribution', {}).get(col, {}):
                outlier_pct = self.results['distribution'][col]['outliers']['percentage']
                if outlier_pct > 10:
                    recommendations.append(
                        f"{col} sütununda %{outlier_pct:.1f} outlier var. "
                        "Outlier analizi ve işleme yapılmalı."
                    )
        
        # Normallik önerileri
        non_normal_cols = []
        for col, dist_stats in self.results.get('distribution', {}).items():
            normality = dist_stats.get('normality_tests', {})
            if normality.get('shapiro_wilk', {}).get('is_normal') == False:
                non_normal_cols.append(col)
        
        if non_normal_cols:
            recommendations.append(
                f"{len(non_normal_cols)} sütun normal dağılım göstermiyor. "
                "Dönüşüm işlemleri (log, sqrt) uygulanabilir."
            )
        
        # Korelasyon önerileri
        if 'correlation' in self.results:
            strong_corrs = self.results['correlation'].get('strong_positive_correlations', [])
            if len(strong_corrs) > 0:
                recommendations.append(
                    f"{len(strong_corrs)} çift sütun arasında güçlü korelasyon var. "
                    "Çoklu doğrusal bağlantı kontrolü yapılmalı."
                )
        
        # Kategorik değişken önerileri
        for col, stats in categorical_stats.items():
            if stats['unique_count'] > 50:
                recommendations.append(
                    f"{col} sütununda {stats['unique_count']} farklı kategori var. "
                    "Kategori birleştirme veya encoding stratejileri uygulanabilir."
                )
        
        return recommendations
    
    def export_results(self, format: str = 'dict') -> Union[Dict, str]:
        """Sonuçları dışa aktar"""
        
        if format == 'dict':
            return self.results
        elif format == 'json':
            import json
            return json.dumps(self.results, indent=2, default=str)
        else:
            raise ValueError("Desteklenen formatlar: 'dict', 'json'")
    
    def get_column_summary(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Belirli bir sütun için detaylı özet"""
        
        if column not in df.columns:
            raise ValueError(f"Sütun '{column}' veri setinde bulunamadı")
        
        series = df[column]
        
        if series.dtype in [np.number, 'int64', 'float64']:
            # Sayısal sütun
            basic_stats = self.calculate_basic_stats(df, [column])
            distribution_stats = self.calculate_distribution_stats(df, [column])
            
            summary = {
                'column_name': column,
                'data_type': str(series.dtype),
                'basic_statistics': basic_stats.get(column, {}),
                'distribution_analysis': distribution_stats.get(column, {}),
                'sample_values': series.dropna().head(10).tolist()
            }
        else:
            # Kategorik sütun
            categorical_stats = self.calculate_categorical_stats(df, [column])
            
            summary = {
                'column_name': column,
                'data_type': str(series.dtype),
                'categorical_statistics': categorical_stats.get(column, {}),
                'sample_values': series.dropna().head(10).tolist()
            }
        
        return summary