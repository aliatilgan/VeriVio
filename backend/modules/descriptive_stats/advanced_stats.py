"""
VeriVio Gelişmiş Betimsel İstatistikler
İleri seviye istatistiksel analizler ve metrikler
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


class AdvancedStatsCalculator:
    """Gelişmiş istatistiksel hesaplamalar sınıfı"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_robust_statistics(self, df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Robust (dayanıklı) istatistikler hesapla"""
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.columns and 
                           df[col].dtype in [np.number, 'int64', 'float64']]
        
        robust_stats = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 3:
                continue
            
            # Robust merkezi eğilim ölçüleri
            median = series.median()
            trimmed_mean_10 = stats.trim_mean(series, 0.1)  # %10 trimmed mean
            trimmed_mean_20 = stats.trim_mean(series, 0.2)  # %20 trimmed mean
            
            # Robust dağılım ölçüleri
            mad = np.median(np.abs(series - median))  # Median Absolute Deviation
            iqr = series.quantile(0.75) - series.quantile(0.25)
            
            # Robust outlier tespiti
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr_multiplier = 1.5
            lower_fence = q1 - iqr_multiplier * iqr
            upper_fence = q3 + iqr_multiplier * iqr
            outliers_iqr = series[(series < lower_fence) | (series > upper_fence)]
            
            # Modified Z-score ile outlier tespiti
            modified_z_scores = 0.6745 * (series - median) / mad
            outliers_modified_z = series[np.abs(modified_z_scores) > 3.5]
            
            # Winsorized statistics
            winsorized_data = stats.mstats.winsorize(series, limits=[0.05, 0.05])
            winsorized_mean = np.mean(winsorized_data)
            winsorized_std = np.std(winsorized_data)
            
            robust_stats[col] = {
                'median': float(median),
                'trimmed_mean_10': float(trimmed_mean_10),
                'trimmed_mean_20': float(trimmed_mean_20),
                'mad': float(mad),
                'iqr': float(iqr),
                'outliers_iqr': {
                    'count': len(outliers_iqr),
                    'percentage': float((len(outliers_iqr) / len(series)) * 100),
                    'values': outliers_iqr.tolist()[:10]
                },
                'outliers_modified_z': {
                    'count': len(outliers_modified_z),
                    'percentage': float((len(outliers_modified_z) / len(series)) * 100),
                    'values': outliers_modified_z.tolist()[:10]
                },
                'winsorized_mean': float(winsorized_mean),
                'winsorized_std': float(winsorized_std),
                'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else None
            }
        
        self.results['robust_statistics'] = robust_stats
        logger.info(f"{len(numeric_cols)} sütun için robust istatistikler hesaplandı")
        
        return robust_stats
    
    def calculate_distribution_fitting(self, df: pd.DataFrame, 
                                      columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Dağılım uyumu analizi"""
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.columns and 
                           df[col].dtype in [np.number, 'int64', 'float64']]
        
        distribution_fitting = {}
        
        # Test edilecek dağılımlar
        distributions = [
            stats.norm,      # Normal
            stats.lognorm,   # Log-normal
            stats.expon,     # Exponential
            stats.gamma,     # Gamma
            stats.beta,      # Beta
            stats.uniform,   # Uniform
            stats.chi2,      # Chi-square
            stats.t          # Student's t
        ]
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 10:
                continue
            
            # Negatif değerler varsa bazı dağılımları çıkar
            if series.min() <= 0:
                test_distributions = [d for d in distributions if d not in [stats.lognorm, stats.gamma, stats.chi2]]
            else:
                test_distributions = distributions
            
            distribution_results = {}
            
            for dist in test_distributions:
                try:
                    # Parametreleri tahmin et
                    params = dist.fit(series)
                    
                    # Kolmogorov-Smirnov testi
                    ks_stat, ks_p = stats.kstest(series, lambda x: dist.cdf(x, *params))
                    
                    # Anderson-Darling testi (normal dağılım için)
                    if dist == stats.norm:
                        ad_stat, ad_critical, ad_significance = stats.anderson(series, dist='norm')
                        ad_p = None
                        for i, sig in enumerate(ad_significance):
                            if ad_stat < ad_critical[i]:
                                ad_p = 1 - sig/100
                                break
                        if ad_p is None:
                            ad_p = 0.01
                    else:
                        ad_stat, ad_p = None, None
                    
                    # AIC ve BIC hesapla
                    log_likelihood = np.sum(dist.logpdf(series, *params))
                    k = len(params)
                    n = len(series)
                    aic = 2 * k - 2 * log_likelihood
                    bic = k * np.log(n) - 2 * log_likelihood
                    
                    distribution_results[dist.name] = {
                        'parameters': params,
                        'ks_statistic': float(ks_stat),
                        'ks_p_value': float(ks_p),
                        'ad_statistic': float(ad_stat) if ad_stat else None,
                        'ad_p_value': float(ad_p) if ad_p else None,
                        'aic': float(aic),
                        'bic': float(bic),
                        'log_likelihood': float(log_likelihood)
                    }
                    
                except Exception as e:
                    logger.warning(f"{col} sütunu için {dist.name} dağılımı test edilemedi: {str(e)}")
                    continue
            
            # En iyi uyumu bul (en düşük AIC)
            if distribution_results:
                best_fit = min(distribution_results.items(), key=lambda x: x[1]['aic'])
                
                distribution_fitting[col] = {
                    'tested_distributions': distribution_results,
                    'best_fit': {
                        'distribution': best_fit[0],
                        'parameters': best_fit[1]['parameters'],
                        'aic': best_fit[1]['aic'],
                        'ks_p_value': best_fit[1]['ks_p_value']
                    }
                }
        
        self.results['distribution_fitting'] = distribution_fitting
        logger.info(f"{len(numeric_cols)} sütun için dağılım uyumu analizi tamamlandı")
        
        return distribution_fitting
    
    def calculate_multivariate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Çok değişkenli istatistikler"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            logger.warning("Çok değişkenli analiz için en az 2 sayısal sütun gerekli")
            return {}
        
        # Sadece sayısal sütunları al
        numeric_data = df[numeric_cols].dropna()
        
        if len(numeric_data) < 3:
            logger.warning("Çok değişkenli analiz için yeterli veri yok")
            return {}
        
        multivariate_stats = {}
        
        # Kovariance matrisi
        cov_matrix = numeric_data.cov()
        
        # Partial korelasyonlar
        try:
            from sklearn.covariance import GraphicalLassoCV
            
            # Precision matrix (inverse covariance)
            gl = GraphicalLassoCV(cv=3)
            gl.fit(numeric_data)
            precision_matrix = gl.precision_
            
            # Partial korelasyonlar precision matrix'ten hesaplanır
            partial_corr = np.zeros_like(precision_matrix)
            for i in range(len(precision_matrix)):
                for j in range(len(precision_matrix)):
                    if i != j:
                        partial_corr[i, j] = -precision_matrix[i, j] / np.sqrt(
                            precision_matrix[i, i] * precision_matrix[j, j]
                        )
            
            partial_corr_df = pd.DataFrame(
                partial_corr, 
                index=numeric_cols, 
                columns=numeric_cols
            )
            
        except ImportError:
            logger.warning("Partial korelasyon hesaplaması için sklearn gerekli")
            partial_corr_df = None
        
        # Mahalanobis uzaklığı
        try:
            mean = numeric_data.mean()
            cov_inv = np.linalg.pinv(cov_matrix)
            
            mahalanobis_distances = []
            for _, row in numeric_data.iterrows():
                diff = row - mean
                distance = np.sqrt(diff.T @ cov_inv @ diff)
                mahalanobis_distances.append(distance)
            
            mahalanobis_distances = np.array(mahalanobis_distances)
            
            # Mahalanobis outliers (chi-square dağılımı ile)
            threshold = stats.chi2.ppf(0.975, df=len(numeric_cols))
            mahalanobis_outliers = numeric_data[mahalanobis_distances > threshold]
            
        except np.linalg.LinAlgError:
            logger.warning("Mahalanobis uzaklığı hesaplanamadı (singular matrix)")
            mahalanobis_distances = None
            mahalanobis_outliers = None
        
        multivariate_stats = {
            'covariance_matrix': cov_matrix.to_dict(),
            'partial_correlations': partial_corr_df.to_dict() if partial_corr_df is not None else None,
            'mahalanobis_analysis': {
                'distances': mahalanobis_distances.tolist() if mahalanobis_distances is not None else None,
                'outliers_count': len(mahalanobis_outliers) if mahalanobis_outliers is not None else 0,
                'outliers_percentage': float((len(mahalanobis_outliers) / len(numeric_data)) * 100) if mahalanobis_outliers is not None else 0
            }
        }
        
        self.results['multivariate_statistics'] = multivariate_stats
        logger.info("Çok değişkenli istatistikler hesaplandı")
        
        return multivariate_stats
    
    def calculate_information_theory_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Bilgi teorisi metrikleri"""
        
        info_metrics = {}
        
        # Kategorik sütunlar için entropy
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            # Entropy hesapla
            value_counts = series.value_counts()
            probabilities = value_counts / len(series)
            col_entropy = entropy(probabilities, base=2)
            
            # Normalized entropy
            max_entropy = np.log2(len(value_counts))
            normalized_entropy = col_entropy / max_entropy if max_entropy > 0 else 0
            
            info_metrics[col] = {
                'entropy': float(col_entropy),
                'max_possible_entropy': float(max_entropy),
                'normalized_entropy': float(normalized_entropy),
                'unique_values': int(len(value_counts)),
                'gini_impurity': float(1 - np.sum(probabilities ** 2))
            }
        
        # Sayısal sütunlar için differential entropy
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 10:
                continue
            
            try:
                # Histogram tabanlı entropy tahmini
                hist, bin_edges = np.histogram(series, bins='auto', density=True)
                bin_width = bin_edges[1] - bin_edges[0]
                
                # Sıfır olmayan değerler için entropy
                non_zero_hist = hist[hist > 0]
                if len(non_zero_hist) > 0:
                    differential_entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist)) * bin_width
                else:
                    differential_entropy = 0
                
                info_metrics[col] = {
                    'differential_entropy': float(differential_entropy),
                    'histogram_bins': len(hist)
                }
                
            except Exception as e:
                logger.warning(f"{col} sütunu için differential entropy hesaplanamadı: {str(e)}")
        
        # Mutual information (kategorik sütunlar arası)
        mutual_info = {}
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                try:
                    # Contingency table
                    contingency = pd.crosstab(df[col1], df[col2])
                    
                    # Mutual information hesapla
                    mi = 0
                    total = contingency.sum().sum()
                    
                    for i in contingency.index:
                        for j in contingency.columns:
                            if contingency.loc[i, j] > 0:
                                p_xy = contingency.loc[i, j] / total
                                p_x = contingency.loc[i, :].sum() / total
                                p_y = contingency.loc[:, j].sum() / total
                                
                                mi += p_xy * np.log2(p_xy / (p_x * p_y))
                    
                    mutual_info[f"{col1}_vs_{col2}"] = float(mi)
                    
                except Exception as e:
                    logger.warning(f"{col1} ve {col2} için mutual information hesaplanamadı: {str(e)}")
        
        info_metrics['mutual_information'] = mutual_info
        
        self.results['information_theory'] = info_metrics
        logger.info("Bilgi teorisi metrikleri hesaplandı")
        
        return info_metrics
    
    def calculate_time_series_statistics(self, df: pd.DataFrame, 
                                        date_column: str,
                                        value_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Zaman serisi istatistikleri"""
        
        if date_column not in df.columns:
            logger.warning(f"Tarih sütunu '{date_column}' bulunamadı")
            return {}
        
        # Tarih sütununu datetime'a dönüştür
        try:
            df_ts = df.copy()
            df_ts[date_column] = pd.to_datetime(df_ts[date_column])
            df_ts = df_ts.sort_values(date_column)
        except:
            logger.warning(f"'{date_column}' sütunu tarih formatına dönüştürülemedi")
            return {}
        
        if value_columns is None:
            value_columns = df_ts.select_dtypes(include=[np.number]).columns.tolist()
        
        ts_stats = {}
        
        for col in value_columns:
            if col not in df_ts.columns or col == date_column:
                continue
            
            series = df_ts[col].dropna()
            
            if len(series) < 10:
                continue
            
            # Temel zaman serisi istatistikleri
            first_diff = series.diff().dropna()
            
            # Trend analizi (linear regression)
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            
            # Seasonality tespiti (basit)
            # Autocorrelation
            autocorr_lags = min(20, len(series) // 4)
            autocorrelations = [series.autocorr(lag=i) for i in range(1, autocorr_lags + 1)]
            
            # Stationarity testi (Augmented Dickey-Fuller)
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(series.dropna())
                is_stationary = adf_result[1] < 0.05
                adf_statistic = adf_result[0]
                adf_p_value = adf_result[1]
            except ImportError:
                logger.warning("Stationarity testi için statsmodels gerekli")
                is_stationary = None
                adf_statistic = None
                adf_p_value = None
            
            ts_stats[col] = {
                'trend_analysis': {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                },
                'variability': {
                    'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else None,
                    'first_difference_std': float(first_diff.std()),
                    'volatility': float(first_diff.std() / series.mean()) if series.mean() != 0 else None
                },
                'autocorrelation': {
                    'lag_1': float(autocorrelations[0]) if autocorrelations else None,
                    'max_autocorr': float(max(autocorrelations)) if autocorrelations else None,
                    'autocorrelations': [float(ac) for ac in autocorrelations[:10]]
                },
                'stationarity': {
                    'is_stationary': is_stationary,
                    'adf_statistic': float(adf_statistic) if adf_statistic else None,
                    'adf_p_value': float(adf_p_value) if adf_p_value else None
                },
                'descriptive': {
                    'observations': len(series),
                    'time_span_days': (df_ts[date_column].max() - df_ts[date_column].min()).days,
                    'missing_values': int(df_ts[col].isna().sum())
                }
            }
        
        self.results['time_series'] = ts_stats
        logger.info(f"{len(value_columns)} sütun için zaman serisi analizi tamamlandı")
        
        return ts_stats
    
    def generate_advanced_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gelişmiş istatistiksel özet rapor"""
        
        # Tüm gelişmiş analizleri çalıştır
        robust_stats = self.calculate_robust_statistics(df)
        distribution_fitting = self.calculate_distribution_fitting(df)
        multivariate_stats = self.calculate_multivariate_statistics(df)
        info_metrics = self.calculate_information_theory_metrics(df)
        
        # Zaman serisi analizi için tarih sütunu ara
        date_columns = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(100))
                date_columns.append(col)
            except:
                continue
        
        ts_stats = {}
        if date_columns:
            ts_stats = self.calculate_time_series_statistics(df, date_columns[0])
        
        advanced_summary = {
            'robust_statistics': robust_stats,
            'distribution_fitting': distribution_fitting,
            'multivariate_statistics': multivariate_stats,
            'information_theory_metrics': info_metrics,
            'time_series_statistics': ts_stats,
            'analysis_metadata': {
                'total_numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'total_categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'date_columns_detected': len(date_columns),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        self.results['advanced_summary'] = advanced_summary
        logger.info("Gelişmiş istatistiksel özet rapor oluşturuldu")
        
        return advanced_summary