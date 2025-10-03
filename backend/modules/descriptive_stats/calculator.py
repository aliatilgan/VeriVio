"""
VeriVio Gelişmiş Betimsel İstatistikler Modülü
Kapsamlı istatistiksel analiz ve dağılım testleri
Otomatik yorum ve görselleştirme desteği
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import normaltest, shapiro, jarque_bera, anderson, kstest
from scipy.stats import skew, kurtosis, entropy
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DescriptiveStatsCalculator:
    """Gelişmiş betimsel istatistikler hesaplayıcısı"""
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.results = {}
        self.interpretations = {}
        self.distribution_tests = {}
        
    def calculate_comprehensive_stats(self, df: pd.DataFrame, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Kapsamlı betimsel istatistikler hesapla"""
        
        if options is None:
            options = {}
            
        logger.info("Kapsamlı betimsel istatistikler hesaplanıyor")
        
        results = {
            'basic_stats': self._calculate_basic_stats(df),
            'advanced_stats': self._calculate_advanced_stats(df),
            'distribution_analysis': self._analyze_distributions(df),
            'correlation_analysis': self._calculate_correlations(df),
            'categorical_analysis': self._analyze_categorical_variables(df),
            'outlier_analysis': self._analyze_outliers_detailed(df),
            'data_quality_metrics': self._calculate_quality_metrics(df),
            'interpretations': self._generate_interpretations(df)
        }
        
        # Güven aralıkları hesapla
        if options.get('confidence_intervals', True):
            results['confidence_intervals'] = self._calculate_confidence_intervals(df, options.get('confidence_level', 0.95))
        
        # Normallik testleri
        if options.get('normality_tests', True):
            results['normality_tests'] = self._perform_normality_tests(df)
        
        # Dağılım uyum testleri
        if options.get('distribution_fitting', False):
            results['distribution_fitting'] = self._fit_distributions(df)
        
        # Summary oluştur
        results['summary'] = self._generate_summary(df, results)
        
        self.results = results
        logger.info("Betimsel istatistikler hesaplama tamamlandı")
        
        return results
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Temel istatistikler"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "Sayısal sütun bulunamadı"}
        
        basic_stats = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            stats_dict = {
                'count': len(series),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'mode': float(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                'std': float(series.std()),
                'var': float(series.var()),
                'min': float(series.min()),
                'max': float(series.max()),
                'range': float(series.max() - series.min()),
                'q1': float(series.quantile(0.25)),
                'q3': float(series.quantile(0.75)),
                'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                'skewness': float(skew(series)),
                'kurtosis': float(kurtosis(series)),
                'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else None
            }
            
            basic_stats[col] = stats_dict
        
        return basic_stats
    
    def _calculate_advanced_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gelişmiş istatistikler"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "Sayısal sütun bulunamadı"}
        
        advanced_stats = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue
            
            # Momentler
            moments = {
                'first_moment': float(series.mean()),
                'second_moment': float(series.var()),
                'third_moment': float(skew(series)),
                'fourth_moment': float(kurtosis(series, fisher=False))
            }
            
            # Percentile'lar
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95, 99]:
                percentiles[f'p{p}'] = float(series.quantile(p/100))
            
            # Robust istatistikler
            robust_stats = {
                'median_absolute_deviation': float(np.median(np.abs(series - series.median()))),
                'trimmed_mean_10': float(stats.trim_mean(series, 0.1)),
                'trimmed_mean_20': float(stats.trim_mean(series, 0.2)),
                'interquartile_mean': float(series[(series >= series.quantile(0.25)) & 
                                                 (series <= series.quantile(0.75))].mean())
            }
            
            # Entropi ve bilgi teorisi
            try:
                hist, _ = np.histogram(series, bins=min(50, len(series)//10))
                hist = hist[hist > 0]  # Sıfır olmayan değerler
                shannon_entropy = float(entropy(hist, base=2))
            except:
                shannon_entropy = None
            
            advanced_stats[col] = {
                'moments': moments,
                'percentiles': percentiles,
                'robust_stats': robust_stats,
                'shannon_entropy': shannon_entropy,
                'unique_values': int(series.nunique()),
                'unique_ratio': float(series.nunique() / len(series))
            }
        
        return advanced_stats
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Dağılım analizi"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_analysis = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 8:
                continue
            
            # Dağılım şekli analizi
            skewness = skew(series)
            kurt = kurtosis(series)
            
            # Dağılım tipi belirleme
            distribution_type = self._classify_distribution(skewness, kurt)
            
            # Normallik değerlendirmesi
            normality_score = self._assess_normality(series)
            
            distribution_analysis[col] = {
                'skewness': float(skewness),
                'kurtosis': float(kurt),
                'distribution_type': distribution_type,
                'normality_score': normality_score,
                'is_approximately_normal': normality_score > 0.7,
                'distribution_description': self._describe_distribution(skewness, kurt)
            }
        
        return distribution_analysis
    
    def _perform_normality_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Normallik testleri"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        normality_tests = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 8:
                continue
            
            tests = {}
            
            # Shapiro-Wilk testi (n < 5000 için)
            if len(series) <= 5000:
                try:
                    stat, p_value = shapiro(series)
                    tests['shapiro_wilk'] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > 0.05
                    }
                except:
                    tests['shapiro_wilk'] = None
            
            # D'Agostino ve Pearson testi
            try:
                stat, p_value = normaltest(series)
                tests['dagostino_pearson'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
            except:
                tests['dagostino_pearson'] = None
            
            # Jarque-Bera testi
            try:
                stat, p_value = jarque_bera(series)
                tests['jarque_bera'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
            except:
                tests['jarque_bera'] = None
            
            # Anderson-Darling testi
            try:
                result = anderson(series, dist='norm')
                tests['anderson_darling'] = {
                    'statistic': float(result.statistic),
                    'critical_values': result.critical_values.tolist(),
                    'significance_levels': result.significance_level.tolist(),
                    'is_normal': result.statistic < result.critical_values[2]  # %5 seviyesi
                }
            except:
                tests['anderson_darling'] = None
            
            # Kolmogorov-Smirnov testi
            try:
                # Standart normal dağılımla karşılaştır
                normalized = (series - series.mean()) / series.std()
                stat, p_value = kstest(normalized, 'norm')
                tests['kolmogorov_smirnov'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
            except:
                tests['kolmogorov_smirnov'] = None
            
            # Genel normallik değerlendirmesi
            normal_count = sum(1 for test in tests.values() 
                             if test and test.get('is_normal', False))
            total_tests = sum(1 for test in tests.values() if test is not None)
            
            tests['overall_assessment'] = {
                'normal_test_count': normal_count,
                'total_tests': total_tests,
                'normality_confidence': normal_count / total_tests if total_tests > 0 else 0,
                'recommendation': 'normal' if normal_count / total_tests > 0.5 else 'non_normal'
            }
            
            normality_tests[col] = tests
        
        return normality_tests
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Korelasyon analizi"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Korelasyon analizi için en az 2 sayısal sütun gerekli"}
        
        correlations = {}
        
        # Pearson korelasyonu
        pearson_corr = df[numeric_cols].corr(method='pearson')
        correlations['pearson'] = pearson_corr.to_dict()
        
        # Spearman korelasyonu
        spearman_corr = df[numeric_cols].corr(method='spearman')
        correlations['spearman'] = spearman_corr.to_dict()
        
        # Kendall korelasyonu
        kendall_corr = df[numeric_cols].corr(method='kendall')
        correlations['kendall'] = kendall_corr.to_dict()
        
        # Güçlü korelasyonları bul
        strong_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Tekrarları önle
                    pearson_val = abs(pearson_corr.loc[col1, col2])
                    if pearson_val > 0.7:
                        strong_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'pearson_correlation': float(pearson_corr.loc[col1, col2]),
                            'spearman_correlation': float(spearman_corr.loc[col1, col2]),
                            'strength': 'very_strong' if pearson_val > 0.9 else 'strong'
                        })
        
        correlations['strong_correlations'] = strong_correlations
        
        return correlations
    
    def _analyze_categorical_variables(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Kategorik değişken analizi"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {"message": "Kategorik sütun bulunamadı"}
        
        categorical_analysis = {}
        
        for col in categorical_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            value_counts = series.value_counts()
            
            analysis = {
                'unique_count': int(series.nunique()),
                'most_frequent': str(value_counts.index[0]),
                'most_frequent_count': int(value_counts.iloc[0]),
                'most_frequent_percentage': float(value_counts.iloc[0] / len(series) * 100),
                'least_frequent': str(value_counts.index[-1]),
                'least_frequent_count': int(value_counts.iloc[-1]),
                'entropy': float(entropy(value_counts.values, base=2)),
                'concentration_ratio': float(value_counts.iloc[0] / value_counts.sum()),
                'frequency_distribution': value_counts.head(10).to_dict()
            }
            
            # Kategorik değişken tipi belirleme
            unique_ratio = series.nunique() / len(series)
            if unique_ratio > 0.9:
                var_type = "high_cardinality"
            elif unique_ratio < 0.1:
                var_type = "low_cardinality"
            else:
                var_type = "medium_cardinality"
            
            analysis['variable_type'] = var_type
            analysis['cardinality_ratio'] = float(unique_ratio)
            
            categorical_analysis[col] = analysis
        
        return categorical_analysis
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Veri kalitesi metrikleri hesapla"""
        quality_metrics = {}
        
        # Genel veri kalitesi
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        quality_metrics['overall'] = {
            'total_cells': int(total_cells),
            'missing_cells': int(missing_cells),
            'missing_percentage': float(missing_percentage),
            'completeness_score': float(100 - missing_percentage)
        }
        
        # Sütun bazında kalite metrikleri
        column_quality = {}
        for col in df.columns:
            series = df[col]
            missing_count = series.isnull().sum()
            missing_pct = (missing_count / len(series)) * 100
            unique_count = series.nunique()
            unique_pct = (unique_count / len(series)) * 100
            
            # Veri tipi tutarlılığı
            if series.dtype in ['int64', 'float64']:
                # Sayısal sütunlar için
                try:
                    numeric_series = pd.to_numeric(series, errors='coerce')
                    conversion_errors = numeric_series.isnull().sum() - missing_count
                    type_consistency = 100 - (conversion_errors / len(series) * 100)
                except:
                    type_consistency = 100
            else:
                # Kategorik sütunlar için
                type_consistency = 100  # String sütunlar genelde tutarlı
            
            column_quality[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_pct),
                'unique_count': int(unique_count),
                'unique_percentage': float(unique_pct),
                'type_consistency': float(type_consistency),
                'data_type': str(series.dtype),
                'quality_score': float((100 - missing_pct + type_consistency) / 2)
            }
        
        quality_metrics['by_column'] = column_quality
        
        # Genel kalite skoru
        avg_column_quality = np.mean([col['quality_score'] for col in column_quality.values()])
        quality_metrics['overall']['quality_score'] = float(avg_column_quality)
        
        # Kalite kategorisi
        if avg_column_quality >= 90:
            quality_category = "excellent"
        elif avg_column_quality >= 75:
            quality_category = "good"
        elif avg_column_quality >= 60:
            quality_category = "fair"
        else:
            quality_category = "poor"
        
        quality_metrics['overall']['quality_category'] = quality_category
        
        return quality_metrics
    
    def _calculate_confidence_intervals(self, df: pd.DataFrame, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Güven aralıkları hesapla"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "Sayısal sütun bulunamadı"}
        
        confidence_intervals = {}
        alpha = 1 - confidence_level
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 2:
                continue
            
            n = len(series)
            mean = series.mean()
            std_err = series.std() / np.sqrt(n)
            
            # t-dağılımı kullanarak güven aralığı
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
            margin_error = t_critical * std_err
            
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
            
            # Medyan için güven aralığı (bootstrap yöntemi)
            try:
                # Basit bootstrap yaklaşımı
                bootstrap_medians = []
                for _ in range(1000):
                    bootstrap_sample = np.random.choice(series, size=n, replace=True)
                    bootstrap_medians.append(np.median(bootstrap_sample))
                
                median_ci_lower = np.percentile(bootstrap_medians, (alpha/2) * 100)
                median_ci_upper = np.percentile(bootstrap_medians, (1 - alpha/2) * 100)
            except:
                median_ci_lower = None
                median_ci_upper = None
            
            confidence_intervals[col] = {
                'confidence_level': confidence_level,
                'sample_size': int(n),
                'mean_ci': {
                    'lower': float(ci_lower),
                    'upper': float(ci_upper),
                    'margin_of_error': float(margin_error)
                },
                'median_ci': {
                    'lower': float(median_ci_lower) if median_ci_lower is not None else None,
                    'upper': float(median_ci_upper) if median_ci_upper is not None else None
                } if median_ci_lower is not None else None
            }
        
        return confidence_intervals
    
    def _fit_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Dağılım uyum testleri"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "Sayısal sütun bulunamadı"}
        
        distribution_fitting = {}
        
        # Test edilecek dağılımlar
        distributions_to_test = [
            ('norm', 'Normal'),
            ('expon', 'Exponential'),
            ('gamma', 'Gamma'),
            ('lognorm', 'Log-Normal'),
            ('uniform', 'Uniform')
        ]
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            distribution_results = {}
            
            for dist_name, dist_label in distributions_to_test:
                try:
                    # Dağılım parametrelerini tahmin et
                    if dist_name == 'norm':
                        params = stats.norm.fit(series)
                        ks_stat, ks_p = stats.kstest(series, lambda x: stats.norm.cdf(x, *params))
                    elif dist_name == 'expon':
                        params = stats.expon.fit(series)
                        ks_stat, ks_p = stats.kstest(series, lambda x: stats.expon.cdf(x, *params))
                    elif dist_name == 'gamma':
                        params = stats.gamma.fit(series)
                        ks_stat, ks_p = stats.kstest(series, lambda x: stats.gamma.cdf(x, *params))
                    elif dist_name == 'lognorm':
                        # Pozitif değerler için log-normal
                        if (series > 0).all():
                            params = stats.lognorm.fit(series)
                            ks_stat, ks_p = stats.kstest(series, lambda x: stats.lognorm.cdf(x, *params))
                        else:
                            continue
                    elif dist_name == 'uniform':
                        params = stats.uniform.fit(series)
                        ks_stat, ks_p = stats.kstest(series, lambda x: stats.uniform.cdf(x, *params))
                    else:
                        continue
                    
                    # AIC ve BIC hesapla (yaklaşık)
                    log_likelihood = np.sum(getattr(stats, dist_name).logpdf(series, *params))
                    k = len(params)  # Parametre sayısı
                    n = len(series)
                    aic = 2 * k - 2 * log_likelihood
                    bic = k * np.log(n) - 2 * log_likelihood
                    
                    distribution_results[dist_name] = {
                        'distribution_name': dist_label,
                        'parameters': [float(p) for p in params],
                        'ks_statistic': float(ks_stat),
                        'ks_p_value': float(ks_p),
                        'fits_well': ks_p > 0.05,
                        'aic': float(aic),
                        'bic': float(bic),
                        'log_likelihood': float(log_likelihood)
                    }
                    
                except Exception as e:
                    distribution_results[dist_name] = {
                        'distribution_name': dist_label,
                        'error': str(e)
                    }
            
            # En iyi uyumu bul
            valid_fits = {k: v for k, v in distribution_results.items() 
                         if 'ks_p_value' in v and v['fits_well']}
            
            if valid_fits:
                best_fit = max(valid_fits.items(), key=lambda x: x[1]['ks_p_value'])
                distribution_results['best_fit'] = {
                    'distribution': best_fit[0],
                    'distribution_name': best_fit[1]['distribution_name'],
                    'ks_p_value': best_fit[1]['ks_p_value']
                }
            else:
                distribution_results['best_fit'] = None
            
            distribution_fitting[col] = distribution_results
        
        return distribution_fitting

    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Dağılım tipini çarpıklık ve basıklık değerlerine göre sınıflandır"""
        # Çarpıklık değerlendirmesi
        if abs(skewness) < 0.5:
            skew_type = "symmetric"
        elif skewness > 0.5:
            skew_type = "right_skewed"
        else:
            skew_type = "left_skewed"
        
        # Basıklık değerlendirmesi (normal dağılım için kurtosis = 3)
        excess_kurtosis = kurtosis - 3
        if abs(excess_kurtosis) < 0.5:
            kurt_type = "mesokurtic"  # Normal basıklık
        elif excess_kurtosis > 0.5:
            kurt_type = "leptokurtic"  # Yüksek basıklık
        else:
            kurt_type = "platykurtic"  # Düşük basıklık
        
        # Dağılım tipini belirle
        if skew_type == "symmetric" and kurt_type == "mesokurtic":
            return "approximately_normal"
        elif skew_type == "symmetric":
            return f"symmetric_{kurt_type}"
        else:
            return f"{skew_type}_{kurt_type}"
    
    def _assess_normality(self, series: pd.Series) -> float:
        """Normallik skorunu hesapla (0-1 arası)"""
        try:
            # Çarpıklık ve basıklık kontrolü
            skewness = abs(skew(series))
            kurt = abs(kurtosis(series) - 3)  # Excess kurtosis
            
            # Normallik skorları
            skew_score = max(0, 1 - skewness / 2)  # Çarpıklık ne kadar az o kadar iyi
            kurt_score = max(0, 1 - kurt / 4)      # Basıklık ne kadar normale yakın o kadar iyi
            
            # Shapiro-Wilk testi (küçük örneklemler için)
            shapiro_score = 0.5  # Varsayılan
            if len(series) <= 5000:
                try:
                    _, p_value = shapiro(series)
                    shapiro_score = min(1.0, p_value * 2)  # p-value'yu 0-1 arasına normalize et
                except:
                    pass
            
            # Genel normallik skoru (ağırlıklı ortalama)
            normality_score = (skew_score * 0.3 + kurt_score * 0.3 + shapiro_score * 0.4)
            return float(normality_score)
            
        except Exception:
            return 0.5  # Hata durumunda orta değer
    
    def _describe_distribution(self, skewness: float, kurtosis: float) -> str:
        """Dağılımı açıklayıcı metin oluştur"""
        description = []
        
        # Çarpıklık açıklaması
        if abs(skewness) < 0.5:
            description.append("simetrik")
        elif skewness > 0.5:
            if skewness > 1.0:
                description.append("oldukça sağa çarpık")
            else:
                description.append("hafif sağa çarpık")
        else:
            if skewness < -1.0:
                description.append("oldukça sola çarpık")
            else:
                description.append("hafif sola çarpık")
        
        # Basıklık açıklaması
        excess_kurtosis = kurtosis - 3
        if abs(excess_kurtosis) < 0.5:
            description.append("normal basıklıkta")
        elif excess_kurtosis > 0.5:
            if excess_kurtosis > 2.0:
                description.append("oldukça sivri")
            else:
                description.append("hafif sivri")
        else:
            if excess_kurtosis < -2.0:
                description.append("oldukça basık")
            else:
                description.append("hafif basık")
        
        return " ve ".join(description) + " dağılım"
    
    def _analyze_outliers_detailed(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detaylı aykırı değer analizi"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {"message": "Sayısal sütun bulunamadı"}
        
        outlier_analysis = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue
            
            # IQR yöntemi ile aykırı değer tespiti
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            outliers_iqr = series[(series < lower_fence) | (series > upper_fence)]
            
            # Z-score yöntemi ile aykırı değer tespiti
            z_scores = np.abs(stats.zscore(series))
            outliers_zscore = series[z_scores > 3]
            
            # Modified Z-score yöntemi
            median = series.median()
            mad = np.median(np.abs(series - median))
            if mad != 0:
                modified_z_scores = 0.6745 * (series - median) / mad
                outliers_modified_z = series[np.abs(modified_z_scores) > 3.5]
            else:
                outliers_modified_z = pd.Series([], dtype=float)
            
            # Aykırı değer istatistikleri
            outlier_stats = {
                'iqr_method': {
                    'count': len(outliers_iqr),
                    'percentage': float((len(outliers_iqr) / len(series)) * 100),
                    'lower_fence': float(lower_fence),
                    'upper_fence': float(upper_fence),
                    'outlier_values': outliers_iqr.head(10).tolist()
                },
                'zscore_method': {
                    'count': len(outliers_zscore),
                    'percentage': float((len(outliers_zscore) / len(series)) * 100),
                    'outlier_values': outliers_zscore.head(10).tolist()
                },
                'modified_zscore_method': {
                    'count': len(outliers_modified_z),
                    'percentage': float((len(outliers_modified_z) / len(series)) * 100),
                    'outlier_values': outliers_modified_z.head(10).tolist()
                },
                'summary': {
                    'total_observations': len(series),
                    'potential_outliers': len(set(outliers_iqr.index) | set(outliers_zscore.index) | set(outliers_modified_z.index)),
                    'outlier_severity': 'low' if len(outliers_iqr) / len(series) < 0.05 else 'moderate' if len(outliers_iqr) / len(series) < 0.1 else 'high'
                }
            }
            
            outlier_analysis[col] = outlier_stats
        
        return outlier_analysis
    
    def _generate_interpretations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Otomatik yorumlar oluştur"""
        interpretations = {}
        
        # Genel veri seti yorumu
        interpretations['dataset_overview'] = self._interpret_dataset_overview(df)
        
        # Sayısal değişkenler için yorumlar
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            interpretations['numeric_variables'] = self._interpret_numeric_variables(df, numeric_cols)
        
        # Kategorik değişkenler için yorumlar
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            interpretations['categorical_variables'] = self._interpret_categorical_variables(df, categorical_cols)
        
        return interpretations
    
    def _interpret_dataset_overview(self, df: pd.DataFrame) -> str:
        """Veri seti genel yorumu"""
        n_rows, n_cols = df.shape
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        missing_ratio = df.isnull().sum().sum() / (n_rows * n_cols) * 100
        
        interpretation = f"""
        Veri seti {n_rows:,} satır ve {n_cols} sütundan oluşmaktadır.
        {numeric_cols} sayısal ve {categorical_cols} kategorik değişken bulunmaktadır.
        Genel eksik veri oranı %{missing_ratio:.1f}'dir.
        """
        
        if missing_ratio < 5:
            interpretation += " Veri kalitesi yüksektir."
        elif missing_ratio < 15:
            interpretation += " Veri kalitesi orta düzeydedir."
        else:
            interpretation += " Veri kalitesi düşüktür, temizleme gerekebilir."
        
        return interpretation.strip()
    
    def _interpret_numeric_variables(self, df: pd.DataFrame, numeric_cols) -> Dict[str, str]:
        """Sayısal değişkenler için yorumlar"""
        interpretations = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std()
            skewness = skew(series)
            
            # Temel istatistik yorumu
            interpretation = f"{col} değişkeni için ortalama {mean_val:.2f}, medyan {median_val:.2f}'dir. "
            
            # Dağılım yorumu
            if abs(skewness) < 0.5:
                interpretation += "Dağılım simetriktir. "
            elif skewness > 0.5:
                interpretation += "Dağılım sağa çarpıktır (pozitif çarpıklık). "
            else:
                interpretation += "Dağılım sola çarpıktır (negatif çarpıklık). "
            
            # Değişkenlik yorumu
            cv = std_val / mean_val if mean_val != 0 else 0
            if cv < 0.1:
                interpretation += "Değişkenlik düşüktür."
            elif cv < 0.3:
                interpretation += "Değişkenlik orta düzeydedir."
            else:
                interpretation += "Değişkenlik yüksektir."
            
            interpretations[col] = interpretation
        
        return interpretations
    
    def _interpret_categorical_variables(self, df: pd.DataFrame, categorical_cols) -> Dict[str, str]:
        """Kategorik değişkenler için yorumlar"""
        interpretations = {}
        
        for col in categorical_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            value_counts = series.value_counts()
            unique_count = series.nunique()
            most_frequent = value_counts.index[0]
            most_frequent_pct = (value_counts.iloc[0] / len(series)) * 100
            
            interpretation = f"{col} değişkeni {unique_count} farklı kategori içermektedir. "
            interpretation += f"En sık görülen kategori '{most_frequent}' (%{most_frequent_pct:.1f}). "
            
            # Dağılım yorumu
            if unique_count / len(series) > 0.9:
                interpretation += "Yüksek kardinalite (çok fazla benzersiz değer)."
            elif most_frequent_pct > 80:
                interpretation += "Tek bir kategori baskındır."
            elif most_frequent_pct < 20:
                interpretation += "Kategoriler dengeli dağılmıştır."
            else:
                interpretation += "Orta düzeyde kategori çeşitliliği vardır."
            
            interpretations[col] = interpretation
        
        return interpretations

    def generate_summary_report(self) -> str:
        """Özet rapor oluştur"""
        if not self.results:
            return "Henüz analiz yapılmamış."
        
        report = "VeriVio Betimsel İstatistikler Raporu\n"
        report += "=" * 40 + "\n\n"
        
        # Temel istatistikler özeti
        if 'basic_stats' in self.results:
            report += "TEMEL İSTATİSTİKLER\n"
            report += "-" * 20 + "\n"
            for col, stats in self.results['basic_stats'].items():
                if isinstance(stats, dict):
                    report += f"{col}:\n"
                    report += f"  Ortalama: {stats['mean']:.2f}\n"
                    report += f"  Medyan: {stats['median']:.2f}\n"
                    report += f"  Standart Sapma: {stats['std']:.2f}\n\n"
        
        # Normallik testi özeti
        if 'normality_tests' in self.results:
            report += "NORMALLİK TESTLERİ\n"
            report += "-" * 20 + "\n"
            for col, tests in self.results['normality_tests'].items():
                if 'overall_assessment' in tests:
                    assessment = tests['overall_assessment']
                    report += f"{col}: {assessment['recommendation'].upper()}\n"
        
        return report
    
    def _generate_summary(self, df: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analiz özeti oluştur"""
        summary = {
            'dataset_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'missing_values_total': df.isnull().sum().sum(),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            },
            'key_findings': [],
            'data_quality': {
                'completeness_score': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1),
                'columns_with_missing': df.columns[df.isnull().any()].tolist(),
                'duplicate_rows': df.duplicated().sum()
            }
        }
        
        # Sayısal değişkenler için önemli bulgular
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and 'basic_stats' in results:
            basic_stats = results['basic_stats']
            
            # En yüksek ve en düşük ortalamaya sahip değişkenler
            means = {col: stats['mean'] for col, stats in basic_stats.items() 
                    if isinstance(stats, dict) and 'mean' in stats}
            
            if means:
                highest_mean_col = max(means, key=means.get)
                lowest_mean_col = min(means, key=means.get)
                
                summary['key_findings'].extend([
                    f"En yüksek ortalama: {highest_mean_col} ({means[highest_mean_col]:.2f})",
                    f"En düşük ortalama: {lowest_mean_col} ({means[lowest_mean_col]:.2f})"
                ])
            
            # En yüksek varyasyona sahip değişken
            stds = {col: stats['std'] for col, stats in basic_stats.items() 
                   if isinstance(stats, dict) and 'std' in stats}
            
            if stds:
                highest_std_col = max(stds, key=stds.get)
                summary['key_findings'].append(
                    f"En yüksek varyasyon: {highest_std_col} (std: {stds[highest_std_col]:.2f})"
                )
        
        # Kategorik değişkenler için bulgular
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                unique_count = df[col].nunique()
                most_frequent = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                summary['key_findings'].append(
                    f"{col}: {unique_count} benzersiz değer, en sık: {most_frequent}"
                )
        
        # Korelasyon bulguları
        if 'correlation_analysis' in results and results['correlation_analysis']:
            corr_data = results['correlation_analysis']
            if 'strong_correlations' in corr_data and corr_data['strong_correlations']:
                strong_corrs = corr_data['strong_correlations']
                if strong_corrs:
                    summary['key_findings'].append(
                        f"Güçlü korelasyonlar tespit edildi: {len(strong_corrs)} çift"
                    )
        
        # Aykırı değer bulguları
        if 'outlier_analysis' in results and results['outlier_analysis']:
            outlier_data = results['outlier_analysis']
            total_outliers = sum(len(col_outliers.get('outlier_indices', [])) 
                               for col_outliers in outlier_data.values() 
                               if isinstance(col_outliers, dict))
            if total_outliers > 0:
                summary['key_findings'].append(f"Toplam {total_outliers} aykırı değer tespit edildi")
        
        return summary
    
    def calculate(self, columns: list) -> Dict[str, Any]:
        """Basit betimsel istatistikler hesapla (main.py uyumluluğu için)"""
        if not hasattr(self, 'df') or self.df is None:
            raise ValueError("DataFrame not set. Use calculate_comprehensive_stats instead.")
            
        result = {}
        numeric_cols = self.df[columns].select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            series = self.df[col]
            result[col] = {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
            }
        return {'descriptive_stats': result, 'message': 'Betimsel istatistikler hesaplandı.'}