"""
Çevre Bilimleri Analizi Sınıfı

Bu modül çevre bilimlerinde kullanılan özel analiz araçlarını içerir.
Zaman serisi analizi, GIS tabanlı analiz, çevresel veri analizi gibi işlevler sunar.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Zaman serisi analizi için
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Prophet için
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class EnvironmentalAnalyzer:
    """
    Çevre bilimleri veri analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        EnvironmentalAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek çevresel veri seti
        """
        self.data = data.copy()
        self.results = {}
        
    def time_series_analysis(self, value_column: str, 
                           date_column: str,
                           seasonal_period: Optional[int] = None) -> Dict[str, Any]:
        """
        Zaman serisi analizi gerçekleştirir
        
        Args:
            value_column: Değer sütunu
            date_column: Tarih sütunu
            seasonal_period: Mevsimsel periyot (opsiyonel)
            
        Returns:
            Zaman serisi analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            if value_column not in self.data.columns:
                return {'error': f'{value_column} sütunu bulunamadı'}
            
            if date_column not in self.data.columns:
                return {'error': f'{date_column} sütunu bulunamadı'}
            
            if not pd.api.types.is_numeric_dtype(self.data[value_column]):
                return {'error': f'{value_column} sayısal bir sütun olmalıdır'}
            
            # Veriyi hazırla
            data_clean = self.data[[date_column, value_column]].dropna()
            
            if len(data_clean) < 20:
                return {'error': 'Zaman serisi analizi için en az 20 gözlem gereklidir'}
            
            # Tarihi datetime'a çevir
            try:
                data_clean[date_column] = pd.to_datetime(data_clean[date_column])
            except:
                return {'error': f'{date_column} tarih formatına çevrilemedi'}
            
            # Tarihe göre sırala
            data_clean = data_clean.sort_values(date_column)
            data_clean = data_clean.set_index(date_column)
            
            ts_data = data_clean[value_column]
            
            # Temel istatistikler
            basic_stats = {
                'count': len(ts_data),
                'mean': float(ts_data.mean()),
                'std': float(ts_data.std()),
                'min': float(ts_data.min()),
                'max': float(ts_data.max()),
                'trend_slope': self._calculate_trend_slope(ts_data)
            }
            
            # Durağanlık testleri
            stationarity_tests = self._test_stationarity(ts_data)
            
            # Mevsimsel ayrıştırma
            seasonal_decomposition = self._seasonal_decomposition(ts_data, seasonal_period)
            
            # Otokorelasyon analizi
            autocorr_analysis = self._autocorrelation_analysis(ts_data)
            
            # ARIMA modeli (eğer statsmodels varsa)
            arima_results = {}
            if STATSMODELS_AVAILABLE:
                arima_results = self._fit_arima_model(ts_data)
            
            # Prophet modeli (eğer prophet varsa)
            prophet_results = {}
            if PROPHET_AVAILABLE:
                prophet_results = self._fit_prophet_model(data_clean.reset_index(), 
                                                        date_column, value_column)
            
            # Anomali tespiti
            anomaly_detection = self._detect_anomalies(ts_data)
            
            # Değişim noktası analizi
            change_point_analysis = self._detect_change_points(ts_data)
            
            # Sonuçları birleştir
            results = {
                'basic_statistics': basic_stats,
                'stationarity_tests': stationarity_tests,
                'seasonal_decomposition': seasonal_decomposition,
                'autocorrelation_analysis': autocorr_analysis,
                'arima_results': arima_results,
                'prophet_results': prophet_results,
                'anomaly_detection': anomaly_detection,
                'change_point_analysis': change_point_analysis,
                'interpretation': self._interpret_time_series_results(
                    basic_stats, stationarity_tests, seasonal_decomposition
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Zaman serisi analizi hatası: {str(e)}'}
    
    def environmental_correlation_analysis(self, environmental_variables: List[str],
                                         target_variable: str,
                                         lag_analysis: bool = True) -> Dict[str, Any]:
        """
        Çevresel değişkenler arası korelasyon analizi
        
        Args:
            environmental_variables: Çevresel değişken sütunları
            target_variable: Hedef değişken
            lag_analysis: Gecikme analizi yapılsın mı
            
        Returns:
            Çevresel korelasyon analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            all_columns = environmental_variables + [target_variable]
            missing_cols = [col for col in all_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal sütunları kontrol et
            numeric_cols = []
            for col in all_columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_cols.append(col)
            
            if len(numeric_cols) < 2:
                return {'error': 'En az 2 sayısal sütun gereklidir'}
            
            data_clean = self.data[numeric_cols].dropna()
            
            if len(data_clean) < 10:
                return {'error': 'Korelasyon analizi için en az 10 gözlem gereklidir'}
            
            # Temel korelasyon analizi
            correlation_matrix = data_clean.corr()
            
            # Hedef değişkenle korelasyonlar
            target_correlations = {}
            if target_variable in correlation_matrix.columns:
                target_corr = correlation_matrix[target_variable].drop(target_variable)
                
                for var in target_corr.index:
                    if var in data_clean.columns:
                        # Pearson korelasyonu
                        pearson_r, pearson_p = pearsonr(data_clean[var], data_clean[target_variable])
                        
                        # Spearman korelasyonu
                        spearman_r, spearman_p = spearmanr(data_clean[var], data_clean[target_variable])
                        
                        target_correlations[var] = {
                            'pearson_correlation': float(pearson_r),
                            'pearson_p_value': float(pearson_p),
                            'spearman_correlation': float(spearman_r),
                            'spearman_p_value': float(spearman_p),
                            'pearson_significant': pearson_p < 0.05,
                            'spearman_significant': spearman_p < 0.05
                        }
            
            # Gecikme analizi
            lag_analysis_results = {}
            if lag_analysis and target_variable in data_clean.columns:
                lag_analysis_results = self._lag_correlation_analysis(
                    data_clean, environmental_variables, target_variable
                )
            
            # Kısmi korelasyon analizi
            partial_correlations = self._partial_correlation_analysis(
                data_clean, environmental_variables, target_variable
            )
            
            # Çevresel faktör grupları analizi
            factor_groups = self._environmental_factor_grouping(
                data_clean, environmental_variables
            )
            
            # Mevsimsel korelasyon (eğer tarih sütunu varsa)
            seasonal_correlations = {}
            date_columns = [col for col in self.data.columns 
                          if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_columns:
                seasonal_correlations = self._seasonal_correlation_analysis(
                    self.data, date_columns[0], environmental_variables, target_variable
                )
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'correlation_matrix': correlation_matrix.to_dict(),
                'target_correlations': target_correlations,
                'lag_analysis': lag_analysis_results,
                'partial_correlations': partial_correlations,
                'factor_groups': factor_groups,
                'seasonal_correlations': seasonal_correlations,
                'interpretation': self._interpret_environmental_correlations(
                    target_correlations, lag_analysis_results
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Çevresel korelasyon analizi hatası: {str(e)}'}
    
    def pollution_trend_analysis(self, pollutant_columns: List[str],
                                date_column: str,
                                location_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Kirlilik trend analizi gerçekleştirir
        
        Args:
            pollutant_columns: Kirletici madde sütunları
            date_column: Tarih sütunu
            location_column: Konum sütunu (opsiyonel)
            
        Returns:
            Kirlilik trend analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            all_columns = pollutant_columns + [date_column]
            if location_column:
                all_columns.append(location_column)
            
            missing_cols = [col for col in all_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal kirletici sütunlarını kontrol et
            numeric_pollutants = []
            for col in pollutant_columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_pollutants.append(col)
            
            if len(numeric_pollutants) == 0:
                return {'error': 'En az 1 sayısal kirletici sütunu gereklidir'}
            
            data_clean = self.data[all_columns].dropna()
            
            if len(data_clean) < 20:
                return {'error': 'Trend analizi için en az 20 gözlem gereklidir'}
            
            # Tarihi datetime'a çevir
            try:
                data_clean[date_column] = pd.to_datetime(data_clean[date_column])
            except:
                return {'error': f'{date_column} tarih formatına çevrilemedi'}
            
            # Kirletici trend analizi
            pollutant_trends = {}
            for pollutant in numeric_pollutants:
                trend_analysis = self._analyze_pollutant_trend(
                    data_clean, pollutant, date_column
                )
                pollutant_trends[pollutant] = trend_analysis
            
            # Konum bazlı analiz
            location_analysis = {}
            if location_column and location_column in data_clean.columns:
                location_analysis = self._location_based_pollution_analysis(
                    data_clean, numeric_pollutants, location_column, date_column
                )
            
            # Mevsimsel analiz
            seasonal_analysis = self._seasonal_pollution_analysis(
                data_clean, numeric_pollutants, date_column
            )
            
            # Kirletici korelasyonları
            pollutant_correlations = data_clean[numeric_pollutants].corr()
            
            # Limit aşımları analizi
            exceedance_analysis = self._pollution_exceedance_analysis(
                data_clean, numeric_pollutants
            )
            
            # Hava kalitesi indeksi (basit)
            air_quality_index = self._calculate_air_quality_index(
                data_clean, numeric_pollutants
            )
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'analysis_period': {
                    'start_date': str(data_clean[date_column].min().date()),
                    'end_date': str(data_clean[date_column].max().date()),
                    'duration_days': (data_clean[date_column].max() - data_clean[date_column].min()).days
                },
                'pollutant_trends': pollutant_trends,
                'location_analysis': location_analysis,
                'seasonal_analysis': seasonal_analysis,
                'pollutant_correlations': pollutant_correlations.to_dict(),
                'exceedance_analysis': exceedance_analysis,
                'air_quality_index': air_quality_index,
                'interpretation': self._interpret_pollution_trends(
                    pollutant_trends, seasonal_analysis, exceedance_analysis
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Kirlilik trend analizi hatası: {str(e)}'}
    
    def climate_change_analysis(self, temperature_column: str,
                              precipitation_column: Optional[str] = None,
                              date_column: str = None,
                              reference_period: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        İklim değişikliği analizi gerçekleştirir
        
        Args:
            temperature_column: Sıcaklık sütunu
            precipitation_column: Yağış sütunu (opsiyonel)
            date_column: Tarih sütunu
            reference_period: Referans periyot (başlangıç, bitiş)
            
        Returns:
            İklim değişikliği analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            required_columns = [temperature_column]
            if date_column:
                required_columns.append(date_column)
            if precipitation_column:
                required_columns.append(precipitation_column)
            
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            if not pd.api.types.is_numeric_dtype(self.data[temperature_column]):
                return {'error': f'{temperature_column} sayısal bir sütun olmalıdır'}
            
            data_clean = self.data[required_columns].dropna()
            
            if len(data_clean) < 30:
                return {'error': 'İklim analizi için en az 30 gözlem gereklidir'}
            
            # Tarih işleme
            if date_column:
                try:
                    data_clean[date_column] = pd.to_datetime(data_clean[date_column])
                    data_clean = data_clean.sort_values(date_column)
                except:
                    return {'error': f'{date_column} tarih formatına çevrilemedi'}
            
            # Sıcaklık trend analizi
            temperature_analysis = self._temperature_trend_analysis(
                data_clean, temperature_column, date_column
            )
            
            # Yağış analizi
            precipitation_analysis = {}
            if precipitation_column and precipitation_column in data_clean.columns:
                if pd.api.types.is_numeric_dtype(data_clean[precipitation_column]):
                    precipitation_analysis = self._precipitation_analysis(
                        data_clean, precipitation_column, date_column
                    )
            
            # Ekstrem olaylar analizi
            extreme_events = self._extreme_weather_analysis(
                data_clean, temperature_column, precipitation_column
            )
            
            # Referans periyot karşılaştırması
            reference_comparison = {}
            if reference_period and date_column:
                reference_comparison = self._reference_period_comparison(
                    data_clean, temperature_column, precipitation_column, 
                    date_column, reference_period
                )
            
            # İklim indeksleri
            climate_indices = self._calculate_climate_indices(
                data_clean, temperature_column, precipitation_column, date_column
            )
            
            # Değişkenlik analizi
            variability_analysis = self._climate_variability_analysis(
                data_clean, temperature_column, precipitation_column, date_column
            )
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'temperature_analysis': temperature_analysis,
                'precipitation_analysis': precipitation_analysis,
                'extreme_events': extreme_events,
                'reference_comparison': reference_comparison,
                'climate_indices': climate_indices,
                'variability_analysis': variability_analysis,
                'interpretation': self._interpret_climate_analysis(
                    temperature_analysis, precipitation_analysis, extreme_events
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'İklim değişikliği analizi hatası: {str(e)}'}
    
    # Yardımcı metodlar
    
    def _calculate_trend_slope(self, ts_data: pd.Series) -> float:
        """
        Zaman serisinin trend eğimini hesaplar
        """
        try:
            x = np.arange(len(ts_data))
            slope, _, _, _, _ = stats.linregress(x, ts_data.values)
            return float(slope)
        except:
            return 0.0
    
    def _test_stationarity(self, ts_data: pd.Series) -> Dict[str, Any]:
        """
        Durağanlık testleri yapar
        """
        results = {}
        
        try:
            # Augmented Dickey-Fuller test
            if STATSMODELS_AVAILABLE:
                adf_result = adfuller(ts_data.dropna())
                results['adf_test'] = {
                    'statistic': float(adf_result[0]),
                    'p_value': float(adf_result[1]),
                    'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                    'is_stationary': adf_result[1] < 0.05
                }
                
                # KPSS test
                kpss_result = kpss(ts_data.dropna())
                results['kpss_test'] = {
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                    'is_stationary': kpss_result[1] > 0.05
                }
            else:
                results['message'] = 'Statsmodels kütüphanesi bulunamadı'
                
        except Exception as e:
            results['error'] = f'Durağanlık testi hatası: {str(e)}'
        
        return results
    
    def _seasonal_decomposition(self, ts_data: pd.Series, 
                              seasonal_period: Optional[int]) -> Dict[str, Any]:
        """
        Mevsimsel ayrıştırma yapar
        """
        try:
            if not STATSMODELS_AVAILABLE:
                return {'message': 'Statsmodels kütüphanesi bulunamadı'}
            
            if seasonal_period is None:
                # Otomatik periyot tespiti (basit)
                seasonal_period = min(12, len(ts_data) // 4)
            
            if len(ts_data) < 2 * seasonal_period:
                return {'error': 'Mevsimsel ayrıştırma için yeterli veri yok'}
            
            decomposition = seasonal_decompose(ts_data, model='additive', 
                                             period=seasonal_period)
            
            return {
                'seasonal_period': seasonal_period,
                'trend_strength': float(np.var(decomposition.trend.dropna()) / np.var(ts_data)),
                'seasonal_strength': float(np.var(decomposition.seasonal) / np.var(ts_data)),
                'residual_variance': float(np.var(decomposition.resid.dropna()))
            }
            
        except Exception as e:
            return {'error': f'Mevsimsel ayrıştırma hatası: {str(e)}'}
    
    def _autocorrelation_analysis(self, ts_data: pd.Series) -> Dict[str, Any]:
        """
        Otokorelasyon analizi yapar
        """
        try:
            # Basit otokorelasyon hesaplama
            autocorr_values = []
            max_lag = min(20, len(ts_data) // 4)
            
            for lag in range(1, max_lag + 1):
                if len(ts_data) > lag:
                    corr = ts_data.autocorr(lag=lag)
                    if not np.isnan(corr):
                        autocorr_values.append({'lag': lag, 'autocorr': float(corr)})
            
            # Ljung-Box test (eğer statsmodels varsa)
            ljung_box_result = {}
            if STATSMODELS_AVAILABLE and len(ts_data) > 10:
                try:
                    lb_result = acorr_ljungbox(ts_data.dropna(), lags=min(10, len(ts_data)//4))
                    ljung_box_result = {
                        'statistic': float(lb_result['lb_stat'].iloc[-1]),
                        'p_value': float(lb_result['lb_pvalue'].iloc[-1]),
                        'significant_autocorr': lb_result['lb_pvalue'].iloc[-1] < 0.05
                    }
                except:
                    ljung_box_result = {'error': 'Ljung-Box testi hesaplanamadı'}
            
            return {
                'autocorrelations': autocorr_values,
                'ljung_box_test': ljung_box_result
            }
            
        except Exception as e:
            return {'error': f'Otokorelasyon analizi hatası: {str(e)}'}
    
    def _fit_arima_model(self, ts_data: pd.Series) -> Dict[str, Any]:
        """
        ARIMA modeli uydurur
        """
        try:
            if not STATSMODELS_AVAILABLE:
                return {'message': 'Statsmodels kütüphanesi bulunamadı'}
            
            if len(ts_data) < 20:
                return {'error': 'ARIMA modeli için yeterli veri yok'}
            
            # Basit ARIMA(1,1,1) modeli
            model = ARIMA(ts_data.dropna(), order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Model istatistikleri
            return {
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic),
                'log_likelihood': float(fitted_model.llf),
                'parameters': fitted_model.params.to_dict(),
                'residual_std': float(np.std(fitted_model.resid))
            }
            
        except Exception as e:
            return {'error': f'ARIMA modeli hatası: {str(e)}'}
    
    def _fit_prophet_model(self, data: pd.DataFrame, 
                          date_column: str, 
                          value_column: str) -> Dict[str, Any]:
        """
        Prophet modeli uydurur
        """
        try:
            if not PROPHET_AVAILABLE:
                return {'message': 'Prophet kütüphanesi bulunamadı'}
            
            if len(data) < 20:
                return {'error': 'Prophet modeli için yeterli veri yok'}
            
            # Prophet için veri formatı
            prophet_data = data[[date_column, value_column]].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Model oluştur ve uydur
            model = Prophet(daily_seasonality=False, weekly_seasonality=False)
            model.fit(prophet_data)
            
            # Gelecek tahminleri
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Model performansı
            y_true = prophet_data['y'].values
            y_pred = forecast['yhat'][:len(y_true)].values
            
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'trend_strength': float(np.std(forecast['trend'])),
                'forecast_period': 30
            }
            
        except Exception as e:
            return {'error': f'Prophet modeli hatası: {str(e)}'}
    
    def _detect_anomalies(self, ts_data: pd.Series) -> Dict[str, Any]:
        """
        Anomali tespiti yapar
        """
        try:
            # IQR yöntemi ile anomali tespiti
            q1 = ts_data.quantile(0.25)
            q3 = ts_data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            anomalies = ts_data[(ts_data < lower_bound) | (ts_data > upper_bound)]
            
            # Z-score yöntemi
            z_scores = np.abs(stats.zscore(ts_data.dropna()))
            z_anomalies = ts_data[z_scores > 3]
            
            return {
                'iqr_method': {
                    'anomaly_count': len(anomalies),
                    'anomaly_percentage': float(len(anomalies) / len(ts_data) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                },
                'zscore_method': {
                    'anomaly_count': len(z_anomalies),
                    'anomaly_percentage': float(len(z_anomalies) / len(ts_data) * 100)
                }
            }
            
        except Exception as e:
            return {'error': f'Anomali tespiti hatası: {str(e)}'}
    
    def _detect_change_points(self, ts_data: pd.Series) -> Dict[str, Any]:
        """
        Değişim noktası tespiti yapar
        """
        try:
            # Basit değişim noktası tespiti (varyans değişimi)
            window_size = max(10, len(ts_data) // 10)
            change_points = []
            
            for i in range(window_size, len(ts_data) - window_size):
                before = ts_data.iloc[i-window_size:i]
                after = ts_data.iloc[i:i+window_size]
                
                # Ortalama değişimi
                mean_change = abs(before.mean() - after.mean())
                
                # Varyans değişimi
                var_change = abs(before.var() - after.var())
                
                # Eşik değerlerini aş
                if mean_change > ts_data.std() or var_change > ts_data.var():
                    change_points.append({
                        'index': i,
                        'mean_change': float(mean_change),
                        'variance_change': float(var_change)
                    })
            
            return {
                'change_point_count': len(change_points),
                'change_points': change_points[:10]  # İlk 10 tanesi
            }
            
        except Exception as e:
            return {'error': f'Değişim noktası tespiti hatası: {str(e)}'}
    
    def _lag_correlation_analysis(self, data: pd.DataFrame, 
                                environmental_vars: List[str], 
                                target_var: str) -> Dict[str, Any]:
        """
        Gecikme korelasyon analizi yapar
        """
        try:
            lag_results = {}
            max_lag = min(10, len(data) // 4)
            
            for var in environmental_vars:
                if var in data.columns and pd.api.types.is_numeric_dtype(data[var]):
                    var_lag_results = []
                    
                    for lag in range(1, max_lag + 1):
                        if len(data) > lag:
                            lagged_var = data[var].shift(lag)
                            valid_indices = ~(lagged_var.isna() | data[target_var].isna())
                            
                            if valid_indices.sum() > 10:
                                corr, p_val = pearsonr(
                                    lagged_var[valid_indices], 
                                    data[target_var][valid_indices]
                                )
                                
                                var_lag_results.append({
                                    'lag': lag,
                                    'correlation': float(corr),
                                    'p_value': float(p_val),
                                    'significant': p_val < 0.05
                                })
                    
                    lag_results[var] = var_lag_results
            
            return lag_results
            
        except Exception as e:
            return {'error': f'Gecikme analizi hatası: {str(e)}'}
    
    def _partial_correlation_analysis(self, data: pd.DataFrame, 
                                    environmental_vars: List[str], 
                                    target_var: str) -> Dict[str, Any]:
        """
        Kısmi korelasyon analizi yapar
        """
        try:
            # Basit kısmi korelasyon hesaplama
            partial_corrs = {}
            
            for var in environmental_vars:
                if var in data.columns and pd.api.types.is_numeric_dtype(data[var]):
                    # Diğer değişkenleri kontrol değişkeni olarak kullan
                    control_vars = [v for v in environmental_vars if v != var and v in data.columns]
                    
                    if len(control_vars) > 0:
                        # Basit regresyon yaklaşımı
                        from sklearn.linear_model import LinearRegression
                        
                        # Kontrol değişkenlerinin etkisini çıkar
                        control_data = data[control_vars].dropna()
                        
                        if len(control_data) > 10:
                            # Target'tan kontrol değişkenlerinin etkisini çıkar
                            reg_target = LinearRegression()
                            reg_target.fit(control_data, data[target_var])
                            target_residuals = data[target_var] - reg_target.predict(control_data)
                            
                            # Var'dan kontrol değişkenlerinin etkisini çıkar
                            reg_var = LinearRegression()
                            reg_var.fit(control_data, data[var])
                            var_residuals = data[var] - reg_var.predict(control_data)
                            
                            # Kalıntılar arası korelasyon
                            partial_corr, partial_p = pearsonr(var_residuals, target_residuals)
                            
                            partial_corrs[var] = {
                                'partial_correlation': float(partial_corr),
                                'p_value': float(partial_p),
                                'significant': partial_p < 0.05
                            }
            
            return partial_corrs
            
        except Exception as e:
            return {'error': f'Kısmi korelasyon analizi hatası: {str(e)}'}
    
    def _environmental_factor_grouping(self, data: pd.DataFrame, 
                                     environmental_vars: List[str]) -> Dict[str, Any]:
        """
        Çevresel faktörleri gruplandırır
        """
        try:
            # Korelasyon bazlı gruplandırma
            numeric_vars = [var for var in environmental_vars 
                          if var in data.columns and pd.api.types.is_numeric_dtype(data[var])]
            
            if len(numeric_vars) < 2:
                return {'message': 'Gruplandırma için yeterli sayısal değişken yok'}
            
            corr_matrix = data[numeric_vars].corr()
            
            # Yüksek korelasyonlu grupları bul
            high_corr_groups = []
            processed_vars = set()
            
            for i, var1 in enumerate(numeric_vars):
                if var1 not in processed_vars:
                    group = [var1]
                    processed_vars.add(var1)
                    
                    for j, var2 in enumerate(numeric_vars):
                        if i != j and var2 not in processed_vars:
                            if abs(corr_matrix.loc[var1, var2]) > 0.7:
                                group.append(var2)
                                processed_vars.add(var2)
                    
                    if len(group) > 1:
                        high_corr_groups.append(group)
            
            return {
                'high_correlation_groups': high_corr_groups,
                'correlation_threshold': 0.7
            }
            
        except Exception as e:
            return {'error': f'Faktör gruplandırma hatası: {str(e)}'}
    
    def _seasonal_correlation_analysis(self, data: pd.DataFrame, 
                                     date_column: str,
                                     environmental_vars: List[str], 
                                     target_var: str) -> Dict[str, Any]:
        """
        Mevsimsel korelasyon analizi yapar
        """
        try:
            # Tarihi datetime'a çevir
            data_copy = data.copy()
            data_copy[date_column] = pd.to_datetime(data_copy[date_column])
            
            # Mevsim bilgisi ekle
            data_copy['month'] = data_copy[date_column].dt.month
            data_copy['season'] = data_copy['month'].map({
                12: 'Kış', 1: 'Kış', 2: 'Kış',
                3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
                6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
                9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
            })
            
            seasonal_correlations = {}
            
            for season in ['İlkbahar', 'Yaz', 'Sonbahar', 'Kış']:
                season_data = data_copy[data_copy['season'] == season]
                
                if len(season_data) > 10:
                    season_corrs = {}
                    
                    for var in environmental_vars:
                        if (var in season_data.columns and 
                            pd.api.types.is_numeric_dtype(season_data[var]) and
                            target_var in season_data.columns):
                            
                            corr, p_val = pearsonr(season_data[var], season_data[target_var])
                            season_corrs[var] = {
                                'correlation': float(corr),
                                'p_value': float(p_val),
                                'significant': p_val < 0.05,
                                'sample_size': len(season_data)
                            }
                    
                    seasonal_correlations[season] = season_corrs
            
            return seasonal_correlations
            
        except Exception as e:
            return {'error': f'Mevsimsel korelasyon analizi hatası: {str(e)}'}
    
    def _analyze_pollutant_trend(self, data: pd.DataFrame, 
                               pollutant: str, 
                               date_column: str) -> Dict[str, Any]:
        """
        Tek bir kirletici için trend analizi yapar
        """
        try:
            # Zaman bazlı trend
            data_sorted = data.sort_values(date_column)
            pollutant_values = data_sorted[pollutant]
            
            # Doğrusal trend
            x = np.arange(len(pollutant_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, pollutant_values)
            
            # Yıllık ortalamalar
            data_sorted['year'] = data_sorted[date_column].dt.year
            yearly_means = data_sorted.groupby('year')[pollutant].mean()
            
            # Yıllık trend
            if len(yearly_means) > 2:
                years = yearly_means.index.values
                yearly_slope, _, yearly_r, yearly_p, _ = stats.linregress(years, yearly_means.values)
            else:
                yearly_slope = yearly_r = yearly_p = 0
            
            return {
                'overall_trend': {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'trend_direction': 'Artış' if slope > 0 else 'Azalış' if slope < 0 else 'Sabit'
                },
                'yearly_trend': {
                    'slope': float(yearly_slope),
                    'r_squared': float(yearly_r ** 2),
                    'p_value': float(yearly_p),
                    'significant': yearly_p < 0.05
                },
                'basic_statistics': {
                    'mean': float(pollutant_values.mean()),
                    'std': float(pollutant_values.std()),
                    'min': float(pollutant_values.min()),
                    'max': float(pollutant_values.max())
                }
            }
            
        except Exception as e:
            return {'error': f'Kirletici trend analizi hatası: {str(e)}'}
    
    def _location_based_pollution_analysis(self, data: pd.DataFrame, 
                                         pollutants: List[str],
                                         location_column: str, 
                                         date_column: str) -> Dict[str, Any]:
        """
        Konum bazlı kirlilik analizi yapar
        """
        try:
            location_analysis = {}
            
            for location in data[location_column].unique():
                location_data = data[data[location_column] == location]
                
                if len(location_data) > 5:
                    location_stats = {}
                    
                    for pollutant in pollutants:
                        if pollutant in location_data.columns:
                            pollutant_data = location_data[pollutant]
                            
                            location_stats[pollutant] = {
                                'mean': float(pollutant_data.mean()),
                                'std': float(pollutant_data.std()),
                                'min': float(pollutant_data.min()),
                                'max': float(pollutant_data.max()),
                                'sample_size': len(pollutant_data)
                            }
                    
                    location_analysis[str(location)] = location_stats
            
            return location_analysis
            
        except Exception as e:
            return {'error': f'Konum bazlı analiz hatası: {str(e)}'}
    
    def _seasonal_pollution_analysis(self, data: pd.DataFrame, 
                                   pollutants: List[str], 
                                   date_column: str) -> Dict[str, Any]:
        """
        Mevsimsel kirlilik analizi yapar
        """
        try:
            # Mevsim bilgisi ekle
            data_copy = data.copy()
            data_copy['month'] = data_copy[date_column].dt.month
            data_copy['season'] = data_copy['month'].map({
                12: 'Kış', 1: 'Kış', 2: 'Kış',
                3: 'İlkbahar', 4: 'İlkbahar', 5: 'İlkbahar',
                6: 'Yaz', 7: 'Yaz', 8: 'Yaz',
                9: 'Sonbahar', 10: 'Sonbahar', 11: 'Sonbahar'
            })
            
            seasonal_analysis = {}
            
            for season in ['İlkbahar', 'Yaz', 'Sonbahar', 'Kış']:
                season_data = data_copy[data_copy['season'] == season]
                
                if len(season_data) > 5:
                    season_stats = {}
                    
                    for pollutant in pollutants:
                        if pollutant in season_data.columns:
                            pollutant_data = season_data[pollutant]
                            
                            season_stats[pollutant] = {
                                'mean': float(pollutant_data.mean()),
                                'std': float(pollutant_data.std()),
                                'min': float(pollutant_data.min()),
                                'max': float(pollutant_data.max()),
                                'sample_size': len(pollutant_data)
                            }
                    
                    seasonal_analysis[season] = season_stats
            
            return seasonal_analysis
            
        except Exception as e:
            return {'error': f'Mevsimsel analiz hatası: {str(e)}'}
    
    def _pollution_exceedance_analysis(self, data: pd.DataFrame, 
                                     pollutants: List[str]) -> Dict[str, Any]:
        """
        Kirlilik limit aşımları analizi yapar
        """
        try:
            # Örnek limit değerleri (gerçek uygulamada standart değerler kullanılmalı)
            standard_limits = {
                'PM10': 50,  # μg/m³
                'PM2.5': 25,  # μg/m³
                'NO2': 40,   # μg/m³
                'SO2': 20,   # μg/m³
                'O3': 120,   # μg/m³
                'CO': 10     # mg/m³
            }
            
            exceedance_analysis = {}
            
            for pollutant in pollutants:
                if pollutant in data.columns:
                    pollutant_data = data[pollutant]
                    
                    # Limit değerini bul (varsayılan olarak 95. percentile kullan)
                    limit_value = standard_limits.get(pollutant, pollutant_data.quantile(0.95))
                    
                    exceedances = pollutant_data > limit_value
                    exceedance_count = exceedances.sum()
                    exceedance_percentage = (exceedance_count / len(pollutant_data)) * 100
                    
                    exceedance_analysis[pollutant] = {
                        'limit_value': float(limit_value),
                        'exceedance_count': int(exceedance_count),
                        'exceedance_percentage': float(exceedance_percentage),
                        'max_exceedance': float(pollutant_data.max() - limit_value) if exceedance_count > 0 else 0,
                        'total_observations': len(pollutant_data)
                    }
            
            return exceedance_analysis
            
        except Exception as e:
            return {'error': f'Limit aşımı analizi hatası: {str(e)}'}
    
    def _calculate_air_quality_index(self, data: pd.DataFrame, 
                                   pollutants: List[str]) -> Dict[str, Any]:
        """
        Basit hava kalitesi indeksi hesaplar
        """
        try:
            # Basit AQI hesaplama (normalize edilmiş değerler)
            aqi_scores = []
            
            for _, row in data.iterrows():
                pollutant_scores = []
                
                for pollutant in pollutants:
                    if pollutant in row and not pd.isna(row[pollutant]):
                        # Normalize et (0-100 arası)
                        pollutant_series = data[pollutant]
                        min_val = pollutant_series.min()
                        max_val = pollutant_series.max()
                        
                        if max_val > min_val:
                            normalized_score = ((row[pollutant] - min_val) / (max_val - min_val)) * 100
                            pollutant_scores.append(normalized_score)
                
                if pollutant_scores:
                    # En yüksek kirletici skorunu al
                    aqi_score = max(pollutant_scores)
                    aqi_scores.append(aqi_score)
            
            if aqi_scores:
                aqi_array = np.array(aqi_scores)
                
                # AQI kategorileri
                def categorize_aqi(score):
                    if score <= 20:
                        return 'İyi'
                    elif score <= 40:
                        return 'Orta'
                    elif score <= 60:
                        return 'Hassas Gruplar için Sağlıksız'
                    elif score <= 80:
                        return 'Sağlıksız'
                    else:
                        return 'Çok Sağlıksız'
                
                aqi_categories = [categorize_aqi(score) for score in aqi_scores]
                category_counts = pd.Series(aqi_categories).value_counts()
                
                return {
                    'mean_aqi': float(aqi_array.mean()),
                    'max_aqi': float(aqi_array.max()),
                    'min_aqi': float(aqi_array.min()),
                    'category_distribution': category_counts.to_dict(),
                    'total_observations': len(aqi_scores)
                }
            else:
                return {'message': 'AQI hesaplanamadı'}
                
        except Exception as e:
            return {'error': f'AQI hesaplama hatası: {str(e)}'}
    
    def _temperature_trend_analysis(self, data: pd.DataFrame, 
                                  temperature_column: str, 
                                  date_column: Optional[str]) -> Dict[str, Any]:
        """
        Sıcaklık trend analizi yapar
        """
        try:
            temp_data = data[temperature_column]
            
            # Temel istatistikler
            basic_stats = {
                'mean': float(temp_data.mean()),
                'std': float(temp_data.std()),
                'min': float(temp_data.min()),
                'max': float(temp_data.max()),
                'range': float(temp_data.max() - temp_data.min())
            }
            
            # Trend analizi
            trend_analysis = {}
            if date_column:
                data_sorted = data.sort_values(date_column)
                temp_sorted = data_sorted[temperature_column]
                
                # Doğrusal trend
                x = np.arange(len(temp_sorted))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, temp_sorted)
                
                trend_analysis = {
                    'slope_per_observation': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'trend_direction': 'Isınma' if slope > 0 else 'Soğuma' if slope < 0 else 'Sabit'
                }
                
                # Yıllık analiz
                if len(data_sorted) > 365:  # En az 1 yıl veri varsa
                    data_sorted['year'] = data_sorted[date_column].dt.year
                    yearly_temps = data_sorted.groupby('year')[temperature_column].mean()
                    
                    if len(yearly_temps) > 2:
                        years = yearly_temps.index.values
                        yearly_slope, _, yearly_r, yearly_p, _ = stats.linregress(years, yearly_temps.values)
                        
                        trend_analysis['yearly_trend'] = {
                            'slope_per_year': float(yearly_slope),
                            'r_squared': float(yearly_r ** 2),
                            'p_value': float(yearly_p),
                            'significant': yearly_p < 0.05
                        }
            
            return {
                'basic_statistics': basic_stats,
                'trend_analysis': trend_analysis
            }
            
        except Exception as e:
            return {'error': f'Sıcaklık trend analizi hatası: {str(e)}'}
    
    def _precipitation_analysis(self, data: pd.DataFrame, 
                              precipitation_column: str, 
                              date_column: Optional[str]) -> Dict[str, Any]:
        """
        Yağış analizi yapar
        """
        try:
            precip_data = data[precipitation_column]
            
            # Temel istatistikler
            basic_stats = {
                'mean': float(precip_data.mean()),
                'std': float(precip_data.std()),
                'min': float(precip_data.min()),
                'max': float(precip_data.max()),
                'total': float(precip_data.sum()),
                'rainy_days': int((precip_data > 0).sum()),
                'dry_days': int((precip_data == 0).sum())
            }
            
            # Yağış yoğunluğu kategorileri
            def categorize_precipitation(value):
                if value == 0:
                    return 'Yağışsız'
                elif value <= 2.5:
                    return 'Hafif'
                elif value <= 10:
                    return 'Orta'
                elif value <= 50:
                    return 'Şiddetli'
                else:
                    return 'Çok Şiddetli'
            
            precip_categories = precip_data.apply(categorize_precipitation)
            category_distribution = precip_categories.value_counts()
            
            # Trend analizi
            trend_analysis = {}
            if date_column:
                data_sorted = data.sort_values(date_column)
                precip_sorted = data_sorted[precipitation_column]
                
                # Doğrusal trend
                x = np.arange(len(precip_sorted))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, precip_sorted)
                
                trend_analysis = {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'trend_direction': 'Artış' if slope > 0 else 'Azalış' if slope < 0 else 'Sabit'
                }
            
            return {
                'basic_statistics': basic_stats,
                'intensity_distribution': category_distribution.to_dict(),
                'trend_analysis': trend_analysis
            }
            
        except Exception as e:
            return {'error': f'Yağış analizi hatası: {str(e)}'}
    
    def _extreme_weather_analysis(self, data: pd.DataFrame, 
                                temperature_column: str, 
                                precipitation_column: Optional[str]) -> Dict[str, Any]:
        """
        Ekstrem hava olayları analizi yapar
        """
        try:
            temp_data = data[temperature_column]
            
            # Sıcaklık ekstremları
            temp_q95 = temp_data.quantile(0.95)
            temp_q05 = temp_data.quantile(0.05)
            
            hot_days = (temp_data > temp_q95).sum()
            cold_days = (temp_data < temp_q05).sum()
            
            extreme_analysis = {
                'temperature_extremes': {
                    'hot_days_count': int(hot_days),
                    'cold_days_count': int(cold_days),
                    'hot_threshold': float(temp_q95),
                    'cold_threshold': float(temp_q05),
                    'extreme_percentage': float((hot_days + cold_days) / len(temp_data) * 100)
                }
            }
            
            # Yağış ekstremları
            if precipitation_column and precipitation_column in data.columns:
                precip_data = data[precipitation_column]
                precip_q95 = precip_data.quantile(0.95)
                
                heavy_rain_days = (precip_data > precip_q95).sum()
                
                extreme_analysis['precipitation_extremes'] = {
                    'heavy_rain_days': int(heavy_rain_days),
                    'heavy_rain_threshold': float(precip_q95),
                    'heavy_rain_percentage': float(heavy_rain_days / len(precip_data) * 100)
                }
            
            return extreme_analysis
            
        except Exception as e:
            return {'error': f'Ekstrem hava analizi hatası: {str(e)}'}
    
    def _reference_period_comparison(self, data: pd.DataFrame, 
                                   temperature_column: str,
                                   precipitation_column: Optional[str], 
                                   date_column: str,
                                   reference_period: Tuple[str, str]) -> Dict[str, Any]:
        """
        Referans periyot karşılaştırması yapar
        """
        try:
            start_date, end_date = reference_period
            
            # Referans periyot verisi
            ref_mask = (data[date_column] >= start_date) & (data[date_column] <= end_date)
            ref_data = data[ref_mask]
            
            # Referans sonrası veri
            post_ref_data = data[data[date_column] > end_date]
            
            if len(ref_data) < 10 or len(post_ref_data) < 10:
                return {'error': 'Karşılaştırma için yeterli veri yok'}
            
            comparison_results = {}
            
            # Sıcaklık karşılaştırması
            ref_temp = ref_data[temperature_column]
            post_temp = post_ref_data[temperature_column]
            
            temp_ttest = stats.ttest_ind(ref_temp, post_temp)
            
            comparison_results['temperature_comparison'] = {
                'reference_mean': float(ref_temp.mean()),
                'post_reference_mean': float(post_temp.mean()),
                'mean_difference': float(post_temp.mean() - ref_temp.mean()),
                't_statistic': float(temp_ttest.statistic),
                'p_value': float(temp_ttest.pvalue),
                'significant_change': temp_ttest.pvalue < 0.05
            }
            
            # Yağış karşılaştırması
            if precipitation_column and precipitation_column in data.columns:
                ref_precip = ref_data[precipitation_column]
                post_precip = post_ref_data[precipitation_column]
                
                precip_ttest = stats.ttest_ind(ref_precip, post_precip)
                
                comparison_results['precipitation_comparison'] = {
                    'reference_mean': float(ref_precip.mean()),
                    'post_reference_mean': float(post_precip.mean()),
                    'mean_difference': float(post_precip.mean() - ref_precip.mean()),
                    't_statistic': float(precip_ttest.statistic),
                    'p_value': float(precip_ttest.pvalue),
                    'significant_change': precip_ttest.pvalue < 0.05
                }
            
            return comparison_results
            
        except Exception as e:
            return {'error': f'Referans periyot karşılaştırması hatası: {str(e)}'}
    
    def _calculate_climate_indices(self, data: pd.DataFrame, 
                                 temperature_column: str,
                                 precipitation_column: Optional[str], 
                                 date_column: Optional[str]) -> Dict[str, Any]:
        """
        İklim indekslerini hesaplar
        """
        try:
            temp_data = data[temperature_column]
            indices = {}
            
            # Sıcaklık bazlı indeksler
            indices['temperature_indices'] = {
                'growing_degree_days': float(temp_data[temp_data > 10].sum()),  # 10°C üzeri günler
                'cooling_degree_days': float(temp_data[temp_data > 18].sum()),  # 18°C üzeri günler
                'heating_degree_days': float((18 - temp_data[temp_data < 18]).sum()),  # 18°C altı günler
                'frost_days': int((temp_data < 0).sum()),  # Donlu günler
                'hot_days': int((temp_data > 25).sum())  # Sıcak günler
            }
            
            # Yağış bazlı indeksler
            if precipitation_column and precipitation_column in data.columns:
                precip_data = data[precipitation_column]
                
                indices['precipitation_indices'] = {
                    'wet_days': int((precip_data > 1).sum()),  # Yağışlı günler (>1mm)
                    'very_wet_days': int((precip_data > 10).sum()),  # Çok yağışlı günler (>10mm)
                    'extremely_wet_days': int((precip_data > 20).sum()),  # Aşırı yağışlı günler (>20mm)
                    'dry_spell_max': self._calculate_max_dry_spell(precip_data),
                    'wet_spell_max': self._calculate_max_wet_spell(precip_data)
                }
            
            return indices
            
        except Exception as e:
            return {'error': f'İklim indeksleri hesaplama hatası: {str(e)}'}
    
    def _calculate_max_dry_spell(self, precip_data: pd.Series) -> int:
        """
        En uzun kuru periyodu hesaplar
        """
        try:
            dry_days = (precip_data <= 1).astype(int)
            max_spell = 0
            current_spell = 0
            
            for is_dry in dry_days:
                if is_dry:
                    current_spell += 1
                    max_spell = max(max_spell, current_spell)
                else:
                    current_spell = 0
            
            return max_spell
            
        except:
            return 0
    
    def _calculate_max_wet_spell(self, precip_data: pd.Series) -> int:
        """
        En uzun yağışlı periyodu hesaplar
        """
        try:
            wet_days = (precip_data > 1).astype(int)
            max_spell = 0
            current_spell = 0
            
            for is_wet in wet_days:
                if is_wet:
                    current_spell += 1
                    max_spell = max(max_spell, current_spell)
                else:
                    current_spell = 0
            
            return max_spell
            
        except:
            return 0
    
    def _climate_variability_analysis(self, data: pd.DataFrame, 
                                    temperature_column: str,
                                    precipitation_column: Optional[str], 
                                    date_column: Optional[str]) -> Dict[str, Any]:
        """
        İklim değişkenliği analizi yapar
        """
        try:
            variability_results = {}
            
            # Sıcaklık değişkenliği
            temp_data = data[temperature_column]
            
            # Yıllık değişkenlik (eğer tarih varsa)
            if date_column:
                data_copy = data.copy()
                data_copy[date_column] = pd.to_datetime(data_copy[date_column])
                data_copy['year'] = data_copy[date_column].dt.year
                
                yearly_temp_stats = data_copy.groupby('year')[temperature_column].agg(['mean', 'std', 'min', 'max'])
                
                variability_results['temperature_variability'] = {
                    'inter_annual_mean_std': float(yearly_temp_stats['mean'].std()),
                    'inter_annual_range_mean': float((yearly_temp_stats['max'] - yearly_temp_stats['min']).mean()),
                    'coefficient_of_variation': float(temp_data.std() / temp_data.mean() * 100)
                }
                
                # Yağış değişkenliği
                if precipitation_column and precipitation_column in data.columns:
                    yearly_precip_stats = data_copy.groupby('year')[precipitation_column].agg(['sum', 'std'])
                    
                    variability_results['precipitation_variability'] = {
                        'inter_annual_total_std': float(yearly_precip_stats['sum'].std()),
                        'coefficient_of_variation': float(data[precipitation_column].std() / data[precipitation_column].mean() * 100)
                    }
            
            return variability_results
            
        except Exception as e:
            return {'error': f'İklim değişkenliği analizi hatası: {str(e)}'}
    
    # Yorum metodları
    
    def _interpret_time_series_results(self, basic_stats: Dict, 
                                     stationarity_tests: Dict, 
                                     seasonal_decomposition: Dict) -> str:
        """
        Zaman serisi analizi sonuçlarını yorumlar
        """
        interpretation = []
        
        # Temel istatistikler yorumu
        mean_val = basic_stats['mean']
        std_val = basic_stats['std']
        trend_slope = basic_stats['trend_slope']
        
        interpretation.append(f"Zaman serisinin ortalaması {mean_val:.2f}, standart sapması {std_val:.2f}'dir.")
        
        if abs(trend_slope) > 0.01:
            trend_direction = "artış" if trend_slope > 0 else "azalış"
            interpretation.append(f"Veriler genel olarak {trend_direction} eğilimi göstermektedir.")
        else:
            interpretation.append("Veriler genel olarak sabit bir eğilim göstermektedir.")
        
        # Durağanlık yorumu
        if 'adf_test' in stationarity_tests:
            adf_result = stationarity_tests['adf_test']
            if adf_result['is_stationary']:
                interpretation.append("ADF testine göre zaman serisi durağandır.")
            else:
                interpretation.append("ADF testine göre zaman serisi durağan değildir, trend veya mevsimsellik içerebilir.")
        
        # Mevsimsellik yorumu
        if 'seasonal_strength' in seasonal_decomposition:
            seasonal_strength = seasonal_decomposition['seasonal_strength']
            if seasonal_strength > 0.1:
                interpretation.append("Verilerde güçlü mevsimsel bileşen tespit edilmiştir.")
            elif seasonal_strength > 0.05:
                interpretation.append("Verilerde orta düzeyde mevsimsel bileşen bulunmaktadır.")
            else:
                interpretation.append("Verilerde belirgin bir mevsimsel bileşen görülmemektedir.")
        
        return " ".join(interpretation)
    
    def _interpret_environmental_correlations(self, target_correlations: Dict, 
                                            lag_analysis: Dict) -> str:
        """
        Çevresel korelasyon sonuçlarını yorumlar
        """
        interpretation = []
        
        if target_correlations:
            # En güçlü korelasyonları bul
            strong_correlations = []
            for var, corr_data in target_correlations.items():
                pearson_r = corr_data['pearson_correlation']
                if abs(pearson_r) > 0.5 and corr_data['pearson_significant']:
                    strength = "güçlü" if abs(pearson_r) > 0.7 else "orta"
                    direction = "pozitif" if pearson_r > 0 else "negatif"
                    strong_correlations.append(f"{var} ({strength} {direction}, r={pearson_r:.3f})")
            
            if strong_correlations:
                interpretation.append(f"Hedef değişkenle anlamlı korelasyon gösteren faktörler: {', '.join(strong_correlations)}.")
            else:
                interpretation.append("Hedef değişkenle güçlü korelasyon gösteren çevresel faktör bulunamamıştır.")
        
        # Gecikme analizi yorumu
        if lag_analysis:
            lag_effects = []
            for var, lag_results in lag_analysis.items():
                for lag_data in lag_results:
                    if lag_data['significant'] and abs(lag_data['correlation']) > 0.3:
                        lag_effects.append(f"{var} ({lag_data['lag']} dönem gecikme)")
            
            if lag_effects:
                interpretation.append(f"Gecikme etkisi gösteren faktörler: {', '.join(lag_effects)}.")
        
        return " ".join(interpretation)
    
    def _interpret_pollution_trends(self, pollutant_trends: Dict, 
                                  seasonal_analysis: Dict, 
                                  exceedance_analysis: Dict) -> str:
        """
        Kirlilik trend sonuçlarını yorumlar
        """
        interpretation = []
        
        # Trend yorumları
        increasing_pollutants = []
        decreasing_pollutants = []
        
        for pollutant, trend_data in pollutant_trends.items():
            overall_trend = trend_data.get('overall_trend', {})
            if overall_trend.get('significant', False):
                if overall_trend['slope'] > 0:
                    increasing_pollutants.append(pollutant)
                else:
                    decreasing_pollutants.append(pollutant)
        
        if increasing_pollutants:
            interpretation.append(f"Artan kirletici seviyeleri: {', '.join(increasing_pollutants)}.")
        
        if decreasing_pollutants:
            interpretation.append(f"Azalan kirletici seviyeleri: {', '.join(decreasing_pollutants)}.")
        
        if not increasing_pollutants and not decreasing_pollutants:
            interpretation.append("Kirletici seviyelerinde anlamlı bir trend tespit edilmemiştir.")
        
        # Mevsimsel yorumlar
        if seasonal_analysis:
            high_seasons = []
            for season, season_data in seasonal_analysis.items():
                season_means = [data['mean'] for data in season_data.values() if 'mean' in data]
                if season_means and np.mean(season_means) > np.mean([np.mean(list(s.values())) for s in seasonal_analysis.values()]):
                    high_seasons.append(season)
            
            if high_seasons:
                interpretation.append(f"En yüksek kirlilik seviyeleri: {', '.join(high_seasons)} mevsimlerinde gözlenmiştir.")
        
        # Limit aşımı yorumları
        if exceedance_analysis:
            high_exceedance = []
            for pollutant, exc_data in exceedance_analysis.items():
                if exc_data['exceedance_percentage'] > 10:
                    high_exceedance.append(f"{pollutant} (%{exc_data['exceedance_percentage']:.1f})")
            
            if high_exceedance:
                interpretation.append(f"Yüksek limit aşımı gösteren kirleticiler: {', '.join(high_exceedance)}.")
        
        return " ".join(interpretation)
    
    def _interpret_climate_analysis(self, temperature_analysis: Dict, 
                                  precipitation_analysis: Dict, 
                                  extreme_events: Dict) -> str:
        """
        İklim analizi sonuçlarını yorumlar
        """
        interpretation = []
        
        # Sıcaklık trend yorumu
        if 'trend_analysis' in temperature_analysis:
            trend_data = temperature_analysis['trend_analysis']
            if trend_data.get('significant', False):
                if trend_data['trend_direction'] == 'Isınma':
                    interpretation.append("Sıcaklıklarda anlamlı bir ısınma trendi tespit edilmiştir.")
                elif trend_data['trend_direction'] == 'Soğuma':
                    interpretation.append("Sıcaklıklarda anlamlı bir soğuma trendi tespit edilmiştir.")
            else:
                interpretation.append("Sıcaklıklarda anlamlı bir trend tespit edilmemiştir.")
        
        # Yağış trend yorumu
        if precipitation_analysis and 'trend_analysis' in precipitation_analysis:
            precip_trend = precipitation_analysis['trend_analysis']
            if precip_trend.get('significant', False):
                direction = "artış" if precip_trend['trend_direction'] == 'Artış' else "azalış"
                interpretation.append(f"Yağış miktarlarında anlamlı bir {direction} trendi gözlenmiştir.")
        
        # Ekstrem olaylar yorumu
        if 'temperature_extremes' in extreme_events:
            temp_extremes = extreme_events['temperature_extremes']
            extreme_percentage = temp_extremes['extreme_percentage']
            
            if extreme_percentage > 20:
                interpretation.append("Yüksek oranda ekstrem sıcaklık olayları tespit edilmiştir.")
            elif extreme_percentage > 10:
                interpretation.append("Orta düzeyde ekstrem sıcaklık olayları gözlenmiştir.")
        
        if 'precipitation_extremes' in extreme_events:
            precip_extremes = extreme_events['precipitation_extremes']
            heavy_rain_percentage = precip_extremes['heavy_rain_percentage']
            
            if heavy_rain_percentage > 10:
                interpretation.append("Yüksek oranda şiddetli yağış olayları tespit edilmiştir.")
        
        return " ".join(interpretation)