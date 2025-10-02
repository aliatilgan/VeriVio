"""
Zaman Serisi Analizi Sınıfı

Bu modül zaman serisi verilerinin analizi için kullanılır.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """
    Zaman serisi analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        TimeSeriesAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek veri seti
        """
        self.data = data.copy()
        self.results = {}
        self.models = {}
        
    def basic_time_series_analysis(self, date_column: str, value_column: str,
                                  freq: str = 'D') -> Dict[str, Any]:
        """
        Temel zaman serisi analizi gerçekleştirir
        
        Args:
            date_column: Tarih sütunu
            value_column: Değer sütunu
            freq: Frekans ('D', 'M', 'Y', etc.)
            
        Returns:
            Temel zaman serisi analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[[date_column, value_column]].dropna()
            data_clean[date_column] = pd.to_datetime(data_clean[date_column])
            data_clean = data_clean.sort_values(date_column)
            data_clean.set_index(date_column, inplace=True)
            
            if len(data_clean) < 10:
                return {'error': 'Zaman serisi analizi için en az 10 gözlem gereklidir'}
            
            ts = data_clean[value_column]
            
            # Temel istatistikler
            basic_stats = {
                'count': int(len(ts)),
                'mean': float(ts.mean()),
                'std': float(ts.std()),
                'min': float(ts.min()),
                'max': float(ts.max()),
                'median': float(ts.median()),
                'skewness': float(ts.skew()),
                'kurtosis': float(ts.kurtosis())
            }
            
            # Trend analizi
            time_index = np.arange(len(ts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, ts.values)
            
            trend_analysis = {
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'trend_direction': 'Artan' if slope > 0 else 'Azalan' if slope < 0 else 'Sabit',
                'trend_strength': 'Güçlü' if abs(r_value) > 0.7 else 'Orta' if abs(r_value) > 0.3 else 'Zayıf',
                'significant': p_value < 0.05
            }
            
            # Değişkenlik analizi
            rolling_std = ts.rolling(window=min(30, len(ts)//4)).std()
            variability_analysis = {
                'coefficient_of_variation': float(ts.std() / ts.mean()) if ts.mean() != 0 else None,
                'range': float(ts.max() - ts.min()),
                'iqr': float(ts.quantile(0.75) - ts.quantile(0.25)),
                'rolling_std_mean': float(rolling_std.mean()),
                'rolling_std_std': float(rolling_std.std())
            }
            
            # Outlier analizi
            Q1 = ts.quantile(0.25)
            Q3 = ts.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ts[(ts < lower_bound) | (ts > upper_bound)]
            
            outlier_analysis = {
                'n_outliers': int(len(outliers)),
                'outlier_percentage': float(len(outliers) / len(ts) * 100),
                'outlier_dates': outliers.index.strftime('%Y-%m-%d').tolist() if len(outliers) > 0 else [],
                'outlier_values': outliers.values.tolist() if len(outliers) > 0 else []
            }
            
            # Normallik testi
            _, normality_p = jarque_bera(ts.values)
            
            # Durağanlık testi (ADF)
            adf_result = adfuller(ts.values)
            stationarity_analysis = {
                'adf_statistic': float(adf_result[0]),
                'adf_p_value': float(adf_result[1]),
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': {k: float(v) for k, v in adf_result[4].items()}
            }
            
            # Otokorelasyon analizi
            autocorr_lags = min(20, len(ts)//4)
            autocorr = acf(ts.values, nlags=autocorr_lags, fft=False)
            significant_lags = [i for i, val in enumerate(autocorr[1:], 1) if abs(val) > 0.2]
            
            autocorr_analysis = {
                'autocorrelation_values': autocorr.tolist(),
                'significant_lags': significant_lags,
                'max_autocorr': float(max(abs(autocorr[1:]))),
                'autocorr_pattern': 'Güçlü' if max(abs(autocorr[1:])) > 0.7 else 'Orta' if max(abs(autocorr[1:])) > 0.3 else 'Zayıf'
            }
            
            result = {
                'analysis_type': 'Basic Time Series Analysis',
                'date_column': date_column,
                'value_column': value_column,
                'time_period': {
                    'start_date': ts.index.min().strftime('%Y-%m-%d'),
                    'end_date': ts.index.max().strftime('%Y-%m-%d'),
                    'duration_days': int((ts.index.max() - ts.index.min()).days),
                    'frequency': freq
                },
                'basic_statistics': basic_stats,
                'trend_analysis': trend_analysis,
                'variability_analysis': variability_analysis,
                'outlier_analysis': outlier_analysis,
                'stationarity_analysis': stationarity_analysis,
                'autocorrelation_analysis': autocorr_analysis,
                'data_quality': {
                    'normality_p_value': float(normality_p),
                    'is_normal': normality_p > 0.05,
                    'missing_values': int(self.data[value_column].isna().sum()),
                    'data_completeness': float((len(ts) / len(self.data)) * 100)
                },
                'interpretation': self._interpret_basic_time_series(
                    trend_analysis, stationarity_analysis, autocorr_analysis, outlier_analysis
                )
            }
            
            self.results['basic_time_series'] = result
            return result
            
        except Exception as e:
            return {'error': f'Temel zaman serisi analizi hatası: {str(e)}'}
    
    def seasonal_decomposition(self, date_column: str, value_column: str,
                              model: str = 'additive', period: Optional[int] = None) -> Dict[str, Any]:
        """
        Mevsimsel ayrıştırma analizi gerçekleştirir
        
        Args:
            date_column: Tarih sütunu
            value_column: Değer sütunu
            model: Ayrıştırma modeli ('additive' veya 'multiplicative')
            period: Mevsimsel periyot
            
        Returns:
            Mevsimsel ayrıştırma analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[[date_column, value_column]].dropna()
            data_clean[date_column] = pd.to_datetime(data_clean[date_column])
            data_clean = data_clean.sort_values(date_column)
            data_clean.set_index(date_column, inplace=True)
            
            if len(data_clean) < 24:  # En az 2 periyot gerekli
                return {'error': 'Mevsimsel ayrıştırma için en az 24 gözlem gereklidir'}
            
            ts = data_clean[value_column]
            
            # Periyot otomatik belirleme
            if period is None:
                # Basit periyot tahmini (en yaygın frekanslar)
                if len(ts) >= 365:
                    period = 365  # Yıllık
                elif len(ts) >= 52:
                    period = 52   # Haftalık
                elif len(ts) >= 12:
                    period = 12   # Aylık
                else:
                    period = 4    # Çeyreklik
            
            # Mevsimsel ayrıştırma
            decomposition = seasonal_decompose(ts, model=model, period=period)
            
            # Bileşen analizi
            trend_component = decomposition.trend.dropna()
            seasonal_component = decomposition.seasonal
            residual_component = decomposition.resid.dropna()
            
            # Trend analizi
            if len(trend_component) > 1:
                trend_slope, _, trend_r, trend_p, _ = stats.linregress(
                    range(len(trend_component)), trend_component.values
                )
            else:
                trend_slope = trend_r = trend_p = 0
            
            # Mevsimsellik gücü
            seasonal_strength = 1 - (residual_component.var() / (residual_component + seasonal_component.dropna()).var())
            
            # Trend gücü
            trend_strength = 1 - (residual_component.var() / (residual_component + trend_component).var())
            
            # Mevsimsel patern analizi
            seasonal_pattern = seasonal_component.iloc[:period].values
            seasonal_amplitude = seasonal_pattern.max() - seasonal_pattern.min()
            seasonal_cv = seasonal_pattern.std() / abs(seasonal_pattern.mean()) if seasonal_pattern.mean() != 0 else 0
            
            result = {
                'analysis_type': 'Seasonal Decomposition',
                'date_column': date_column,
                'value_column': value_column,
                'decomposition_model': model,
                'period': period,
                'components': {
                    'original': ts.values.tolist(),
                    'trend': decomposition.trend.values.tolist(),
                    'seasonal': decomposition.seasonal.values.tolist(),
                    'residual': decomposition.resid.values.tolist(),
                    'dates': ts.index.strftime('%Y-%m-%d').tolist()
                },
                'trend_analysis': {
                    'slope': float(trend_slope),
                    'r_squared': float(trend_r**2),
                    'p_value': float(trend_p),
                    'direction': 'Artan' if trend_slope > 0 else 'Azalan' if trend_slope < 0 else 'Sabit',
                    'strength': float(trend_strength)
                },
                'seasonal_analysis': {
                    'strength': float(seasonal_strength),
                    'amplitude': float(seasonal_amplitude),
                    'coefficient_of_variation': float(seasonal_cv),
                    'pattern': seasonal_pattern.tolist(),
                    'peak_season': int(np.argmax(seasonal_pattern)),
                    'trough_season': int(np.argmin(seasonal_pattern))
                },
                'residual_analysis': {
                    'mean': float(residual_component.mean()),
                    'std': float(residual_component.std()),
                    'min': float(residual_component.min()),
                    'max': float(residual_component.max()),
                    'autocorrelation': float(acf(residual_component.values, nlags=1, fft=False)[1])
                },
                'model_quality': {
                    'explained_variance': float(1 - residual_component.var() / ts.var()),
                    'seasonal_dominance': seasonal_strength > trend_strength,
                    'residual_randomness': abs(residual_component.mean()) < residual_component.std()
                },
                'interpretation': self._interpret_seasonal_decomposition(
                    seasonal_strength, trend_strength, model, period
                )
            }
            
            self.results['seasonal_decomposition'] = result
            return result
            
        except Exception as e:
            return {'error': f'Mevsimsel ayrıştırma hatası: {str(e)}'}
    
    def arima_modeling(self, date_column: str, value_column: str,
                      order: Tuple[int, int, int] = (1, 1, 1),
                      forecast_steps: int = 10) -> Dict[str, Any]:
        """
        ARIMA modeli oluşturur ve tahmin yapar
        
        Args:
            date_column: Tarih sütunu
            value_column: Değer sütunu
            order: ARIMA(p,d,q) parametreleri
            forecast_steps: Tahmin adım sayısı
            
        Returns:
            ARIMA modeli sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[[date_column, value_column]].dropna()
            data_clean[date_column] = pd.to_datetime(data_clean[date_column])
            data_clean = data_clean.sort_values(date_column)
            data_clean.set_index(date_column, inplace=True)
            
            if len(data_clean) < 20:
                return {'error': 'ARIMA modeli için en az 20 gözlem gereklidir'}
            
            ts = data_clean[value_column]
            
            # Train-test split
            train_size = int(len(ts) * 0.8)
            train_data = ts[:train_size]
            test_data = ts[train_size:]
            
            # ARIMA modeli
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            
            # Model özeti
            model_summary = {
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic),
                'log_likelihood': float(fitted_model.llf),
                'parameters': {
                    'ar_params': fitted_model.arparams.tolist() if len(fitted_model.arparams) > 0 else [],
                    'ma_params': fitted_model.maparams.tolist() if len(fitted_model.maparams) > 0 else [],
                    'sigma2': float(fitted_model.sigma2)
                }
            }
            
            # In-sample tahminler
            fitted_values = fitted_model.fittedvalues
            residuals = train_data - fitted_values
            
            # Out-of-sample tahminler (test seti)
            if len(test_data) > 0:
                forecast_test = fitted_model.forecast(steps=len(test_data))
                test_mse = np.mean((test_data - forecast_test)**2)
                test_mae = np.mean(np.abs(test_data - forecast_test))
                test_mape = np.mean(np.abs((test_data - forecast_test) / test_data)) * 100
            else:
                forecast_test = []
                test_mse = test_mae = test_mape = None
            
            # Gelecek tahminleri
            future_forecast = fitted_model.forecast(steps=forecast_steps)
            forecast_conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
            
            # Residual analizi
            residual_stats = {
                'mean': float(residuals.mean()),
                'std': float(residuals.std()),
                'ljung_box_p': float(fitted_model.test_serial_correlation('ljungbox')[0]['lb_pvalue'].iloc[-1]),
                'jarque_bera_p': float(fitted_model.test_normality('jarquebera')[0]['jarque_bera_pvalue']),
                'heteroscedasticity_p': float(fitted_model.test_heteroskedasticity('breakvar')[0]['breakvar_pvalue'])
            }
            
            # Model performansı
            train_mse = np.mean(residuals**2)
            train_mae = np.mean(np.abs(residuals))
            train_mape = np.mean(np.abs(residuals / train_data)) * 100
            
            result = {
                'analysis_type': 'ARIMA Modeling',
                'date_column': date_column,
                'value_column': value_column,
                'model_order': order,
                'sample_size': len(ts),
                'train_size': len(train_data),
                'test_size': len(test_data),
                'model_summary': model_summary,
                'performance_metrics': {
                    'train_mse': float(train_mse),
                    'train_mae': float(train_mae),
                    'train_mape': float(train_mape),
                    'test_mse': float(test_mse) if test_mse is not None else None,
                    'test_mae': float(test_mae) if test_mae is not None else None,
                    'test_mape': float(test_mape) if test_mape is not None else None
                },
                'residual_diagnostics': residual_stats,
                'forecasts': {
                    'future_values': future_forecast.tolist(),
                    'confidence_intervals': {
                        'lower': forecast_conf_int.iloc[:, 0].tolist(),
                        'upper': forecast_conf_int.iloc[:, 1].tolist()
                    },
                    'forecast_dates': pd.date_range(
                        start=ts.index[-1] + pd.Timedelta(days=1),
                        periods=forecast_steps,
                        freq='D'
                    ).strftime('%Y-%m-%d').tolist()
                },
                'model_diagnostics': {
                    'residuals_white_noise': residual_stats['ljung_box_p'] > 0.05,
                    'residuals_normal': residual_stats['jarque_bera_p'] > 0.05,
                    'homoscedastic': residual_stats['heteroscedasticity_p'] > 0.05,
                    'model_adequacy': (residual_stats['ljung_box_p'] > 0.05 and 
                                     residual_stats['jarque_bera_p'] > 0.05)
                },
                'interpretation': self._interpret_arima_model(
                    order, model_summary, residual_stats, train_mape
                )
            }
            
            # Modeli sakla
            self.models['arima'] = fitted_model
            self.results['arima'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'ARIMA modeli hatası: {str(e)}'}
    
    def exponential_smoothing(self, date_column: str, value_column: str,
                             trend: str = 'add', seasonal: str = 'add',
                             seasonal_periods: int = 12, forecast_steps: int = 10) -> Dict[str, Any]:
        """
        Exponential Smoothing (Holt-Winters) modeli oluşturur
        
        Args:
            date_column: Tarih sütunu
            value_column: Değer sütunu
            trend: Trend tipi ('add', 'mul', None)
            seasonal: Mevsimsellik tipi ('add', 'mul', None)
            seasonal_periods: Mevsimsel periyot
            forecast_steps: Tahmin adım sayısı
            
        Returns:
            Exponential Smoothing modeli sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[[date_column, value_column]].dropna()
            data_clean[date_column] = pd.to_datetime(data_clean[date_column])
            data_clean = data_clean.sort_values(date_column)
            data_clean.set_index(date_column, inplace=True)
            
            if len(data_clean) < seasonal_periods * 2:
                return {'error': f'Exponential Smoothing için en az {seasonal_periods * 2} gözlem gereklidir'}
            
            ts = data_clean[value_column]
            
            # Train-test split
            train_size = int(len(ts) * 0.8)
            train_data = ts[:train_size]
            test_data = ts[train_size:]
            
            # Exponential Smoothing modeli
            model = ExponentialSmoothing(
                train_data,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None
            )
            fitted_model = model.fit()
            
            # Model parametreleri
            model_params = {
                'alpha': float(fitted_model.params['smoothing_level']),
                'beta': float(fitted_model.params.get('smoothing_trend', 0)),
                'gamma': float(fitted_model.params.get('smoothing_seasonal', 0)),
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic),
                'sse': float(fitted_model.sse)
            }
            
            # In-sample tahminler
            fitted_values = fitted_model.fittedvalues
            residuals = train_data - fitted_values
            
            # Out-of-sample tahminler
            if len(test_data) > 0:
                forecast_test = fitted_model.forecast(steps=len(test_data))
                test_mse = np.mean((test_data - forecast_test)**2)
                test_mae = np.mean(np.abs(test_data - forecast_test))
                test_mape = np.mean(np.abs((test_data - forecast_test) / test_data)) * 100
            else:
                forecast_test = []
                test_mse = test_mae = test_mape = None
            
            # Gelecek tahminleri
            future_forecast = fitted_model.forecast(steps=forecast_steps)
            
            # Model performansı
            train_mse = np.mean(residuals**2)
            train_mae = np.mean(np.abs(residuals))
            train_mape = np.mean(np.abs(residuals / train_data)) * 100
            
            result = {
                'analysis_type': 'Exponential Smoothing',
                'date_column': date_column,
                'value_column': value_column,
                'model_config': {
                    'trend': trend,
                    'seasonal': seasonal,
                    'seasonal_periods': seasonal_periods
                },
                'sample_size': len(ts),
                'train_size': len(train_data),
                'test_size': len(test_data),
                'model_parameters': model_params,
                'performance_metrics': {
                    'train_mse': float(train_mse),
                    'train_mae': float(train_mae),
                    'train_mape': float(train_mape),
                    'test_mse': float(test_mse) if test_mse is not None else None,
                    'test_mae': float(test_mae) if test_mae is not None else None,
                    'test_mape': float(test_mape) if test_mape is not None else None
                },
                'forecasts': {
                    'future_values': future_forecast.tolist(),
                    'forecast_dates': pd.date_range(
                        start=ts.index[-1] + pd.Timedelta(days=1),
                        periods=forecast_steps,
                        freq='D'
                    ).strftime('%Y-%m-%d').tolist()
                },
                'model_components': {
                    'level': float(fitted_model.level.iloc[-1]) if hasattr(fitted_model, 'level') else None,
                    'trend': float(fitted_model.trend.iloc[-1]) if hasattr(fitted_model, 'trend') else None,
                    'seasonal': fitted_model.season.iloc[-seasonal_periods:].tolist() if hasattr(fitted_model, 'season') else None
                },
                'interpretation': self._interpret_exponential_smoothing(
                    trend, seasonal, model_params, train_mape
                )
            }
            
            # Modeli sakla
            self.models['exponential_smoothing'] = fitted_model
            self.results['exponential_smoothing'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Exponential Smoothing hatası: {str(e)}'}
    
    def _interpret_basic_time_series(self, trend_analysis: Dict, stationarity_analysis: Dict,
                                   autocorr_analysis: Dict, outlier_analysis: Dict) -> str:
        """Temel zaman serisi analizi sonucunu yorumlar"""
        interpretation = "Zaman serisi analizi: "
        
        # Trend yorumu
        if trend_analysis['significant']:
            interpretation += f"{trend_analysis['trend_direction']} trend tespit edildi ({trend_analysis['trend_strength']} güçte). "
        else:
            interpretation += "Anlamlı bir trend tespit edilmedi. "
        
        # Durağanlık yorumu
        if stationarity_analysis['is_stationary']:
            interpretation += "Seri durağandır. "
        else:
            interpretation += "Seri durağan değildir, fark alma gerekebilir. "
        
        # Otokorelasyon yorumu
        if autocorr_analysis['autocorr_pattern'] == 'Güçlü':
            interpretation += "Güçlü otokorelasyon tespit edildi, geçmiş değerler gelecek değerleri önemli ölçüde etkiliyor. "
        
        # Outlier yorumu
        if outlier_analysis['outlier_percentage'] > 5:
            interpretation += f"Yüksek outlier oranı (%{outlier_analysis['outlier_percentage']:.1f}) tespit edildi."
        
        return interpretation
    
    def _interpret_seasonal_decomposition(self, seasonal_strength: float, trend_strength: float,
                                        model: str, period: int) -> str:
        """Mevsimsel ayrıştırma sonucunu yorumlar"""
        interpretation = f"{model.title()} mevsimsel ayrıştırma (periyot: {period}): "
        
        if seasonal_strength > 0.6:
            interpretation += "Güçlü mevsimsel patern tespit edildi. "
        elif seasonal_strength > 0.3:
            interpretation += "Orta düzeyde mevsimsel patern tespit edildi. "
        else:
            interpretation += "Zayıf mevsimsel patern tespit edildi. "
        
        if trend_strength > 0.6:
            interpretation += "Güçlü trend bileşeni mevcut. "
        elif trend_strength > 0.3:
            interpretation += "Orta düzeyde trend bileşeni mevcut. "
        
        if seasonal_strength > trend_strength:
            interpretation += "Mevsimsellik trend'den daha baskın."
        else:
            interpretation += "Trend mevsimsellik'ten daha baskın."
        
        return interpretation
    
    def _interpret_arima_model(self, order: Tuple, model_summary: Dict,
                             residual_stats: Dict, mape: float) -> str:
        """ARIMA modeli sonucunu yorumlar"""
        p, d, q = order
        interpretation = f"ARIMA({p},{d},{q}) modeli: "
        
        # Model kalitesi
        if mape < 10:
            interpretation += "Mükemmel tahmin performansı. "
        elif mape < 20:
            interpretation += "İyi tahmin performansı. "
        elif mape < 50:
            interpretation += "Orta tahmin performansı. "
        else:
            interpretation += "Zayıf tahmin performansı. "
        
        # Model tanıları
        if residual_stats['ljung_box_p'] > 0.05:
            interpretation += "Residualler white noise özelliği gösteriyor. "
        else:
            interpretation += "Residuallerde otokorelasyon tespit edildi, model yetersiz olabilir. "
        
        if residual_stats['jarque_bera_p'] > 0.05:
            interpretation += "Residualler normal dağılıyor."
        else:
            interpretation += "Residualler normal dağılmıyor."
        
        return interpretation
    
    def _interpret_exponential_smoothing(self, trend: str, seasonal: str,
                                       model_params: Dict, mape: float) -> str:
        """Exponential Smoothing sonucunu yorumlar"""
        interpretation = f"Exponential Smoothing (trend: {trend}, seasonal: {seasonal}): "
        
        # Model kalitesi
        if mape < 10:
            interpretation += "Mükemmel tahmin performansı. "
        elif mape < 20:
            interpretation += "İyi tahmin performansı. "
        elif mape < 50:
            interpretation += "Orta tahmin performansı. "
        else:
            interpretation += "Zayıf tahmin performansı. "
        
        # Parametreler
        alpha = model_params['alpha']
        if alpha > 0.8:
            interpretation += "Yüksek alpha değeri, son gözlemlere yüksek ağırlık veriyor. "
        elif alpha < 0.2:
            interpretation += "Düşük alpha değeri, geçmiş gözlemlere yüksek ağırlık veriyor. "
        
        if trend and model_params['beta'] > 0.5:
            interpretation += "Güçlü trend smoothing parametresi."
        
        return interpretation
    
    def get_time_series_summary(self) -> Dict[str, Any]:
        """
        Gerçekleştirilen tüm zaman serisi analizlerinin özetini döndürür
        
        Returns:
            Zaman serisi analizi özetleri
        """
        if not self.results:
            return {'message': 'Henüz zaman serisi analizi gerçekleştirilmedi'}
        
        summary = {
            'total_analyses': len(self.results),
            'analyses_performed': list(self.results.keys()),
            'models_available': list(self.models.keys()),
            'analysis_details': self.results
        }
        
        # En iyi modeli bul (en düşük MAPE)
        best_model = None
        best_mape = float('inf')
        
        for analysis_name, result in self.results.items():
            if 'error' not in result and 'performance_metrics' in result:
                mape = result['performance_metrics'].get('train_mape')
                if mape and mape < best_mape:
                    best_mape = mape
                    best_model = analysis_name
        
        if best_model:
            summary['best_model'] = {
                'name': best_model,
                'mape': best_mape,
                'quality': 'Mükemmel' if best_mape < 10 else 'İyi' if best_mape < 20 else 'Orta' if best_mape < 50 else 'Zayıf'
            }
        
        return summary