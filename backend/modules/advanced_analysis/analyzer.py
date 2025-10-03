"""
VeriVio Gelişmiş Analiz Modülü
Yapısal Eşitlik Modellemesi (SEM), Zaman Serisi, Makine Öğrenmesi, Survival Analizi
Meta analiz, çok düzeyli analiz ve diğer ileri seviye istatistiksel yöntemler
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from lifelines import KaplanMeierFitter, CoxPHFitter, LogNormalFitter
from lifelines.statistics import logrank_test
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

# Time series analysis
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class AdvancedAnalyzer:
    """Gelişmiş analiz sınıfı"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        self.models = {}
        self.scalers = {}
        
        # Analiz kriterleri
        self.ml_criteria = {
            'test_size': 0.2,
            'cv_folds': 5,
            'random_state': 42
        }
        
        self.time_series_criteria = {
            'min_periods': 10,
            'seasonality_threshold': 0.1
        }
    
    def machine_learning_analysis(self, target_column: str, 
                                 feature_columns: List[str],
                                 problem_type: str = 'auto') -> Dict[str, Any]:
        """Makine öğrenmesi analizi"""
        try:
            # Veriyi hazırla
            features = self.data[feature_columns].select_dtypes(include=[np.number])
            target = self.data[target_column]
            
            # Eksik değerleri temizle
            clean_data = pd.concat([features, target], axis=1).dropna()
            X = clean_data[features.columns]
            y = clean_data[target_column]
            
            if len(X) < 20:
                raise ValueError("Makine öğrenmesi için en az 20 gözlem gerekli")
            
            # Problem türünü belirle
            if problem_type == 'auto':
                if target.dtype == 'object' or len(target.unique()) <= 10:
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.ml_criteria['test_size'], 
                random_state=self.ml_criteria['random_state']
            )
            
            # Özellik ölçeklendirme
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Label encoding (sınıflandırma için)
            label_encoder = None
            if problem_type == 'classification' and y.dtype == 'object':
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_test = label_encoder.transform(y_test)
            
            # Modelleri tanımla
            if problem_type == 'classification':
                models = {
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'SVM': SVC(random_state=42, probability=True),
                    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                    'Neural Network': MLPClassifier(random_state=42, max_iter=500)
                }
            else:
                models = {
                    'Random Forest': RandomForestRegressor(random_state=42),
                    'Linear Regression': LinearRegression(),
                    'SVM': SVR(),
                    'XGBoost': xgb.XGBRegressor(random_state=42),
                    'Neural Network': MLPRegressor(random_state=42, max_iter=500)
                }
            
            # Modelleri eğit ve değerlendir
            model_results = {}
            best_model = None
            best_score = -np.inf if problem_type == 'classification' else np.inf
            
            for model_name, model in models.items():
                try:
                    # Cross-validation
                    if problem_type == 'classification':
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                                  cv=self.ml_criteria['cv_folds'], 
                                                  scoring='accuracy')
                    else:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                                  cv=self.ml_criteria['cv_folds'], 
                                                  scoring='r2')
                    
                    # Modeli eğit
                    model.fit(X_train_scaled, y_train)
                    
                    # Tahmin yap
                    y_pred = model.predict(X_test_scaled)
                    
                    # Performans metrikleri
                    if problem_type == 'classification':
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        # ROC AUC (binary classification için)
                        try:
                            if len(np.unique(y_test)) == 2:
                                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                                roc_auc = roc_auc_score(y_test, y_pred_proba)
                            else:
                                roc_auc = None
                        except:
                            roc_auc = None
                        
                        metrics = {
                            'accuracy': float(accuracy),
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1),
                            'roc_auc': float(roc_auc) if roc_auc else None,
                            'cv_mean': float(cv_scores.mean()),
                            'cv_std': float(cv_scores.std())
                        }
                        
                        # En iyi model (accuracy'ye göre)
                        if accuracy > best_score:
                            best_score = accuracy
                            best_model = model_name
                    
                    else:  # regression
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        
                        metrics = {
                            'mse': float(mse),
                            'mae': float(mae),
                            'rmse': float(rmse),
                            'r2_score': float(r2),
                            'cv_mean': float(cv_scores.mean()),
                            'cv_std': float(cv_scores.std())
                        }
                        
                        # En iyi model (R² skoruna göre)
                        if r2 > best_score:
                            best_score = r2
                            best_model = model_name
                    
                    # Özellik önemleri (varsa)
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(features.columns, model.feature_importances_))
                    elif hasattr(model, 'coef_'):
                        if problem_type == 'classification' and len(model.coef_.shape) > 1:
                            # Multi-class için ortalama al
                            coef = np.mean(np.abs(model.coef_), axis=0)
                        else:
                            coef = model.coef_.flatten() if hasattr(model.coef_, 'flatten') else model.coef_
                        feature_importance = dict(zip(features.columns, np.abs(coef)))
                    
                    model_results[model_name] = {
                        'metrics': metrics,
                        'feature_importance': feature_importance,
                        'predictions_sample': y_pred[:10].tolist() if len(y_pred) >= 10 else y_pred.tolist()
                    }
                    
                except Exception as e:
                    model_results[model_name] = {'error': str(e)}
                    logger.warning(f"{model_name} modeli hatası: {str(e)}")
            
            result = {
                'model_type': 'Makine Öğrenmesi Analizi',
                'problem_type': problem_type,
                'target_variable': target_column,
                'feature_variables': feature_columns,
                'sample_size': len(clean_data),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'model_results': model_results,
                'best_model': best_model,
                'best_score': float(best_score),
                'data_preprocessing': {
                    'scaling_applied': True,
                    'label_encoding_applied': label_encoder is not None,
                    'missing_values_removed': len(self.data) - len(clean_data)
                },
                'interpretation': self._generate_ml_interpretation(
                    problem_type, best_model, best_score, model_results
                )
            }
            
            # En iyi modeli kaydet
            if best_model and best_model in models:
                self.models['machine_learning'] = {
                    'model': models[best_model],
                    'scaler': scaler,
                    'label_encoder': label_encoder
                }
            
            self.results['machine_learning'] = result
            
            logger.info(f"Makine öğrenmesi analizi tamamlandı: {problem_type}")
            return result
            
        except Exception as e:
            logger.error(f"Makine öğrenmesi analizi hatası: {str(e)}")
            return {'error': f'Makine öğrenmesi analizi hatası: {str(e)}'}
    
    def time_series_analysis(self, time_column: str, value_column: str,
                           forecast_periods: int = 30) -> Dict[str, Any]:
        """Zaman serisi analizi"""
        try:
            if not STATSMODELS_AVAILABLE:
                return {'error': 'Statsmodels kütüphanesi bulunamadı'}
            
            # Veriyi hazırla
            ts_data = self.data[[time_column, value_column]].copy()
            ts_data = ts_data.dropna()
            
            # Zaman sütununu datetime'a çevir
            ts_data[time_column] = pd.to_datetime(ts_data[time_column])
            ts_data = ts_data.sort_values(time_column)
            ts_data.set_index(time_column, inplace=True)
            
            if len(ts_data) < self.time_series_criteria['min_periods']:
                raise ValueError(f"Zaman serisi analizi için en az {self.time_series_criteria['min_periods']} gözlem gerekli")
            
            # Temel istatistikler
            series = ts_data[value_column]
            basic_stats = {
                'count': len(series),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'trend': 'Artan' if series.iloc[-1] > series.iloc[0] else 'Azalan'
            }
            
            # Durağanlık testleri
            stationarity_tests = self._test_stationarity(series)
            
            # Mevsimsellik analizi
            seasonality_analysis = self._analyze_seasonality(series)
            
            # ARIMA modeli
            arima_results = self._fit_arima_model(series, forecast_periods)
            
            # Prophet modeli (varsa)
            prophet_results = None
            if PROPHET_AVAILABLE:
                prophet_results = self._fit_prophet_model(ts_data, value_column, forecast_periods)
            
            result = {
                'model_type': 'Zaman Serisi Analizi',
                'time_variable': time_column,
                'value_variable': value_column,
                'sample_size': len(ts_data),
                'time_range': {
                    'start': str(ts_data.index.min()),
                    'end': str(ts_data.index.max()),
                    'frequency': str(ts_data.index.freq) if ts_data.index.freq else 'Belirsiz'
                },
                'basic_statistics': basic_stats,
                'stationarity_tests': stationarity_tests,
                'seasonality_analysis': seasonality_analysis,
                'arima_model': arima_results,
                'prophet_model': prophet_results,
                'forecast_periods': forecast_periods,
                'interpretation': self._generate_time_series_interpretation(
                    basic_stats, stationarity_tests, seasonality_analysis, arima_results
                )
            }
            
            self.results['time_series'] = result
            
            logger.info("Zaman serisi analizi tamamlandı")
            return result
            
        except Exception as e:
            logger.error(f"Zaman serisi analizi hatası: {str(e)}")
            return {'error': f'Zaman serisi analizi hatası: {str(e)}'}
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Durağanlık testleri"""
        try:
            # Augmented Dickey-Fuller Test
            adf_result = adfuller(series.dropna())
            
            # KPSS Test
            kpss_result = kpss(series.dropna())
            
            return {
                'adf_test': {
                    'statistic': float(adf_result[0]),
                    'p_value': float(adf_result[1]),
                    'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                    'is_stationary': adf_result[1] < 0.05
                },
                'kpss_test': {
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                    'is_stationary': kpss_result[1] > 0.05
                }
            }
            
        except Exception as e:
            logger.warning(f"Durağanlık testi hatası: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Mevsimsellik analizi"""
        try:
            # Seasonal decomposition
            if len(series) >= 24:  # En az 2 yıl veri
                decomposition = seasonal_decompose(series, model='additive', period=12)
                
                seasonal_strength = np.var(decomposition.seasonal) / np.var(series)
                trend_strength = np.var(decomposition.trend.dropna()) / np.var(series)
                
                return {
                    'has_seasonality': seasonal_strength > self.time_series_criteria['seasonality_threshold'],
                    'seasonal_strength': float(seasonal_strength),
                    'trend_strength': float(trend_strength),
                    'decomposition_available': True
                }
            else:
                return {
                    'has_seasonality': False,
                    'seasonal_strength': 0.0,
                    'trend_strength': 0.0,
                    'decomposition_available': False,
                    'note': 'Yetersiz veri için mevsimsellik analizi yapılamadı'
                }
                
        except Exception as e:
            logger.warning(f"Mevsimsellik analizi hatası: {str(e)}")
            return {'error': str(e)}
    
    def _fit_arima_model(self, series: pd.Series, forecast_periods: int) -> Dict[str, Any]:
        """ARIMA modeli"""
        try:
            # Otomatik ARIMA model seçimi (basit)
            best_aic = np.inf
            best_order = (1, 1, 1)
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # En iyi modeli fit et
            final_model = ARIMA(series, order=best_order)
            fitted_final_model = final_model.fit()
            
            # Tahmin yap
            forecast = fitted_final_model.forecast(steps=forecast_periods)
            forecast_ci = fitted_final_model.get_forecast(steps=forecast_periods).conf_int()
            
            # Model diagnostikleri
            residuals = fitted_final_model.resid
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            return {
                'model_order': best_order,
                'aic': float(fitted_final_model.aic),
                'bic': float(fitted_final_model.bic),
                'log_likelihood': float(fitted_final_model.llf),
                'forecast': forecast.tolist(),
                'forecast_confidence_interval': {
                    'lower': forecast_ci.iloc[:, 0].tolist(),
                    'upper': forecast_ci.iloc[:, 1].tolist()
                },
                'residual_diagnostics': {
                    'ljung_box_p_value': float(ljung_box['lb_pvalue'].iloc[-1]),
                    'residuals_autocorrelated': ljung_box['lb_pvalue'].iloc[-1] < 0.05
                },
                'model_summary': str(fitted_final_model.summary()).split('\n')[:10]  # İlk 10 satır
            }
            
        except Exception as e:
            logger.warning(f"ARIMA modeli hatası: {str(e)}")
            return {'error': str(e)}
    
    def _fit_prophet_model(self, data: pd.DataFrame, value_column: str, 
                          forecast_periods: int) -> Dict[str, Any]:
        """Prophet modeli"""
        try:
            # Prophet için veri formatı
            prophet_data = data.reset_index()
            prophet_data.columns = ['ds', 'y']
            
            # Model oluştur ve eğit
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(prophet_data)
            
            # Gelecek tarihler oluştur
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = model.predict(future)
            
            # Sadece tahmin kısmını al
            forecast_only = forecast.tail(forecast_periods)
            
            return {
                'forecast': forecast_only['yhat'].tolist(),
                'forecast_lower': forecast_only['yhat_lower'].tolist(),
                'forecast_upper': forecast_only['yhat_upper'].tolist(),
                'trend': forecast_only['trend'].tolist(),
                'seasonal': forecast_only.get('seasonal', [0] * forecast_periods),
                'model_components': {
                    'trend': 'Dahil',
                    'weekly_seasonality': 'Dahil',
                    'yearly_seasonality': 'Dahil'
                }
            }
            
        except Exception as e:
            logger.warning(f"Prophet modeli hatası: {str(e)}")
            return {'error': str(e)}
    
    def survival_analysis(self, duration_column: str, event_column: str,
                         group_column: Optional[str] = None) -> Dict[str, Any]:
        """Survival analizi"""
        try:
            # Veriyi hazırla
            survival_data = self.data[[duration_column, event_column]].copy()
            if group_column:
                survival_data[group_column] = self.data[group_column]
            
            survival_data = survival_data.dropna()
            
            if len(survival_data) < 10:
                raise ValueError("Survival analizi için en az 10 gözlem gerekli")
            
            # Kaplan-Meier estimator
            kmf = KaplanMeierFitter()
            kmf.fit(survival_data[duration_column], survival_data[event_column])
            
            # Temel istatistikler
            median_survival = kmf.median_survival_time_
            survival_function = kmf.survival_function_
            
            result = {
                'model_type': 'Survival Analizi',
                'duration_variable': duration_column,
                'event_variable': event_column,
                'sample_size': len(survival_data),
                'events_observed': int(survival_data[event_column].sum()),
                'censored_observations': int(len(survival_data) - survival_data[event_column].sum()),
                'median_survival_time': float(median_survival) if not pd.isna(median_survival) else None,
                'kaplan_meier': {
                    'survival_times': survival_function.index.tolist(),
                    'survival_probabilities': survival_function.iloc[:, 0].tolist()
                }
            }
            
            # Grup karşılaştırması (varsa)
            if group_column:
                group_comparison = self._compare_survival_groups(
                    survival_data, duration_column, event_column, group_column
                )
                result['group_comparison'] = group_comparison
            
            # Cox Proportional Hazards Model (grup varsa)
            if group_column:
                cox_results = self._fit_cox_model(survival_data, duration_column, event_column, group_column)
                result['cox_model'] = cox_results
            
            result['interpretation'] = self._generate_survival_interpretation(result)
            
            self.results['survival_analysis'] = result
            
            logger.info("Survival analizi tamamlandı")
            return result
            
        except Exception as e:
            logger.error(f"Survival analizi hatası: {str(e)}")
            return {'error': f'Survival analizi hatası: {str(e)}'}
    
    def _compare_survival_groups(self, data: pd.DataFrame, duration_col: str,
                               event_col: str, group_col: str) -> Dict[str, Any]:
        """Survival gruplarını karşılaştır"""
        try:
            groups = data[group_col].unique()
            
            if len(groups) != 2:
                return {'error': 'Grup karşılaştırması sadece 2 grup için destekleniyor'}
            
            group1_data = data[data[group_col] == groups[0]]
            group2_data = data[data[group_col] == groups[1]]
            
            # Log-rank test
            logrank_result = logrank_test(
                group1_data[duration_col], group2_data[duration_col],
                group1_data[event_col], group2_data[event_col]
            )
            
            # Her grup için Kaplan-Meier
            kmf1 = KaplanMeierFitter()
            kmf1.fit(group1_data[duration_col], group1_data[event_col], label=str(groups[0]))
            
            kmf2 = KaplanMeierFitter()
            kmf2.fit(group2_data[duration_col], group2_data[event_col], label=str(groups[1]))
            
            return {
                'groups': [str(g) for g in groups],
                'group_sizes': [len(group1_data), len(group2_data)],
                'median_survival_times': [
                    float(kmf1.median_survival_time_) if not pd.isna(kmf1.median_survival_time_) else None,
                    float(kmf2.median_survival_time_) if not pd.isna(kmf2.median_survival_time_) else None
                ],
                'logrank_test': {
                    'statistic': float(logrank_result.test_statistic),
                    'p_value': float(logrank_result.p_value),
                    'significant_difference': logrank_result.p_value < 0.05
                }
            }
            
        except Exception as e:
            logger.warning(f"Grup karşılaştırması hatası: {str(e)}")
            return {'error': str(e)}
    
    def _fit_cox_model(self, data: pd.DataFrame, duration_col: str,
                      event_col: str, group_col: str) -> Dict[str, Any]:
        """Cox Proportional Hazards Model"""
        try:
            # Grup değişkenini dummy variable'a çevir
            cox_data = data.copy()
            cox_data = pd.get_dummies(cox_data, columns=[group_col], prefix='group')
            
            # Cox model
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col=duration_col, event_col=event_col)
            
            return {
                'coefficients': cph.params_.to_dict(),
                'hazard_ratios': np.exp(cph.params_).to_dict(),
                'p_values': cph.summary['p'].to_dict(),
                'confidence_intervals': {
                    'lower': cph.confidence_intervals_.iloc[:, 0].to_dict(),
                    'upper': cph.confidence_intervals_.iloc[:, 1].to_dict()
                },
                'concordance_index': float(cph.concordance_index_),
                'log_likelihood': float(cph.log_likelihood_)
            }
            
        except Exception as e:
            logger.warning(f"Cox modeli hatası: {str(e)}")
            return {'error': str(e)}
    
    def meta_analysis(self, effect_sizes: List[float], 
                     standard_errors: List[float],
                     study_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Meta analiz"""
        try:
            if len(effect_sizes) != len(standard_errors):
                raise ValueError("Etki büyüklükleri ve standart hatalar aynı uzunlukta olmalı")
            
            if len(effect_sizes) < 2:
                raise ValueError("Meta analiz için en az 2 çalışma gerekli")
            
            effect_sizes = np.array(effect_sizes)
            standard_errors = np.array(standard_errors)
            variances = standard_errors ** 2
            weights = 1 / variances
            
            # Fixed-effects model
            fixed_effect = np.sum(weights * effect_sizes) / np.sum(weights)
            fixed_se = np.sqrt(1 / np.sum(weights))
            fixed_ci_lower = fixed_effect - 1.96 * fixed_se
            fixed_ci_upper = fixed_effect + 1.96 * fixed_se
            
            # Heterogeneity test (Q statistic)
            q_statistic = np.sum(weights * (effect_sizes - fixed_effect) ** 2)
            df = len(effect_sizes) - 1
            q_p_value = 1 - stats.chi2.cdf(q_statistic, df)
            
            # I² statistic
            i_squared = max(0, (q_statistic - df) / q_statistic) * 100
            
            # Random-effects model (DerSimonian-Laird)
            if q_statistic > df:
                tau_squared = (q_statistic - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights))
                random_weights = 1 / (variances + tau_squared)
                random_effect = np.sum(random_weights * effect_sizes) / np.sum(random_weights)
                random_se = np.sqrt(1 / np.sum(random_weights))
                random_ci_lower = random_effect - 1.96 * random_se
                random_ci_upper = random_effect + 1.96 * random_se
            else:
                tau_squared = 0
                random_effect = fixed_effect
                random_se = fixed_se
                random_ci_lower = fixed_ci_lower
                random_ci_upper = fixed_ci_upper
            
            # Study details
            if study_names is None:
                study_names = [f"Study_{i+1}" for i in range(len(effect_sizes))]
            
            study_details = []
            for i, name in enumerate(study_names):
                study_details.append({
                    'study_name': name,
                    'effect_size': float(effect_sizes[i]),
                    'standard_error': float(standard_errors[i]),
                    'weight_fixed': float(weights[i] / np.sum(weights) * 100),
                    'weight_random': float(random_weights[i] / np.sum(random_weights) * 100) if 'random_weights' in locals() else float(weights[i] / np.sum(weights) * 100)
                })
            
            result = {
                'model_type': 'Meta Analiz',
                'number_of_studies': len(effect_sizes),
                'fixed_effects_model': {
                    'pooled_effect': float(fixed_effect),
                    'standard_error': float(fixed_se),
                    'confidence_interval': [float(fixed_ci_lower), float(fixed_ci_upper)],
                    'z_score': float(fixed_effect / fixed_se),
                    'p_value': float(2 * (1 - stats.norm.cdf(abs(fixed_effect / fixed_se))))
                },
                'random_effects_model': {
                    'pooled_effect': float(random_effect),
                    'standard_error': float(random_se),
                    'confidence_interval': [float(random_ci_lower), float(random_ci_upper)],
                    'z_score': float(random_effect / random_se),
                    'p_value': float(2 * (1 - stats.norm.cdf(abs(random_effect / random_se)))),
                    'tau_squared': float(tau_squared)
                },
                'heterogeneity': {
                    'q_statistic': float(q_statistic),
                    'degrees_of_freedom': df,
                    'q_p_value': float(q_p_value),
                    'i_squared': float(i_squared),
                    'interpretation': self._interpret_heterogeneity(i_squared)
                },
                'study_details': study_details,
                'interpretation': self._generate_meta_analysis_interpretation(
                    fixed_effect, random_effect, i_squared, q_p_value
                )
            }
            
            self.results['meta_analysis'] = result
            
            logger.info("Meta analiz tamamlandı")
            return result
            
        except Exception as e:
            logger.error(f"Meta analiz hatası: {str(e)}")
            return {'error': f'Meta analiz hatası: {str(e)}'}
    
    def _interpret_heterogeneity(self, i_squared: float) -> str:
        """Heterogeneity yorumlama"""
        if i_squared < 25:
            return "Düşük heterogeneity"
        elif i_squared < 50:
            return "Orta heterogeneity"
        elif i_squared < 75:
            return "Yüksek heterogeneity"
        else:
            return "Çok yüksek heterogeneity"
    
    def _generate_ml_interpretation(self, problem_type: str, best_model: str,
                                  best_score: float, model_results: Dict) -> str:
        """Makine öğrenmesi yorumu"""
        interpretation = f"Makine Öğrenmesi Analizi Sonuçları ({problem_type.title()}):\n\n"
        
        if best_model:
            interpretation += f"• En iyi performans: {best_model}\n"
            
            if problem_type == 'classification':
                interpretation += f"• Doğruluk oranı: %{best_score*100:.1f}\n"
                
                if best_model in model_results:
                    metrics = model_results[best_model]['metrics']
                    interpretation += f"• Precision: {metrics.get('precision', 0):.3f}\n"
                    interpretation += f"• Recall: {metrics.get('recall', 0):.3f}\n"
                    interpretation += f"• F1-Score: {metrics.get('f1_score', 0):.3f}\n"
            else:
                interpretation += f"• R² skoru: {best_score:.3f}\n"
                
                if best_model in model_results:
                    metrics = model_results[best_model]['metrics']
                    interpretation += f"• RMSE: {metrics.get('rmse', 0):.3f}\n"
                    interpretation += f"• MAE: {metrics.get('mae', 0):.3f}\n"
        
        interpretation += f"\n• Toplam {len(model_results)} model karşılaştırıldı\n"
        interpretation += "• Cross-validation ile model performansları değerlendirildi"
        
        return interpretation
    
    def _generate_time_series_interpretation(self, basic_stats: Dict, 
                                           stationarity_tests: Dict,
                                           seasonality_analysis: Dict,
                                           arima_results: Dict) -> str:
        """Zaman serisi yorumu"""
        interpretation = "Zaman Serisi Analizi Sonuçları:\n\n"
        interpretation += f"• Veri trendi: {basic_stats['trend']}\n"
        interpretation += f"• Ortalama değer: {basic_stats['mean']:.2f}\n"
        
        if 'adf_test' in stationarity_tests:
            if stationarity_tests['adf_test']['is_stationary']:
                interpretation += "• Seri durağan (ADF testi)\n"
            else:
                interpretation += "• Seri durağan değil (ADF testi)\n"
        
        if seasonality_analysis.get('has_seasonality'):
            interpretation += "• Mevsimsellik tespit edildi\n"
        else:
            interpretation += "• Belirgin mevsimsellik yok\n"
        
        if 'model_order' in arima_results:
            order = arima_results['model_order']
            interpretation += f"• En uygun ARIMA modeli: ARIMA{order}\n"
            interpretation += f"• Model AIC değeri: {arima_results['aic']:.2f}\n"
        
        return interpretation
    
    def _generate_survival_interpretation(self, result: Dict) -> str:
        """Survival analizi yorumu"""
        interpretation = "Survival Analizi Sonuçları:\n\n"
        interpretation += f"• Toplam gözlem: {result['sample_size']}\n"
        interpretation += f"• Gözlenen olaylar: {result['events_observed']}\n"
        interpretation += f"• Sansürlenen gözlemler: {result['censored_observations']}\n"
        
        if result['median_survival_time']:
            interpretation += f"• Medyan survival süresi: {result['median_survival_time']:.2f}\n"
        else:
            interpretation += "• Medyan survival süresi hesaplanamadı\n"
        
        if 'group_comparison' in result:
            group_comp = result['group_comparison']
            if 'logrank_test' in group_comp:
                if group_comp['logrank_test']['significant_difference']:
                    interpretation += "• Gruplar arasında anlamlı fark var (Log-rank test)\n"
                else:
                    interpretation += "• Gruplar arasında anlamlı fark yok (Log-rank test)\n"
        
        return interpretation
    
    def _generate_meta_analysis_interpretation(self, fixed_effect: float,
                                             random_effect: float, i_squared: float,
                                             q_p_value: float) -> str:
        """Meta analiz yorumu"""
        interpretation = "Meta Analiz Sonuçları:\n\n"
        interpretation += f"• Sabit etkiler modeli: {fixed_effect:.3f}\n"
        interpretation += f"• Rastgele etkiler modeli: {random_effect:.3f}\n"
        interpretation += f"• Heterogeneity (I²): %{i_squared:.1f}\n"
        
        if q_p_value < 0.05:
            interpretation += "• Çalışmalar arasında anlamlı heterogeneity var\n"
            interpretation += "• Rastgele etkiler modeli önerilir\n"
        else:
            interpretation += "• Çalışmalar arasında anlamlı heterogeneity yok\n"
            interpretation += "• Sabit etkiler modeli kullanılabilir\n"
        
        return interpretation
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Analiz özetini döndür"""
        return {
            'total_analyses': len(self.results),
            'available_analyses': list(self.results.keys()),
            'analysis_types': [result.get('model_type', 'Unknown') for result in self.results.values()],
            'results': self.results
        }