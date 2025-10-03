"""
VeriVio Kapsamlı Regresyon Analizi Modülü
Doğrusal, lojistik, çoklu regresyon ve ileri analiz teknikleri
Otomatik model seçimi, tanı testleri ve yorumlama
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import jarque_bera
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor, TheilSenRegressor, RANSACRegressor
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, linear_harvey_collier,
    acorr_ljungbox
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ComprehensiveRegressionAnalyzer:
    """Kapsamlı regresyon analizi sınıfı"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        self.models = {}
        self.scalers = {}
        
        # Model performans kriterleri
        self.performance_thresholds = {
            'r2_excellent': 0.9,
            'r2_good': 0.7,
            'r2_moderate': 0.5,
            'r2_poor': 0.3
        }
    
    def _prepare_data(self, target: str, features: List[str], 
                     test_size: float = 0.2, scale_features: bool = False) -> Tuple:
        """Veriyi model için hazırla"""
        # Eksik değerleri temizle
        data_clean = self.data[features + [target]].dropna()
        
        if len(data_clean) < 10:
            raise ValueError("Model için yeterli veri yok (en az 10 gözlem gerekli)")
        
        X = data_clean[features]
        y = data_clean[target]
        
        # Özellik ölçeklendirme
        if scale_features:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            self.scalers[f"{target}_scaler"] = scaler
            X = X_scaled
        
        # Eğitim-test ayrımı
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            return X_train, X_test, y_train, y_test, X, y
        else:
            return X, None, y, None, X, y
    
    def _calculate_model_diagnostics(self, model, X: pd.DataFrame, y: pd.Series,
                                   y_pred: np.ndarray) -> Dict[str, Any]:
        """Model tanı testleri"""
        diagnostics = {}
        
        try:
            # Artıklar
            residuals = y - y_pred
            
            # Normallik testi (Jarque-Bera)
            jb_stat, jb_p = jarque_bera(residuals)
            diagnostics['normality_test'] = {
                'jarque_bera_statistic': float(jb_stat),
                'jarque_bera_p_value': float(jb_p),
                'is_normal': jb_p > 0.05
            }
            
            # Durbin-Watson testi (otokorelasyon)
            dw_stat = durbin_watson(residuals)
            diagnostics['autocorrelation_test'] = {
                'durbin_watson_statistic': float(dw_stat),
                'interpretation': self._interpret_durbin_watson(dw_stat)
            }
            
            # Heteroskedastisite testleri
            if hasattr(model, 'coef_') and len(X.columns) > 1:
                try:
                    # Breusch-Pagan testi
                    X_with_const = sm.add_constant(X)
                    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
                    
                    diagnostics['heteroscedasticity_test'] = {
                        'breusch_pagan_statistic': float(bp_stat),
                        'breusch_pagan_p_value': float(bp_p),
                        'is_homoscedastic': bp_p > 0.05
                    }
                except:
                    diagnostics['heteroscedasticity_test'] = {'error': 'Test hesaplanamadı'}
            
            # VIF (Variance Inflation Factor)
            if len(X.columns) > 1:
                try:
                    X_with_const = sm.add_constant(X)
                    vif_data = []
                    for i in range(1, X_with_const.shape[1]):  # Sabit terimi atla
                        vif = variance_inflation_factor(X_with_const.values, i)
                        vif_data.append({
                            'feature': X.columns[i-1],
                            'vif': float(vif),
                            'multicollinearity_concern': vif > 5
                        })
                    
                    diagnostics['multicollinearity_test'] = {
                        'vif_values': vif_data,
                        'max_vif': max([v['vif'] for v in vif_data]),
                        'has_multicollinearity': any([v['multicollinearity_concern'] for v in vif_data])
                    }
                except:
                    diagnostics['multicollinearity_test'] = {'error': 'VIF hesaplanamadı'}
            
        except Exception as e:
            logger.warning(f"Tanı testleri hesaplanamadı: {str(e)}")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Durbin-Watson istatistiğini yorumla"""
        if dw_stat < 1.5:
            return "Pozitif otokorelasyon var"
        elif dw_stat > 2.5:
            return "Negatif otokorelasyon var"
        else:
            return "Otokorelasyon yok"
    
    def _interpret_r2(self, r2: float) -> str:
        """R² değerini yorumla"""
        if r2 >= self.performance_thresholds['r2_excellent']:
            return "Mükemmel"
        elif r2 >= self.performance_thresholds['r2_good']:
            return "İyi"
        elif r2 >= self.performance_thresholds['r2_moderate']:
            return "Orta"
        elif r2 >= self.performance_thresholds['r2_poor']:
            return "Zayıf"
        else:
            return "Çok zayıf"
    
    def linear_regression(self, target: str, features: List[str],
                         test_size: float = 0.2, include_diagnostics: bool = True) -> Dict[str, Any]:
        """Doğrusal regresyon analizi"""
        try:
            # Veriyi hazırla
            X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_data(
                target, features, test_size
            )
            
            # Model oluştur ve eğit
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test) if X_test is not None else None
            y_full_pred = model.predict(X_full)
            
            # Performans metrikleri
            train_r2 = r2_score(y_train, y_train_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            
            test_metrics = {}
            if y_test is not None:
                test_r2 = r2_score(y_test, y_test_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_metrics = {
                    'r2': float(test_r2),
                    'mse': float(test_mse),
                    'mae': float(test_mae),
                    'rmse': float(np.sqrt(test_mse))
                }
            
            # Çapraz doğrulama
            cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='r2')
            
            # Katsayılar ve istatistikler
            coefficients = []
            for i, feature in enumerate(features):
                coefficients.append({
                    'feature': feature,
                    'coefficient': float(model.coef_[i]),
                    'abs_coefficient': float(abs(model.coef_[i]))
                })
            
            # Statsmodels ile detaylı istatistikler
            try:
                X_sm = sm.add_constant(X_full)
                sm_model = sm.OLS(y_full, X_sm).fit()
                
                detailed_stats = {
                    'intercept': {
                        'value': float(sm_model.params[0]),
                        'std_err': float(sm_model.bse[0]),
                        't_value': float(sm_model.tvalues[0]),
                        'p_value': float(sm_model.pvalues[0])
                    },
                    'coefficients_stats': []
                }
                
                for i, feature in enumerate(features):
                    detailed_stats['coefficients_stats'].append({
                        'feature': feature,
                        'coefficient': float(sm_model.params[i+1]),
                        'std_error': float(sm_model.bse[i+1]),
                        't_value': float(sm_model.tvalues[i+1]),
                        'p_value': float(sm_model.pvalues[i+1]),
                        'is_significant': sm_model.pvalues[i+1] < 0.05,
                        'confidence_interval': [
                            float(sm_model.conf_int().iloc[i+1, 0]),
                            float(sm_model.conf_int().iloc[i+1, 1])
                        ]
                    })
                
                # F-istatistiği
                detailed_stats['f_statistic'] = {
                    'value': float(sm_model.fvalue),
                    'p_value': float(sm_model.f_pvalue),
                    'is_significant': sm_model.f_pvalue < 0.05
                }
                
                # AIC, BIC
                detailed_stats['information_criteria'] = {
                    'aic': float(sm_model.aic),
                    'bic': float(sm_model.bic),
                    'log_likelihood': float(sm_model.llf)
                }
                
            except Exception as e:
                detailed_stats = {'error': f'Detaylı istatistikler hesaplanamadı: {str(e)}'}
            
            # Model tanı testleri
            diagnostics = {}
            if include_diagnostics:
                diagnostics = self._calculate_model_diagnostics(
                    model, X_full, y_full, y_full_pred
                )
            
            result = {
                'model_type': 'Doğrusal Regresyon',
                'target_variable': target,
                'features': features,
                'sample_size': len(X_full),
                'train_size': len(X_train),
                'test_size': len(X_test) if X_test is not None else 0,
                'intercept': float(model.intercept_),
                'coefficients': coefficients,
                'performance_metrics': {
                    'train': {
                        'r2': float(train_r2),
                        'mse': float(train_mse),
                        'mae': float(train_mae),
                        'rmse': float(np.sqrt(train_mse))
                    },
                    'test': test_metrics,
                    'cross_validation': {
                        'mean_r2': float(cv_scores.mean()),
                        'std_r2': float(cv_scores.std()),
                        'scores': cv_scores.tolist()
                    }
                },
                'model_quality': {
                    'r2_interpretation': self._interpret_r2(train_r2),
                    'overfitting_check': abs(train_r2 - cv_scores.mean()) > 0.1,
                    'feature_importance': sorted(coefficients, key=lambda x: x['abs_coefficient'], reverse=True)
                },
                'detailed_statistics': detailed_stats,
                'diagnostics': diagnostics,
                'interpretation': self._generate_linear_regression_interpretation(
                    train_r2, coefficients, detailed_stats
                )
            }
            
            # Modeli kaydet
            self.models[f'linear_regression_{target}'] = model
            self.results['linear_regression'] = result
            
            logger.info(f"Doğrusal regresyon tamamlandı: {target} ~ {features}")
            return result
            
        except Exception as e:
            logger.error(f"Doğrusal regresyon hatası: {str(e)}")
            return {'error': f'Doğrusal regresyon hatası: {str(e)}'}
    
    def logistic_regression(self, target: str, features: List[str],
                           test_size: float = 0.2, regularization: str = 'none',
                           C: float = 1.0) -> Dict[str, Any]:
        """Lojistik regresyon analizi"""
        try:
            # Veriyi hazırla
            X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_data(
                target, features, test_size, scale_features=True
            )
            
            # Hedef değişkenin binary olup olmadığını kontrol et
            unique_values = y_full.nunique()
            if unique_values > 2:
                return {'error': 'Lojistik regresyon için hedef değişken binary olmalıdır'}
            
            # Regularizasyon parametresi
            if regularization == 'l1':
                penalty = 'l1'
                solver = 'liblinear'
            elif regularization == 'l2':
                penalty = 'l2'
                solver = 'liblinear'
            elif regularization == 'elasticnet':
                penalty = 'elasticnet'
                solver = 'saga'
            else:
                penalty = 'none'
                solver = 'lbfgs'
            
            # Model oluştur ve eğit
            if penalty == 'elasticnet':
                model = LogisticRegression(
                    penalty=penalty, C=C, solver=solver, 
                    l1_ratio=0.5, random_state=42, max_iter=1000
                )
            else:
                model = LogisticRegression(
                    penalty=penalty, C=C, solver=solver, 
                    random_state=42, max_iter=1000
                )
            
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            
            test_metrics = {}
            if X_test is not None:
                y_test_pred = model.predict(X_test)
                y_test_proba = model.predict_proba(X_test)[:, 1]
                
                test_metrics = {
                    'accuracy': float(accuracy_score(y_test, y_test_pred)),
                    'precision': float(precision_score(y_test, y_test_pred, average='weighted')),
                    'recall': float(recall_score(y_test, y_test_pred, average='weighted')),
                    'f1_score': float(f1_score(y_test, y_test_pred, average='weighted')),
                    'roc_auc': float(roc_auc_score(y_test, y_test_proba)),
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
                }
            
            # Çapraz doğrulama
            cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')
            
            # Katsayılar ve odds ratios
            coefficients = []
            for i, feature in enumerate(features):
                coef = model.coef_[0][i]
                odds_ratio = np.exp(coef)
                coefficients.append({
                    'feature': feature,
                    'coefficient': float(coef),
                    'odds_ratio': float(odds_ratio),
                    'interpretation': self._interpret_odds_ratio(odds_ratio)
                })
            
            # Performans metrikleri
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average='weighted')
            train_recall = recall_score(y_train, y_train_pred, average='weighted')
            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            train_auc = roc_auc_score(y_train, y_train_proba)
            
            result = {
                'model_type': 'Lojistik Regresyon',
                'target_variable': target,
                'features': features,
                'sample_size': len(X_full),
                'train_size': len(X_train),
                'test_size': len(X_test) if X_test is not None else 0,
                'regularization': regularization,
                'C_parameter': C,
                'intercept': float(model.intercept_[0]),
                'coefficients': coefficients,
                'performance_metrics': {
                    'train': {
                        'accuracy': float(train_accuracy),
                        'precision': float(train_precision),
                        'recall': float(train_recall),
                        'f1_score': float(train_f1),
                        'roc_auc': float(train_auc),
                        'confusion_matrix': confusion_matrix(y_train, y_train_pred).tolist()
                    },
                    'test': test_metrics,
                    'cross_validation': {
                        'mean_accuracy': float(cv_scores.mean()),
                        'std_accuracy': float(cv_scores.std()),
                        'scores': cv_scores.tolist()
                    }
                },
                'model_quality': {
                    'accuracy_interpretation': self._interpret_accuracy(train_accuracy),
                    'overfitting_check': abs(train_accuracy - cv_scores.mean()) > 0.1,
                    'feature_importance': sorted(coefficients, key=lambda x: abs(x['coefficient']), reverse=True)
                },
                'interpretation': self._generate_logistic_regression_interpretation(
                    train_accuracy, train_auc, coefficients
                )
            }
            
            # Modeli kaydet
            self.models[f'logistic_regression_{target}'] = model
            self.results['logistic_regression'] = result
            
            logger.info(f"Lojistik regresyon tamamlandı: {target} ~ {features}")
            return result
            
        except Exception as e:
            logger.error(f"Lojistik regresyon hatası: {str(e)}")
            return {'error': f'Lojistik regresyon hatası: {str(e)}'}
    
    def multiple_regression_with_selection(self, target: str, features: List[str],
                                         selection_method: str = 'forward',
                                         test_size: float = 0.2) -> Dict[str, Any]:
        """Özellik seçimi ile çoklu regresyon"""
        try:
            # Veriyi hazırla
            X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_data(
                target, features, test_size
            )
            
            # Özellik seçimi
            if selection_method == 'rfe':
                # Recursive Feature Elimination
                base_model = LinearRegression()
                selector = RFE(base_model, n_features_to_select=min(5, len(features)))
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test) if X_test is not None else None
                X_full_selected = selector.transform(X_full)
                selected_features = [features[i] for i in range(len(features)) if selector.support_[i]]
                
            elif selection_method == 'kbest':
                # K-Best features
                k = min(5, len(features))
                selector = SelectKBest(score_func=f_regression, k=k)
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test) if X_test is not None else None
                X_full_selected = selector.transform(X_full)
                selected_indices = selector.get_support(indices=True)
                selected_features = [features[i] for i in selected_indices]
                
            else:  # forward/backward selection using statsmodels
                selected_features = self._stepwise_selection(X_full, y_full, features, method=selection_method)
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features] if X_test is not None else None
                X_full_selected = X_full[selected_features]
            
            # Seçilen özelliklerle model eğit
            model = LinearRegression()
            model.fit(X_train_selected, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train_selected)
            y_test_pred = model.predict(X_test_selected) if X_test_selected is not None else None
            y_full_pred = model.predict(X_full_selected)
            
            # Performans metrikleri
            train_r2 = r2_score(y_train, y_train_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            
            test_metrics = {}
            if y_test is not None:
                test_r2 = r2_score(y_test, y_test_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_metrics = {
                    'r2': float(test_r2),
                    'mse': float(test_mse),
                    'rmse': float(np.sqrt(test_mse))
                }
            
            # Katsayılar
            coefficients = []
            for i, feature in enumerate(selected_features):
                coefficients.append({
                    'feature': feature,
                    'coefficient': float(model.coef_[i]),
                    'abs_coefficient': float(abs(model.coef_[i]))
                })
            
            result = {
                'model_type': 'Özellik Seçimli Çoklu Regresyon',
                'target_variable': target,
                'original_features': features,
                'selected_features': selected_features,
                'selection_method': selection_method,
                'feature_reduction': f"{len(features)} -> {len(selected_features)}",
                'sample_size': len(X_full),
                'intercept': float(model.intercept_),
                'coefficients': coefficients,
                'performance_metrics': {
                    'train': {
                        'r2': float(train_r2),
                        'mse': float(train_mse),
                        'rmse': float(np.sqrt(train_mse))
                    },
                    'test': test_metrics
                },
                'model_quality': {
                    'r2_interpretation': self._interpret_r2(train_r2),
                    'feature_importance': sorted(coefficients, key=lambda x: x['abs_coefficient'], reverse=True)
                },
                'interpretation': self._generate_feature_selection_interpretation(
                    len(features), len(selected_features), train_r2, selected_features
                )
            }
            
            # Modeli kaydet
            self.models[f'multiple_regression_{target}'] = model
            self.results['multiple_regression'] = result
            
            logger.info(f"Özellik seçimli çoklu regresyon tamamlandı: {target}")
            return result
            
        except Exception as e:
            logger.error(f"Çoklu regresyon hatası: {str(e)}")
            return {'error': f'Çoklu regresyon hatası: {str(e)}'}
    
    def polynomial_regression(self, target: str, features: List[str],
                             degree: int = 2, test_size: float = 0.2) -> Dict[str, Any]:
        """Polinom regresyon analizi"""
        try:
            # Veriyi hazırla
            X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_data(
                target, features, test_size
            )
            
            # Polinom özellikleri oluştur
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test) if X_test is not None else None
            X_full_poly = poly_features.transform(X_full)
            
            # Model eğit
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train_poly)
            y_test_pred = model.predict(X_test_poly) if X_test_poly is not None else None
            
            # Performans metrikleri
            train_r2 = r2_score(y_train, y_train_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            
            test_metrics = {}
            if y_test is not None:
                test_r2 = r2_score(y_test, y_test_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_metrics = {
                    'r2': float(test_r2),
                    'mse': float(test_mse),
                    'rmse': float(np.sqrt(test_mse))
                }
            
            # Özellik isimleri
            feature_names = poly_features.get_feature_names_out(features)
            
            result = {
                'model_type': 'Polinom Regresyon',
                'target_variable': target,
                'original_features': features,
                'polynomial_degree': degree,
                'total_features_after_expansion': len(feature_names),
                'sample_size': len(X_full),
                'intercept': float(model.intercept_),
                'performance_metrics': {
                    'train': {
                        'r2': float(train_r2),
                        'mse': float(train_mse),
                        'rmse': float(np.sqrt(train_mse))
                    },
                    'test': test_metrics
                },
                'model_quality': {
                    'r2_interpretation': self._interpret_r2(train_r2),
                    'complexity_warning': len(feature_names) > len(X_full) / 10
                },
                'interpretation': self._generate_polynomial_regression_interpretation(
                    degree, train_r2, len(feature_names)
                )
            }
            
            # Modeli kaydet
            self.models[f'polynomial_regression_{target}'] = {
                'model': model,
                'poly_features': poly_features
            }
            self.results['polynomial_regression'] = result
            
            logger.info(f"Polinom regresyon tamamlandı: {target} (derece: {degree})")
            return result
            
        except Exception as e:
            logger.error(f"Polinom regresyon hatası: {str(e)}")
            return {'error': f'Polinom regresyon hatası: {str(e)}'}
    
    def regularized_regression(self, target: str, features: List[str],
                              method: str = 'ridge', test_size: float = 0.2,
                              alpha: float = 1.0) -> Dict[str, Any]:
        """Düzenlenmiş regresyon (Ridge, Lasso, ElasticNet)"""
        try:
            # Veriyi hazırla (ölçeklendirme ile)
            X_train, X_test, y_train, y_test, X_full, y_full = self._prepare_data(
                target, features, test_size, scale_features=True
            )
            
            # Model seçimi
            if method.lower() == 'ridge':
                model = Ridge(alpha=alpha, random_state=42)
            elif method.lower() == 'lasso':
                model = Lasso(alpha=alpha, random_state=42, max_iter=1000)
            elif method.lower() == 'elasticnet':
                model = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42, max_iter=1000)
            else:
                return {'error': 'Desteklenmeyen düzenleme yöntemi (ridge, lasso, elasticnet)'}
            
            # Model eğit
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test) if X_test is not None else None
            
            # Performans metrikleri
            train_r2 = r2_score(y_train, y_train_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            
            test_metrics = {}
            if y_test is not None:
                test_r2 = r2_score(y_test, y_test_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_metrics = {
                    'r2': float(test_r2),
                    'mse': float(test_mse),
                    'rmse': float(np.sqrt(test_mse))
                }
            
            # Katsayılar
            coefficients = []
            non_zero_features = 0
            for i, feature in enumerate(features):
                coef = model.coef_[i]
                if abs(coef) > 1e-10:
                    non_zero_features += 1
                coefficients.append({
                    'feature': feature,
                    'coefficient': float(coef),
                    'abs_coefficient': float(abs(coef)),
                    'is_selected': abs(coef) > 1e-10
                })
            
            # Çapraz doğrulama ile optimal alpha bulma
            alphas = np.logspace(-4, 2, 20)
            cv_scores = []
            
            for a in alphas:
                if method.lower() == 'ridge':
                    temp_model = Ridge(alpha=a, random_state=42)
                elif method.lower() == 'lasso':
                    temp_model = Lasso(alpha=a, random_state=42, max_iter=1000)
                else:
                    temp_model = ElasticNet(alpha=a, l1_ratio=0.5, random_state=42, max_iter=1000)
                
                scores = cross_val_score(temp_model, X_full, y_full, cv=5, scoring='r2')
                cv_scores.append(scores.mean())
            
            optimal_alpha = alphas[np.argmax(cv_scores)]
            
            result = {
                'model_type': f'{method.title()} Regresyon',
                'target_variable': target,
                'features': features,
                'regularization_method': method,
                'alpha_parameter': alpha,
                'optimal_alpha': float(optimal_alpha),
                'sample_size': len(X_full),
                'intercept': float(model.intercept_),
                'coefficients': coefficients,
                'feature_selection': {
                    'total_features': len(features),
                    'selected_features': non_zero_features,
                    'selection_ratio': non_zero_features / len(features)
                },
                'performance_metrics': {
                    'train': {
                        'r2': float(train_r2),
                        'mse': float(train_mse),
                        'rmse': float(np.sqrt(train_mse))
                    },
                    'test': test_metrics
                },
                'model_quality': {
                    'r2_interpretation': self._interpret_r2(train_r2),
                    'regularization_effect': f"{len(features) - non_zero_features} özellik sıfırlandı" if method.lower() == 'lasso' else "Katsayılar küçültüldü"
                },
                'interpretation': self._generate_regularized_regression_interpretation(
                    method, train_r2, non_zero_features, len(features)
                )
            }
            
            # Modeli kaydet
            self.models[f'{method}_regression_{target}'] = model
            self.results[f'{method}_regression'] = result
            
            logger.info(f"{method.title()} regresyon tamamlandı: {target}")
            return result
            
        except Exception as e:
            logger.error(f"{method} regresyon hatası: {str(e)}")
            return {'error': f'{method} regresyon hatası: {str(e)}'}
    
    def _stepwise_selection(self, X: pd.DataFrame, y: pd.Series, 
                           features: List[str], method: str = 'forward') -> List[str]:
        """Adımsal özellik seçimi"""
        if method == 'forward':
            selected = []
            remaining = features.copy()
            
            while remaining:
                best_feature = None
                best_aic = float('inf')
                
                for feature in remaining:
                    test_features = selected + [feature]
                    X_test = sm.add_constant(X[test_features])
                    try:
                        model = sm.OLS(y, X_test).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_feature = feature
                    except:
                        continue
                
                if best_feature and len(selected) < min(5, len(features)):
                    selected.append(best_feature)
                    remaining.remove(best_feature)
                else:
                    break
            
            return selected if selected else features[:3]  # En az 3 özellik döndür
        
        else:  # backward
            selected = features.copy()
            
            while len(selected) > 1:
                worst_feature = None
                best_aic = float('inf')
                
                for feature in selected:
                    test_features = [f for f in selected if f != feature]
                    X_test = sm.add_constant(X[test_features])
                    try:
                        model = sm.OLS(y, X_test).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            worst_feature = feature
                    except:
                        continue
                
                if worst_feature and len(selected) > 3:
                    selected.remove(worst_feature)
                else:
                    break
            
            return selected
    
    def _interpret_odds_ratio(self, odds_ratio: float) -> str:
        """Odds ratio yorumlama"""
        if odds_ratio > 1.5:
            return f"Güçlü pozitif etki ({odds_ratio:.2f}x artış)"
        elif odds_ratio > 1.1:
            return f"Orta pozitif etki ({odds_ratio:.2f}x artış)"
        elif odds_ratio > 0.9:
            return "Minimal etki"
        elif odds_ratio > 0.67:
            return f"Orta negatif etki ({1/odds_ratio:.2f}x azalış)"
        else:
            return f"Güçlü negatif etki ({1/odds_ratio:.2f}x azalış)"
    
    def _interpret_accuracy(self, accuracy: float) -> str:
        """Doğruluk yorumlama"""
        if accuracy >= 0.95:
            return "Mükemmel"
        elif accuracy >= 0.85:
            return "Çok iyi"
        elif accuracy >= 0.75:
            return "İyi"
        elif accuracy >= 0.65:
            return "Orta"
        else:
            return "Zayıf"
    
    def _generate_linear_regression_interpretation(self, r2: float, coefficients: List[Dict],
                                                 detailed_stats: Dict) -> str:
        """Doğrusal regresyon yorumu"""
        interpretation = f"Doğrusal Regresyon Analizi Sonuçları:\n\n"
        interpretation += f"• Model açıklama gücü: R² = {r2:.3f} ({self._interpret_r2(r2)})\n"
        interpretation += f"• Model varyansın %{r2*100:.1f}'ini açıklıyor\n\n"
        
        interpretation += "En etkili değişkenler:\n"
        for i, coef in enumerate(coefficients[:3]):
            direction = "pozitif" if coef['coefficient'] > 0 else "negatif"
            interpretation += f"• {coef['feature']}: {direction} etki (katsayı: {coef['coefficient']:.3f})\n"
        
        if 'f_statistic' in detailed_stats and detailed_stats['f_statistic']['is_significant']:
            interpretation += f"\n• Model genel olarak anlamlıdır (F-test p < 0.05)"
        
        return interpretation
    
    def _generate_logistic_regression_interpretation(self, accuracy: float, auc: float,
                                                   coefficients: List[Dict]) -> str:
        """Lojistik regresyon yorumu"""
        interpretation = f"Lojistik Regresyon Analizi Sonuçları:\n\n"
        interpretation += f"• Model doğruluğu: {accuracy:.3f} ({self._interpret_accuracy(accuracy)})\n"
        interpretation += f"• ROC AUC: {auc:.3f} (ayırt etme gücü)\n\n"
        
        interpretation += "En etkili değişkenler:\n"
        for i, coef in enumerate(coefficients[:3]):
            interpretation += f"• {coef['feature']}: {coef['interpretation']}\n"
        
        return interpretation
    
    def _generate_feature_selection_interpretation(self, original_count: int, selected_count: int,
                                                 r2: float, selected_features: List[str]) -> str:
        """Özellik seçimi yorumu"""
        interpretation = f"Özellik Seçimi ile Çoklu Regresyon Sonuçları:\n\n"
        interpretation += f"• {original_count} özellikten {selected_count} tanesi seçildi\n"
        interpretation += f"• Model açıklama gücü: R² = {r2:.3f}\n"
        interpretation += f"• Seçilen özellikler: {', '.join(selected_features)}\n"
        
        return interpretation
    
    def _generate_polynomial_regression_interpretation(self, degree: int, r2: float,
                                                     feature_count: int) -> str:
        """Polinom regresyon yorumu"""
        interpretation = f"Polinom Regresyon Analizi Sonuçları:\n\n"
        interpretation += f"• Polinom derecesi: {degree}\n"
        interpretation += f"• Toplam özellik sayısı: {feature_count}\n"
        interpretation += f"• Model açıklama gücü: R² = {r2:.3f}\n"
        
        if feature_count > 20:
            interpretation += "• Uyarı: Yüksek boyutlu model, aşırı öğrenme riski var\n"
        
        return interpretation
    
    def _generate_regularized_regression_interpretation(self, method: str, r2: float,
                                                      selected_features: int, total_features: int) -> str:
        """Düzenlenmiş regresyon yorumu"""
        interpretation = f"{method.title()} Regresyon Analizi Sonuçları:\n\n"
        interpretation += f"• Model açıklama gücü: R² = {r2:.3f}\n"
        interpretation += f"• Aktif özellik sayısı: {selected_features}/{total_features}\n"
        
        if method.lower() == 'lasso':
            interpretation += f"• Lasso {total_features - selected_features} özelliği otomatik olarak eledi\n"
        elif method.lower() == 'ridge':
            interpretation += "• Ridge tüm özellikleri kullandı ancak katsayıları küçülttü\n"
        
        return interpretation
    
    def compare_models(self) -> Dict[str, Any]:
        """Modelleri karşılaştır"""
        if not self.results:
            return {'error': 'Karşılaştırılacak model yok'}
        
        comparison = {
            'model_count': len(self.results),
            'models': [],
            'best_model': None,
            'comparison_summary': {}
        }
        
        best_r2 = -1
        best_model_name = None
        
        for model_name, result in self.results.items():
            if 'error' not in result:
                model_info = {
                    'name': model_name,
                    'type': result.get('model_type', 'Unknown'),
                    'target': result.get('target_variable', 'Unknown')
                }
                
                # Performans metriklerini al
                if 'performance_metrics' in result:
                    train_metrics = result['performance_metrics'].get('train', {})
                    if 'r2' in train_metrics:
                        r2 = train_metrics['r2']
                        model_info['r2'] = r2
                        if r2 > best_r2:
                            best_r2 = r2
                            best_model_name = model_name
                    
                    if 'accuracy' in train_metrics:
                        model_info['accuracy'] = train_metrics['accuracy']
                
                comparison['models'].append(model_info)
        
        comparison['best_model'] = best_model_name
        comparison['comparison_summary'] = {
            'best_r2': best_r2,
            'model_types': list(set([m['type'] for m in comparison['models']]))
        }
        
        return comparison
    
    def comprehensive_regression_analysis(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Kapsamlı regresyon analizi gerçekleştirir"""
        try:
            # Parametreleri al
            target_column = params.get('target_column')
            feature_columns = params.get('columns', [])
            regression_type = params.get('regression_type', 'linear')
            test_size = params.get('test_size', 0.2)
            cross_validation = params.get('cross_validation', True)
            
            if not target_column:
                return {'error': 'Hedef değişken belirtilmedi'}
            
            if not feature_columns:
                # Hedef değişken dışındaki tüm sayısal sütunları kullan
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [col for col in numeric_cols if col != target_column]
            
            if not feature_columns:
                return {'error': 'Analiz için uygun özellik bulunamadı'}
            
            # Regresyon türüne göre analiz yap
            if regression_type == 'linear':
                result = self.linear_regression(target_column, feature_columns, test_size)
            elif regression_type == 'multiple':
                result = self.multiple_regression_with_selection(target_column, feature_columns, 'forward', test_size)
            elif regression_type == 'polynomial':
                degree = params.get('polynomial_degree', 2)
                result = self.polynomial_regression(target_column, feature_columns, degree, test_size)
            elif regression_type in ['ridge', 'lasso', 'elastic_net']:
                alpha = params.get('alpha', 1.0)
                result = self.regularized_regression(target_column, feature_columns, regression_type, alpha, test_size)
            else:
                # Varsayılan olarak doğrusal regresyon
                result = self.linear_regression(target_column, feature_columns, test_size)
            
            # Sonuçları kaydet
            analysis_key = f"{regression_type}_{target_column}"
            self.results[analysis_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Kapsamlı regresyon analizi hatası: {str(e)}")
            return {'error': f'Regresyon analizi hatası: {str(e)}'}

    def get_model_summary(self) -> Dict[str, Any]:
        """Model özetini döndür"""
        return {
            'total_models': len(self.results),
            'available_models': list(self.results.keys()),
            'model_types': [result.get('model_type', 'Unknown') for result in self.results.values()],
            'results': self.results
        }