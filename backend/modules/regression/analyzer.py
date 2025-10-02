"""
Regresyon Analizi Sınıfı

Bu modül çeşitli regresyon analizi türlerini gerçekleştirmek için kullanılır.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from scipy import stats
from scipy.stats import jarque_bera, durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class RegressionAnalyzer:
    """
    Çeşitli regresyon analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        RegressionAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek veri seti
        """
        self.data = data.copy()
        self.results = {}
        self.models = {}
        self.scalers = {}
        
    def linear_regression(self, target_column: str, feature_columns: List[str],
                         test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Doğrusal regresyon analizi gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_columns: Özellik değişkenleri
            test_size: Test seti oranı
            random_state: Rastgele durum
            
        Returns:
            Regresyon analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            if len(data_clean) < 10:
                return {'error': 'Regresyon analizi için en az 10 gözlem gereklidir'}
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Modeli eğit
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Model performansı
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Katsayılar ve istatistikler
            coefficients = {}
            for i, feature in enumerate(feature_columns):
                coefficients[feature] = {
                    'coefficient': float(model.coef_[i]),
                    'abs_coefficient': float(abs(model.coef_[i]))
                }
            
            # Residual analizi
            residuals_train = y_train - y_train_pred
            residuals_test = y_test - y_test_pred
            
            # Normallik testi (residuals)
            _, residuals_normality_p = jarque_bera(residuals_train)
            
            # Durbin-Watson testi (otokorelasyon)
            dw_statistic = durbin_watson(residuals_train)
            
            # Feature importance (mutlak katsayılar)
            feature_importance = sorted(
                [(feature, abs(model.coef_[i])) for i, feature in enumerate(feature_columns)],
                key=lambda x: x[1], reverse=True
            )
            
            result = {
                'model_type': 'Linear Regression',
                'target_column': target_column,
                'feature_columns': feature_columns,
                'sample_size': len(data_clean),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'intercept': float(model.intercept_),
                'coefficients': coefficients,
                'feature_importance': [(feat, float(imp)) for feat, imp in feature_importance],
                'performance_metrics': {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                },
                'residual_analysis': {
                    'residuals_normality_p': float(residuals_normality_p),
                    'residuals_normal': residuals_normality_p > 0.05,
                    'durbin_watson': float(dw_statistic),
                    'autocorrelation_concern': dw_statistic < 1.5 or dw_statistic > 2.5
                },
                'model_diagnostics': {
                    'overfitting_risk': train_r2 - test_r2 > 0.1,
                    'r2_difference': float(train_r2 - test_r2),
                    'model_quality': self._assess_model_quality(test_r2)
                },
                'interpretation': self._interpret_linear_regression(
                    test_r2, feature_importance, residuals_normality_p > 0.05
                )
            }
            
            # Modeli sakla
            self.models['linear_regression'] = model
            self.results['linear_regression'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Doğrusal regresyon hatası: {str(e)}'}
    
    def logistic_regression(self, target_column: str, feature_columns: List[str],
                           test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Lojistik regresyon analizi gerçekleştirir
        
        Args:
            target_column: Hedef değişken (binary)
            feature_columns: Özellik değişkenleri
            test_size: Test seti oranı
            random_state: Rastgele durum
            
        Returns:
            Lojistik regresyon analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            # Binary kontrolü
            unique_values = y.unique()
            if len(unique_values) != 2:
                return {'error': 'Lojistik regresyon için hedef değişken binary olmalıdır'}
            
            if len(data_clean) < 10:
                return {'error': 'Lojistik regresyon için en az 10 gözlem gereklidir'}
            
            # Veriyi standardize et
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Modeli eğit
            model = LogisticRegression(random_state=random_state, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            # Model performansı
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_precision = precision_score(y_train, y_train_pred, average='binary')
            test_precision = precision_score(y_test, y_test_pred, average='binary')
            train_recall = recall_score(y_train, y_train_pred, average='binary')
            test_recall = recall_score(y_test, y_test_pred, average='binary')
            train_f1 = f1_score(y_train, y_train_pred, average='binary')
            test_f1 = f1_score(y_test, y_test_pred, average='binary')
            
            # AUC-ROC
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            
            # Katsayılar ve odds ratios
            coefficients = {}
            for i, feature in enumerate(feature_columns):
                coef = model.coef_[0][i]
                odds_ratio = np.exp(coef)
                coefficients[feature] = {
                    'coefficient': float(coef),
                    'odds_ratio': float(odds_ratio),
                    'abs_coefficient': float(abs(coef))
                }
            
            # Feature importance
            feature_importance = sorted(
                [(feature, abs(model.coef_[0][i])) for i, feature in enumerate(feature_columns)],
                key=lambda x: x[1], reverse=True
            )
            
            # Confusion matrix
            cm_test = confusion_matrix(y_test, y_test_pred)
            
            result = {
                'model_type': 'Logistic Regression',
                'target_column': target_column,
                'feature_columns': feature_columns,
                'sample_size': len(data_clean),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'target_classes': unique_values.tolist(),
                'intercept': float(model.intercept_[0]),
                'coefficients': coefficients,
                'feature_importance': [(feat, float(imp)) for feat, imp in feature_importance],
                'performance_metrics': {
                    'train_accuracy': float(train_accuracy),
                    'test_accuracy': float(test_accuracy),
                    'train_precision': float(train_precision),
                    'test_precision': float(test_precision),
                    'train_recall': float(train_recall),
                    'test_recall': float(test_recall),
                    'train_f1': float(train_f1),
                    'test_f1': float(test_f1),
                    'train_auc': float(train_auc),
                    'test_auc': float(test_auc),
                    'cv_accuracy_mean': float(cv_scores.mean()),
                    'cv_accuracy_std': float(cv_scores.std())
                },
                'confusion_matrix': cm_test.tolist(),
                'model_diagnostics': {
                    'overfitting_risk': train_accuracy - test_accuracy > 0.1,
                    'accuracy_difference': float(train_accuracy - test_accuracy),
                    'model_quality': self._assess_classification_quality(test_accuracy, test_auc)
                },
                'interpretation': self._interpret_logistic_regression(
                    test_accuracy, test_auc, feature_importance
                )
            }
            
            # Modeli ve scaler'ı sakla
            self.models['logistic_regression'] = model
            self.scalers['logistic_regression'] = scaler
            self.results['logistic_regression'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Lojistik regresyon hatası: {str(e)}'}
    
    def polynomial_regression(self, target_column: str, feature_column: str,
                             degree: int = 2, test_size: float = 0.2,
                             random_state: int = 42) -> Dict[str, Any]:
        """
        Polinom regresyon analizi gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_column: Özellik değişkeni (tek değişken)
            degree: Polinom derecesi
            test_size: Test seti oranı
            random_state: Rastgele durum
            
        Returns:
            Polinom regresyon analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[[feature_column, target_column]].dropna()
            X = data_clean[[feature_column]]
            y = data_clean[target_column]
            
            if len(data_clean) < 10:
                return {'error': 'Polinom regresyon için en az 10 gözlem gereklidir'}
            
            # Polinom özellikleri oluştur
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly_features.fit_transform(X)
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X_poly, y, test_size=test_size, random_state=random_state
            )
            
            # Modeli eğit
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Model performansı
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
            
            # Linear regresyon ile karşılaştırma
            linear_model = LinearRegression()
            linear_model.fit(X[y_train.index], y_train)
            linear_test_pred = linear_model.predict(X[y_test.index])
            linear_r2 = r2_score(y_test, linear_test_pred)
            
            # Polinom katsayıları
            feature_names = poly_features.get_feature_names_out([feature_column])
            coefficients = {}
            for i, name in enumerate(feature_names):
                coefficients[name] = float(model.coef_[i])
            
            result = {
                'model_type': 'Polynomial Regression',
                'target_column': target_column,
                'feature_column': feature_column,
                'polynomial_degree': degree,
                'sample_size': len(data_clean),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'intercept': float(model.intercept_),
                'coefficients': coefficients,
                'performance_metrics': {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                },
                'comparison_with_linear': {
                    'linear_r2': float(linear_r2),
                    'polynomial_r2': float(test_r2),
                    'improvement': float(test_r2 - linear_r2),
                    'is_better': test_r2 > linear_r2
                },
                'model_diagnostics': {
                    'overfitting_risk': train_r2 - test_r2 > 0.15,
                    'r2_difference': float(train_r2 - test_r2),
                    'model_quality': self._assess_model_quality(test_r2)
                },
                'interpretation': self._interpret_polynomial_regression(
                    test_r2, linear_r2, degree, train_r2 - test_r2 > 0.15
                )
            }
            
            # Modeli sakla
            self.models['polynomial_regression'] = {
                'model': model,
                'poly_features': poly_features
            }
            self.results['polynomial_regression'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Polinom regresyon hatası: {str(e)}'}
    
    def ridge_regression(self, target_column: str, feature_columns: List[str],
                        alpha: float = 1.0, test_size: float = 0.2,
                        random_state: int = 42) -> Dict[str, Any]:
        """
        Ridge regresyon analizi gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_columns: Özellik değişkenleri
            alpha: Regularization parametresi
            test_size: Test seti oranı
            random_state: Rastgele durum
            
        Returns:
            Ridge regresyon analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            if len(data_clean) < 10:
                return {'error': 'Ridge regresyon için en az 10 gözlem gereklidir'}
            
            # Veriyi standardize et
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            
            # Modeli eğit
            model = Ridge(alpha=alpha, random_state=random_state)
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Model performansı
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            
            # Linear regresyon ile karşılaştırma
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            linear_test_pred = linear_model.predict(X_test)
            linear_r2 = r2_score(y_test, linear_test_pred)
            
            # Katsayılar
            coefficients = {}
            for i, feature in enumerate(feature_columns):
                coefficients[feature] = {
                    'coefficient': float(model.coef_[i]),
                    'abs_coefficient': float(abs(model.coef_[i]))
                }
            
            # Feature importance
            feature_importance = sorted(
                [(feature, abs(model.coef_[i])) for i, feature in enumerate(feature_columns)],
                key=lambda x: x[1], reverse=True
            )
            
            result = {
                'model_type': 'Ridge Regression',
                'target_column': target_column,
                'feature_columns': feature_columns,
                'alpha': alpha,
                'sample_size': len(data_clean),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'intercept': float(model.intercept_),
                'coefficients': coefficients,
                'feature_importance': [(feat, float(imp)) for feat, imp in feature_importance],
                'performance_metrics': {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                },
                'comparison_with_linear': {
                    'linear_r2': float(linear_r2),
                    'ridge_r2': float(test_r2),
                    'improvement': float(test_r2 - linear_r2),
                    'is_better': test_r2 > linear_r2
                },
                'regularization_effect': {
                    'alpha_value': alpha,
                    'coefficient_shrinkage': float(np.mean([abs(c['coefficient']) for c in coefficients.values()])),
                    'overfitting_reduction': max(0, linear_r2 - test_r2) if linear_r2 > test_r2 else 0
                },
                'interpretation': self._interpret_ridge_regression(
                    test_r2, linear_r2, alpha, feature_importance
                )
            }
            
            # Modeli ve scaler'ı sakla
            self.models['ridge_regression'] = model
            self.scalers['ridge_regression'] = scaler
            self.results['ridge_regression'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Ridge regresyon hatası: {str(e)}'}
    
    def lasso_regression(self, target_column: str, feature_columns: List[str],
                        alpha: float = 1.0, test_size: float = 0.2,
                        random_state: int = 42) -> Dict[str, Any]:
        """
        Lasso regresyon analizi gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_columns: Özellik değişkenleri
            alpha: Regularization parametresi
            test_size: Test seti oranı
            random_state: Rastgele durum
            
        Returns:
            Lasso regresyon analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            if len(data_clean) < 10:
                return {'error': 'Lasso regresyon için en az 10 gözlem gereklidir'}
            
            # Veriyi standardize et
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            
            # Modeli eğit
            model = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
            model.fit(X_train, y_train)
            
            # Tahminler
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Model performansı
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            
            # Özellik seçimi
            selected_features = [feature for i, feature in enumerate(feature_columns) 
                               if abs(model.coef_[i]) > 1e-6]
            eliminated_features = [feature for i, feature in enumerate(feature_columns) 
                                 if abs(model.coef_[i]) <= 1e-6]
            
            # Katsayılar
            coefficients = {}
            for i, feature in enumerate(feature_columns):
                coefficients[feature] = {
                    'coefficient': float(model.coef_[i]),
                    'abs_coefficient': float(abs(model.coef_[i])),
                    'selected': abs(model.coef_[i]) > 1e-6
                }
            
            # Feature importance (sadece seçilen özellikler)
            feature_importance = sorted(
                [(feature, abs(model.coef_[i])) for i, feature in enumerate(feature_columns) 
                 if abs(model.coef_[i]) > 1e-6],
                key=lambda x: x[1], reverse=True
            )
            
            result = {
                'model_type': 'Lasso Regression',
                'target_column': target_column,
                'feature_columns': feature_columns,
                'alpha': alpha,
                'sample_size': len(data_clean),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'intercept': float(model.intercept_),
                'coefficients': coefficients,
                'feature_selection': {
                    'selected_features': selected_features,
                    'eliminated_features': eliminated_features,
                    'num_selected': len(selected_features),
                    'num_eliminated': len(eliminated_features),
                    'selection_ratio': len(selected_features) / len(feature_columns)
                },
                'feature_importance': [(feat, float(imp)) for feat, imp in feature_importance],
                'performance_metrics': {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                },
                'regularization_effect': {
                    'alpha_value': alpha,
                    'sparsity_level': len(eliminated_features) / len(feature_columns),
                    'model_complexity': len(selected_features)
                },
                'interpretation': self._interpret_lasso_regression(
                    test_r2, alpha, selected_features, eliminated_features
                )
            }
            
            # Modeli ve scaler'ı sakla
            self.models['lasso_regression'] = model
            self.scalers['lasso_regression'] = scaler
            self.results['lasso_regression'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Lasso regresyon hatası: {str(e)}'}
    
    def _assess_model_quality(self, r2_score: float) -> str:
        """R² skoruna göre model kalitesini değerlendirir"""
        if r2_score >= 0.9:
            return "Mükemmel"
        elif r2_score >= 0.8:
            return "Çok iyi"
        elif r2_score >= 0.7:
            return "İyi"
        elif r2_score >= 0.5:
            return "Orta"
        elif r2_score >= 0.3:
            return "Zayıf"
        else:
            return "Çok zayıf"
    
    def _assess_classification_quality(self, accuracy: float, auc: float) -> str:
        """Accuracy ve AUC skorlarına göre sınıflandırma kalitesini değerlendirir"""
        avg_score = (accuracy + auc) / 2
        if avg_score >= 0.95:
            return "Mükemmel"
        elif avg_score >= 0.85:
            return "Çok iyi"
        elif avg_score >= 0.75:
            return "İyi"
        elif avg_score >= 0.65:
            return "Orta"
        elif avg_score >= 0.55:
            return "Zayıf"
        else:
            return "Çok zayıf"
    
    def _interpret_linear_regression(self, r2: float, feature_importance: List, 
                                   residuals_normal: bool) -> str:
        """Doğrusal regresyon sonucunu yorumlar"""
        quality = self._assess_model_quality(r2)
        top_feature = feature_importance[0][0] if feature_importance else "bilinmiyor"
        
        interpretation = f"Model kalitesi {quality.lower()}dir (R² = {r2:.3f}). "
        interpretation += f"En önemli özellik: {top_feature}. "
        
        if not residuals_normal:
            interpretation += "Residuallerin normal dağılmaması model varsayımlarında sorun olduğunu gösterir."
        else:
            interpretation += "Model varsayımları genel olarak karşılanmaktadır."
        
        return interpretation
    
    def _interpret_logistic_regression(self, accuracy: float, auc: float, 
                                     feature_importance: List) -> str:
        """Lojistik regresyon sonucunu yorumlar"""
        quality = self._assess_classification_quality(accuracy, auc)
        top_feature = feature_importance[0][0] if feature_importance else "bilinmiyor"
        
        interpretation = f"Model kalitesi {quality.lower()}dir (Accuracy = {accuracy:.3f}, AUC = {auc:.3f}). "
        interpretation += f"En önemli özellik: {top_feature}. "
        
        if auc > 0.8:
            interpretation += "Model iyi ayırt edici güce sahiptir."
        elif auc > 0.7:
            interpretation += "Model orta düzeyde ayırt edici güce sahiptir."
        else:
            interpretation += "Model zayıf ayırt edici güce sahiptir."
        
        return interpretation
    
    def _interpret_polynomial_regression(self, poly_r2: float, linear_r2: float, 
                                       degree: int, overfitting: bool) -> str:
        """Polinom regresyon sonucunu yorumlar"""
        improvement = poly_r2 - linear_r2
        
        interpretation = f"Derece {degree} polinom modeli "
        
        if improvement > 0.05:
            interpretation += f"doğrusal modelden önemli ölçüde daha iyi performans gösteriyor (iyileşme: {improvement:.3f}). "
        elif improvement > 0:
            interpretation += f"doğrusal modelden hafif daha iyi performans gösteriyor (iyileşme: {improvement:.3f}). "
        else:
            interpretation += f"doğrusal modelden daha kötü performans gösteriyor (kötüleşme: {abs(improvement):.3f}). "
        
        if overfitting:
            interpretation += "Aşırı öğrenme riski tespit edildi."
        
        return interpretation
    
    def _interpret_ridge_regression(self, ridge_r2: float, linear_r2: float, 
                                  alpha: float, feature_importance: List) -> str:
        """Ridge regresyon sonucunu yorumlar"""
        improvement = ridge_r2 - linear_r2
        top_feature = feature_importance[0][0] if feature_importance else "bilinmiyor"
        
        interpretation = f"Ridge regresyon (α={alpha}) "
        
        if improvement > 0:
            interpretation += f"doğrusal regresyondan daha iyi performans gösteriyor (iyileşme: {improvement:.3f}). "
        else:
            interpretation += f"doğrusal regresyonla benzer performans gösteriyor. "
        
        interpretation += f"En önemli özellik: {top_feature}. "
        interpretation += "Regularization aşırı öğrenmeyi azaltmaya yardımcı olur."
        
        return interpretation
    
    def _interpret_lasso_regression(self, lasso_r2: float, alpha: float, 
                                  selected_features: List, eliminated_features: List) -> str:
        """Lasso regresyon sonucunu yorumlar"""
        interpretation = f"Lasso regresyon (α={alpha}) "
        interpretation += f"{len(selected_features)} özellik seçti ve {len(eliminated_features)} özelliği eledi. "
        
        if len(selected_features) < len(selected_features) + len(eliminated_features):
            interpretation += "Model özellik seçimi yaparak karmaşıklığı azalttı. "
        
        interpretation += f"Model performansı: R² = {lasso_r2:.3f}. "
        
        if selected_features:
            interpretation += f"En önemli özellikler: {', '.join(selected_features[:3])}."
        
        return interpretation
    
    def predict(self, model_name: str, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Eğitilmiş model ile tahmin yapar
        
        Args:
            model_name: Model adı
            new_data: Tahmin yapılacak veri
            
        Returns:
            Tahmin sonuçları
        """
        try:
            if model_name not in self.models:
                return {'error': f'{model_name} modeli bulunamadı'}
            
            model = self.models[model_name]
            
            # Model tipine göre tahmin
            if model_name == 'polynomial_regression':
                poly_features = model['poly_features']
                actual_model = model['model']
                X_poly = poly_features.transform(new_data)
                predictions = actual_model.predict(X_poly)
            else:
                # Standardization gerekiyorsa uygula
                if model_name in self.scalers:
                    scaler = self.scalers[model_name]
                    new_data_scaled = scaler.transform(new_data)
                    predictions = model.predict(new_data_scaled)
                else:
                    predictions = model.predict(new_data)
            
            return {
                'model_name': model_name,
                'predictions': predictions.tolist(),
                'num_predictions': len(predictions)
            }
            
        except Exception as e:
            return {'error': f'Tahmin hatası: {str(e)}'}
    
    def get_regression_summary(self) -> Dict[str, Any]:
        """
        Gerçekleştirilen tüm regresyon analizlerinin özetini döndürür
        
        Returns:
            Regresyon analizi özetleri
        """
        if not self.results:
            return {'message': 'Henüz regresyon analizi gerçekleştirilmedi'}
        
        summary = {
            'total_models': len(self.results),
            'models_trained': list(self.results.keys()),
            'available_for_prediction': list(self.models.keys()),
            'model_details': self.results
        }
        
        # En iyi modeli bul (R² veya accuracy'ye göre)
        best_model = None
        best_score = -1
        
        for model_name, result in self.results.items():
            if 'error' not in result:
                if 'performance_metrics' in result:
                    if 'test_r2' in result['performance_metrics']:
                        score = result['performance_metrics']['test_r2']
                    elif 'test_accuracy' in result['performance_metrics']:
                        score = result['performance_metrics']['test_accuracy']
                    else:
                        continue
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_name
        
        if best_model:
            summary['best_model'] = {
                'name': best_model,
                'score': best_score,
                'metric': 'R²' if 'test_r2' in self.results[best_model]['performance_metrics'] else 'Accuracy'
            }
        
        return summary