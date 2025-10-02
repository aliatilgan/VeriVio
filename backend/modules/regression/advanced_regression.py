"""
Gelişmiş Regresyon Analizi Sınıfı

Bu modül daha karmaşık regresyon analizi türlerini gerçekleştirmek için kullanılır.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class AdvancedRegressionAnalyzer:
    """
    Gelişmiş regresyon analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        AdvancedRegressionAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek veri seti
        """
        self.data = data.copy()
        self.results = {}
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
    def random_forest_regression(self, target_column: str, feature_columns: List[str],
                                n_estimators: int = 100, max_depth: Optional[int] = None,
                                test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Random Forest regresyon analizi gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_columns: Özellik değişkenleri
            n_estimators: Ağaç sayısı
            max_depth: Maksimum derinlik
            test_size: Test seti oranı
            random_state: Rastgele durum
            
        Returns:
            Random Forest regresyon analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            if len(data_clean) < 20:
                return {'error': 'Random Forest için en az 20 gözlem gereklidir'}
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Modeli eğit
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
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
            
            # Feature importance
            feature_importance = list(zip(feature_columns, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Out-of-bag score
            oob_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                oob_score=True,
                n_jobs=-1
            )
            oob_model.fit(X_train, y_train)
            oob_score = oob_model.oob_score_
            
            result = {
                'model_type': 'Random Forest Regression',
                'target_column': target_column,
                'feature_columns': feature_columns,
                'hyperparameters': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth if max_depth else 'None'
                },
                'sample_size': len(data_clean),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_importance': [(feat, float(imp)) for feat, imp in feature_importance],
                'performance_metrics': {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std()),
                    'oob_score': float(oob_score)
                },
                'model_diagnostics': {
                    'overfitting_risk': train_r2 - test_r2 > 0.1,
                    'r2_difference': float(train_r2 - test_r2),
                    'model_quality': self._assess_model_quality(test_r2),
                    'feature_diversity': len([imp for _, imp in feature_importance if imp > 0.01])
                },
                'interpretation': self._interpret_random_forest(
                    test_r2, feature_importance[:3], train_r2 - test_r2 > 0.1
                )
            }
            
            # Modeli sakla
            self.models['random_forest'] = model
            self.results['random_forest'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Random Forest regresyon hatası: {str(e)}'}
    
    def gradient_boosting_regression(self, target_column: str, feature_columns: List[str],
                                   n_estimators: int = 100, learning_rate: float = 0.1,
                                   max_depth: int = 3, test_size: float = 0.2,
                                   random_state: int = 42) -> Dict[str, Any]:
        """
        Gradient Boosting regresyon analizi gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_columns: Özellik değişkenleri
            n_estimators: Ağaç sayısı
            learning_rate: Öğrenme oranı
            max_depth: Maksimum derinlik
            test_size: Test seti oranı
            random_state: Rastgele durum
            
        Returns:
            Gradient Boosting regresyon analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            if len(data_clean) < 20:
                return {'error': 'Gradient Boosting için en az 20 gözlem gereklidir'}
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Modeli eğit
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
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
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Feature importance
            feature_importance = list(zip(feature_columns, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Training deviance (loss over iterations)
            train_deviance = model.train_score_
            
            result = {
                'model_type': 'Gradient Boosting Regression',
                'target_column': target_column,
                'feature_columns': feature_columns,
                'hyperparameters': {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth
                },
                'sample_size': len(data_clean),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_importance': [(feat, float(imp)) for feat, imp in feature_importance],
                'performance_metrics': {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                },
                'training_info': {
                    'final_train_loss': float(train_deviance[-1]),
                    'loss_improvement': float(train_deviance[0] - train_deviance[-1]),
                    'convergence_achieved': len(train_deviance) < n_estimators
                },
                'model_diagnostics': {
                    'overfitting_risk': train_r2 - test_r2 > 0.15,
                    'r2_difference': float(train_r2 - test_r2),
                    'model_quality': self._assess_model_quality(test_r2)
                },
                'interpretation': self._interpret_gradient_boosting(
                    test_r2, feature_importance[:3], learning_rate, n_estimators
                )
            }
            
            # Modeli sakla
            self.models['gradient_boosting'] = model
            self.results['gradient_boosting'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Gradient Boosting regresyon hatası: {str(e)}'}
    
    def support_vector_regression(self, target_column: str, feature_columns: List[str],
                                 kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1,
                                 test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Support Vector Regression analizi gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_columns: Özellik değişkenleri
            kernel: Kernel tipi ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parametresi
            epsilon: Epsilon parametresi
            test_size: Test seti oranı
            random_state: Rastgele durum
            
        Returns:
            SVR analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            if len(data_clean) < 20:
                return {'error': 'SVR için en az 20 gözlem gereklidir'}
            
            # Veriyi standardize et
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            
            # Modeli eğit
            model = SVR(kernel=kernel, C=C, epsilon=epsilon)
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
            
            # Support vectors bilgisi
            n_support_vectors = len(model.support_)
            support_vector_ratio = n_support_vectors / len(X_train)
            
            result = {
                'model_type': 'Support Vector Regression',
                'target_column': target_column,
                'feature_columns': feature_columns,
                'hyperparameters': {
                    'kernel': kernel,
                    'C': C,
                    'epsilon': epsilon
                },
                'sample_size': len(data_clean),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'performance_metrics': {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                },
                'model_info': {
                    'n_support_vectors': int(n_support_vectors),
                    'support_vector_ratio': float(support_vector_ratio),
                    'kernel_used': kernel
                },
                'model_diagnostics': {
                    'overfitting_risk': train_r2 - test_r2 > 0.1,
                    'r2_difference': float(train_r2 - test_r2),
                    'model_quality': self._assess_model_quality(test_r2),
                    'complexity_level': 'High' if support_vector_ratio > 0.5 else 'Medium' if support_vector_ratio > 0.2 else 'Low'
                },
                'interpretation': self._interpret_svr(
                    test_r2, kernel, support_vector_ratio, C
                )
            }
            
            # Modeli ve scaler'ı sakla
            self.models['svr'] = model
            self.scalers['svr'] = scaler
            self.results['svr'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'SVR hatası: {str(e)}'}
    
    def neural_network_regression(self, target_column: str, feature_columns: List[str],
                                 hidden_layer_sizes: Tuple = (100,), activation: str = 'relu',
                                 learning_rate_init: float = 0.001, max_iter: int = 500,
                                 test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Neural Network regresyon analizi gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_columns: Özellik değişkenleri
            hidden_layer_sizes: Gizli katman boyutları
            activation: Aktivasyon fonksiyonu
            learning_rate_init: Başlangıç öğrenme oranı
            max_iter: Maksimum iterasyon sayısı
            test_size: Test seti oranı
            random_state: Rastgele durum
            
        Returns:
            Neural Network regresyon analizi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            if len(data_clean) < 50:
                return {'error': 'Neural Network için en az 50 gözlem gereklidir'}
            
            # Veriyi standardize et
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
            
            # Veriyi böl
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            
            # Modeli eğit
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
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
            cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')  # 3-fold due to computational cost
            
            result = {
                'model_type': 'Neural Network Regression',
                'target_column': target_column,
                'feature_columns': feature_columns,
                'hyperparameters': {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': activation,
                    'learning_rate_init': learning_rate_init,
                    'max_iter': max_iter
                },
                'sample_size': len(data_clean),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'performance_metrics': {
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                },
                'training_info': {
                    'n_iterations': int(model.n_iter_),
                    'converged': model.n_iter_ < max_iter,
                    'final_loss': float(model.loss_),
                    'n_layers': len(hidden_layer_sizes) + 2,  # +input +output
                    'total_parameters': sum(hidden_layer_sizes) + len(feature_columns) + 1
                },
                'model_diagnostics': {
                    'overfitting_risk': train_r2 - test_r2 > 0.15,
                    'r2_difference': float(train_r2 - test_r2),
                    'model_quality': self._assess_model_quality(test_r2),
                    'training_stability': 'Good' if model.n_iter_ < max_iter * 0.8 else 'Concerning'
                },
                'interpretation': self._interpret_neural_network(
                    test_r2, hidden_layer_sizes, model.n_iter_, max_iter
                )
            }
            
            # Modeli ve scaler'ı sakla
            self.models['neural_network'] = model
            self.scalers['neural_network'] = scaler
            self.results['neural_network'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Neural Network regresyon hatası: {str(e)}'}
    
    def automated_feature_selection(self, target_column: str, feature_columns: List[str],
                                   method: str = 'rfe', n_features: int = 5) -> Dict[str, Any]:
        """
        Otomatik özellik seçimi gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_columns: Özellik değişkenleri
            method: Seçim yöntemi ('rfe', 'univariate', 'pca')
            n_features: Seçilecek özellik sayısı
            
        Returns:
            Özellik seçimi sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            if len(data_clean) < 10:
                return {'error': 'Özellik seçimi için en az 10 gözlem gereklidir'}
            
            if method == 'rfe':
                # Recursive Feature Elimination
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                selector = RFE(estimator, n_features_to_select=n_features)
                selector.fit(X, y)
                
                selected_features = [feature_columns[i] for i in range(len(feature_columns)) 
                                   if selector.support_[i]]
                feature_rankings = list(zip(feature_columns, selector.ranking_))
                feature_rankings.sort(key=lambda x: x[1])
                
            elif method == 'univariate':
                # Univariate Feature Selection
                selector = SelectKBest(score_func=f_regression, k=n_features)
                selector.fit(X, y)
                
                selected_features = [feature_columns[i] for i in range(len(feature_columns)) 
                                   if selector.get_support()[i]]
                scores = selector.scores_
                feature_rankings = list(zip(feature_columns, scores))
                feature_rankings.sort(key=lambda x: x[1], reverse=True)
                
            elif method == 'pca':
                # Principal Component Analysis
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                pca = PCA(n_components=n_features)
                X_pca = pca.fit_transform(X_scaled)
                
                # PCA bileşenlerinin açıkladığı varyans
                explained_variance = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                # En önemli özellikleri bul (component loadings)
                components = pca.components_
                feature_importance_pca = np.abs(components).mean(axis=0)
                feature_rankings = list(zip(feature_columns, feature_importance_pca))
                feature_rankings.sort(key=lambda x: x[1], reverse=True)
                
                selected_features = [feat for feat, _ in feature_rankings[:n_features]]
                
                result = {
                    'method': method,
                    'n_features_selected': n_features,
                    'selected_features': selected_features,
                    'feature_rankings': [(feat, float(score)) for feat, score in feature_rankings],
                    'pca_info': {
                        'explained_variance_ratio': explained_variance.tolist(),
                        'cumulative_variance': cumulative_variance.tolist(),
                        'total_variance_explained': float(cumulative_variance[-1])
                    },
                    'interpretation': self._interpret_pca_selection(
                        n_features, cumulative_variance[-1], selected_features
                    )
                }
                
                # PCA transformer'ı sakla
                self.feature_selectors[f'{method}_selector'] = {
                    'pca': pca,
                    'scaler': scaler
                }
                
                return result
            
            else:
                return {'error': f'Bilinmeyen özellik seçim yöntemi: {method}'}
            
            # Seçilen özelliklerle model performansını test et
            X_selected = X[selected_features]
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42
            )
            
            # Basit model ile test
            test_model = RandomForestRegressor(n_estimators=50, random_state=42)
            test_model.fit(X_train, y_train)
            y_pred = test_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            
            # Tüm özelliklerle karşılaştırma
            full_model = RandomForestRegressor(n_estimators=50, random_state=42)
            X_train_full, X_test_full, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            full_model.fit(X_train_full, y_train)
            y_pred_full = full_model.predict(X_test_full)
            full_r2 = r2_score(y_test, y_pred_full)
            
            result = {
                'method': method,
                'n_features_original': len(feature_columns),
                'n_features_selected': len(selected_features),
                'selected_features': selected_features,
                'feature_rankings': [(feat, float(score)) for feat, score in feature_rankings],
                'performance_comparison': {
                    'selected_features_r2': float(test_r2),
                    'all_features_r2': float(full_r2),
                    'performance_difference': float(test_r2 - full_r2),
                    'dimensionality_reduction': (len(feature_columns) - len(selected_features)) / len(feature_columns)
                },
                'interpretation': self._interpret_feature_selection(
                    method, len(selected_features), len(feature_columns), test_r2, full_r2
                )
            }
            
            # Selector'ı sakla
            self.feature_selectors[f'{method}_selector'] = selector
            
            return result
            
        except Exception as e:
            return {'error': f'Özellik seçimi hatası: {str(e)}'}
    
    def hyperparameter_tuning(self, target_column: str, feature_columns: List[str],
                             model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Hiperparametre optimizasyonu gerçekleştirir
        
        Args:
            target_column: Hedef değişken
            feature_columns: Özellik değişkenleri
            model_type: Model tipi ('random_forest', 'gradient_boosting', 'svr')
            
        Returns:
            Hiperparametre optimizasyonu sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[feature_columns + [target_column]].dropna()
            X = data_clean[feature_columns]
            y = data_clean[target_column]
            
            if len(data_clean) < 50:
                return {'error': 'Hiperparametre optimizasyonu için en az 50 gözlem gereklidir'}
            
            # Model ve parametre grid'ini tanımla
            if model_type == 'random_forest':
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            elif model_type == 'svr':
                # SVR için veriyi standardize et
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns, index=X.index)
                
                model = SVR()
                param_grid = {
                    'kernel': ['linear', 'rbf'],
                    'C': [0.1, 1, 10],
                    'epsilon': [0.01, 0.1, 0.2]
                }
            else:
                return {'error': f'Desteklenmeyen model tipi: {model_type}'}
            
            # Grid Search
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
            )
            grid_search.fit(X, y)
            
            # En iyi modeli test et
            best_model = grid_search.best_estimator_
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            
            # Varsayılan parametrelerle karşılaştırma
            if model_type == 'random_forest':
                default_model = RandomForestRegressor(random_state=42)
            elif model_type == 'gradient_boosting':
                default_model = GradientBoostingRegressor(random_state=42)
            elif model_type == 'svr':
                default_model = SVR()
            
            default_model.fit(X_train, y_train)
            default_pred = default_model.predict(X_test)
            default_r2 = r2_score(y_test, default_pred)
            
            result = {
                'model_type': model_type,
                'target_column': target_column,
                'feature_columns': feature_columns,
                'best_parameters': grid_search.best_params_,
                'best_cv_score': float(grid_search.best_score_),
                'test_performance': {
                    'optimized_r2': float(test_r2),
                    'default_r2': float(default_r2),
                    'improvement': float(test_r2 - default_r2)
                },
                'grid_search_results': {
                    'n_parameter_combinations': len(grid_search.cv_results_['params']),
                    'best_rank': int(grid_search.best_index_ + 1),
                    'cv_std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
                },
                'interpretation': self._interpret_hyperparameter_tuning(
                    model_type, test_r2, default_r2, grid_search.best_params_
                )
            }
            
            # En iyi modeli sakla
            self.models[f'{model_type}_optimized'] = best_model
            if model_type == 'svr':
                self.scalers[f'{model_type}_optimized'] = scaler
            
            return result
            
        except Exception as e:
            return {'error': f'Hiperparametre optimizasyonu hatası: {str(e)}'}
    
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
    
    def _interpret_random_forest(self, r2: float, top_features: List, overfitting: bool) -> str:
        """Random Forest sonucunu yorumlar"""
        quality = self._assess_model_quality(r2)
        interpretation = f"Random Forest modeli {quality.lower()} performans gösteriyor (R² = {r2:.3f}). "
        
        if top_features:
            top_feature_names = [feat for feat, _ in top_features]
            interpretation += f"En önemli özellikler: {', '.join(top_feature_names)}. "
        
        if overfitting:
            interpretation += "Aşırı öğrenme riski tespit edildi, model karmaşıklığı azaltılabilir."
        else:
            interpretation += "Model genelleme yeteneği iyidir."
        
        return interpretation
    
    def _interpret_gradient_boosting(self, r2: float, top_features: List, 
                                   learning_rate: float, n_estimators: int) -> str:
        """Gradient Boosting sonucunu yorumlar"""
        quality = self._assess_model_quality(r2)
        interpretation = f"Gradient Boosting modeli {quality.lower()} performans gösteriyor (R² = {r2:.3f}). "
        
        if top_features:
            top_feature_names = [feat for feat, _ in top_features]
            interpretation += f"En önemli özellikler: {', '.join(top_feature_names)}. "
        
        if learning_rate < 0.1:
            interpretation += "Düşük öğrenme oranı daha stabil öğrenme sağlar. "
        
        interpretation += f"Model {n_estimators} ağaç kullanarak eğitildi."
        
        return interpretation
    
    def _interpret_svr(self, r2: float, kernel: str, support_ratio: float, C: float) -> str:
        """SVR sonucunu yorumlar"""
        quality = self._assess_model_quality(r2)
        interpretation = f"SVR modeli {quality.lower()} performans gösteriyor (R² = {r2:.3f}). "
        interpretation += f"Kullanılan kernel: {kernel}. "
        
        if support_ratio > 0.5:
            interpretation += "Yüksek support vector oranı karmaşık bir karar sınırı olduğunu gösterir. "
        elif support_ratio < 0.2:
            interpretation += "Düşük support vector oranı basit bir karar sınırı olduğunu gösterir. "
        
        if C > 1:
            interpretation += "Yüksek C değeri daha az regularization anlamına gelir."
        else:
            interpretation += "Düşük C değeri daha fazla regularization anlamına gelir."
        
        return interpretation
    
    def _interpret_neural_network(self, r2: float, hidden_layers: Tuple, 
                                 iterations: int, max_iter: int) -> str:
        """Neural Network sonucunu yorumlar"""
        quality = self._assess_model_quality(r2)
        interpretation = f"Neural Network modeli {quality.lower()} performans gösteriyor (R² = {r2:.3f}). "
        
        interpretation += f"Ağ yapısı: {len(hidden_layers)} gizli katman, "
        interpretation += f"toplam {sum(hidden_layers)} nöron. "
        
        if iterations >= max_iter * 0.9:
            interpretation += "Model tam olarak yakınsayamadı, daha fazla iterasyon gerekebilir."
        else:
            interpretation += f"Model {iterations} iterasyonda yakınsadı."
        
        return interpretation
    
    def _interpret_feature_selection(self, method: str, n_selected: int, n_total: int,
                                   selected_r2: float, full_r2: float) -> str:
        """Özellik seçimi sonucunu yorumlar"""
        reduction_ratio = (n_total - n_selected) / n_total
        performance_diff = selected_r2 - full_r2
        
        interpretation = f"{method.upper()} yöntemi ile {n_total} özellikten {n_selected} tanesi seçildi "
        interpretation += f"(%{reduction_ratio*100:.1f} azalma). "
        
        if performance_diff > -0.05:
            interpretation += "Özellik azaltma performansı önemli ölçüde etkilemedi. "
        elif performance_diff > -0.1:
            interpretation += "Özellik azaltma performansta hafif düşüşe neden oldu. "
        else:
            interpretation += "Özellik azaltma performansta önemli düşüşe neden oldu. "
        
        interpretation += "Bu, boyut azaltma ve model basitliği açısından faydalıdır."
        
        return interpretation
    
    def _interpret_pca_selection(self, n_components: int, variance_explained: float,
                               selected_features: List) -> str:
        """PCA özellik seçimi sonucunu yorumlar"""
        interpretation = f"PCA ile {n_components} bileşen seçildi, "
        interpretation += f"toplam varyansın %{variance_explained*100:.1f}'ini açıklıyor. "
        
        if variance_explained > 0.9:
            interpretation += "Çok yüksek varyans açıklaması, bilgi kaybı minimal. "
        elif variance_explained > 0.8:
            interpretation += "İyi varyans açıklaması, kabul edilebilir bilgi kaybı. "
        else:
            interpretation += "Orta düzeyde varyans açıklaması, önemli bilgi kaybı olabilir. "
        
        interpretation += f"En önemli orijinal özellikler: {', '.join(selected_features[:3])}."
        
        return interpretation
    
    def _interpret_hyperparameter_tuning(self, model_type: str, optimized_r2: float,
                                       default_r2: float, best_params: Dict) -> str:
        """Hiperparametre optimizasyonu sonucunu yorumlar"""
        improvement = optimized_r2 - default_r2
        
        interpretation = f"{model_type.replace('_', ' ').title()} modeli için hiperparametre optimizasyonu "
        
        if improvement > 0.05:
            interpretation += f"önemli iyileşme sağladı (iyileşme: {improvement:.3f}). "
        elif improvement > 0.01:
            interpretation += f"hafif iyileşme sağladı (iyileşme: {improvement:.3f}). "
        else:
            interpretation += "önemli iyileşme sağlamadı. "
        
        interpretation += f"En iyi parametreler: {best_params}. "
        interpretation += f"Final performans: R² = {optimized_r2:.3f}."
        
        return interpretation
    
    def get_advanced_summary(self) -> Dict[str, Any]:
        """
        Gerçekleştirilen tüm gelişmiş regresyon analizlerinin özetini döndürür
        
        Returns:
            Gelişmiş regresyon analizi özetleri
        """
        if not self.results:
            return {'message': 'Henüz gelişmiş regresyon analizi gerçekleştirilmedi'}
        
        summary = {
            'total_models': len(self.results),
            'models_trained': list(self.results.keys()),
            'available_for_prediction': list(self.models.keys()),
            'feature_selectors_available': list(self.feature_selectors.keys()),
            'model_details': self.results
        }
        
        # En iyi modeli bul
        best_model = None
        best_score = -1
        
        for model_name, result in self.results.items():
            if 'error' not in result and 'performance_metrics' in result:
                if 'test_r2' in result['performance_metrics']:
                    score = result['performance_metrics']['test_r2']
                    if score > best_score:
                        best_score = score
                        best_model = model_name
        
        if best_model:
            summary['best_model'] = {
                'name': best_model,
                'r2_score': best_score,
                'quality': self._assess_model_quality(best_score)
            }
        
        return summary