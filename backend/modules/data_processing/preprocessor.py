"""
VeriVio Veri Ön İşleme Modülü
Veri dönüştürme, özellik mühendisliği ve hazırlama işlemleri
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
from sklearn.decomposition import PCA
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Veri ön işleme ve özellik mühendisliği sınıfı"""
    
    def __init__(self):
        self.encoders = {}
        self.feature_selectors = {}
        self.pca_transformer = None
        self.preprocessing_steps = []
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                   method: str = "auto",
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Kategorik değişkenleri encode et"""
        
        df_encoded = df.copy()
        
        # Kategorik sütunları belirle
        if columns is None:
            categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            categorical_cols = [col for col in columns if col in df_encoded.columns]
        
        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()
            
            # Encoding yöntemini belirle
            if method == "auto":
                if unique_count <= 2:
                    encoding_method = "label"
                elif unique_count <= 10:
                    encoding_method = "onehot"
                else:
                    encoding_method = "label"
            else:
                encoding_method = method
            
            # Encoding uygula
            if encoding_method == "label":
                df_encoded = self._apply_label_encoding(df_encoded, col)
            elif encoding_method == "onehot":
                df_encoded = self._apply_onehot_encoding(df_encoded, col)
            
            self.preprocessing_steps.append(f"{col} sütunu {encoding_method} encoding ile dönüştürüldü")
        
        logger.info(f"{len(categorical_cols)} kategorik sütun encode edildi")
        return df_encoded
    
    def _apply_label_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Label encoding uygula"""
        
        encoder = LabelEncoder()
        df_encoded = df.copy()
        
        # NaN değerleri geçici olarak doldur
        temp_value = "MISSING_VALUE_TEMP"
        df_encoded[column] = df_encoded[column].fillna(temp_value)
        
        # Encoding uygula
        df_encoded[column] = encoder.fit_transform(df_encoded[column].astype(str))
        
        # NaN değerleri geri koy
        mask = df[column].isna()
        df_encoded.loc[mask, column] = np.nan
        
        # Encoder'ı sakla
        self.encoders[column] = encoder
        
        return df_encoded
    
    def _apply_onehot_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """One-hot encoding uygula"""
        
        df_encoded = df.copy()
        
        # One-hot encoding
        dummies = pd.get_dummies(df_encoded[column], prefix=column, dummy_na=True)
        
        # Orijinal sütunu kaldır ve dummy sütunları ekle
        df_encoded = df_encoded.drop(columns=[column])
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        return df_encoded
    
    def create_datetime_features(self, df: pd.DataFrame, 
                                datetime_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Datetime sütunlarından özellikler çıkar"""
        
        df_features = df.copy()
        
        # Datetime sütunları belirle
        if datetime_columns is None:
            datetime_columns = df_features.select_dtypes(include=['datetime64']).columns.tolist()
            
            # String sütunlardan datetime olabilecekleri tespit et
            for col in df_features.select_dtypes(include=['object']).columns:
                try:
                    pd.to_datetime(df_features[col].dropna().head(100))
                    datetime_columns.append(col)
                except:
                    continue
        
        for col in datetime_columns:
            if col not in df_features.columns:
                continue
                
            # Datetime'a dönüştür
            try:
                df_features[col] = pd.to_datetime(df_features[col])
            except:
                logger.warning(f"{col} sütunu datetime'a dönüştürülemedi")
                continue
            
            # Özellikler çıkar
            df_features[f"{col}_year"] = df_features[col].dt.year
            df_features[f"{col}_month"] = df_features[col].dt.month
            df_features[f"{col}_day"] = df_features[col].dt.day
            df_features[f"{col}_dayofweek"] = df_features[col].dt.dayofweek
            df_features[f"{col}_quarter"] = df_features[col].dt.quarter
            df_features[f"{col}_is_weekend"] = df_features[col].dt.dayofweek.isin([5, 6]).astype(int)
            
            # Saat bilgisi varsa
            if df_features[col].dt.hour.nunique() > 1:
                df_features[f"{col}_hour"] = df_features[col].dt.hour
                df_features[f"{col}_is_business_hour"] = df_features[col].dt.hour.between(9, 17).astype(int)
            
            self.preprocessing_steps.append(f"{col} sütunundan datetime özellikleri çıkarıldı")
        
        logger.info(f"{len(datetime_columns)} datetime sütunundan özellikler çıkarıldı")
        return df_features
    
    def create_numerical_features(self, df: pd.DataFrame, 
                                 operations: List[str] = ["log", "sqrt", "square"]) -> pd.DataFrame:
        """Sayısal sütunlardan yeni özellikler oluştur"""
        
        df_features = df.copy()
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if df_features[col].min() <= 0 and "log" in operations:
                # Log için pozitif değerler gerekli
                operations_for_col = [op for op in operations if op != "log"]
            else:
                operations_for_col = operations
            
            for operation in operations_for_col:
                try:
                    if operation == "log":
                        df_features[f"{col}_log"] = np.log(df_features[col])
                    elif operation == "sqrt":
                        df_features[f"{col}_sqrt"] = np.sqrt(np.abs(df_features[col]))
                    elif operation == "square":
                        df_features[f"{col}_square"] = df_features[col] ** 2
                    elif operation == "reciprocal":
                        df_features[f"{col}_reciprocal"] = 1 / (df_features[col] + 1e-8)
                except:
                    logger.warning(f"{col} sütunu için {operation} işlemi uygulanamadı")
        
        logger.info(f"{len(numeric_cols)} sayısal sütun için özellik mühendisliği yapıldı")
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   max_interactions: int = 10) -> pd.DataFrame:
        """Sütunlar arası etkileşim özellikleri oluştur"""
        
        df_interactions = df.copy()
        numeric_cols = df_interactions.select_dtypes(include=[np.number]).columns.tolist()
        
        interaction_count = 0
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                # Çarpım etkileşimi
                df_interactions[f"{col1}_x_{col2}"] = df_interactions[col1] * df_interactions[col2]
                
                # Oran etkileşimi (sıfıra bölme kontrolü)
                df_interactions[f"{col1}_div_{col2}"] = df_interactions[col1] / (df_interactions[col2] + 1e-8)
                
                interaction_count += 2
                
                if interaction_count >= max_interactions:
                    break
        
        logger.info(f"{interaction_count} etkileşim özelliği oluşturuldu")
        return df_interactions
    
    def select_features(self, df: pd.DataFrame, target_column: str, 
                       method: str = "univariate", k: int = 10) -> pd.DataFrame:
        """Özellik seçimi yap"""
        
        if target_column not in df.columns:
            logger.warning(f"Hedef sütun {target_column} bulunamadı")
            return df
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Sadece sayısal sütunları kullan
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols]
        
        if len(X_numeric.columns) == 0:
            logger.warning("Özellik seçimi için sayısal sütun bulunamadı")
            return df
        
        # Hedef değişken tipine göre scoring fonksiyonu seç
        if y.dtype in ['object', 'category'] or y.nunique() < 20:
            # Sınıflandırma
            score_func = f_classif
        else:
            # Regresyon
            score_func = f_regression
        
        # Özellik seçimi
        k_best = min(k, len(X_numeric.columns))
        selector = SelectKBest(score_func=score_func, k=k_best)
        
        try:
            X_selected = selector.fit_transform(X_numeric, y)
            selected_features = X_numeric.columns[selector.get_support()].tolist()
            
            # Seçilen özellikleri ve hedef sütunu birleştir
            df_selected = pd.concat([
                pd.DataFrame(X_selected, columns=selected_features, index=df.index),
                y
            ], axis=1)
            
            # Kategorik sütunları da ekle
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                df_selected = pd.concat([df_selected, df[categorical_cols]], axis=1)
            
            self.feature_selectors[target_column] = selector
            self.preprocessing_steps.append(f"{len(selected_features)} özellik seçildi ({method} yöntemi)")
            
            logger.info(f"Özellik seçimi tamamlandı: {len(selected_features)} özellik seçildi")
            return df_selected
            
        except Exception as e:
            logger.error(f"Özellik seçimi hatası: {str(e)}")
            return df
    
    def apply_pca(self, df: pd.DataFrame, n_components: Union[int, float] = 0.95,
                  exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """PCA (Temel Bileşen Analizi) uygula"""
        
        # Hariç tutulacak sütunları belirle
        if exclude_columns is None:
            exclude_columns = []
        
        # Sayısal sütunları seç
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        pca_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        if len(pca_cols) < 2:
            logger.warning("PCA için en az 2 sayısal sütun gerekli")
            return df
        
        X = df[pca_cols]
        
        # PCA uygula
        self.pca_transformer = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca_transformer.fit_transform(X)
        
        # PCA sütunları oluştur
        pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
        
        # Hariç tutulan sütunları ekle
        excluded_data = df[exclude_columns] if exclude_columns else pd.DataFrame(index=df.index)
        categorical_data = df.select_dtypes(include=['object', 'category'])
        
        # Sonucu birleştir
        df_result = pd.concat([df_pca, excluded_data, categorical_data], axis=1)
        
        explained_variance = self.pca_transformer.explained_variance_ratio_.sum()
        self.preprocessing_steps.append(
            f"PCA uygulandı: {X_pca.shape[1]} bileşen, %{explained_variance*100:.1f} varyans açıklandı"
        )
        
        logger.info(f"PCA tamamlandı: {X_pca.shape[1]} bileşen oluşturuldu")
        return df_result
    
    def create_binned_features(self, df: pd.DataFrame, 
                              columns: Optional[List[str]] = None,
                              n_bins: int = 5) -> pd.DataFrame:
        """Sayısal değişkenleri kategorik gruplara ayır"""
        
        df_binned = df.copy()
        
        if columns is None:
            columns = df_binned.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df_binned.columns:
                continue
                
            try:
                # Quantile-based binning
                df_binned[f"{col}_binned"] = pd.qcut(
                    df_binned[col], 
                    q=n_bins, 
                    labels=[f"{col}_Q{i+1}" for i in range(n_bins)],
                    duplicates='drop'
                )
                
                self.preprocessing_steps.append(f"{col} sütunu {n_bins} gruba ayrıldı")
                
            except Exception as e:
                logger.warning(f"{col} sütunu gruplandırılamadı: {str(e)}")
        
        return df_binned
    
    def detect_and_handle_multicollinearity(self, df: pd.DataFrame, 
                                           threshold: float = 0.95) -> pd.DataFrame:
        """Çoklu doğrusal bağlantı tespit et ve işle"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return df
        
        # Korelasyon matrisi
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Üst üçgen matris
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Yüksek korelasyonlu sütunları bul
        high_corr_pairs = []
        for col in upper_triangle.columns:
            for idx in upper_triangle.index:
                if upper_triangle.loc[idx, col] > threshold:
                    high_corr_pairs.append((idx, col, upper_triangle.loc[idx, col]))
        
        # Kaldırılacak sütunları belirle
        columns_to_drop = set()
        for col1, col2, corr_val in high_corr_pairs:
            # Daha az varyansa sahip olanı kaldır
            if df[col1].var() < df[col2].var():
                columns_to_drop.add(col1)
            else:
                columns_to_drop.add(col2)
        
        # Sütunları kaldır
        df_cleaned = df.drop(columns=list(columns_to_drop))
        
        if columns_to_drop:
            self.preprocessing_steps.append(
                f"Çoklu doğrusal bağlantı nedeniyle {len(columns_to_drop)} sütun kaldırıldı"
            )
            logger.info(f"Çoklu doğrusal bağlantı temizlendi: {len(columns_to_drop)} sütun kaldırıldı")
        
        return df_cleaned
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """Ön işleme raporu"""
        
        return {
            "steps_applied": self.preprocessing_steps,
            "encoders_used": list(self.encoders.keys()),
            "feature_selectors": list(self.feature_selectors.keys()),
            "pca_applied": self.pca_transformer is not None,
            "pca_components": self.pca_transformer.n_components_ if self.pca_transformer else None,
            "total_steps": len(self.preprocessing_steps)
        }
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Yeni veri için aynı dönüşümleri uygula"""
        
        df_transformed = df.copy()
        
        # Label encoders uygula
        for col, encoder in self.encoders.items():
            if col in df_transformed.columns:
                try:
                    # Bilinmeyen kategoriler için -1 kullan
                    df_transformed[col] = df_transformed[col].map(
                        lambda x: encoder.transform([str(x)])[0] if str(x) in encoder.classes_ else -1
                    )
                except:
                    logger.warning(f"{col} sütunu için encoder uygulanamadı")
        
        # PCA uygula
        if self.pca_transformer:
            numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= self.pca_transformer.n_features_in_:
                try:
                    X_pca = self.pca_transformer.transform(df_transformed[numeric_cols])
                    pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
                    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df_transformed.index)
                    
                    # PCA dışındaki sütunları koru
                    non_numeric = df_transformed.select_dtypes(exclude=[np.number])
                    df_transformed = pd.concat([df_pca, non_numeric], axis=1)
                except:
                    logger.warning("PCA dönüşümü uygulanamadı")
        
        return df_transformed