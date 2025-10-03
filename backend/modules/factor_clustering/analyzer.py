"""
VeriVio Faktör Analizi ve Kümeleme Modülü
Keşfedici faktör analizi, doğrulayıcı faktör analizi, K-Means, hiyerarşik kümeleme
Otomatik faktör sayısı belirleme ve küme optimizasyonu
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class FactorClusteringAnalyzer:
    """Faktör analizi ve kümeleme analizi sınıfı"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        self.models = {}
        self.scalers = {}
        
        # Analiz kriterleri
        self.factor_criteria = {
            'eigenvalue_threshold': 1.0,
            'variance_threshold': 0.6,
            'loading_threshold': 0.4
        }
        
        self.cluster_criteria = {
            'min_clusters': 2,
            'max_clusters': 10,
            'silhouette_threshold': 0.5
        }
    
    def _prepare_data_for_factor_analysis(self, columns: List[str], 
                                        standardize: bool = True) -> pd.DataFrame:
        """Faktör analizi için veriyi hazırla"""
        # Sadece sayısal sütunları al
        numeric_data = self.data[columns].select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 3:
            raise ValueError("Faktör analizi için en az 3 sayısal değişken gerekli")
        
        # Eksik değerleri temizle
        clean_data = numeric_data.dropna()
        
        if len(clean_data) < 50:
            raise ValueError("Faktör analizi için en az 50 gözlem gerekli")
        
        # Standardizasyon
        if standardize:
            scaler = StandardScaler()
            scaled_data = pd.DataFrame(
                scaler.fit_transform(clean_data),
                columns=clean_data.columns,
                index=clean_data.index
            )
            self.scalers['factor_analysis'] = scaler
            return scaled_data
        
        return clean_data
    
    def _calculate_kmo_bartlett(self, data: pd.DataFrame) -> Dict[str, Any]:
        """KMO ve Bartlett testlerini hesapla"""
        try:
            # Korelasyon matrisi
            corr_matrix = data.corr()
            
            # Bartlett's Test of Sphericity
            n = len(data)
            p = len(data.columns)
            
            # Determinant hesapla
            det_corr = np.linalg.det(corr_matrix)
            
            if det_corr <= 0:
                bartlett_stat = np.nan
                bartlett_p = np.nan
            else:
                bartlett_stat = -(n - 1 - (2 * p + 5) / 6) * np.log(det_corr)
                df = p * (p - 1) / 2
                bartlett_p = 1 - stats.chi2.cdf(bartlett_stat, df)
            
            # KMO (Kaiser-Meyer-Olkin) hesaplama
            # Basitleştirilmiş KMO hesaplaması
            corr_inv = np.linalg.pinv(corr_matrix)
            partial_corr = np.zeros_like(corr_matrix)
            
            for i in range(p):
                for j in range(p):
                    if i != j:
                        partial_corr.iloc[i, j] = -corr_inv[i, j] / np.sqrt(corr_inv[i, i] * corr_inv[j, j])
            
            # KMO hesaplama
            sum_sq_corr = np.sum(corr_matrix.values**2) - p  # Diagonal hariç
            sum_sq_partial = np.sum(partial_corr.values**2)
            
            kmo = sum_sq_corr / (sum_sq_corr + sum_sq_partial)
            
            return {
                'kmo_value': float(kmo),
                'kmo_interpretation': self._interpret_kmo(kmo),
                'bartlett_statistic': float(bartlett_stat) if not np.isnan(bartlett_stat) else None,
                'bartlett_p_value': float(bartlett_p) if not np.isnan(bartlett_p) else None,
                'bartlett_significant': bartlett_p < 0.05 if not np.isnan(bartlett_p) else None,
                'factor_analysis_suitable': kmo > 0.6 and (bartlett_p < 0.05 if not np.isnan(bartlett_p) else False)
            }
            
        except Exception as e:
            logger.warning(f"KMO/Bartlett hesaplama hatası: {str(e)}")
            return {
                'kmo_value': None,
                'kmo_interpretation': 'Hesaplanamadı',
                'bartlett_statistic': None,
                'bartlett_p_value': None,
                'bartlett_significant': None,
                'factor_analysis_suitable': False,
                'error': str(e)
            }
    
    def _interpret_kmo(self, kmo: float) -> str:
        """KMO değerini yorumla"""
        if kmo >= 0.9:
            return "Mükemmel"
        elif kmo >= 0.8:
            return "Çok iyi"
        elif kmo >= 0.7:
            return "İyi"
        elif kmo >= 0.6:
            return "Orta"
        elif kmo >= 0.5:
            return "Zayıf"
        else:
            return "Kabul edilemez"
    
    def _determine_optimal_factors(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimal faktör sayısını belirle"""
        try:
            # PCA ile eigenvalue analizi
            pca = PCA()
            pca.fit(data)
            
            eigenvalues = pca.explained_variance_
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # Kaiser kriteri (eigenvalue > 1)
            kaiser_factors = np.sum(eigenvalues > self.factor_criteria['eigenvalue_threshold'])
            
            # Scree plot kriteri (elbow method)
            scree_factors = self._find_elbow_point(eigenvalues)
            
            # Varyans kriteri (%60 varyans açıklama)
            variance_factors = np.argmax(cumulative_variance >= self.factor_criteria['variance_threshold']) + 1
            
            # Paralel analiz (basitleştirilmiş)
            parallel_factors = self._parallel_analysis(data, eigenvalues)
            
            # Önerilen faktör sayısı (çoğunluk kararı)
            factor_suggestions = [kaiser_factors, scree_factors, variance_factors, parallel_factors]
            factor_suggestions = [f for f in factor_suggestions if f > 0]
            
            if factor_suggestions:
                recommended_factors = int(np.median(factor_suggestions))
            else:
                recommended_factors = min(3, len(data.columns) // 2)
            
            return {
                'eigenvalues': eigenvalues.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'kaiser_criterion': kaiser_factors,
                'scree_criterion': scree_factors,
                'variance_criterion': variance_factors,
                'parallel_analysis': parallel_factors,
                'recommended_factors': recommended_factors,
                'factor_suggestions': factor_suggestions
            }
            
        except Exception as e:
            logger.error(f"Optimal faktör sayısı belirleme hatası: {str(e)}")
            return {
                'recommended_factors': min(3, len(data.columns) // 2),
                'error': str(e)
            }
    
    def _find_elbow_point(self, eigenvalues: np.ndarray) -> int:
        """Scree plot'ta elbow noktasını bul"""
        try:
            if len(eigenvalues) < 3:
                return 1
            
            # İkinci türev yöntemi
            diffs = np.diff(eigenvalues)
            second_diffs = np.diff(diffs)
            
            # En büyük ikinci türev noktası
            elbow_idx = np.argmax(np.abs(second_diffs)) + 2
            
            return min(elbow_idx, len(eigenvalues))
            
        except:
            return min(3, len(eigenvalues))
    
    def _parallel_analysis(self, data: pd.DataFrame, real_eigenvalues: np.ndarray) -> int:
        """Paralel analiz ile faktör sayısı belirleme"""
        try:
            n_vars = data.shape[1]
            n_obs = data.shape[0]
            n_iterations = 100
            
            # Rastgele veri ile eigenvalue'ları hesapla
            random_eigenvalues = []
            
            for _ in range(n_iterations):
                random_data = np.random.normal(size=(n_obs, n_vars))
                random_corr = np.corrcoef(random_data.T)
                random_eig = np.linalg.eigvals(random_corr)
                random_eig = np.sort(random_eig)[::-1]
                random_eigenvalues.append(random_eig)
            
            # Ortalama rastgele eigenvalue'lar
            mean_random_eig = np.mean(random_eigenvalues, axis=0)
            
            # Gerçek eigenvalue'ların rastgele eigenvalue'lardan büyük olduğu faktör sayısı
            parallel_factors = np.sum(real_eigenvalues > mean_random_eig)
            
            return max(1, parallel_factors)
            
        except:
            return min(3, len(data.columns) // 2)
    
    def exploratory_factor_analysis(self, columns: List[str], 
                                  n_factors: Optional[int] = None,
                                  rotation: str = 'varimax') -> Dict[str, Any]:
        """Keşfedici faktör analizi"""
        try:
            # Veriyi hazırla
            data = self._prepare_data_for_factor_analysis(columns)
            
            # KMO ve Bartlett testleri
            adequacy_tests = self._calculate_kmo_bartlett(data)
            
            if not adequacy_tests.get('factor_analysis_suitable', False):
                logger.warning("Veri faktör analizi için uygun görünmüyor")
            
            # Optimal faktör sayısını belirle
            factor_analysis = self._determine_optimal_factors(data)
            
            if n_factors is None:
                n_factors = factor_analysis['recommended_factors']
            
            n_factors = max(1, min(n_factors, len(columns) - 1))
            
            # Faktör analizi uygula
            fa_model = FactorAnalysis(n_components=n_factors, random_state=42)
            fa_model.fit(data)
            
            # Faktör yükleri
            loadings = fa_model.components_.T
            loadings_df = pd.DataFrame(
                loadings,
                index=data.columns,
                columns=[f'Factor_{i+1}' for i in range(n_factors)]
            )
            
            # Communalities (ortak varyanslar)
            communalities = np.sum(loadings**2, axis=1)
            
            # Faktör skorları
            factor_scores = fa_model.transform(data)
            factor_scores_df = pd.DataFrame(
                factor_scores,
                index=data.index,
                columns=[f'Factor_{i+1}' for i in range(n_factors)]
            )
            
            # Açıklanan varyans
            explained_variance = np.var(factor_scores, axis=0)
            total_variance = np.sum(explained_variance)
            explained_variance_ratio = explained_variance / len(data.columns)
            
            # Faktör yorumlama
            factor_interpretation = self._interpret_factors(loadings_df)
            
            # Model uyum iyiliği
            # Yeniden yapılandırılmış korelasyon matrisi
            reconstructed_corr = np.dot(loadings, loadings.T)
            original_corr = data.corr().values
            
            # Residual korelasyonlar
            residual_corr = original_corr - reconstructed_corr
            rmsr = np.sqrt(np.mean(residual_corr[np.triu_indices_from(residual_corr, k=1)]**2))
            
            result = {
                'model_type': 'Keşfedici Faktör Analizi',
                'variables': columns,
                'n_factors': n_factors,
                'rotation_method': rotation,
                'sample_size': len(data),
                'adequacy_tests': adequacy_tests,
                'factor_determination': factor_analysis,
                'factor_loadings': loadings_df.round(3).to_dict(),
                'communalities': dict(zip(data.columns, communalities.round(3))),
                'factor_scores_sample': factor_scores_df.head(10).round(3).to_dict(),
                'explained_variance': {
                    'factor_variances': explained_variance.tolist(),
                    'explained_variance_ratio': explained_variance_ratio.tolist(),
                    'cumulative_variance_ratio': np.cumsum(explained_variance_ratio).tolist(),
                    'total_variance_explained': float(total_variance / len(data.columns))
                },
                'model_fit': {
                    'rmsr': float(rmsr),
                    'rmsr_interpretation': 'İyi' if rmsr < 0.08 else 'Orta' if rmsr < 0.1 else 'Zayıf'
                },
                'factor_interpretation': factor_interpretation,
                'interpretation': self._generate_efa_interpretation(
                    n_factors, total_variance / len(data.columns), adequacy_tests, factor_interpretation
                )
            }
            
            # Modeli kaydet
            self.models['exploratory_factor_analysis'] = fa_model
            self.results['exploratory_factor_analysis'] = result
            
            logger.info(f"Keşfedici faktör analizi tamamlandı: {n_factors} faktör")
            return result
            
        except Exception as e:
            logger.error(f"Keşfedici faktör analizi hatası: {str(e)}")
            return {'error': f'Keşfedici faktör analizi hatası: {str(e)}'}
    
    def _interpret_factors(self, loadings_df: pd.DataFrame) -> Dict[str, Any]:
        """Faktörleri yorumla"""
        interpretation = {}
        
        for factor in loadings_df.columns:
            # Yüksek yüklü değişkenler
            high_loadings = loadings_df[factor][
                abs(loadings_df[factor]) >= self.factor_criteria['loading_threshold']
            ].sort_values(key=abs, ascending=False)
            
            interpretation[factor] = {
                'high_loading_variables': high_loadings.to_dict(),
                'dominant_variables': high_loadings.head(3).index.tolist(),
                'factor_strength': 'Güçlü' if len(high_loadings) >= 3 else 'Orta' if len(high_loadings) >= 2 else 'Zayıf',
                'suggested_name': self._suggest_factor_name(high_loadings.index.tolist())
            }
        
        return interpretation
    
    def _suggest_factor_name(self, variables: List[str]) -> str:
        """Faktör için isim öner"""
        if not variables:
            return "Belirsiz Faktör"
        
        # Basit isim önerisi (değişken isimlerinden ortak kelimeler)
        common_words = []
        for var in variables[:3]:  # İlk 3 değişken
            words = var.lower().split('_')
            common_words.extend(words)
        
        if common_words:
            # En sık geçen kelime
            word_counts = {}
            for word in common_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            most_common = max(word_counts, key=word_counts.get)
            return f"{most_common.title()} Faktörü"
        
        return f"Faktör ({', '.join(variables[:2])}...)"
    
    def principal_component_analysis(self, columns: List[str], 
                                   n_components: Optional[int] = None) -> Dict[str, Any]:
        """Temel bileşenler analizi"""
        try:
            # Veriyi hazırla
            data = self._prepare_data_for_factor_analysis(columns)
            
            # PCA uygula
            if n_components is None:
                pca = PCA()
            else:
                n_components = min(n_components, len(columns))
                pca = PCA(n_components=n_components)
            
            pca_scores = pca.fit_transform(data)
            
            # Bileşen yükleri
            components_df = pd.DataFrame(
                pca.components_.T,
                index=data.columns,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)]
            )
            
            # PCA skorları
            scores_df = pd.DataFrame(
                pca_scores,
                index=data.index,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)]
            )
            
            # Açıklanan varyans
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Bileşen yorumlama
            component_interpretation = {}
            for i, pc in enumerate(components_df.columns):
                loadings = components_df[pc]
                high_loadings = loadings[abs(loadings) >= 0.4].sort_values(key=abs, ascending=False)
                
                component_interpretation[pc] = {
                    'explained_variance': float(explained_variance_ratio[i]),
                    'cumulative_variance': float(cumulative_variance[i]),
                    'high_loading_variables': high_loadings.to_dict(),
                    'dominant_variables': high_loadings.head(3).index.tolist()
                }
            
            result = {
                'model_type': 'Temel Bileşenler Analizi',
                'variables': columns,
                'n_components': pca.n_components_,
                'sample_size': len(data),
                'eigenvalues': pca.explained_variance_.tolist(),
                'explained_variance_ratio': explained_variance_ratio.tolist(),
                'cumulative_variance_ratio': cumulative_variance.tolist(),
                'component_loadings': components_df.round(3).to_dict(),
                'component_scores_sample': scores_df.head(10).round(3).to_dict(),
                'component_interpretation': component_interpretation,
                'total_variance_explained': float(cumulative_variance[-1]),
                'interpretation': self._generate_pca_interpretation(
                    pca.n_components_, cumulative_variance[-1], component_interpretation
                )
            }
            
            # Modeli kaydet
            self.models['principal_component_analysis'] = pca
            self.results['principal_component_analysis'] = result
            
            logger.info(f"Temel bileşenler analizi tamamlandı: {pca.n_components_} bileşen")
            return result
            
        except Exception as e:
            logger.error(f"Temel bileşenler analizi hatası: {str(e)}")
            return {'error': f'Temel bileşenler analizi hatası: {str(e)}'}
    
    def _prepare_data_for_clustering(self, columns: List[str], 
                                   standardize: bool = True) -> pd.DataFrame:
        """Kümeleme için veriyi hazırla"""
        # Sadece sayısal sütunları al
        numeric_data = self.data[columns].select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            raise ValueError("Kümeleme için en az 2 sayısal değişken gerekli")
        
        # Eksik değerleri temizle
        clean_data = numeric_data.dropna()
        
        if len(clean_data) < 10:
            raise ValueError("Kümeleme için en az 10 gözlem gerekli")
        
        # Standardizasyon
        if standardize:
            scaler = StandardScaler()
            scaled_data = pd.DataFrame(
                scaler.fit_transform(clean_data),
                columns=clean_data.columns,
                index=clean_data.index
            )
            self.scalers['clustering'] = scaler
            return scaled_data
        
        return clean_data
    
    def _determine_optimal_clusters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimal küme sayısını belirle"""
        try:
            max_clusters = min(self.cluster_criteria['max_clusters'], len(data) // 2)
            cluster_range = range(self.cluster_criteria['min_clusters'], max_clusters + 1)
            
            # Elbow method (WCSS)
            wcss = []
            silhouette_scores = []
            calinski_scores = []
            davies_bouldin_scores = []
            
            for k in cluster_range:
                # K-Means
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                
                # WCSS
                wcss.append(kmeans.inertia_)
                
                # Silhouette Score
                if k > 1:
                    sil_score = silhouette_score(data, labels)
                    silhouette_scores.append(sil_score)
                    
                    # Calinski-Harabasz Index
                    cal_score = calinski_harabasz_score(data, labels)
                    calinski_scores.append(cal_score)
                    
                    # Davies-Bouldin Index
                    db_score = davies_bouldin_score(data, labels)
                    davies_bouldin_scores.append(db_score)
            
            # Optimal küme sayısı belirleme
            # Elbow method
            elbow_k = self._find_elbow_point(np.array(wcss)) + self.cluster_criteria['min_clusters']
            
            # Silhouette method
            if silhouette_scores:
                silhouette_k = cluster_range[1:][np.argmax(silhouette_scores)]
            else:
                silhouette_k = 3
            
            # Calinski-Harabasz method
            if calinski_scores:
                calinski_k = cluster_range[1:][np.argmax(calinski_scores)]
            else:
                calinski_k = 3
            
            # Davies-Bouldin method (minimum is better)
            if davies_bouldin_scores:
                davies_bouldin_k = cluster_range[1:][np.argmin(davies_bouldin_scores)]
            else:
                davies_bouldin_k = 3
            
            # Önerilen küme sayısı (çoğunluk kararı)
            suggestions = [elbow_k, silhouette_k, calinski_k, davies_bouldin_k]
            recommended_k = int(np.median(suggestions))
            
            return {
                'cluster_range': list(cluster_range),
                'wcss_values': wcss,
                'silhouette_scores': silhouette_scores,
                'calinski_harabasz_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores,
                'elbow_method': elbow_k,
                'silhouette_method': silhouette_k,
                'calinski_method': calinski_k,
                'davies_bouldin_method': davies_bouldin_k,
                'recommended_clusters': recommended_k,
                'all_suggestions': suggestions
            }
            
        except Exception as e:
            logger.error(f"Optimal küme sayısı belirleme hatası: {str(e)}")
            return {
                'recommended_clusters': 3,
                'error': str(e)
            }
    
    def kmeans_clustering(self, columns: List[str], 
                         n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """K-Means kümeleme analizi"""
        try:
            # Veriyi hazırla
            data = self._prepare_data_for_clustering(columns)
            
            # Optimal küme sayısını belirle
            cluster_analysis = self._determine_optimal_clusters(data)
            
            if n_clusters is None:
                n_clusters = cluster_analysis['recommended_clusters']
            
            n_clusters = max(2, min(n_clusters, len(data) // 2))
            
            # K-Means uygula
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Küme merkezleri
            cluster_centers = pd.DataFrame(
                kmeans.cluster_centers_,
                columns=data.columns,
                index=[f'Cluster_{i}' for i in range(n_clusters)]
            )
            
            # Küme istatistikleri
            cluster_stats = self._calculate_cluster_statistics(data, cluster_labels)
            
            # Model performansı
            performance = self._evaluate_clustering_performance(data, cluster_labels)
            
            # Küme profilleri
            cluster_profiles = self._create_cluster_profiles(data, cluster_labels)
            
            result = {
                'model_type': 'K-Means Kümeleme',
                'variables': columns,
                'n_clusters': n_clusters,
                'sample_size': len(data),
                'cluster_optimization': cluster_analysis,
                'cluster_centers': cluster_centers.round(3).to_dict(),
                'cluster_labels': cluster_labels.tolist(),
                'cluster_statistics': cluster_stats,
                'cluster_profiles': cluster_profiles,
                'performance_metrics': performance,
                'model_parameters': {
                    'inertia': float(kmeans.inertia_),
                    'n_iterations': int(kmeans.n_iter_)
                },
                'interpretation': self._generate_kmeans_interpretation(
                    n_clusters, performance, cluster_profiles
                )
            }
            
            # Modeli kaydet
            self.models['kmeans_clustering'] = kmeans
            self.results['kmeans_clustering'] = result
            
            logger.info(f"K-Means kümeleme tamamlandı: {n_clusters} küme")
            return result
            
        except Exception as e:
            logger.error(f"K-Means kümeleme hatası: {str(e)}")
            return {'error': f'K-Means kümeleme hatası: {str(e)}'}
    
    def hierarchical_clustering(self, columns: List[str], 
                              linkage_method: str = 'ward',
                              n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Hiyerarşik kümeleme analizi"""
        try:
            # Veriyi hazırla
            data = self._prepare_data_for_clustering(columns)
            
            # Optimal küme sayısını belirle
            if n_clusters is None:
                cluster_analysis = self._determine_optimal_clusters(data)
                n_clusters = cluster_analysis['recommended_clusters']
            
            n_clusters = max(2, min(n_clusters, len(data) // 2))
            
            # Hiyerarşik kümeleme uygula
            linkage_matrix = linkage(data, method=linkage_method)
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
            
            # Agglomerative clustering ile de kontrol et
            agg_clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage=linkage_method
            )
            agg_labels = agg_clustering.fit_predict(data)
            
            # Küme istatistikleri
            cluster_stats = self._calculate_cluster_statistics(data, cluster_labels)
            
            # Model performansı
            performance = self._evaluate_clustering_performance(data, cluster_labels)
            
            # Küme profilleri
            cluster_profiles = self._create_cluster_profiles(data, cluster_labels)
            
            # Dendrogram bilgisi
            dendrogram_info = {
                'linkage_matrix_shape': linkage_matrix.shape,
                'linkage_method': linkage_method,
                'cophenetic_correlation': self._calculate_cophenetic_correlation(data, linkage_matrix)
            }
            
            result = {
                'model_type': 'Hiyerarşik Kümeleme',
                'variables': columns,
                'n_clusters': n_clusters,
                'linkage_method': linkage_method,
                'sample_size': len(data),
                'cluster_labels': cluster_labels.tolist(),
                'cluster_statistics': cluster_stats,
                'cluster_profiles': cluster_profiles,
                'performance_metrics': performance,
                'dendrogram_info': dendrogram_info,
                'interpretation': self._generate_hierarchical_interpretation(
                    n_clusters, linkage_method, performance, cluster_profiles
                )
            }
            
            # Modeli kaydet
            self.models['hierarchical_clustering'] = {
                'linkage_matrix': linkage_matrix,
                'agg_clustering': agg_clustering
            }
            self.results['hierarchical_clustering'] = result
            
            logger.info(f"Hiyerarşik kümeleme tamamlandı: {n_clusters} küme")
            return result
            
        except Exception as e:
            logger.error(f"Hiyerarşik kümeleme hatası: {str(e)}")
            return {'error': f'Hiyerarşik kümeleme hatası: {str(e)}'}
    
    def _calculate_cophenetic_correlation(self, data: pd.DataFrame, 
                                        linkage_matrix: np.ndarray) -> float:
        """Cophenetic korelasyon hesapla"""
        try:
            from scipy.cluster.hierarchy import cophenet
            from scipy.spatial.distance import pdist
            
            distances = pdist(data)
            cophenetic_dists, _ = cophenet(linkage_matrix, distances)
            correlation = np.corrcoef(distances, cophenetic_dists)[0, 1]
            
            return float(correlation)
            
        except:
            return 0.0
    
    def _calculate_cluster_statistics(self, data: pd.DataFrame, 
                                    cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Küme istatistiklerini hesapla"""
        stats = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_data = data[cluster_labels == cluster_id]
            
            stats[f'Cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
                'mean_values': cluster_data.mean().to_dict(),
                'std_values': cluster_data.std().to_dict(),
                'min_values': cluster_data.min().to_dict(),
                'max_values': cluster_data.max().to_dict()
            }
        
        return stats
    
    def _evaluate_clustering_performance(self, data: pd.DataFrame, 
                                       cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Kümeleme performansını değerlendir"""
        try:
            performance = {}
            
            # Silhouette Score
            if len(np.unique(cluster_labels)) > 1:
                sil_score = silhouette_score(data, cluster_labels)
                performance['silhouette_score'] = float(sil_score)
                performance['silhouette_interpretation'] = self._interpret_silhouette(sil_score)
                
                # Calinski-Harabasz Index
                cal_score = calinski_harabasz_score(data, cluster_labels)
                performance['calinski_harabasz_score'] = float(cal_score)
                
                # Davies-Bouldin Index
                db_score = davies_bouldin_score(data, cluster_labels)
                performance['davies_bouldin_score'] = float(db_score)
                performance['davies_bouldin_interpretation'] = 'İyi' if db_score < 1 else 'Orta' if db_score < 2 else 'Zayıf'
            
            # Küme içi ve küme arası mesafeler
            intra_cluster_distances = []
            inter_cluster_distances = []
            
            for cluster_id in np.unique(cluster_labels):
                cluster_data = data[cluster_labels == cluster_id]
                if len(cluster_data) > 1:
                    # Küme içi mesafe
                    intra_dist = pdist(cluster_data).mean()
                    intra_cluster_distances.append(intra_dist)
            
            performance['mean_intra_cluster_distance'] = float(np.mean(intra_cluster_distances)) if intra_cluster_distances else 0
            
            return performance
            
        except Exception as e:
            logger.warning(f"Performans değerlendirme hatası: {str(e)}")
            return {'error': str(e)}
    
    def _interpret_silhouette(self, score: float) -> str:
        """Silhouette skorunu yorumla"""
        if score >= 0.7:
            return "Mükemmel"
        elif score >= 0.5:
            return "İyi"
        elif score >= 0.25:
            return "Orta"
        else:
            return "Zayıf"
    
    def _create_cluster_profiles(self, data: pd.DataFrame, 
                               cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Küme profillerini oluştur"""
        profiles = {}
        
        # Genel istatistikler
        overall_mean = data.mean()
        
        for cluster_id in np.unique(cluster_labels):
            cluster_data = data[cluster_labels == cluster_id]
            cluster_mean = cluster_data.mean()
            
            # Küme karakteristikleri
            characteristics = {}
            for column in data.columns:
                diff = cluster_mean[column] - overall_mean[column]
                std_diff = diff / data[column].std() if data[column].std() > 0 else 0
                
                if abs(std_diff) > 0.5:
                    if std_diff > 0:
                        characteristics[column] = f"Yüksek ({diff:.2f})"
                    else:
                        characteristics[column] = f"Düşük ({diff:.2f})"
                else:
                    characteristics[column] = "Ortalama"
            
            profiles[f'Cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'characteristics': characteristics,
                'dominant_features': self._find_dominant_features(characteristics),
                'cluster_description': self._describe_cluster(characteristics)
            }
        
        return profiles
    
    def _find_dominant_features(self, characteristics: Dict[str, str]) -> List[str]:
        """Kümenin baskın özelliklerini bul"""
        dominant = []
        for feature, desc in characteristics.items():
            if 'Yüksek' in desc or 'Düşük' in desc:
                dominant.append(f"{feature}: {desc}")
        
        return dominant[:3]  # En fazla 3 özellik
    
    def _describe_cluster(self, characteristics: Dict[str, str]) -> str:
        """Kümeyi tanımla"""
        high_features = [k for k, v in characteristics.items() if 'Yüksek' in v]
        low_features = [k for k, v in characteristics.items() if 'Düşük' in v]
        
        description = "Bu küme "
        
        if high_features:
            description += f"yüksek {', '.join(high_features[:2])} "
        
        if low_features:
            if high_features:
                description += "ve "
            description += f"düşük {', '.join(low_features[:2])} "
        
        description += "değerleri ile karakterize edilir."
        
        return description
    
    def _generate_efa_interpretation(self, n_factors: int, total_variance: float,
                                   adequacy_tests: Dict, factor_interpretation: Dict) -> str:
        """Keşfedici faktör analizi yorumu"""
        interpretation = f"Keşfedici Faktör Analizi Sonuçları:\n\n"
        interpretation += f"• {n_factors} faktör belirlendi\n"
        interpretation += f"• Toplam varyansın %{total_variance*100:.1f}'i açıklandı\n"
        
        if adequacy_tests.get('kmo_value'):
            interpretation += f"• KMO değeri: {adequacy_tests['kmo_value']:.3f} ({adequacy_tests['kmo_interpretation']})\n"
        
        if adequacy_tests.get('bartlett_significant'):
            interpretation += "• Bartlett testi anlamlı (faktör analizi uygun)\n"
        
        interpretation += "\nFaktör Yorumları:\n"
        for factor, info in factor_interpretation.items():
            interpretation += f"• {factor}: {info['suggested_name']} ({info['factor_strength']})\n"
        
        return interpretation
    
    def _generate_pca_interpretation(self, n_components: int, total_variance: float,
                                   component_interpretation: Dict) -> str:
        """Temel bileşenler analizi yorumu"""
        interpretation = f"Temel Bileşenler Analizi Sonuçları:\n\n"
        interpretation += f"• {n_components} bileşen analiz edildi\n"
        interpretation += f"• Toplam varyansın %{total_variance*100:.1f}'i açıklandı\n\n"
        
        interpretation += "Bileşen Yorumları:\n"
        for pc, info in component_interpretation.items():
            interpretation += f"• {pc}: Varyansın %{info['explained_variance']*100:.1f}'ini açıklıyor\n"
            if info['dominant_variables']:
                interpretation += f"  Baskın değişkenler: {', '.join(info['dominant_variables'][:3])}\n"
        
        return interpretation
    
    def _generate_kmeans_interpretation(self, n_clusters: int, performance: Dict,
                                      cluster_profiles: Dict) -> str:
        """K-Means kümeleme yorumu"""
        interpretation = f"K-Means Kümeleme Analizi Sonuçları:\n\n"
        interpretation += f"• {n_clusters} küme oluşturuldu\n"
        
        if 'silhouette_score' in performance:
            interpretation += f"• Silhouette skoru: {performance['silhouette_score']:.3f} ({performance['silhouette_interpretation']})\n"
        
        interpretation += "\nKüme Profilleri:\n"
        for cluster, profile in cluster_profiles.items():
            interpretation += f"• {cluster}: {profile['size']} gözlem\n"
            interpretation += f"  {profile['cluster_description']}\n"
        
        return interpretation
    
    def _generate_hierarchical_interpretation(self, n_clusters: int, linkage_method: str,
                                            performance: Dict, cluster_profiles: Dict) -> str:
        """Hiyerarşik kümeleme yorumu"""
        interpretation = f"Hiyerarşik Kümeleme Analizi Sonuçları:\n\n"
        interpretation += f"• {n_clusters} küme oluşturuldu ({linkage_method} bağlantı yöntemi)\n"
        
        if 'silhouette_score' in performance:
            interpretation += f"• Silhouette skoru: {performance['silhouette_score']:.3f} ({performance['silhouette_interpretation']})\n"
        
        interpretation += "\nKüme Profilleri:\n"
        for cluster, profile in cluster_profiles.items():
            interpretation += f"• {cluster}: {profile['size']} gözlem\n"
            if profile['dominant_features']:
                interpretation += f"  Özellikler: {', '.join(profile['dominant_features'][:2])}\n"
        
        return interpretation
    
    def compare_clustering_methods(self, columns: List[str]) -> Dict[str, Any]:
        """Farklı kümeleme yöntemlerini karşılaştır"""
        try:
            data = self._prepare_data_for_clustering(columns)
            
            # Optimal küme sayısını belirle
            cluster_analysis = self._determine_optimal_clusters(data)
            n_clusters = cluster_analysis['recommended_clusters']
            
            methods = {
                'K-Means': KMeans(n_clusters=n_clusters, random_state=42),
                'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
                'Spectral': SpectralClustering(n_clusters=n_clusters, random_state=42)
            }
            
            comparison = {
                'data_info': {
                    'variables': columns,
                    'sample_size': len(data),
                    'optimal_clusters': n_clusters
                },
                'methods': {}
            }
            
            best_method = None
            best_score = -1
            
            for method_name, model in methods.items():
                try:
                    labels = model.fit_predict(data)
                    performance = self._evaluate_clustering_performance(data, labels)
                    
                    comparison['methods'][method_name] = {
                        'performance': performance,
                        'cluster_sizes': [int(np.sum(labels == i)) for i in range(n_clusters)]
                    }
                    
                    # En iyi yöntemi belirle (Silhouette skoruna göre)
                    if 'silhouette_score' in performance:
                        if performance['silhouette_score'] > best_score:
                            best_score = performance['silhouette_score']
                            best_method = method_name
                
                except Exception as e:
                    comparison['methods'][method_name] = {'error': str(e)}
            
            comparison['best_method'] = best_method
            comparison['recommendation'] = f"{best_method} yöntemi en iyi performansı gösterdi" if best_method else "Karşılaştırma tamamlanamadı"
            
            return comparison
            
        except Exception as e:
            logger.error(f"Kümeleme yöntemleri karşılaştırma hatası: {str(e)}")
            return {'error': f'Kümeleme yöntemleri karşılaştırma hatası: {str(e)}'}
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Analiz özetini döndür"""
        return {
            'total_analyses': len(self.results),
            'available_analyses': list(self.results.keys()),
            'analysis_types': [result.get('model_type', 'Unknown') for result in self.results.values()],
            'results': self.results
        }