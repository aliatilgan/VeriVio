"""
Kümeleme Analizi Sınıfı

Bu modül veri setlerinin kümeleme analizleri için kullanılır.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class ClusteringAnalyzer:
    """
    Kümeleme analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        ClusteringAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek veri seti
        """
        self.data = data.copy()
        self.results = {}
        self.models = {}
        self.scaled_data = None
        self.feature_columns = None
        
    def prepare_data(self, feature_columns: List[str], scaling_method: str = 'standard') -> Dict[str, Any]:
        """
        Kümeleme için veriyi hazırlar
        
        Args:
            feature_columns: Kümeleme için kullanılacak özellik sütunları
            scaling_method: Ölçeklendirme yöntemi ('standard', 'minmax', 'none')
            
        Returns:
            Veri hazırlama sonuçları
        """
        try:
            # Özellik sütunlarını kontrol et
            missing_cols = [col for col in feature_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal olmayan sütunları kontrol et
            numeric_cols = []
            for col in feature_columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_cols.append(col)
            
            if len(numeric_cols) < 2:
                return {'error': 'Kümeleme için en az 2 sayısal sütun gereklidir'}
            
            # Eksik değerleri temizle
            data_clean = self.data[numeric_cols].dropna()
            
            if len(data_clean) < 10:
                return {'error': 'Kümeleme için en az 10 gözlem gereklidir'}
            
            # Ölçeklendirme
            if scaling_method == 'standard':
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_clean)
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data_clean)
            else:
                scaled_data = data_clean.values
                scaler = None
            
            self.scaled_data = scaled_data
            self.feature_columns = numeric_cols
            self.original_data = data_clean
            
            # Veri özeti
            data_summary = {
                'n_samples': len(data_clean),
                'n_features': len(numeric_cols),
                'feature_columns': numeric_cols,
                'scaling_method': scaling_method,
                'data_shape': scaled_data.shape,
                'feature_statistics': {
                    col: {
                        'mean': float(data_clean[col].mean()),
                        'std': float(data_clean[col].std()),
                        'min': float(data_clean[col].min()),
                        'max': float(data_clean[col].max())
                    } for col in numeric_cols
                }
            }
            
            result = {
                'analysis_type': 'Data Preparation for Clustering',
                'data_summary': data_summary,
                'preparation_successful': True,
                'message': f'{len(data_clean)} gözlem ve {len(numeric_cols)} özellik ile kümeleme için hazır'
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Veri hazırlama hatası: {str(e)}'}
    
    def kmeans_clustering(self, n_clusters: int = 3, random_state: int = 42) -> Dict[str, Any]:
        """
        K-Means kümeleme analizi gerçekleştirir
        
        Args:
            n_clusters: Küme sayısı
            random_state: Rastgele durum
            
        Returns:
            K-Means kümeleme sonuçları
        """
        try:
            if self.scaled_data is None:
                return {'error': 'Önce prepare_data() metodunu çalıştırın'}
            
            # K-Means modeli
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_data)
            
            # Küme merkezleri (orijinal ölçekte)
            cluster_centers_scaled = kmeans.cluster_centers_
            
            # Küme istatistikleri
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = self.original_data[cluster_mask]
                
                cluster_stats[f'cluster_{i}'] = {
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100),
                    'center_scaled': cluster_centers_scaled[i].tolist(),
                    'feature_means': {
                        col: float(cluster_data[col].mean()) 
                        for col in self.feature_columns
                    },
                    'feature_stds': {
                        col: float(cluster_data[col].std()) 
                        for col in self.feature_columns
                    }
                }
            
            # Kümeleme kalite metrikleri
            silhouette_avg = silhouette_score(self.scaled_data, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(self.scaled_data, cluster_labels)
            davies_bouldin = davies_bouldin_score(self.scaled_data, cluster_labels)
            
            # Inertia (within-cluster sum of squares)
            inertia = kmeans.inertia_
            
            # Küme içi ve küme arası mesafeler
            within_cluster_distances = []
            between_cluster_distances = []
            
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = self.scaled_data[cluster_mask]
                
                if len(cluster_data) > 1:
                    # Küme içi mesafeler
                    within_dist = pdist(cluster_data).mean()
                    within_cluster_distances.append(within_dist)
                
                # Küme merkezleri arası mesafeler
                for j in range(i+1, n_clusters):
                    between_dist = np.linalg.norm(
                        cluster_centers_scaled[i] - cluster_centers_scaled[j]
                    )
                    between_cluster_distances.append(between_dist)
            
            result = {
                'analysis_type': 'K-Means Clustering',
                'n_clusters': n_clusters,
                'n_samples': len(cluster_labels),
                'n_features': len(self.feature_columns),
                'feature_columns': self.feature_columns,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_statistics': cluster_stats,
                'quality_metrics': {
                    'silhouette_score': float(silhouette_avg),
                    'calinski_harabasz_score': float(calinski_harabasz),
                    'davies_bouldin_score': float(davies_bouldin),
                    'inertia': float(inertia),
                    'within_cluster_distance_mean': float(np.mean(within_cluster_distances)) if within_cluster_distances else 0,
                    'between_cluster_distance_mean': float(np.mean(between_cluster_distances)) if between_cluster_distances else 0
                },
                'model_parameters': {
                    'algorithm': 'K-Means',
                    'n_clusters': n_clusters,
                    'random_state': random_state,
                    'n_iterations': int(kmeans.n_iter_)
                },
                'interpretation': self._interpret_kmeans_results(
                    silhouette_avg, calinski_harabasz, davies_bouldin, cluster_stats
                )
            }
            
            # Modeli sakla
            self.models['kmeans'] = kmeans
            self.results['kmeans'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'K-Means kümeleme hatası: {str(e)}'}
    
    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """
        DBSCAN kümeleme analizi gerçekleştirir
        
        Args:
            eps: Epsilon parametresi (komşuluk yarıçapı)
            min_samples: Minimum örnek sayısı
            
        Returns:
            DBSCAN kümeleme sonuçları
        """
        try:
            if self.scaled_data is None:
                return {'error': 'Önce prepare_data() metodunu çalıştırın'}
            
            # DBSCAN modeli
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(self.scaled_data)
            
            # Küme sayısı ve gürültü noktaları
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            if n_clusters == 0:
                return {
                    'error': 'DBSCAN hiç küme bulamadı. eps veya min_samples parametrelerini ayarlayın',
                    'n_noise_points': n_noise,
                    'suggestion': 'eps değerini artırın veya min_samples değerini azaltın'
                }
            
            # Küme istatistikleri
            cluster_stats = {}
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Gürültü noktaları
                    cluster_mask = cluster_labels == label
                    cluster_stats['noise'] = {
                        'size': int(np.sum(cluster_mask)),
                        'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100)
                    }
                else:
                    cluster_mask = cluster_labels == label
                    cluster_data = self.original_data[cluster_mask]
                    
                    cluster_stats[f'cluster_{label}'] = {
                        'size': int(np.sum(cluster_mask)),
                        'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100),
                        'feature_means': {
                            col: float(cluster_data[col].mean()) 
                            for col in self.feature_columns
                        },
                        'feature_stds': {
                            col: float(cluster_data[col].std()) 
                            for col in self.feature_columns
                        }
                    }
            
            # Kalite metrikleri (gürültü noktaları hariç)
            if n_clusters > 1:
                non_noise_mask = cluster_labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette_avg = silhouette_score(
                        self.scaled_data[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    calinski_harabasz = calinski_harabasz_score(
                        self.scaled_data[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                    davies_bouldin = davies_bouldin_score(
                        self.scaled_data[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                else:
                    silhouette_avg = calinski_harabasz = davies_bouldin = None
            else:
                silhouette_avg = calinski_harabasz = davies_bouldin = None
            
            # Core samples
            core_samples = dbscan.core_sample_indices_
            
            result = {
                'analysis_type': 'DBSCAN Clustering',
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'n_core_samples': len(core_samples),
                'n_samples': len(cluster_labels),
                'n_features': len(self.feature_columns),
                'feature_columns': self.feature_columns,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_statistics': cluster_stats,
                'quality_metrics': {
                    'silhouette_score': float(silhouette_avg) if silhouette_avg is not None else None,
                    'calinski_harabasz_score': float(calinski_harabasz) if calinski_harabasz is not None else None,
                    'davies_bouldin_score': float(davies_bouldin) if davies_bouldin is not None else None,
                    'noise_ratio': float(n_noise / len(cluster_labels))
                },
                'model_parameters': {
                    'algorithm': 'DBSCAN',
                    'eps': eps,
                    'min_samples': min_samples
                },
                'interpretation': self._interpret_dbscan_results(
                    n_clusters, n_noise, len(cluster_labels), silhouette_avg
                )
            }
            
            # Modeli sakla
            self.models['dbscan'] = dbscan
            self.results['dbscan'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'DBSCAN kümeleme hatası: {str(e)}'}
    
    def hierarchical_clustering(self, n_clusters: int = 3, linkage_method: str = 'ward') -> Dict[str, Any]:
        """
        Hiyerarşik kümeleme analizi gerçekleştirir
        
        Args:
            n_clusters: Küme sayısı
            linkage_method: Bağlantı yöntemi ('ward', 'complete', 'average', 'single')
            
        Returns:
            Hiyerarşik kümeleme sonuçları
        """
        try:
            if self.scaled_data is None:
                return {'error': 'Önce prepare_data() metodunu çalıştırın'}
            
            # Hiyerarşik kümeleme modeli
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage=linkage_method
            )
            cluster_labels = hierarchical.fit_predict(self.scaled_data)
            
            # Dendrogram için linkage matrisi
            linkage_matrix = linkage(self.scaled_data, method=linkage_method)
            
            # Küme istatistikleri
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_data = self.original_data[cluster_mask]
                
                cluster_stats[f'cluster_{i}'] = {
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100),
                    'feature_means': {
                        col: float(cluster_data[col].mean()) 
                        for col in self.feature_columns
                    },
                    'feature_stds': {
                        col: float(cluster_data[col].std()) 
                        for col in self.feature_columns
                    }
                }
            
            # Kalite metrikleri
            silhouette_avg = silhouette_score(self.scaled_data, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(self.scaled_data, cluster_labels)
            davies_bouldin = davies_bouldin_score(self.scaled_data, cluster_labels)
            
            # Dendrogram yükseklikleri
            heights = linkage_matrix[:, 2]
            optimal_clusters_suggestion = self._suggest_optimal_clusters_from_dendrogram(heights)
            
            result = {
                'analysis_type': 'Hierarchical Clustering',
                'n_clusters': n_clusters,
                'linkage_method': linkage_method,
                'n_samples': len(cluster_labels),
                'n_features': len(self.feature_columns),
                'feature_columns': self.feature_columns,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_statistics': cluster_stats,
                'quality_metrics': {
                    'silhouette_score': float(silhouette_avg),
                    'calinski_harabasz_score': float(calinski_harabasz),
                    'davies_bouldin_score': float(davies_bouldin)
                },
                'dendrogram_info': {
                    'linkage_matrix_shape': linkage_matrix.shape,
                    'max_height': float(heights.max()),
                    'min_height': float(heights.min()),
                    'height_differences': np.diff(sorted(heights, reverse=True))[:10].tolist(),
                    'suggested_optimal_clusters': optimal_clusters_suggestion
                },
                'model_parameters': {
                    'algorithm': 'Agglomerative Clustering',
                    'n_clusters': n_clusters,
                    'linkage': linkage_method
                },
                'interpretation': self._interpret_hierarchical_results(
                    silhouette_avg, linkage_method, optimal_clusters_suggestion
                )
            }
            
            # Modeli sakla
            self.models['hierarchical'] = hierarchical
            self.results['hierarchical'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Hiyerarşik kümeleme hatası: {str(e)}'}
    
    def gaussian_mixture_clustering(self, n_components: int = 3, random_state: int = 42) -> Dict[str, Any]:
        """
        Gaussian Mixture Model kümeleme analizi gerçekleştirir
        
        Args:
            n_components: Bileşen sayısı
            random_state: Rastgele durum
            
        Returns:
            GMM kümeleme sonuçları
        """
        try:
            if self.scaled_data is None:
                return {'error': 'Önce prepare_data() metodunu çalıştırın'}
            
            # Gaussian Mixture Model
            gmm = GaussianMixture(n_components=n_components, random_state=random_state)
            gmm.fit(self.scaled_data)
            
            # Küme etiketleri ve olasılıkları
            cluster_labels = gmm.predict(self.scaled_data)
            cluster_probabilities = gmm.predict_proba(self.scaled_data)
            
            # Model parametreleri
            means = gmm.means_
            covariances = gmm.covariances_
            weights = gmm.weights_
            
            # Küme istatistikleri
            cluster_stats = {}
            for i in range(n_components):
                cluster_mask = cluster_labels == i
                cluster_data = self.original_data[cluster_mask]
                
                # Ortalama olasılık
                avg_probability = cluster_probabilities[cluster_mask, i].mean() if np.sum(cluster_mask) > 0 else 0
                
                cluster_stats[f'cluster_{i}'] = {
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100),
                    'weight': float(weights[i]),
                    'average_probability': float(avg_probability),
                    'feature_means': {
                        col: float(cluster_data[col].mean()) if len(cluster_data) > 0 else 0
                        for col in self.feature_columns
                    },
                    'feature_stds': {
                        col: float(cluster_data[col].std()) if len(cluster_data) > 0 else 0
                        for col in self.feature_columns
                    },
                    'gaussian_mean': means[i].tolist(),
                    'gaussian_covariance_diagonal': np.diag(covariances[i]).tolist()
                }
            
            # Kalite metrikleri
            silhouette_avg = silhouette_score(self.scaled_data, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(self.scaled_data, cluster_labels)
            davies_bouldin = davies_bouldin_score(self.scaled_data, cluster_labels)
            
            # Model kalitesi
            aic = gmm.aic(self.scaled_data)
            bic = gmm.bic(self.scaled_data)
            log_likelihood = gmm.score(self.scaled_data)
            
            # Belirsizlik analizi
            max_probabilities = cluster_probabilities.max(axis=1)
            uncertainty_analysis = {
                'mean_max_probability': float(max_probabilities.mean()),
                'std_max_probability': float(max_probabilities.std()),
                'high_uncertainty_ratio': float(np.sum(max_probabilities < 0.7) / len(max_probabilities)),
                'very_high_uncertainty_ratio': float(np.sum(max_probabilities < 0.5) / len(max_probabilities))
            }
            
            result = {
                'analysis_type': 'Gaussian Mixture Model',
                'n_components': n_components,
                'n_samples': len(cluster_labels),
                'n_features': len(self.feature_columns),
                'feature_columns': self.feature_columns,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_probabilities': cluster_probabilities.tolist(),
                'cluster_statistics': cluster_stats,
                'quality_metrics': {
                    'silhouette_score': float(silhouette_avg),
                    'calinski_harabasz_score': float(calinski_harabasz),
                    'davies_bouldin_score': float(davies_bouldin),
                    'aic': float(aic),
                    'bic': float(bic),
                    'log_likelihood': float(log_likelihood)
                },
                'uncertainty_analysis': uncertainty_analysis,
                'model_parameters': {
                    'algorithm': 'Gaussian Mixture Model',
                    'n_components': n_components,
                    'random_state': random_state,
                    'converged': bool(gmm.converged_),
                    'n_iterations': int(gmm.n_iter_)
                },
                'interpretation': self._interpret_gmm_results(
                    silhouette_avg, uncertainty_analysis, aic, bic
                )
            }
            
            # Modeli sakla
            self.models['gmm'] = gmm
            self.results['gmm'] = result
            
            return result
            
        except Exception as e:
            return {'error': f'Gaussian Mixture Model hatası: {str(e)}'}
    
    def optimal_clusters_analysis(self, max_clusters: int = 10, methods: List[str] = ['elbow', 'silhouette']) -> Dict[str, Any]:
        """
        Optimal küme sayısını belirlemek için analiz yapar
        
        Args:
            max_clusters: Maksimum küme sayısı
            methods: Kullanılacak yöntemler ('elbow', 'silhouette', 'gap')
            
        Returns:
            Optimal küme sayısı analizi sonuçları
        """
        try:
            if self.scaled_data is None:
                return {'error': 'Önce prepare_data() metodunu çalıştırın'}
            
            if max_clusters < 2:
                max_clusters = min(10, len(self.scaled_data) // 2)
            
            results = {}
            
            # Elbow method
            if 'elbow' in methods:
                inertias = []
                k_range = range(1, max_clusters + 1)
                
                for k in k_range:
                    if k == 1:
                        inertias.append(np.sum(np.var(self.scaled_data, axis=0)) * len(self.scaled_data))
                    else:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(self.scaled_data)
                        inertias.append(kmeans.inertia_)
                
                # Elbow noktasını bul
                elbow_point = self._find_elbow_point(list(k_range), inertias)
                
                results['elbow'] = {
                    'k_values': list(k_range),
                    'inertias': inertias,
                    'optimal_k': elbow_point,
                    'inertia_differences': np.diff(inertias).tolist()
                }
            
            # Silhouette method
            if 'silhouette' in methods:
                silhouette_scores = []
                k_range = range(2, max_clusters + 1)
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(self.scaled_data)
                    silhouette_avg = silhouette_score(self.scaled_data, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                
                optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
                
                results['silhouette'] = {
                    'k_values': list(k_range),
                    'silhouette_scores': silhouette_scores,
                    'optimal_k': optimal_k_silhouette,
                    'max_silhouette_score': float(max(silhouette_scores))
                }
            
            # Gap statistic (basitleştirilmiş)
            if 'gap' in methods:
                gap_stats = []
                k_range = range(1, max_clusters + 1)
                
                for k in k_range:
                    if k == 1:
                        gap_stats.append(0)
                    else:
                        # Gerçek veri için log(inertia)
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(self.scaled_data)
                        real_inertia = kmeans.inertia_
                        
                        # Rastgele veri için log(inertia) (basitleştirilmiş)
                        random_data = np.random.uniform(
                            self.scaled_data.min(axis=0),
                            self.scaled_data.max(axis=0),
                            self.scaled_data.shape
                        )
                        kmeans_random = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans_random.fit(random_data)
                        random_inertia = kmeans_random.inertia_
                        
                        gap = np.log(random_inertia) - np.log(real_inertia)
                        gap_stats.append(gap)
                
                # En yüksek gap değeri
                optimal_k_gap = k_range[np.argmax(gap_stats)]
                
                results['gap'] = {
                    'k_values': list(k_range),
                    'gap_statistics': gap_stats,
                    'optimal_k': optimal_k_gap,
                    'max_gap_statistic': float(max(gap_stats))
                }
            
            # Genel öneri
            optimal_suggestions = []
            if 'elbow' in results:
                optimal_suggestions.append(results['elbow']['optimal_k'])
            if 'silhouette' in results:
                optimal_suggestions.append(results['silhouette']['optimal_k'])
            if 'gap' in results:
                optimal_suggestions.append(results['gap']['optimal_k'])
            
            # En sık önerilen değer
            if optimal_suggestions:
                from collections import Counter
                suggestion_counts = Counter(optimal_suggestions)
                most_common_k = suggestion_counts.most_common(1)[0][0]
            else:
                most_common_k = 3
            
            final_result = {
                'analysis_type': 'Optimal Clusters Analysis',
                'methods_used': methods,
                'max_clusters_tested': max_clusters,
                'n_samples': len(self.scaled_data),
                'n_features': len(self.feature_columns),
                'results_by_method': results,
                'recommendations': {
                    'individual_suggestions': optimal_suggestions,
                    'consensus_optimal_k': most_common_k,
                    'confidence': 'Yüksek' if len(set(optimal_suggestions)) == 1 else 'Orta' if len(set(optimal_suggestions)) <= 2 else 'Düşük'
                },
                'interpretation': self._interpret_optimal_clusters(results, most_common_k)
            }
            
            self.results['optimal_clusters'] = final_result
            return final_result
            
        except Exception as e:
            return {'error': f'Optimal küme analizi hatası: {str(e)}'}
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Elbow noktasını bulur"""
        if len(k_values) < 3:
            return k_values[0] if k_values else 2
        
        # İkinci türev yöntemi
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        if second_derivatives:
            elbow_idx = np.argmax(second_derivatives) + 1
            return k_values[elbow_idx]
        else:
            return k_values[1] if len(k_values) > 1 else 2
    
    def _suggest_optimal_clusters_from_dendrogram(self, heights: np.ndarray) -> int:
        """Dendrogram yüksekliklerinden optimal küme sayısını önerir"""
        if len(heights) < 2:
            return 2
        
        # En büyük yükseklik farkını bul
        height_diffs = np.diff(sorted(heights, reverse=True))
        if len(height_diffs) > 0:
            max_diff_idx = np.argmax(height_diffs)
            return min(max_diff_idx + 2, len(heights))
        else:
            return 2
    
    def _interpret_kmeans_results(self, silhouette: float, calinski: float, 
                                davies: float, cluster_stats: Dict) -> str:
        """K-Means sonuçlarını yorumlar"""
        interpretation = "K-Means kümeleme: "
        
        if silhouette > 0.7:
            interpretation += "Mükemmel kümeleme kalitesi. "
        elif silhouette > 0.5:
            interpretation += "İyi kümeleme kalitesi. "
        elif silhouette > 0.25:
            interpretation += "Orta kümeleme kalitesi. "
        else:
            interpretation += "Zayıf kümeleme kalitesi. "
        
        # Küme boyutları
        cluster_sizes = [stats['size'] for stats in cluster_stats.values()]
        size_variation = np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
        
        if size_variation < 0.3:
            interpretation += "Kümeler dengeli boyutlarda. "
        elif size_variation < 0.7:
            interpretation += "Kümeler orta düzeyde farklı boyutlarda. "
        else:
            interpretation += "Kümeler çok farklı boyutlarda. "
        
        return interpretation
    
    def _interpret_dbscan_results(self, n_clusters: int, n_noise: int, 
                                total_samples: int, silhouette: Optional[float]) -> str:
        """DBSCAN sonuçlarını yorumlar"""
        interpretation = f"DBSCAN kümeleme: {n_clusters} küme bulundu. "
        
        noise_ratio = n_noise / total_samples
        if noise_ratio < 0.05:
            interpretation += "Çok az gürültü noktası. "
        elif noise_ratio < 0.15:
            interpretation += "Makul düzeyde gürültü noktası. "
        else:
            interpretation += "Yüksek gürültü oranı, parametreleri ayarlayın. "
        
        if silhouette is not None:
            if silhouette > 0.5:
                interpretation += "İyi kümeleme kalitesi."
            else:
                interpretation += "Kümeleme kalitesi geliştirilmeli."
        
        return interpretation
    
    def _interpret_hierarchical_results(self, silhouette: float, linkage_method: str, 
                                      optimal_suggestion: int) -> str:
        """Hiyerarşik kümeleme sonuçlarını yorumlar"""
        interpretation = f"Hiyerarşik kümeleme ({linkage_method} bağlantı): "
        
        if silhouette > 0.5:
            interpretation += "İyi kümeleme kalitesi. "
        else:
            interpretation += "Kümeleme kalitesi orta. "
        
        interpretation += f"Dendrogram analizi {optimal_suggestion} küme öneriyor."
        
        return interpretation
    
    def _interpret_gmm_results(self, silhouette: float, uncertainty: Dict, 
                             aic: float, bic: float) -> str:
        """GMM sonuçlarını yorumlar"""
        interpretation = "Gaussian Mixture Model: "
        
        if silhouette > 0.5:
            interpretation += "İyi kümeleme kalitesi. "
        else:
            interpretation += "Orta kümeleme kalitesi. "
        
        if uncertainty['high_uncertainty_ratio'] < 0.2:
            interpretation += "Düşük belirsizlik, net kümeler. "
        elif uncertainty['high_uncertainty_ratio'] < 0.4:
            interpretation += "Orta belirsizlik. "
        else:
            interpretation += "Yüksek belirsizlik, kümeler iç içe geçmiş olabilir. "
        
        return interpretation
    
    def _interpret_optimal_clusters(self, results: Dict, optimal_k: int) -> str:
        """Optimal küme analizi sonuçlarını yorumlar"""
        interpretation = f"Optimal küme sayısı analizi: {optimal_k} küme öneriliyor. "
        
        methods_agree = len(set([
            results.get('elbow', {}).get('optimal_k'),
            results.get('silhouette', {}).get('optimal_k'),
            results.get('gap', {}).get('optimal_k')
        ])) == 1
        
        if methods_agree:
            interpretation += "Tüm yöntemler aynı sonucu veriyor, güvenilir öneri."
        else:
            interpretation += "Yöntemler farklı sonuçlar veriyor, veri yapısı karmaşık olabilir."
        
        return interpretation
    
    def get_clustering_summary(self) -> Dict[str, Any]:
        """
        Gerçekleştirilen tüm kümeleme analizlerinin özetini döndürür
        
        Returns:
            Kümeleme analizi özetleri
        """
        if not self.results:
            return {'message': 'Henüz kümeleme analizi gerçekleştirilmedi'}
        
        summary = {
            'total_analyses': len(self.results),
            'analyses_performed': list(self.results.keys()),
            'models_available': list(self.models.keys()),
            'data_info': {
                'n_samples': len(self.scaled_data) if self.scaled_data is not None else 0,
                'n_features': len(self.feature_columns) if self.feature_columns else 0,
                'feature_columns': self.feature_columns
            },
            'analysis_details': self.results
        }
        
        # En iyi modeli bul (en yüksek silhouette score)
        best_model = None
        best_silhouette = -1
        
        for analysis_name, result in self.results.items():
            if 'error' not in result and 'quality_metrics' in result:
                silhouette = result['quality_metrics'].get('silhouette_score')
                if silhouette and silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_model = analysis_name
        
        if best_model:
            summary['best_model'] = {
                'name': best_model,
                'silhouette_score': best_silhouette,
                'quality': 'Mükemmel' if best_silhouette > 0.7 else 'İyi' if best_silhouette > 0.5 else 'Orta' if best_silhouette > 0.25 else 'Zayıf'
            }
        
        return summary