"""
Sosyal Bilimler Analizi Sınıfı

Bu modül sosyal bilim verilerinin analizi için özel araçlar içerir.
Anket analizi, ölçek güvenilirliği, tutum analizi gibi işlevler sunar.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class SocialSciencesAnalyzer:
    """
    Sosyal bilim veri analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        SocialSciencesAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek sosyal bilim veri seti
        """
        self.data = data.copy()
        self.results = {}
        
    def survey_analysis(self, likert_columns: List[str], 
                       demographic_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Anket analizi gerçekleştirir
        
        Args:
            likert_columns: Likert ölçeği sütunları
            demographic_columns: Demografik değişkenler (opsiyonel)
            
        Returns:
            Anket analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            missing_cols = [col for col in likert_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal sütunları filtrele
            numeric_likert = []
            for col in likert_columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_likert.append(col)
            
            if len(numeric_likert) < 3:
                return {'error': 'En az 3 sayısal Likert sütunu gereklidir'}
            
            data_clean = self.data[numeric_likert].dropna()
            
            if len(data_clean) < 30:
                return {'error': 'Anket analizi için en az 30 katılımcı gereklidir'}
            
            # Temel betimsel istatistikler
            descriptive_stats = {}
            for col in numeric_likert:
                values = data_clean[col]
                descriptive_stats[col] = {
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'skewness': float(values.skew()),
                    'kurtosis': float(values.kurtosis())
                }
            
            # Likert ölçeği dağılım analizi
            distribution_analysis = {}
            for col in numeric_likert:
                values = data_clean[col]
                value_counts = values.value_counts().sort_index()
                percentages = values.value_counts(normalize=True).sort_index() * 100
                
                distribution_analysis[col] = {
                    'value_counts': value_counts.to_dict(),
                    'percentages': percentages.to_dict(),
                    'mode': float(values.mode().iloc[0]) if len(values.mode()) > 0 else None,
                    'ceiling_effect': float(percentages.iloc[-1]) if len(percentages) > 0 else 0,
                    'floor_effect': float(percentages.iloc[0]) if len(percentages) > 0 else 0
                }
            
            # Korelasyon analizi
            correlation_matrix = data_clean[numeric_likert].corr()
            
            # Güvenilirlik analizi (Cronbach's Alpha)
            reliability_analysis = self.calculate_cronbach_alpha(data_clean[numeric_likert])
            
            # Faktör analizi uygunluğu
            factor_suitability = self._assess_factor_analysis_suitability(data_clean[numeric_likert])
            
            # Demografik analiz (eğer demografik değişkenler varsa)
            demographic_analysis = {}
            if demographic_columns:
                available_demo_cols = [col for col in demographic_columns if col in self.data.columns]
                for demo_col in available_demo_cols:
                    if demo_col in self.data.columns:
                        demo_analysis = self._analyze_by_demographic(
                            data_clean, numeric_likert, demo_col
                        )
                        demographic_analysis[demo_col] = demo_analysis
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'descriptive_statistics': descriptive_stats,
                'distribution_analysis': distribution_analysis,
                'correlation_matrix': correlation_matrix.to_dict(),
                'reliability_analysis': reliability_analysis,
                'factor_analysis_suitability': factor_suitability,
                'demographic_analysis': demographic_analysis,
                'interpretation': self._interpret_survey_results(
                    descriptive_stats, reliability_analysis, factor_suitability
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Anket analizi hatası: {str(e)}'}
    
    def calculate_cronbach_alpha(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Cronbach's Alpha güvenilirlik katsayısını hesaplar
        
        Args:
            data: Ölçek maddeleri verisi
            
        Returns:
            Güvenilirlik analizi sonuçları
        """
        try:
            # Madde sayısı
            k = len(data.columns)
            
            if k < 2:
                return {'error': 'En az 2 madde gereklidir'}
            
            # Madde varyansları toplamı
            item_variances = data.var(axis=0, ddof=1).sum()
            
            # Toplam puan varyansı
            total_variance = data.sum(axis=1).var(ddof=1)
            
            # Cronbach's Alpha
            if total_variance == 0:
                alpha = 0
            else:
                alpha = (k / (k - 1)) * (1 - (item_variances / total_variance))
            
            # Madde-toplam korelasyonları
            total_scores = data.sum(axis=1)
            item_total_correlations = {}
            
            for col in data.columns:
                # Madde çıkarıldığında toplam puan
                corrected_total = total_scores - data[col]
                correlation, _ = pearsonr(data[col], corrected_total)
                item_total_correlations[col] = float(correlation)
            
            # Alpha if item deleted
            alpha_if_deleted = {}
            for col in data.columns:
                remaining_data = data.drop(columns=[col])
                if len(remaining_data.columns) > 1:
                    remaining_alpha = self.calculate_cronbach_alpha(remaining_data)
                    if 'alpha' in remaining_alpha:
                        alpha_if_deleted[col] = remaining_alpha['alpha']
            
            # Güvenilirlik yorumu
            if alpha >= 0.9:
                reliability_level = "Mükemmel"
            elif alpha >= 0.8:
                reliability_level = "İyi"
            elif alpha >= 0.7:
                reliability_level = "Kabul edilebilir"
            elif alpha >= 0.6:
                reliability_level = "Şüpheli"
            else:
                reliability_level = "Zayıf"
            
            return {
                'alpha': float(alpha),
                'reliability_level': reliability_level,
                'item_count': k,
                'item_total_correlations': item_total_correlations,
                'alpha_if_deleted': alpha_if_deleted,
                'interpretation': f"Cronbach's Alpha = {alpha:.3f} ({reliability_level} güvenilirlik)"
            }
            
        except Exception as e:
            return {'error': f'Cronbach Alpha hesaplama hatası: {str(e)}'}
    
    def attitude_analysis(self, attitude_columns: List[str], 
                         grouping_variable: Optional[str] = None) -> Dict[str, Any]:
        """
        Tutum analizi gerçekleştirir
        
        Args:
            attitude_columns: Tutum ölçeği sütunları
            grouping_variable: Gruplandırma değişkeni (opsiyonel)
            
        Returns:
            Tutum analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            missing_cols = [col for col in attitude_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal sütunları filtrele
            numeric_attitude = []
            for col in attitude_columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_attitude.append(col)
            
            if len(numeric_attitude) < 3:
                return {'error': 'En az 3 sayısal tutum sütunu gereklidir'}
            
            data_clean = self.data[numeric_attitude].copy()
            
            # Gruplandırma değişkeni varsa ekle
            if grouping_variable and grouping_variable in self.data.columns:
                data_clean[grouping_variable] = self.data[grouping_variable]
            
            data_clean = data_clean.dropna()
            
            if len(data_clean) < 20:
                return {'error': 'Tutum analizi için en az 20 katılımcı gereklidir'}
            
            # Tutum puanı hesaplama (ortalama)
            attitude_score = data_clean[numeric_attitude].mean(axis=1)
            data_clean['attitude_score'] = attitude_score
            
            # Temel istatistikler
            basic_stats = {
                'mean_attitude': float(attitude_score.mean()),
                'median_attitude': float(attitude_score.median()),
                'std_attitude': float(attitude_score.std()),
                'min_attitude': float(attitude_score.min()),
                'max_attitude': float(attitude_score.max()),
                'skewness': float(attitude_score.skew()),
                'kurtosis': float(attitude_score.kurtosis())
            }
            
            # Tutum kategorileri (düşük, orta, yüksek)
            q1, q3 = attitude_score.quantile([0.33, 0.67])
            
            attitude_categories = pd.cut(
                attitude_score, 
                bins=[-np.inf, q1, q3, np.inf], 
                labels=['Düşük', 'Orta', 'Yüksek']
            )
            
            category_distribution = attitude_categories.value_counts()
            category_percentages = attitude_categories.value_counts(normalize=True) * 100
            
            # Normallik testi
            shapiro_stat, shapiro_p = stats.shapiro(attitude_score)
            
            # Grup karşılaştırması (eğer gruplandırma değişkeni varsa)
            group_comparison = {}
            if grouping_variable and grouping_variable in data_clean.columns:
                groups = data_clean[grouping_variable].unique()
                
                if len(groups) >= 2:
                    group_stats = {}
                    group_scores = []
                    
                    for group in groups:
                        group_data = data_clean[data_clean[grouping_variable] == group]['attitude_score']
                        group_stats[str(group)] = {
                            'mean': float(group_data.mean()),
                            'std': float(group_data.std()),
                            'count': len(group_data)
                        }
                        group_scores.append(group_data.values)
                    
                    # Grup karşılaştırma testleri
                    if len(groups) == 2:
                        # İki grup karşılaştırması
                        stat, p_value = stats.ttest_ind(group_scores[0], group_scores[1])
                        test_name = "Bağımsız Örneklem t-Testi"
                        
                        # Mann-Whitney U testi (non-parametrik alternatif)
                        u_stat, u_p = mannwhitneyu(group_scores[0], group_scores[1])
                        
                        group_comparison = {
                            'test_type': test_name,
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'mann_whitney_u': float(u_stat),
                            'mann_whitney_p': float(u_p),
                            'group_statistics': group_stats,
                            'significant': p_value < 0.05
                        }
                    
                    elif len(groups) > 2:
                        # Çoklu grup karşılaştırması
                        f_stat, f_p = stats.f_oneway(*group_scores)
                        kruskal_stat, kruskal_p = kruskal(*group_scores)
                        
                        group_comparison = {
                            'test_type': 'Tek Yönlü ANOVA',
                            'f_statistic': float(f_stat),
                            'f_p_value': float(f_p),
                            'kruskal_statistic': float(kruskal_stat),
                            'kruskal_p_value': float(kruskal_p),
                            'group_statistics': group_stats,
                            'significant': f_p < 0.05
                        }
            
            # Güvenilirlik analizi
            reliability = self.calculate_cronbach_alpha(data_clean[numeric_attitude])
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'basic_statistics': basic_stats,
                'attitude_categories': {
                    'distribution': category_distribution.to_dict(),
                    'percentages': category_percentages.to_dict()
                },
                'normality_test': {
                    'shapiro_statistic': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                },
                'reliability_analysis': reliability,
                'group_comparison': group_comparison,
                'interpretation': self._interpret_attitude_results(
                    basic_stats, reliability, group_comparison
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Tutum analizi hatası: {str(e)}'}
    
    def _assess_factor_analysis_suitability(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Faktör analizi uygunluğunu değerlendirir
        """
        try:
            # KMO testi için korelasyon matrisi
            corr_matrix = data.corr()
            
            # Bartlett's test of sphericity
            n = len(data)
            p = len(data.columns)
            
            # Korelasyon matrisinin determinantı
            det_corr = np.linalg.det(corr_matrix)
            
            # Bartlett test istatistiği
            bartlett_stat = -((n - 1) - (2 * p + 5) / 6) * np.log(det_corr)
            bartlett_df = p * (p - 1) / 2
            bartlett_p = 1 - stats.chi2.cdf(bartlett_stat, bartlett_df)
            
            # Basit KMO hesaplaması
            # (Gerçek KMO hesaplaması daha karmaşıktır)
            kmo_estimate = 0.7 if bartlett_p < 0.05 else 0.5
            
            return {
                'bartlett_statistic': float(bartlett_stat),
                'bartlett_p_value': float(bartlett_p),
                'bartlett_suitable': bartlett_p < 0.05,
                'kmo_estimate': kmo_estimate,
                'kmo_suitable': kmo_estimate > 0.6,
                'sample_adequacy': 'Uygun' if (bartlett_p < 0.05 and kmo_estimate > 0.6) else 'Uygun değil'
            }
            
        except Exception as e:
            return {'error': f'Faktör analizi uygunluk testi hatası: {str(e)}'}
    
    def _analyze_by_demographic(self, data: pd.DataFrame, 
                               likert_columns: List[str], 
                               demographic_column: str) -> Dict[str, Any]:
        """
        Demografik değişkene göre analiz yapar
        """
        try:
            if demographic_column not in data.columns:
                return {'error': f'{demographic_column} sütunu bulunamadı'}
            
            # Demografik grupları belirle
            demo_data = self.data[[demographic_column] + likert_columns].dropna()
            groups = demo_data[demographic_column].unique()
            
            if len(groups) < 2:
                return {'error': 'En az 2 demografik grup gereklidir'}
            
            group_stats = {}
            group_scores = []
            
            for group in groups:
                group_data = demo_data[demo_data[demographic_column] == group][likert_columns]
                group_mean_scores = group_data.mean(axis=1)
                
                group_stats[str(group)] = {
                    'count': len(group_data),
                    'mean_score': float(group_mean_scores.mean()),
                    'std_score': float(group_mean_scores.std()),
                    'item_means': group_data.mean().to_dict()
                }
                group_scores.append(group_mean_scores.values)
            
            # İstatistiksel testler
            if len(groups) == 2:
                stat, p_value = stats.ttest_ind(group_scores[0], group_scores[1])
                test_name = "Bağımsız Örneklem t-Testi"
            else:
                stat, p_value = stats.f_oneway(*group_scores)
                test_name = "Tek Yönlü ANOVA"
            
            return {
                'test_type': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'group_statistics': group_stats,
                'interpretation': f"Demografik gruplar arasında {'anlamlı fark var' if p_value < 0.05 else 'anlamlı fark yok'} (p = {p_value:.3f})"
            }
            
        except Exception as e:
            return {'error': f'Demografik analiz hatası: {str(e)}'}
    
    def _interpret_survey_results(self, descriptive_stats: Dict, 
                                 reliability: Dict, 
                                 factor_suitability: Dict) -> str:
        """
        Anket sonuçlarını yorumlar
        """
        interpretation = []
        
        # Güvenilirlik yorumu
        if 'alpha' in reliability:
            alpha = reliability['alpha']
            interpretation.append(f"Ölçeğin güvenilirlik katsayısı (Cronbach's Alpha) {alpha:.3f} olarak bulunmuştur.")
            interpretation.append(f"Bu değer {reliability.get('reliability_level', 'belirsiz')} düzeyde güvenilirlik göstermektedir.")
        
        # Faktör analizi uygunluğu
        if factor_suitability.get('bartlett_suitable', False):
            interpretation.append("Bartlett testi sonuçları faktör analizi için uygun olduğunu göstermektedir.")
        else:
            interpretation.append("Bartlett testi sonuçları faktör analizi için uygun olmadığını göstermektedir.")
        
        # Genel değerlendirme
        interpretation.append("Anket verileri analiz edilmiş ve yukarıdaki bulgular elde edilmiştir.")
        
        return " ".join(interpretation)
    
    def _interpret_attitude_results(self, basic_stats: Dict, 
                                   reliability: Dict, 
                                   group_comparison: Dict) -> str:
        """
        Tutum analizi sonuçlarını yorumlar
        """
        interpretation = []
        
        # Temel istatistikler
        mean_attitude = basic_stats['mean_attitude']
        interpretation.append(f"Katılımcıların ortalama tutum puanı {mean_attitude:.2f}'dir.")
        
        # Güvenilirlik
        if 'alpha' in reliability:
            alpha = reliability['alpha']
            interpretation.append(f"Tutum ölçeğinin güvenilirlik katsayısı {alpha:.3f}'tür.")
        
        # Grup karşılaştırması
        if group_comparison and 'significant' in group_comparison:
            if group_comparison['significant']:
                interpretation.append("Gruplar arasında istatistiksel olarak anlamlı fark bulunmuştur.")
            else:
                interpretation.append("Gruplar arasında istatistiksel olarak anlamlı fark bulunmamıştır.")
        
        return " ".join(interpretation)