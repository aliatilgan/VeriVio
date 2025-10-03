"""
Eğitim Analizi Sınıfı

Bu modül eğitim verilerinin analizi için özel araçlar içerir.
Çok düzeyli analiz, başarı ölçümü, öğrenci performans analizi gibi işlevler sunar.
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


class EducationAnalyzer:
    """
    Eğitim veri analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        EducationAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek eğitim veri seti
        """
        self.data = data.copy()
        self.results = {}
        
    def academic_performance_analysis(self, score_columns: List[str], 
                                    student_id_column: Optional[str] = None,
                                    class_column: Optional[str] = None,
                                    school_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Akademik başarı analizi gerçekleştirir
        
        Args:
            score_columns: Başarı puanı sütunları
            student_id_column: Öğrenci ID sütunu (opsiyonel)
            class_column: Sınıf sütunu (opsiyonel)
            school_column: Okul sütunu (opsiyonel)
            
        Returns:
            Akademik başarı analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            missing_cols = [col for col in score_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal sütunları filtrele
            numeric_scores = []
            for col in score_columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_scores.append(col)
            
            if len(numeric_scores) < 1:
                return {'error': 'En az 1 sayısal başarı sütunu gereklidir'}
            
            data_clean = self.data[numeric_scores].copy()
            
            # Ek sütunları ekle
            if student_id_column and student_id_column in self.data.columns:
                data_clean['student_id'] = self.data[student_id_column]
            if class_column and class_column in self.data.columns:
                data_clean['class'] = self.data[class_column]
            if school_column and school_column in self.data.columns:
                data_clean['school'] = self.data[school_column]
            
            data_clean = data_clean.dropna()
            
            if len(data_clean) < 10:
                return {'error': 'Başarı analizi için en az 10 öğrenci gereklidir'}
            
            # Genel başarı puanı hesaplama
            overall_score = data_clean[numeric_scores].mean(axis=1)
            data_clean['overall_score'] = overall_score
            
            # Temel istatistikler
            basic_stats = {}
            for col in numeric_scores + ['overall_score']:
                values = data_clean[col]
                basic_stats[col] = {
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q1': float(values.quantile(0.25)),
                    'q3': float(values.quantile(0.75)),
                    'skewness': float(values.skew()),
                    'kurtosis': float(values.kurtosis())
                }
            
            # Başarı kategorileri
            performance_categories = self._categorize_performance(overall_score)
            category_distribution = performance_categories.value_counts()
            category_percentages = performance_categories.value_counts(normalize=True) * 100
            
            # Korelasyon analizi (dersler arası)
            correlation_matrix = data_clean[numeric_scores].corr()
            
            # Normallik testleri
            normality_tests = {}
            for col in numeric_scores + ['overall_score']:
                values = data_clean[col]
                shapiro_stat, shapiro_p = stats.shapiro(values)
                normality_tests[col] = {
                    'shapiro_statistic': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            
            # Sınıf düzeyinde analiz
            class_analysis = {}
            if 'class' in data_clean.columns:
                class_analysis = self._analyze_by_class(data_clean, numeric_scores)
            
            # Okul düzeyinde analiz
            school_analysis = {}
            if 'school' in data_clean.columns:
                school_analysis = self._analyze_by_school(data_clean, numeric_scores)
            
            # Çok düzeyli analiz (eğer hem sınıf hem okul varsa)
            multilevel_analysis = {}
            if 'class' in data_clean.columns and 'school' in data_clean.columns:
                multilevel_analysis = self._multilevel_analysis(data_clean, numeric_scores)
            
            # Başarı trendleri (eğer zaman serisi varsa)
            trend_analysis = self._analyze_performance_trends(data_clean, numeric_scores)
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'basic_statistics': basic_stats,
                'performance_categories': {
                    'distribution': category_distribution.to_dict(),
                    'percentages': category_percentages.to_dict()
                },
                'correlation_matrix': correlation_matrix.to_dict(),
                'normality_tests': normality_tests,
                'class_analysis': class_analysis,
                'school_analysis': school_analysis,
                'multilevel_analysis': multilevel_analysis,
                'trend_analysis': trend_analysis,
                'interpretation': self._interpret_academic_performance(
                    basic_stats, category_distribution, class_analysis, school_analysis
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Akademik başarı analizi hatası: {str(e)}'}
    
    def learning_outcome_assessment(self, pre_test_columns: List[str], 
                                   post_test_columns: List[str],
                                   intervention_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Öğrenme çıktıları değerlendirmesi gerçekleştirir
        
        Args:
            pre_test_columns: Ön test sütunları
            post_test_columns: Son test sütunları
            intervention_column: Müdahale/yöntem sütunu (opsiyonel)
            
        Returns:
            Öğrenme çıktıları değerlendirme sonuçları
        """
        try:
            # Veriyi kontrol et
            all_columns = pre_test_columns + post_test_columns
            missing_cols = [col for col in all_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            if len(pre_test_columns) != len(post_test_columns):
                return {'error': 'Ön test ve son test sütun sayıları eşit olmalıdır'}
            
            # Sayısal sütunları filtrele
            numeric_pre = [col for col in pre_test_columns if pd.api.types.is_numeric_dtype(self.data[col])]
            numeric_post = [col for col in post_test_columns if pd.api.types.is_numeric_dtype(self.data[col])]
            
            if len(numeric_pre) < 1 or len(numeric_post) < 1:
                return {'error': 'En az 1 sayısal ön test ve son test sütunu gereklidir'}
            
            # Eşleşen sütunları al
            min_length = min(len(numeric_pre), len(numeric_post))
            numeric_pre = numeric_pre[:min_length]
            numeric_post = numeric_post[:min_length]
            
            data_clean = self.data[numeric_pre + numeric_post].copy()
            
            # Müdahale sütunu varsa ekle
            if intervention_column and intervention_column in self.data.columns:
                data_clean['intervention'] = self.data[intervention_column]
            
            data_clean = data_clean.dropna()
            
            if len(data_clean) < 10:
                return {'error': 'Öğrenme çıktıları analizi için en az 10 öğrenci gereklidir'}
            
            # Kazanım hesaplamaları
            gain_scores = {}
            effect_sizes = {}
            paired_tests = {}
            
            for i, (pre_col, post_col) in enumerate(zip(numeric_pre, numeric_post)):
                pre_scores = data_clean[pre_col]
                post_scores = data_clean[post_col]
                
                # Kazanım puanı
                gain = post_scores - pre_scores
                gain_scores[f'gain_{i+1}'] = {
                    'mean_gain': float(gain.mean()),
                    'std_gain': float(gain.std()),
                    'min_gain': float(gain.min()),
                    'max_gain': float(gain.max()),
                    'positive_gains': int((gain > 0).sum()),
                    'negative_gains': int((gain < 0).sum()),
                    'no_change': int((gain == 0).sum())
                }
                
                # Etki büyüklüğü (Cohen's d)
                pooled_std = np.sqrt((pre_scores.var() + post_scores.var()) / 2)
                cohens_d = gain.mean() / pooled_std if pooled_std > 0 else 0
                effect_sizes[f'effect_size_{i+1}'] = float(cohens_d)
                
                # Eşleştirilmiş t-testi
                t_stat, t_p = stats.ttest_rel(post_scores, pre_scores)
                paired_tests[f'test_{i+1}'] = {
                    'pre_mean': float(pre_scores.mean()),
                    'post_mean': float(post_scores.mean()),
                    't_statistic': float(t_stat),
                    'p_value': float(t_p),
                    'significant': t_p < 0.05,
                    'cohens_d': float(cohens_d)
                }
            
            # Genel kazanım analizi
            overall_pre = data_clean[numeric_pre].mean(axis=1)
            overall_post = data_clean[numeric_post].mean(axis=1)
            overall_gain = overall_post - overall_pre
            
            overall_analysis = {
                'pre_test_mean': float(overall_pre.mean()),
                'post_test_mean': float(overall_post.mean()),
                'mean_gain': float(overall_gain.mean()),
                'gain_std': float(overall_gain.std()),
                'improvement_rate': float((overall_gain > 0).mean() * 100),
                'decline_rate': float((overall_gain < 0).mean() * 100)
            }
            
            # Müdahale grupları analizi
            intervention_analysis = {}
            if 'intervention' in data_clean.columns:
                intervention_analysis = self._analyze_by_intervention(
                    data_clean, numeric_pre, numeric_post
                )
            
            # Öğrenme etkililik kategorileri
            effectiveness_categories = self._categorize_learning_effectiveness(overall_gain)
            effectiveness_distribution = effectiveness_categories.value_counts()
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'gain_scores': gain_scores,
                'effect_sizes': effect_sizes,
                'paired_tests': paired_tests,
                'overall_analysis': overall_analysis,
                'intervention_analysis': intervention_analysis,
                'effectiveness_distribution': effectiveness_distribution.to_dict(),
                'interpretation': self._interpret_learning_outcomes(
                    overall_analysis, effect_sizes, intervention_analysis
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Öğrenme çıktıları analizi hatası: {str(e)}'}
    
    def _categorize_performance(self, scores: pd.Series) -> pd.Series:
        """
        Başarı puanlarını kategorilere ayırır
        """
        # Percentile bazlı kategorileme
        q1, q2, q3 = scores.quantile([0.25, 0.5, 0.75])
        
        categories = pd.cut(
            scores,
            bins=[-np.inf, q1, q2, q3, np.inf],
            labels=['Düşük', 'Orta-Alt', 'Orta-Üst', 'Yüksek']
        )
        
        return categories
    
    def _categorize_learning_effectiveness(self, gains: pd.Series) -> pd.Series:
        """
        Öğrenme kazanımlarını etkililik kategorilerine ayırır
        """
        # Kazanım bazlı kategorileme
        categories = pd.cut(
            gains,
            bins=[-np.inf, -5, 0, 10, np.inf],
            labels=['Gerileme', 'Değişim Yok', 'Az Gelişim', 'Yüksek Gelişim']
        )
        
        return categories
    
    def _analyze_by_class(self, data: pd.DataFrame, score_columns: List[str]) -> Dict[str, Any]:
        """
        Sınıf düzeyinde analiz yapar
        """
        try:
            classes = data['class'].unique()
            class_stats = {}
            class_scores = []
            
            for class_name in classes:
                class_data = data[data['class'] == class_name]
                class_overall = class_data[score_columns].mean(axis=1)
                
                class_stats[str(class_name)] = {
                    'count': len(class_data),
                    'mean_score': float(class_overall.mean()),
                    'std_score': float(class_overall.std()),
                    'min_score': float(class_overall.min()),
                    'max_score': float(class_overall.max())
                }
                class_scores.append(class_overall.values)
            
            # ANOVA testi
            if len(classes) > 2:
                f_stat, f_p = stats.f_oneway(*class_scores)
                test_result = {
                    'test_type': 'Tek Yönlü ANOVA',
                    'f_statistic': float(f_stat),
                    'p_value': float(f_p),
                    'significant': f_p < 0.05
                }
            else:
                t_stat, t_p = stats.ttest_ind(class_scores[0], class_scores[1])
                test_result = {
                    'test_type': 'Bağımsız Örneklem t-Testi',
                    't_statistic': float(t_stat),
                    'p_value': float(t_p),
                    'significant': t_p < 0.05
                }
            
            return {
                'class_statistics': class_stats,
                'statistical_test': test_result,
                'interpretation': f"Sınıflar arasında {'anlamlı fark var' if test_result['significant'] else 'anlamlı fark yok'}"
            }
            
        except Exception as e:
            return {'error': f'Sınıf analizi hatası: {str(e)}'}
    
    def _analyze_by_school(self, data: pd.DataFrame, score_columns: List[str]) -> Dict[str, Any]:
        """
        Okul düzeyinde analiz yapar
        """
        try:
            schools = data['school'].unique()
            school_stats = {}
            school_scores = []
            
            for school_name in schools:
                school_data = data[data['school'] == school_name]
                school_overall = school_data[score_columns].mean(axis=1)
                
                school_stats[str(school_name)] = {
                    'count': len(school_data),
                    'mean_score': float(school_overall.mean()),
                    'std_score': float(school_overall.std()),
                    'class_count': len(school_data['class'].unique()) if 'class' in school_data.columns else 1
                }
                school_scores.append(school_overall.values)
            
            # ANOVA testi
            if len(schools) > 2:
                f_stat, f_p = stats.f_oneway(*school_scores)
                test_result = {
                    'test_type': 'Tek Yönlü ANOVA',
                    'f_statistic': float(f_stat),
                    'p_value': float(f_p),
                    'significant': f_p < 0.05
                }
            elif len(schools) == 2:
                t_stat, t_p = stats.ttest_ind(school_scores[0], school_scores[1])
                test_result = {
                    'test_type': 'Bağımsız Örneklem t-Testi',
                    't_statistic': float(t_stat),
                    'p_value': float(t_p),
                    'significant': t_p < 0.05
                }
            else:
                test_result = {'error': 'Karşılaştırma için en az 2 okul gereklidir'}
            
            return {
                'school_statistics': school_stats,
                'statistical_test': test_result,
                'interpretation': f"Okullar arasında {'anlamlı fark var' if test_result.get('significant', False) else 'anlamlı fark yok'}"
            }
            
        except Exception as e:
            return {'error': f'Okul analizi hatası: {str(e)}'}
    
    def _multilevel_analysis(self, data: pd.DataFrame, score_columns: List[str]) -> Dict[str, Any]:
        """
        Çok düzeyli analiz yapar (öğrenci-sınıf-okul)
        """
        try:
            overall_scores = data[score_columns].mean(axis=1)
            
            # Varyans bileşenleri analizi
            total_variance = overall_scores.var()
            
            # Okul düzeyinde varyans
            school_means = data.groupby('school')[score_columns].mean().mean(axis=1)
            between_school_variance = school_means.var()
            
            # Sınıf düzeyinde varyans (okul içi)
            class_means = data.groupby(['school', 'class'])[score_columns].mean().mean(axis=1)
            within_school_between_class_variance = class_means.groupby(level=0).var().mean()
            
            # Öğrenci düzeyinde varyans (sınıf içi)
            within_class_variance = total_variance - between_school_variance - within_school_between_class_variance
            
            # ICC hesaplamaları
            icc_school = between_school_variance / total_variance
            icc_class = within_school_between_class_variance / total_variance
            icc_student = within_class_variance / total_variance
            
            return {
                'variance_components': {
                    'total_variance': float(total_variance),
                    'between_school_variance': float(between_school_variance),
                    'between_class_variance': float(within_school_between_class_variance),
                    'within_class_variance': float(within_class_variance)
                },
                'intraclass_correlations': {
                    'school_level_icc': float(icc_school),
                    'class_level_icc': float(icc_class),
                    'student_level_icc': float(icc_student)
                },
                'interpretation': f"Toplam varyansın %{icc_school*100:.1f}'i okul, %{icc_class*100:.1f}'i sınıf, %{icc_student*100:.1f}'i öğrenci düzeyindedir."
            }
            
        except Exception as e:
            return {'error': f'Çok düzeyli analiz hatası: {str(e)}'}
    
    def _analyze_performance_trends(self, data: pd.DataFrame, score_columns: List[str]) -> Dict[str, Any]:
        """
        Başarı trendlerini analiz eder
        """
        try:
            # Eğer sütun isimleri zaman bilgisi içeriyorsa trend analizi yap
            time_pattern_columns = []
            for col in score_columns:
                if any(time_word in col.lower() for time_word in ['dönem', 'ay', 'hafta', 'test1', 'test2', 'test3']):
                    time_pattern_columns.append(col)
            
            if len(time_pattern_columns) < 2:
                return {'message': 'Trend analizi için zaman serisi verisi bulunamadı'}
            
            # Zaman serisi analizi
            trend_data = data[time_pattern_columns].mean()
            
            # Trend yönü
            if len(trend_data) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    range(len(trend_data)), trend_data.values
                )
                
                trend_direction = 'Artan' if slope > 0 else 'Azalan' if slope < 0 else 'Sabit'
                
                return {
                    'trend_data': trend_data.to_dict(),
                    'slope': float(slope),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'trend_direction': trend_direction,
                    'significant_trend': p_value < 0.05,
                    'interpretation': f"Başarı puanlarında {trend_direction.lower()} trend {'var' if p_value < 0.05 else 'yok'} (R² = {r_value**2:.3f})"
                }
            
            return {'message': 'Trend analizi için yeterli veri noktası yok'}
            
        except Exception as e:
            return {'error': f'Trend analizi hatası: {str(e)}'}
    
    def _analyze_by_intervention(self, data: pd.DataFrame, 
                               pre_columns: List[str], 
                               post_columns: List[str]) -> Dict[str, Any]:
        """
        Müdahale gruplarına göre analiz yapar
        """
        try:
            interventions = data['intervention'].unique()
            intervention_stats = {}
            gain_scores_by_group = []
            
            for intervention in interventions:
                group_data = data[data['intervention'] == intervention]
                
                # Grup için kazanım hesaplama
                pre_scores = group_data[pre_columns].mean(axis=1)
                post_scores = group_data[post_columns].mean(axis=1)
                gains = post_scores - pre_scores
                
                intervention_stats[str(intervention)] = {
                    'count': len(group_data),
                    'pre_mean': float(pre_scores.mean()),
                    'post_mean': float(post_scores.mean()),
                    'mean_gain': float(gains.mean()),
                    'gain_std': float(gains.std()),
                    'improvement_rate': float((gains > 0).mean() * 100)
                }
                gain_scores_by_group.append(gains.values)
            
            # Gruplar arası karşılaştırma
            if len(interventions) > 2:
                f_stat, f_p = stats.f_oneway(*gain_scores_by_group)
                test_result = {
                    'test_type': 'Tek Yönlü ANOVA',
                    'f_statistic': float(f_stat),
                    'p_value': float(f_p),
                    'significant': f_p < 0.05
                }
            elif len(interventions) == 2:
                t_stat, t_p = stats.ttest_ind(gain_scores_by_group[0], gain_scores_by_group[1])
                test_result = {
                    'test_type': 'Bağımsız Örneklem t-Testi',
                    't_statistic': float(t_stat),
                    'p_value': float(t_p),
                    'significant': t_p < 0.05
                }
            else:
                test_result = {'error': 'Karşılaştırma için en az 2 müdahale grubu gereklidir'}
            
            return {
                'intervention_statistics': intervention_stats,
                'statistical_test': test_result,
                'interpretation': f"Müdahale grupları arasında {'anlamlı fark var' if test_result.get('significant', False) else 'anlamlı fark yok'}"
            }
            
        except Exception as e:
            return {'error': f'Müdahale analizi hatası: {str(e)}'}
    
    def _interpret_academic_performance(self, basic_stats: Dict, 
                                      category_distribution: pd.Series,
                                      class_analysis: Dict, 
                                      school_analysis: Dict) -> str:
        """
        Akademik başarı sonuçlarını yorumlar
        """
        interpretation = []
        
        # Genel başarı durumu
        if 'overall_score' in basic_stats:
            mean_score = basic_stats['overall_score']['mean']
            interpretation.append(f"Öğrencilerin ortalama başarı puanı {mean_score:.2f}'dir.")
        
        # Başarı dağılımı
        if len(category_distribution) > 0:
            highest_category = category_distribution.index[0]
            highest_percentage = category_distribution.iloc[0] / category_distribution.sum() * 100
            interpretation.append(f"Öğrencilerin %{highest_percentage:.1f}'i {highest_category.lower()} başarı kategorisindedir.")
        
        # Sınıf karşılaştırması
        if class_analysis and 'statistical_test' in class_analysis:
            if class_analysis['statistical_test'].get('significant', False):
                interpretation.append("Sınıflar arasında başarı açısından anlamlı farklar bulunmuştur.")
            else:
                interpretation.append("Sınıflar arasında başarı açısından anlamlı fark bulunmamıştır.")
        
        # Okul karşılaştırması
        if school_analysis and 'statistical_test' in school_analysis:
            if school_analysis['statistical_test'].get('significant', False):
                interpretation.append("Okullar arasında başarı açısından anlamlı farklar bulunmuştur.")
            else:
                interpretation.append("Okullar arasında başarı açısından anlamlı fark bulunmamıştır.")
        
        return " ".join(interpretation)
    
    def _interpret_learning_outcomes(self, overall_analysis: Dict, 
                                   effect_sizes: Dict, 
                                   intervention_analysis: Dict) -> str:
        """
        Öğrenme çıktıları sonuçlarını yorumlar
        """
        interpretation = []
        
        # Genel kazanım
        mean_gain = overall_analysis['mean_gain']
        improvement_rate = overall_analysis['improvement_rate']
        
        interpretation.append(f"Ortalama öğrenme kazanımı {mean_gain:.2f} puandır.")
        interpretation.append(f"Öğrencilerin %{improvement_rate:.1f}'inde gelişim gözlenmiştir.")
        
        # Etki büyüklüğü
        if effect_sizes:
            avg_effect_size = np.mean(list(effect_sizes.values()))
            if avg_effect_size >= 0.8:
                effect_interpretation = "büyük"
            elif avg_effect_size >= 0.5:
                effect_interpretation = "orta"
            elif avg_effect_size >= 0.2:
                effect_interpretation = "küçük"
            else:
                effect_interpretation = "ihmal edilebilir"
            
            interpretation.append(f"Ortalama etki büyüklüğü {avg_effect_size:.2f} olup {effect_interpretation} düzeydedir.")
        
        # Müdahale etkisi
        if intervention_analysis and 'statistical_test' in intervention_analysis:
            if intervention_analysis['statistical_test'].get('significant', False):
                interpretation.append("Farklı müdahale yöntemleri arasında anlamlı farklar bulunmuştur.")
            else:
                interpretation.append("Farklı müdahale yöntemleri arasında anlamlı fark bulunmamıştır.")
        
        return " ".join(interpretation)