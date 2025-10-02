"""
Hipotez Testleri Sınıfı

Bu modül çeşitli istatistiksel hipotez testlerini gerçekleştirmek için kullanılır.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, wilcoxon, kruskal
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class HypothesisTester:
    """
    Çeşitli hipotez testlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        HypothesisTester sınıfını başlatır
        
        Args:
            data: Test edilecek veri seti
        """
        self.data = data.copy()
        self.results = {}
        
    def one_sample_t_test(self, column: str, population_mean: float, 
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Tek örneklem t-testi gerçekleştirir
        
        Args:
            column: Test edilecek sütun
            population_mean: Popülasyon ortalaması
            alpha: Anlamlılık düzeyi
            
        Returns:
            Test sonuçları
        """
        try:
            data_col = self.data[column].dropna()
            
            # Normallik testi
            _, normality_p = stats.shapiro(data_col)
            
            # t-testi
            t_stat, p_value = stats.ttest_1samp(data_col, population_mean)
            
            # Etki büyüklüğü (Cohen's d)
            cohen_d = (data_col.mean() - population_mean) / data_col.std()
            
            # Güven aralığı
            n = len(data_col)
            se = data_col.std() / np.sqrt(n)
            t_critical = stats.t.ppf(1 - alpha/2, n-1)
            ci_lower = data_col.mean() - t_critical * se
            ci_upper = data_col.mean() + t_critical * se
            
            result = {
                'test_type': 'One Sample T-Test',
                'column': column,
                'sample_mean': float(data_col.mean()),
                'population_mean': population_mean,
                'sample_size': n,
                'sample_std': float(data_col.std()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'cohen_d': float(cohen_d),
                'effect_size_interpretation': self._interpret_cohen_d(cohen_d),
                'confidence_interval': (float(ci_lower), float(ci_upper)),
                'normality_test': {
                    'shapiro_p_value': float(normality_p),
                    'is_normal': normality_p > 0.05
                },
                'interpretation': self._interpret_one_sample_t_test(
                    data_col.mean(), population_mean, p_value, alpha, cohen_d
                )
            }
            
            self.results['one_sample_t_test'] = result
            return result
            
        except Exception as e:
            return {'error': f'Tek örneklem t-testi hatası: {str(e)}'}
    
    def two_sample_t_test(self, column1: str, column2: str = None, 
                         group_column: str = None, equal_var: bool = True,
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        İki örneklem t-testi gerçekleştirir
        
        Args:
            column1: İlk sütun veya test edilecek sütun
            column2: İkinci sütun (opsiyonel)
            group_column: Grup sütunu (opsiyonel)
            equal_var: Varyansların eşit olup olmadığı
            alpha: Anlamlılık düzeyi
            
        Returns:
            Test sonuçları
        """
        try:
            if group_column:
                # Grup sütunu kullanılıyorsa
                groups = self.data[group_column].unique()
                if len(groups) != 2:
                    return {'error': 'Grup sütunu tam olarak 2 grup içermelidir'}
                
                group1_data = self.data[self.data[group_column] == groups[0]][column1].dropna()
                group2_data = self.data[self.data[group_column] == groups[1]][column1].dropna()
                group1_name, group2_name = str(groups[0]), str(groups[1])
                
            else:
                # İki ayrı sütun kullanılıyorsa
                if column2 is None:
                    return {'error': 'column2 veya group_column belirtilmelidir'}
                
                group1_data = self.data[column1].dropna()
                group2_data = self.data[column2].dropna()
                group1_name, group2_name = column1, column2
            
            # Normallik testleri
            _, norm1_p = stats.shapiro(group1_data)
            _, norm2_p = stats.shapiro(group2_data)
            
            # Levene testi (varyans homojenliği)
            _, levene_p = stats.levene(group1_data, group2_data)
            
            # t-testi
            if equal_var and levene_p > 0.05:
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=True)
                test_variant = "Student's t-test"
            else:
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                test_variant = "Welch's t-test"
            
            # Etki büyüklüğü (Cohen's d)
            pooled_std = np.sqrt(((len(group1_data)-1)*group1_data.var() + 
                                (len(group2_data)-1)*group2_data.var()) / 
                               (len(group1_data) + len(group2_data) - 2))
            cohen_d = (group1_data.mean() - group2_data.mean()) / pooled_std
            
            result = {
                'test_type': 'Two Sample T-Test',
                'test_variant': test_variant,
                'group1_name': group1_name,
                'group2_name': group2_name,
                'group1_stats': {
                    'mean': float(group1_data.mean()),
                    'std': float(group1_data.std()),
                    'size': len(group1_data)
                },
                'group2_stats': {
                    'mean': float(group2_data.mean()),
                    'std': float(group2_data.std()),
                    'size': len(group2_data)
                },
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'cohen_d': float(cohen_d),
                'effect_size_interpretation': self._interpret_cohen_d(cohen_d),
                'normality_tests': {
                    'group1_shapiro_p': float(norm1_p),
                    'group2_shapiro_p': float(norm2_p),
                    'both_normal': norm1_p > 0.05 and norm2_p > 0.05
                },
                'levene_test': {
                    'p_value': float(levene_p),
                    'equal_variances': levene_p > 0.05
                },
                'interpretation': self._interpret_two_sample_t_test(
                    group1_data.mean(), group2_data.mean(), p_value, alpha, cohen_d
                )
            }
            
            self.results['two_sample_t_test'] = result
            return result
            
        except Exception as e:
            return {'error': f'İki örneklem t-testi hatası: {str(e)}'}
    
    def paired_t_test(self, column1: str, column2: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Eşleştirilmiş t-testi gerçekleştirir
        
        Args:
            column1: İlk sütun
            column2: İkinci sütun
            alpha: Anlamlılık düzeyi
            
        Returns:
            Test sonuçları
        """
        try:
            # Eksik değerleri çıkar
            data_clean = self.data[[column1, column2]].dropna()
            group1_data = data_clean[column1]
            group2_data = data_clean[column2]
            
            # Farkları hesapla
            differences = group1_data - group2_data
            
            # Normallik testi (farklar için)
            _, norm_p = stats.shapiro(differences)
            
            # Eşleştirilmiş t-testi
            t_stat, p_value = stats.ttest_rel(group1_data, group2_data)
            
            # Etki büyüklüğü
            cohen_d = differences.mean() / differences.std()
            
            # Güven aralığı (farklar için)
            n = len(differences)
            se = differences.std() / np.sqrt(n)
            t_critical = stats.t.ppf(1 - alpha/2, n-1)
            ci_lower = differences.mean() - t_critical * se
            ci_upper = differences.mean() + t_critical * se
            
            result = {
                'test_type': 'Paired T-Test',
                'column1': column1,
                'column2': column2,
                'sample_size': n,
                'mean_difference': float(differences.mean()),
                'std_difference': float(differences.std()),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'cohen_d': float(cohen_d),
                'effect_size_interpretation': self._interpret_cohen_d(cohen_d),
                'confidence_interval_difference': (float(ci_lower), float(ci_upper)),
                'normality_test_differences': {
                    'shapiro_p_value': float(norm_p),
                    'is_normal': norm_p > 0.05
                },
                'interpretation': self._interpret_paired_t_test(
                    differences.mean(), p_value, alpha, cohen_d
                )
            }
            
            self.results['paired_t_test'] = result
            return result
            
        except Exception as e:
            return {'error': f'Eşleştirilmiş t-testi hatası: {str(e)}'}
    
    def one_way_anova(self, column: str, group_column: str, 
                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Tek yönlü ANOVA testi gerçekleştirir
        
        Args:
            column: Test edilecek sürekli değişken
            group_column: Grup değişkeni
            alpha: Anlamlılık düzeyi
            
        Returns:
            Test sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[[column, group_column]].dropna()
            groups = data_clean[group_column].unique()
            
            if len(groups) < 2:
                return {'error': 'En az 2 grup gereklidir'}
            
            # Grupları ayır
            group_data = [data_clean[data_clean[group_column] == group][column].values 
                         for group in groups]
            
            # Normallik testleri
            normality_tests = {}
            for i, group in enumerate(groups):
                group_values = group_data[i]
                if len(group_values) >= 3:
                    _, p_val = stats.shapiro(group_values)
                    normality_tests[str(group)] = {
                        'shapiro_p_value': float(p_val),
                        'is_normal': p_val > 0.05
                    }
            
            # Levene testi (varyans homojenliği)
            _, levene_p = stats.levene(*group_data)
            
            # ANOVA testi
            f_stat, p_value = stats.f_oneway(*group_data)
            
            # Eta kare (etki büyüklüğü)
            ss_between = sum([len(group) * (np.mean(group) - np.mean(data_clean[column]))**2 
                             for group in group_data])
            ss_total = sum([(x - np.mean(data_clean[column]))**2 
                           for x in data_clean[column]])
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Grup istatistikleri
            group_stats = {}
            for i, group in enumerate(groups):
                group_values = group_data[i]
                group_stats[str(group)] = {
                    'mean': float(np.mean(group_values)),
                    'std': float(np.std(group_values, ddof=1)),
                    'size': len(group_values)
                }
            
            result = {
                'test_type': 'One-Way ANOVA',
                'column': column,
                'group_column': group_column,
                'num_groups': len(groups),
                'total_sample_size': len(data_clean),
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'eta_squared': float(eta_squared),
                'effect_size_interpretation': self._interpret_eta_squared(eta_squared),
                'group_statistics': group_stats,
                'normality_tests': normality_tests,
                'levene_test': {
                    'p_value': float(levene_p),
                    'equal_variances': levene_p > 0.05
                },
                'interpretation': self._interpret_anova(p_value, alpha, eta_squared, len(groups))
            }
            
            # Post-hoc analiz (eğer anlamlıysa)
            if p_value < alpha and len(groups) > 2:
                result['post_hoc_needed'] = True
                result['post_hoc_recommendation'] = "Tukey HSD testi önerilir"
            
            self.results['one_way_anova'] = result
            return result
            
        except Exception as e:
            return {'error': f'ANOVA testi hatası: {str(e)}'}
    
    def chi_square_test(self, column1: str, column2: str, 
                       alpha: float = 0.05) -> Dict[str, Any]:
        """
        Ki-kare bağımsızlık testi gerçekleştirir
        
        Args:
            column1: İlk kategorik değişken
            column2: İkinci kategorik değişken
            alpha: Anlamlılık düzeyi
            
        Returns:
            Test sonuçları
        """
        try:
            # Çapraz tablo oluştur
            contingency_table = pd.crosstab(self.data[column1], self.data[column2])
            
            # Ki-kare testi
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Cramer's V (etki büyüklüğü)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            
            # Beklenen frekanslar kontrolü
            min_expected = np.min(expected)
            cells_below_5 = np.sum(expected < 5)
            total_cells = expected.size
            
            result = {
                'test_type': 'Chi-Square Test of Independence',
                'column1': column1,
                'column2': column2,
                'contingency_table': contingency_table.to_dict(),
                'expected_frequencies': expected.tolist(),
                'chi2_statistic': float(chi2_stat),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'cramers_v': float(cramers_v),
                'effect_size_interpretation': self._interpret_cramers_v(cramers_v),
                'assumptions_check': {
                    'min_expected_frequency': float(min_expected),
                    'cells_below_5': int(cells_below_5),
                    'total_cells': int(total_cells),
                    'percentage_below_5': float(cells_below_5 / total_cells * 100),
                    'assumptions_met': min_expected >= 5 and cells_below_5 / total_cells <= 0.2
                },
                'interpretation': self._interpret_chi_square(p_value, alpha, cramers_v)
            }
            
            # Fisher's exact test önerisi (2x2 tablo için)
            if contingency_table.shape == (2, 2) and min_expected < 5:
                result['fisher_exact_recommended'] = True
                
            self.results['chi_square_test'] = result
            return result
            
        except Exception as e:
            return {'error': f'Ki-kare testi hatası: {str(e)}'}
    
    def mann_whitney_u_test(self, column1: str, column2: str = None,
                           group_column: str = None, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Mann-Whitney U testi (Wilcoxon rank-sum test) gerçekleştirir
        
        Args:
            column1: İlk sütun veya test edilecek sütun
            column2: İkinci sütun (opsiyonel)
            group_column: Grup sütunu (opsiyonel)
            alpha: Anlamlılık düzeyi
            
        Returns:
            Test sonuçları
        """
        try:
            if group_column:
                groups = self.data[group_column].unique()
                if len(groups) != 2:
                    return {'error': 'Grup sütunu tam olarak 2 grup içermelidir'}
                
                group1_data = self.data[self.data[group_column] == groups[0]][column1].dropna()
                group2_data = self.data[self.data[group_column] == groups[1]][column1].dropna()
                group1_name, group2_name = str(groups[0]), str(groups[1])
                
            else:
                if column2 is None:
                    return {'error': 'column2 veya group_column belirtilmelidir'}
                
                group1_data = self.data[column1].dropna()
                group2_data = self.data[column2].dropna()
                group1_name, group2_name = column1, column2
            
            # Mann-Whitney U testi
            u_stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            
            # Etki büyüklüğü (r = Z / sqrt(N))
            n1, n2 = len(group1_data), len(group2_data)
            z_score = stats.norm.ppf(p_value/2)  # Yaklaşık Z skoru
            effect_size_r = abs(z_score) / np.sqrt(n1 + n2)
            
            result = {
                'test_type': 'Mann-Whitney U Test',
                'group1_name': group1_name,
                'group2_name': group2_name,
                'group1_stats': {
                    'median': float(group1_data.median()),
                    'mean_rank': float(group1_data.rank().mean()),
                    'size': n1
                },
                'group2_stats': {
                    'median': float(group2_data.median()),
                    'mean_rank': float(group2_data.rank().mean()),
                    'size': n2
                },
                'u_statistic': float(u_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'effect_size_r': float(effect_size_r),
                'effect_size_interpretation': self._interpret_effect_size_r(effect_size_r),
                'interpretation': self._interpret_mann_whitney(
                    group1_data.median(), group2_data.median(), p_value, alpha, effect_size_r
                )
            }
            
            self.results['mann_whitney_u_test'] = result
            return result
            
        except Exception as e:
            return {'error': f'Mann-Whitney U testi hatası: {str(e)}'}
    
    def wilcoxon_signed_rank_test(self, column1: str, column2: str, 
                                 alpha: float = 0.05) -> Dict[str, Any]:
        """
        Wilcoxon işaretli sıralar testi gerçekleştirir
        
        Args:
            column1: İlk sütun
            column2: İkinci sütun
            alpha: Anlamlılık düzeyi
            
        Returns:
            Test sonuçları
        """
        try:
            # Eksik değerleri çıkar
            data_clean = self.data[[column1, column2]].dropna()
            group1_data = data_clean[column1]
            group2_data = data_clean[column2]
            
            # Farkları hesapla
            differences = group1_data - group2_data
            
            # Sıfır farkları çıkar
            non_zero_diff = differences[differences != 0]
            
            if len(non_zero_diff) == 0:
                return {'error': 'Tüm farklar sıfır, test gerçekleştirilemez'}
            
            # Wilcoxon testi
            w_stat, p_value = wilcoxon(non_zero_diff)
            
            # Etki büyüklüğü
            z_score = stats.norm.ppf(p_value/2)
            effect_size_r = abs(z_score) / np.sqrt(len(non_zero_diff))
            
            result = {
                'test_type': 'Wilcoxon Signed-Rank Test',
                'column1': column1,
                'column2': column2,
                'sample_size': len(data_clean),
                'non_zero_differences': len(non_zero_diff),
                'median_difference': float(differences.median()),
                'w_statistic': float(w_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'effect_size_r': float(effect_size_r),
                'effect_size_interpretation': self._interpret_effect_size_r(effect_size_r),
                'interpretation': self._interpret_wilcoxon(
                    differences.median(), p_value, alpha, effect_size_r
                )
            }
            
            self.results['wilcoxon_signed_rank_test'] = result
            return result
            
        except Exception as e:
            return {'error': f'Wilcoxon testi hatası: {str(e)}'}
    
    def kruskal_wallis_test(self, column: str, group_column: str, 
                           alpha: float = 0.05) -> Dict[str, Any]:
        """
        Kruskal-Wallis H testi gerçekleştirir
        
        Args:
            column: Test edilecek sürekli değişken
            group_column: Grup değişkeni
            alpha: Anlamlılık düzeyi
            
        Returns:
            Test sonuçları
        """
        try:
            # Veriyi hazırla
            data_clean = self.data[[column, group_column]].dropna()
            groups = data_clean[group_column].unique()
            
            if len(groups) < 2:
                return {'error': 'En az 2 grup gereklidir'}
            
            # Grupları ayır
            group_data = [data_clean[data_clean[group_column] == group][column].values 
                         for group in groups]
            
            # Kruskal-Wallis testi
            h_stat, p_value = kruskal(*group_data)
            
            # Eta kare (etki büyüklüğü)
            n = len(data_clean)
            eta_squared = (h_stat - len(groups) + 1) / (n - len(groups))
            
            # Grup istatistikleri
            group_stats = {}
            for i, group in enumerate(groups):
                group_values = group_data[i]
                group_stats[str(group)] = {
                    'median': float(np.median(group_values)),
                    'mean_rank': float(np.mean(stats.rankdata(np.concatenate(group_data))[
                        sum(len(g) for g in group_data[:i]):sum(len(g) for g in group_data[:i+1])
                    ])),
                    'size': len(group_values)
                }
            
            result = {
                'test_type': 'Kruskal-Wallis H Test',
                'column': column,
                'group_column': group_column,
                'num_groups': len(groups),
                'total_sample_size': n,
                'h_statistic': float(h_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'eta_squared': float(eta_squared),
                'effect_size_interpretation': self._interpret_eta_squared(eta_squared),
                'group_statistics': group_stats,
                'interpretation': self._interpret_kruskal_wallis(p_value, alpha, eta_squared, len(groups))
            }
            
            # Post-hoc analiz önerisi
            if p_value < alpha and len(groups) > 2:
                result['post_hoc_needed'] = True
                result['post_hoc_recommendation'] = "Dunn testi önerilir"
            
            self.results['kruskal_wallis_test'] = result
            return result
            
        except Exception as e:
            return {'error': f'Kruskal-Wallis testi hatası: {str(e)}'}
    
    def _interpret_cohen_d(self, cohen_d: float) -> str:
        """Cohen's d etki büyüklüğünü yorumlar"""
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            return "Çok küçük etki"
        elif abs_d < 0.5:
            return "Küçük etki"
        elif abs_d < 0.8:
            return "Orta etki"
        else:
            return "Büyük etki"
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Eta kare etki büyüklüğünü yorumlar"""
        if eta_squared < 0.01:
            return "Çok küçük etki"
        elif eta_squared < 0.06:
            return "Küçük etki"
        elif eta_squared < 0.14:
            return "Orta etki"
        else:
            return "Büyük etki"
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Cramer's V etki büyüklüğünü yorumlar"""
        if cramers_v < 0.1:
            return "Çok küçük etki"
        elif cramers_v < 0.3:
            return "Küçük etki"
        elif cramers_v < 0.5:
            return "Orta etki"
        else:
            return "Büyük etki"
    
    def _interpret_effect_size_r(self, r: float) -> str:
        """r etki büyüklüğünü yorumlar"""
        if r < 0.1:
            return "Çok küçük etki"
        elif r < 0.3:
            return "Küçük etki"
        elif r < 0.5:
            return "Orta etki"
        else:
            return "Büyük etki"
    
    def _interpret_one_sample_t_test(self, sample_mean: float, pop_mean: float, 
                                   p_value: float, alpha: float, cohen_d: float) -> str:
        """Tek örneklem t-testi sonucunu yorumlar"""
        direction = "büyük" if sample_mean > pop_mean else "küçük"
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        return (f"Örneklem ortalaması ({sample_mean:.3f}) popülasyon ortalamasından ({pop_mean:.3f}) "
                f"istatistiksel olarak {significance} şekilde {direction}dür (p={p_value:.3f}). "
                f"Etki büyüklüğü {self._interpret_cohen_d(cohen_d).lower()}dir.")
    
    def _interpret_two_sample_t_test(self, mean1: float, mean2: float, 
                                   p_value: float, alpha: float, cohen_d: float) -> str:
        """İki örneklem t-testi sonucunu yorumlar"""
        direction = "büyük" if mean1 > mean2 else "küçük"
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        return (f"İki grup arasında istatistiksel olarak {significance} bir fark vardır (p={p_value:.3f}). "
                f"Birinci grubun ortalaması ({mean1:.3f}) ikinci gruptan ({mean2:.3f}) {direction}dür. "
                f"Etki büyüklüğü {self._interpret_cohen_d(cohen_d).lower()}dir.")
    
    def _interpret_paired_t_test(self, mean_diff: float, p_value: float, 
                               alpha: float, cohen_d: float) -> str:
        """Eşleştirilmiş t-testi sonucunu yorumlar"""
        direction = "artış" if mean_diff > 0 else "azalış"
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        return (f"İki ölçüm arasında istatistiksel olarak {significance} bir {direction} vardır "
                f"(ortalama fark: {mean_diff:.3f}, p={p_value:.3f}). "
                f"Etki büyüklüğü {self._interpret_cohen_d(cohen_d).lower()}dir.")
    
    def _interpret_anova(self, p_value: float, alpha: float, eta_squared: float, num_groups: int) -> str:
        """ANOVA sonucunu yorumlar"""
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        interpretation = (f"{num_groups} grup arasında istatistiksel olarak {significance} bir fark vardır "
                         f"(p={p_value:.3f}). Etki büyüklüğü {self._interpret_eta_squared(eta_squared).lower()}dir.")
        
        if p_value < alpha and num_groups > 2:
            interpretation += " Hangi gruplar arasında fark olduğunu belirlemek için post-hoc analiz gereklidir."
        
        return interpretation
    
    def _interpret_chi_square(self, p_value: float, alpha: float, cramers_v: float) -> str:
        """Ki-kare testi sonucunu yorumlar"""
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        return (f"İki kategorik değişken arasında istatistiksel olarak {significance} bir ilişki vardır "
                f"(p={p_value:.3f}). İlişkinin gücü {self._interpret_cramers_v(cramers_v).lower()}dir.")
    
    def _interpret_mann_whitney(self, median1: float, median2: float, 
                              p_value: float, alpha: float, effect_size_r: float) -> str:
        """Mann-Whitney U testi sonucunu yorumlar"""
        direction = "büyük" if median1 > median2 else "küçük"
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        return (f"İki grup arasında istatistiksel olarak {significance} bir fark vardır (p={p_value:.3f}). "
                f"Birinci grubun medyanı ({median1:.3f}) ikinci gruptan ({median2:.3f}) {direction}dür. "
                f"Etki büyüklüğü {self._interpret_effect_size_r(effect_size_r).lower()}dir.")
    
    def _interpret_wilcoxon(self, median_diff: float, p_value: float, 
                          alpha: float, effect_size_r: float) -> str:
        """Wilcoxon testi sonucunu yorumlar"""
        direction = "artış" if median_diff > 0 else "azalış"
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        return (f"İki ölçüm arasında istatistiksel olarak {significance} bir {direction} vardır "
                f"(medyan fark: {median_diff:.3f}, p={p_value:.3f}). "
                f"Etki büyüklüğü {self._interpret_effect_size_r(effect_size_r).lower()}dir.")
    
    def _interpret_kruskal_wallis(self, p_value: float, alpha: float, 
                                eta_squared: float, num_groups: int) -> str:
        """Kruskal-Wallis testi sonucunu yorumlar"""
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        interpretation = (f"{num_groups} grup arasında istatistiksel olarak {significance} bir fark vardır "
                         f"(p={p_value:.3f}). Etki büyüklüğü {self._interpret_eta_squared(eta_squared).lower()}dir.")
        
        if p_value < alpha and num_groups > 2:
            interpretation += " Hangi gruplar arasında fark olduğunu belirlemek için post-hoc analiz gereklidir."
        
        return interpretation
    
    def get_test_summary(self) -> Dict[str, Any]:
        """
        Gerçekleştirilen tüm testlerin özetini döndürür
        
        Returns:
            Test özetleri
        """
        if not self.results:
            return {'message': 'Henüz test gerçekleştirilmedi'}
        
        summary = {
            'total_tests': len(self.results),
            'tests_performed': list(self.results.keys()),
            'significant_tests': [],
            'non_significant_tests': [],
            'test_details': self.results
        }
        
        for test_name, result in self.results.items():
            if 'error' not in result:
                if result.get('is_significant', False):
                    summary['significant_tests'].append(test_name)
                else:
                    summary['non_significant_tests'].append(test_name)
        
        return summary