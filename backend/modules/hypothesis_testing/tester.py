"""
VeriVio Kapsamlı Hipotez Testleri Modülü
Tüm istatistiksel hipotez testleri ve otomatik yorumlama
Kullanıcı dostu açıklamalar ve detaylı analiz
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    chi2_contingency, fisher_exact, mannwhitneyu, wilcoxon, kruskal,
    bartlett, fligner, anderson, jarque_bera, normaltest, levene,
    pearsonr, spearmanr, kendalltau, pointbiserialr, 
    friedmanchisquare, ranksums, mood, ansari,
    combine_pvalues, binomtest, poisson_means_test
)
from statsmodels.stats.diagnostic import lilliefors, het_white, het_breuschpagan
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.weightstats import ttest_ind, ztest
from statsmodels.stats.power import ttest_power
from statsmodels.stats.meta_analysis import combine_effects
import pingouin as pg
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ComprehensiveHypothesisTester:
    """Kapsamlı hipotez testleri sınıfı"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}
        self.test_history = []
        
        # Etki büyüklüğü yorumlama sözlükleri
        self.cohen_d_interpretation = {
            (0, 0.2): "Çok küçük etki",
            (0.2, 0.5): "Küçük etki", 
            (0.5, 0.8): "Orta etki",
            (0.8, 1.2): "Büyük etki",
            (1.2, float('inf')): "Çok büyük etki"
        }
        
        self.eta_squared_interpretation = {
            (0, 0.01): "Çok küçük etki",
            (0.01, 0.06): "Küçük etki",
            (0.06, 0.14): "Orta etki", 
            (0.14, float('inf')): "Büyük etki"
        }
    
    def _interpret_effect_size(self, effect_size: float, effect_type: str) -> str:
        """Etki büyüklüğünü yorumla"""
        if effect_type == "cohen_d":
            interpretation_dict = self.cohen_d_interpretation
        elif effect_type == "eta_squared":
            interpretation_dict = self.eta_squared_interpretation
        else:
            return "Bilinmeyen etki türü"
        
        abs_effect = abs(effect_size)
        for (lower, upper), interpretation in interpretation_dict.items():
            if lower <= abs_effect < upper:
                return interpretation
        return "Çok büyük etki"
    
    def _generate_interpretation(self, test_name: str, p_value: float, alpha: float, 
                               effect_size: float = None, additional_info: Dict = None) -> str:
        """Otomatik test yorumu oluştur"""
        is_significant = p_value < alpha
        
        base_interpretation = f"{test_name} sonuçları:\n"
        
        if is_significant:
            base_interpretation += f"• p-değeri ({p_value:.4f}) < α ({alpha}), sonuç istatistiksel olarak anlamlıdır.\n"
            base_interpretation += "• Null hipotez reddedilir, alternatif hipotez kabul edilir.\n"
        else:
            base_interpretation += f"• p-değeri ({p_value:.4f}) ≥ α ({alpha}), sonuç istatistiksel olarak anlamlı değildir.\n"
            base_interpretation += "• Null hipotez reddedilemez.\n"
        
        if effect_size is not None:
            effect_interpretation = self._interpret_effect_size(effect_size, "cohen_d")
            base_interpretation += f"• Etki büyüklüğü: {effect_size:.3f} ({effect_interpretation})\n"
        
        if additional_info:
            for key, value in additional_info.items():
                base_interpretation += f"• {key}: {value}\n"
        
        return base_interpretation
    
    def run_comprehensive_test(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Kapsamlı test çalıştırıcı - test türüne göre uygun metodu çağırır"""
        try:
            test_type = params.get('test_type', '')
            
            if test_type == 't_test_one_sample':
                return self.one_sample_t_test(
                    column=params.get('column', params.get('columns', [''])[0]),
                    population_mean=params.get('test_value', 0),
                    alpha=params.get('alpha', 0.05),
                    alternative=params.get('alternative', 'two-sided')
                )
            
            elif test_type == 't_test_two_sample':
                return self.two_sample_t_test(
                    column1=params.get('column', params.get('columns', [''])[0]),
                    group_column=params.get('group_column'),
                    alpha=params.get('alpha', 0.05),
                    alternative=params.get('alternative', 'two-sided')
                )
            
            elif test_type == 'anova_one_way':
                return self.one_way_anova(
                    dependent_var=params.get('dependent_var', params.get('columns', [''])[0]),
                    independent_var=params.get('independent_var', params.get('group_column')),
                    alpha=params.get('alpha', 0.05),
                    post_hoc=params.get('post_hoc', True)
                )
            
            elif test_type == 'kruskal_wallis':
                return self.kruskal_wallis_test(
                    dependent_var=params.get('dependent_var', params.get('columns', [''])[0]),
                    independent_var=params.get('independent_var', params.get('group_column')),
                    alpha=params.get('alpha', 0.05),
                    post_hoc=params.get('post_hoc', True)
                )
            
            elif test_type == 'chi_square':
                return self.chi_square_test(
                    column1=params.get('column1', params.get('columns', [''])[0]),
                    column2=params.get('column2', params.get('columns', ['', ''])[1] if len(params.get('columns', [])) > 1 else ''),
                    alpha=params.get('alpha', 0.05)
                )
            
            elif test_type == 'correlation':
                return self.correlation_test(
                    column1=params.get('column1', params.get('columns', [''])[0]),
                    column2=params.get('column2', params.get('columns', ['', ''])[1] if len(params.get('columns', [])) > 1 else ''),
                    method=params.get('method', 'pearson'),
                    alpha=params.get('alpha', 0.05)
                )
            
            elif test_type == 't_test_paired':
                return self.paired_t_test(
                    col1=params.get('paired_col_1'),
                    col2=params.get('paired_col_2'),
                    alpha=params.get('alpha', 0.05)
                )
            
            elif test_type == 'manova':
                return self.manova_test(
                    dependent_vars=params.get('dependent_columns', []),
                    independent_formula=params.get('independent_formula', ''),
                    alpha=params.get('alpha', 0.05)
                )
            
            elif test_type == 'mixed_anova':
                return self.mixed_anova_test(
                    dv=params.get('dv_column'),
                    within=params.get('within_column'),
                    between=params.get('between_column'),
                    subject=params.get('subject_column'),
                    alpha=params.get('alpha', 0.05)
                )
            
            elif test_type == 'wilcoxon_signed_rank':
                return self.wilcoxon_signed_rank_test(
                    col1=params.get('paired_col_1'),
                    col2=params.get('paired_col_2'),
                    alpha=params.get('alpha', 0.05)
                )
            
            else:
                return {'error': f'Desteklenmeyen test türü: {test_type}'}
                
        except Exception as e:
            logger.error(f"Kapsamlı test hatası: {str(e)}")
            return {'error': f'Test çalıştırma hatası: {str(e)}'}
    
    def one_sample_t_test(self, column: str, population_mean: float, 
                         alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """Tek örneklem t-testi"""
        try:
            data_col = self.data[column].dropna()
            n = len(data_col)
            
            if n < 2:
                return {'error': 'Tek örneklem t-testi için en az 2 gözlem gerekli'}
            
            # Normallik kontrolü
            if n <= 5000:
                _, normality_p = stats.shapiro(data_col)
                is_normal = normality_p > 0.05
            else:
                _, normality_p = stats.normaltest(data_col)
                is_normal = normality_p > 0.05
            
            # t-testi
            if alternative == 'two-sided':
                t_stat, p_value = stats.ttest_1samp(data_col, population_mean)
            elif alternative == 'greater':
                t_stat, p_value_two = stats.ttest_1samp(data_col, population_mean)
                p_value = p_value_two / 2 if t_stat > 0 else 1 - p_value_two / 2
            elif alternative == 'less':
                t_stat, p_value_two = stats.ttest_1samp(data_col, population_mean)
                p_value = p_value_two / 2 if t_stat < 0 else 1 - p_value_two / 2
            
            # Etki büyüklüğü (Cohen's d)
            cohen_d = (data_col.mean() - population_mean) / data_col.std()
            
            # Güven aralığı
            se = data_col.std() / np.sqrt(n)
            t_critical = stats.t.ppf(1 - alpha/2, n-1)
            ci_lower = data_col.mean() - t_critical * se
            ci_upper = data_col.mean() + t_critical * se
            
            # Güç analizi
            power = ttest_power(cohen_d, n, alpha, alternative)
            
            result = {
                'test_type': 'Tek Örneklem T-Testi',
                'column': column,
                'sample_mean': float(data_col.mean()),
                'population_mean': population_mean,
                'sample_size': n,
                'sample_std': float(data_col.std()),
                't_statistic': float(t_stat),
                'degrees_of_freedom': n - 1,
                'p_value': float(p_value),
                'alpha': alpha,
                'alternative': alternative,
                'is_significant': p_value < alpha,
                'cohen_d': float(cohen_d),
                'effect_size_interpretation': self._interpret_effect_size(cohen_d, "cohen_d"),
                'confidence_interval': (float(ci_lower), float(ci_upper)),
                'statistical_power': float(power),
                'normality_assumption': {
                    'p_value': float(normality_p),
                    'is_satisfied': is_normal,
                    'recommendation': 'Parametrik test uygun' if is_normal else 'Non-parametrik test önerilir'
                },
                'interpretation': self._generate_interpretation(
                    'Tek Örneklem T-Testi', p_value, alpha, cohen_d,
                    {'Örneklem ortalaması': f'{data_col.mean():.3f}',
                     'Test edilen değer': f'{population_mean:.3f}'}
                )
            }
            
            self.results['one_sample_t_test'] = result
            self.test_history.append(result)
            logger.info(f"Tek örneklem t-testi tamamlandı: {column}")
            
            return result
            
        except Exception as e:
            logger.error(f"Tek örneklem t-testi hatası: {str(e)}")
            return {'error': f'Tek örneklem t-testi hatası: {str(e)}'}
    
    def two_sample_t_test(self, column1: str, column2: str = None, 
                         group_column: str = None, equal_var: bool = True,
                         alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """İki örneklem t-testi"""
        try:
            if column2 is not None:
                # İki ayrı sütun
                group1 = self.data[column1].dropna()
                group2 = self.data[column2].dropna()
                group_names = [column1, column2]
            elif group_column is not None:
                # Gruplandırılmış veri
                groups = self.data.groupby(group_column)[column1].apply(lambda x: x.dropna())
                if len(groups) != 2:
                    return {'error': 'İki örneklem t-testi için tam olarak 2 grup gerekli'}
                group_names = list(groups.index)
                group1, group2 = groups.iloc[0], groups.iloc[1]
            else:
                return {'error': 'column2 veya group_column belirtilmelidir'}
            
            n1, n2 = len(group1), len(group2)
            
            if n1 < 2 or n2 < 2:
                return {'error': 'Her grup için en az 2 gözlem gerekli'}
            
            # Normallik testleri
            if n1 <= 5000:
                _, norm_p1 = stats.shapiro(group1)
            else:
                _, norm_p1 = stats.normaltest(group1)
                
            if n2 <= 5000:
                _, norm_p2 = stats.shapiro(group2)
            else:
                _, norm_p2 = stats.normaltest(group2)
            
            # Varyans homojenliği testi
            levene_stat, levene_p = levene(group1, group2)
            equal_variances = levene_p > 0.05
            
            # t-testi
            if equal_var and equal_variances:
                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
                test_variant = "Student's t-test"
            else:
                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                test_variant = "Welch's t-test"
            
            # Tek yönlü test için p-değeri düzeltmesi
            if alternative != 'two-sided':
                if alternative == 'greater':
                    p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
                elif alternative == 'less':
                    p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
            
            # Etki büyüklüğü (Cohen's d)
            pooled_std = np.sqrt(((n1-1)*group1.var() + (n2-1)*group2.var()) / (n1+n2-2))
            cohen_d = (group1.mean() - group2.mean()) / pooled_std
            
            # Güven aralığı
            se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
            t_critical = stats.t.ppf(1 - alpha/2, df)
            mean_diff = group1.mean() - group2.mean()
            ci_lower = mean_diff - t_critical * se_diff
            ci_upper = mean_diff + t_critical * se_diff
            
            # Güç analizi
            power = ttest_power(cohen_d, min(n1, n2), alpha, alternative)
            
            result = {
                'test_type': 'İki Örneklem T-Testi',
                'test_variant': test_variant,
                'group_names': group_names,
                'group1_stats': {
                    'name': group_names[0],
                    'mean': float(group1.mean()),
                    'std': float(group1.std()),
                    'size': n1
                },
                'group2_stats': {
                    'name': group_names[1],
                    'mean': float(group2.mean()),
                    'std': float(group2.std()),
                    'size': n2
                },
                'mean_difference': float(mean_diff),
                't_statistic': float(t_stat),
                'degrees_of_freedom': df,
                'p_value': float(p_value),
                'alpha': alpha,
                'alternative': alternative,
                'is_significant': p_value < alpha,
                'cohen_d': float(cohen_d),
                'effect_size_interpretation': self._interpret_effect_size(cohen_d, "cohen_d"),
                'confidence_interval_diff': (float(ci_lower), float(ci_upper)),
                'statistical_power': float(power),
                'assumptions': {
                    'normality_group1': {
                        'p_value': float(norm_p1),
                        'is_satisfied': norm_p1 > 0.05
                    },
                    'normality_group2': {
                        'p_value': float(norm_p2),
                        'is_satisfied': norm_p2 > 0.05
                    },
                    'equal_variances': {
                        'levene_statistic': float(levene_stat),
                        'levene_p_value': float(levene_p),
                        'is_satisfied': equal_variances,
                        'test_used': test_variant
                    }
                },
                'interpretation': self._generate_interpretation(
                    'İki Örneklem T-Testi', p_value, alpha, cohen_d,
                    {f'{group_names[0]} ortalaması': f'{group1.mean():.3f}',
                     f'{group_names[1]} ortalaması': f'{group2.mean():.3f}',
                     'Ortalama farkı': f'{mean_diff:.3f}'}
                )
            }
            
            self.results['two_sample_t_test'] = result
            self.test_history.append(result)
            logger.info(f"İki örneklem t-testi tamamlandı: {group_names}")
            
            return result
            
        except Exception as e:
            logger.error(f"İki örneklem t-testi hatası: {str(e)}")
            return {'error': f'İki örneklem t-testi hatası: {str(e)}'}
    
    def paired_t_test(self, column1: str, column2: str, alpha: float = 0.05,
                     alternative: str = 'two-sided') -> Dict[str, Any]:
        """Eşleştirilmiş t-testi"""
        try:
            # Eşleştirilmiş verileri al
            paired_data = self.data[[column1, column2]].dropna()
            
            if len(paired_data) < 2:
                return {'error': 'Eşleştirilmiş t-test için en az 2 çift gözlem gerekli'}
            
            group1 = paired_data[column1]
            group2 = paired_data[column2]
            differences = group1 - group2
            
            # Farkların normallik testi
            if len(differences) <= 5000:
                _, norm_p = stats.shapiro(differences)
            else:
                _, norm_p = stats.normaltest(differences)
            
            # Eşleştirilmiş t-testi
            t_stat, p_value = stats.ttest_rel(group1, group2)
            
            # Tek yönlü test için p-değeri düzeltmesi
            if alternative != 'two-sided':
                if alternative == 'greater':
                    p_value = p_value / 2 if t_stat > 0 else 1 - p_value / 2
                elif alternative == 'less':
                    p_value = p_value / 2 if t_stat < 0 else 1 - p_value / 2
            
            # Etki büyüklüğü
            cohen_d = differences.mean() / differences.std()
            
            # Güven aralığı
            n = len(differences)
            se = differences.std() / np.sqrt(n)
            t_critical = stats.t.ppf(1 - alpha/2, n-1)
            ci_lower = differences.mean() - t_critical * se
            ci_upper = differences.mean() + t_critical * se
            
            # Güç analizi
            power = ttest_power(cohen_d, n, alpha, alternative)
            
            result = {
                'test_type': 'Eşleştirilmiş T-Testi',
                'column1': column1,
                'column2': column2,
                'sample_size': n,
                'group1_mean': float(group1.mean()),
                'group2_mean': float(group2.mean()),
                'mean_difference': float(differences.mean()),
                'difference_std': float(differences.std()),
                't_statistic': float(t_stat),
                'degrees_of_freedom': n - 1,
                'p_value': float(p_value),
                'alpha': alpha,
                'alternative': alternative,
                'is_significant': p_value < alpha,
                'cohen_d': float(cohen_d),
                'effect_size_interpretation': self._interpret_effect_size(cohen_d, "cohen_d"),
                'confidence_interval_diff': (float(ci_lower), float(ci_upper)),
                'statistical_power': float(power),
                'normality_assumption': {
                    'differences_normality_p': float(norm_p),
                    'is_satisfied': norm_p > 0.05,
                    'recommendation': 'Parametrik test uygun' if norm_p > 0.05 else 'Wilcoxon signed-rank test önerilir'
                },
                'interpretation': self._generate_interpretation(
                    'Eşleştirilmiş T-Testi', p_value, alpha, cohen_d,
                    {f'{column1} ortalaması': f'{group1.mean():.3f}',
                     f'{column2} ortalaması': f'{group2.mean():.3f}',
                     'Ortalama fark': f'{differences.mean():.3f}'}
                )
            }
            
            self.results['paired_t_test'] = result
            self.test_history.append(result)
            logger.info(f"Eşleştirilmiş t-testi tamamlandı: {column1} vs {column2}")
            
            return result
            
        except Exception as e:
            logger.error(f"Eşleştirilmiş t-testi hatası: {str(e)}")
            return {'error': f'Eşleştirilmiş t-testi hatası: {str(e)}'}
    
    def one_way_anova(self, dependent_var: str, independent_var: str, 
                     alpha: float = 0.05, post_hoc: bool = True) -> Dict[str, Any]:
        """Tek yönlü ANOVA"""
        try:
            # Debug: Check if columns exist
            logger.info(f"ANOVA Debug - Data columns: {list(self.data.columns)}")
            logger.info(f"ANOVA Debug - Looking for dependent_var: {dependent_var}, independent_var: {independent_var}")
            
            if dependent_var not in self.data.columns:
                return {'error': f'Bağımlı değişken "{dependent_var}" veri setinde bulunamadı'}
            
            if independent_var not in self.data.columns:
                return {'error': f'Bağımsız değişken "{independent_var}" veri setinde bulunamadı'}
            
            # Grupları ayır
            groups = self.data.groupby(independent_var)[dependent_var].apply(lambda x: x.dropna())
            
            if len(groups) < 2:
                return {'error': 'ANOVA için en az 2 grup gerekli'}
            
            group_data = [group for group in groups if len(group) >= 2]
            group_names = [name for name, group in groups.items() if len(group) >= 2]
            
            if len(group_data) < 2:
                return {'error': 'Her grupta en az 2 gözlem gerekli'}
            
            # ANOVA
            f_stat, p_value = stats.f_oneway(*group_data)
            
            # Grup istatistikleri
            group_stats = []
            total_n = 0
            grand_mean = 0
            
            for i, (name, data) in enumerate(zip(group_names, group_data)):
                n = len(data)
                mean = data.mean()
                std = data.std()
                total_n += n
                grand_mean += mean * n
                
                group_stats.append({
                    'group': str(name),
                    'n': n,
                    'mean': float(mean),
                    'std': float(std),
                    'min': float(data.min()),
                    'max': float(data.max())
                })
            
            grand_mean /= total_n
            
            # Etki büyüklüğü (Eta squared)
            ss_between = sum([len(group) * (group.mean() - grand_mean)**2 for group in group_data])
            ss_total = sum([(group - grand_mean)**2 for group in group_data for value in group])
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Varsayım testleri
            # Normallik testi (her grup için)
            normality_tests = []
            for i, (name, data) in enumerate(zip(group_names, group_data)):
                if len(data) <= 5000:
                    _, norm_p = stats.shapiro(data)
                else:
                    _, norm_p = stats.normaltest(data)
                
                normality_tests.append({
                    'group': str(name),
                    'shapiro_p': float(norm_p),
                    'is_normal': norm_p > 0.05
                })
            
            # Varyans homojenliği (Levene testi)
            levene_stat, levene_p = levene(*group_data)
            
            # Post-hoc testler
            post_hoc_results = None
            if post_hoc and p_value < alpha:
                try:
                    # Tukey HSD
                    tukey_results = pg.pairwise_tukey(
                        data=self.data[[dependent_var, independent_var]].dropna(),
                        dv=dependent_var,
                        between=independent_var
                    )
                    
                    post_hoc_results = {
                        'tukey_hsd': tukey_results.to_dict('records')
                    }
                except:
                    post_hoc_results = {'error': 'Post-hoc testler hesaplanamadı'}
            
            result = {
                'test_type': 'Tek Yönlü ANOVA',
                'dependent_variable': dependent_var,
                'independent_variable': independent_var,
                'group_count': len(group_data),
                'total_sample_size': total_n,
                'f_statistic': float(f_stat),
                'degrees_of_freedom': (len(group_data) - 1, total_n - len(group_data)),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'eta_squared': float(eta_squared),
                'effect_size_interpretation': self._interpret_effect_size(eta_squared, "eta_squared"),
                'group_statistics': group_stats,
                'assumptions': {
                    'normality_tests': normality_tests,
                    'homogeneity_of_variance': {
                        'levene_statistic': float(levene_stat),
                        'levene_p_value': float(levene_p),
                        'is_satisfied': levene_p > 0.05
                    }
                },
                'post_hoc_tests': post_hoc_results,
                'interpretation': self._generate_interpretation(
                    'Tek Yönlü ANOVA', p_value, alpha, eta_squared,
                    {'Grup sayısı': len(group_data),
                     'Toplam örneklem': total_n,
                     'F istatistiği': f'{f_stat:.3f}'}
                )
            }
            
            self.results['one_way_anova'] = result
            self.test_history.append(result)
            logger.info(f"Tek yönlü ANOVA tamamlandı: {dependent_var} ~ {independent_var}")
            
            return result
            
        except Exception as e:
            logger.error(f"ANOVA hatası: {str(e)}")
            return {'error': f'ANOVA hatası: {str(e)}'}
    
    def chi_square_test(self, column1: str, column2: str = None, 
                       alpha: float = 0.05, correction: bool = True) -> Dict[str, Any]:
        """Ki-kare bağımsızlık testi"""
        try:
            if column2 is None:
                # Tek değişken için goodness of fit testi
                observed = self.data[column1].value_counts()
                expected = [len(self.data) / len(observed)] * len(observed)
                chi2_stat, p_value = stats.chisquare(observed, expected)
                
                result = {
                    'test_type': 'Ki-Kare Uyum İyiliği Testi',
                    'column': column1,
                    'categories': len(observed),
                    'chi2_statistic': float(chi2_stat),
                    'degrees_of_freedom': len(observed) - 1,
                    'p_value': float(p_value),
                    'alpha': alpha,
                    'is_significant': p_value < alpha,
                    'observed_frequencies': observed.to_dict(),
                    'interpretation': self._generate_interpretation(
                        'Ki-Kare Uyum İyiliği Testi', p_value, alpha
                    )
                }
            else:
                # İki değişken için bağımsızlık testi
                contingency_table = pd.crosstab(self.data[column1], self.data[column2])
                
                if contingency_table.size == 0:
                    return {'error': 'Çapraz tablo oluşturulamadı'}
                
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table, correction=correction)
                
                # Cramér's V (etki büyüklüğü)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                
                # Hücre katkıları
                residuals = (contingency_table - expected) / np.sqrt(expected)
                
                result = {
                    'test_type': 'Ki-Kare Bağımsızlık Testi',
                    'column1': column1,
                    'column2': column2,
                    'contingency_table': contingency_table.to_dict(),
                    'expected_frequencies': pd.DataFrame(expected, 
                                                       index=contingency_table.index,
                                                       columns=contingency_table.columns).to_dict(),
                    'chi2_statistic': float(chi2_stat),
                    'degrees_of_freedom': int(dof),
                    'p_value': float(p_value),
                    'alpha': alpha,
                    'is_significant': p_value < alpha,
                    'cramers_v': float(cramers_v),
                    'effect_size_interpretation': self._interpret_effect_size(cramers_v, "cramers_v"),
                    'standardized_residuals': residuals.to_dict(),
                    'yates_correction': correction,
                    'interpretation': self._generate_interpretation(
                        'Ki-Kare Bağımsızlık Testi', p_value, alpha, cramers_v,
                        {'Cramér\'s V': f'{cramers_v:.3f}',
                         'Serbestlik derecesi': dof}
                    )
                }
            
            self.results['chi_square_test'] = result
            self.test_history.append(result)
            logger.info(f"Ki-kare testi tamamlandı: {column1} vs {column2}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ki-kare testi hatası: {str(e)}")
            return {'error': f'Ki-kare testi hatası: {str(e)}'}
    
    def mann_whitney_u_test(self, column1: str, column2: str = None,
                           group_column: str = None, alpha: float = 0.05,
                           alternative: str = 'two-sided') -> Dict[str, Any]:
        """Mann-Whitney U testi (Wilcoxon rank-sum test)"""
        try:
            if column2 is not None:
                group1 = self.data[column1].dropna()
                group2 = self.data[column2].dropna()
                group_names = [column1, column2]
            elif group_column is not None:
                groups = self.data.groupby(group_column)[column1].apply(lambda x: x.dropna())
                if len(groups) != 2:
                    return {'error': 'Mann-Whitney U testi için tam olarak 2 grup gerekli'}
                group_names = list(groups.index)
                group1, group2 = groups.iloc[0], groups.iloc[1]
            else:
                return {'error': 'column2 veya group_column belirtilmelidir'}
            
            if len(group1) < 1 or len(group2) < 1:
                return {'error': 'Her grup için en az 1 gözlem gerekli'}
            
            # Mann-Whitney U testi
            u_stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
            
            # Etki büyüklüğü (r = Z / sqrt(N))
            n1, n2 = len(group1), len(group2)
            n_total = n1 + n2
            z_score = stats.norm.ppf(1 - p_value/2) if alternative == 'two-sided' else stats.norm.ppf(1 - p_value)
            effect_size_r = abs(z_score) / np.sqrt(n_total)
            
            # Rank-biserial correlation
            rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
            
            result = {
                'test_type': 'Mann-Whitney U Testi',
                'group_names': group_names,
                'group1_stats': {
                    'name': group_names[0],
                    'size': n1,
                    'median': float(group1.median()),
                    'mean_rank': float(stats.rankdata(np.concatenate([group1, group2]))[:n1].mean())
                },
                'group2_stats': {
                    'name': group_names[1],
                    'size': n2,
                    'median': float(group2.median()),
                    'mean_rank': float(stats.rankdata(np.concatenate([group1, group2]))[n1:].mean())
                },
                'u_statistic': float(u_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'alternative': alternative,
                'is_significant': p_value < alpha,
                'effect_size_r': float(effect_size_r),
                'rank_biserial_correlation': float(rank_biserial),
                'interpretation': self._generate_interpretation(
                    'Mann-Whitney U Testi', p_value, alpha, effect_size_r,
                    {f'{group_names[0]} medyanı': f'{group1.median():.3f}',
                     f'{group_names[1]} medyanı': f'{group2.median():.3f}',
                     'Non-parametrik test': 'Dağılım varsayımı gerektirmez'}
                )
            }
            
            self.results['mann_whitney_u_test'] = result
            self.test_history.append(result)
            logger.info(f"Mann-Whitney U testi tamamlandı: {group_names}")
            
            return result
            
        except Exception as e:
            logger.error(f"Mann-Whitney U testi hatası: {str(e)}")
            return {'error': f'Mann-Whitney U testi hatası: {str(e)}'}
    
    def wilcoxon_signed_rank_test(self, column1: str, column2: str, 
                                 alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """Wilcoxon işaretli sıralar testi"""
        try:
            paired_data = self.data[[column1, column2]].dropna()
            
            if len(paired_data) < 1:
                return {'error': 'Wilcoxon testi için en az 1 çift gözlem gerekli'}
            
            group1 = paired_data[column1]
            group2 = paired_data[column2]
            differences = group1 - group2
            
            # Sıfır farkları çıkar
            non_zero_diff = differences[differences != 0]
            
            if len(non_zero_diff) < 1:
                return {'error': 'Tüm farklar sıfır, test yapılamaz'}
            
            # Wilcoxon işaretli sıralar testi
            w_stat, p_value = wilcoxon(non_zero_diff, alternative=alternative)
            
            # Etki büyüklüğü
            n = len(non_zero_diff)
            z_score = stats.norm.ppf(1 - p_value/2) if alternative == 'two-sided' else stats.norm.ppf(1 - p_value)
            effect_size_r = abs(z_score) / np.sqrt(n)
            
            result = {
                'test_type': 'Wilcoxon İşaretli Sıralar Testi',
                'column1': column1,
                'column2': column2,
                'sample_size': len(paired_data),
                'non_zero_differences': n,
                'median_difference': float(non_zero_diff.median()),
                'w_statistic': float(w_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'alternative': alternative,
                'is_significant': p_value < alpha,
                'effect_size_r': float(effect_size_r),
                'interpretation': self._generate_interpretation(
                    'Wilcoxon İşaretli Sıralar Testi', p_value, alpha, effect_size_r,
                    {'Medyan fark': f'{non_zero_diff.median():.3f}',
                     'Non-parametrik test': 'Normallik varsayımı gerektirmez'}
                )
            }
            
            self.results['wilcoxon_signed_rank_test'] = result
            self.test_history.append(result)
            logger.info(f"Wilcoxon işaretli sıralar testi tamamlandı: {column1} vs {column2}")
            
            return result
            
        except Exception as e:
            logger.error(f"Wilcoxon testi hatası: {str(e)}")
            return {'error': f'Wilcoxon testi hatası: {str(e)}'}
    
    def kruskal_wallis_test(self, dependent_var: str, independent_var: str,
                           alpha: float = 0.05, post_hoc: bool = True) -> Dict[str, Any]:
        """Kruskal-Wallis H testi"""
        try:
            groups = self.data.groupby(independent_var)[dependent_var].apply(lambda x: x.dropna())
            
            if len(groups) < 2:
                return {'error': 'Kruskal-Wallis testi için en az 2 grup gerekli'}
            
            group_data = [group for group in groups if len(group) >= 1]
            group_names = [name for name, group in groups.items() if len(group) >= 1]
            
            if len(group_data) < 2:
                return {'error': 'Her grupta en az 1 gözlem gerekli'}
            
            # Kruskal-Wallis testi
            h_stat, p_value = kruskal(*group_data)
            
            # Grup istatistikleri
            group_stats = []
            total_n = sum(len(group) for group in group_data)
            
            for name, data in zip(group_names, group_data):
                ranks = stats.rankdata(np.concatenate(group_data))
                start_idx = sum(len(group_data[i]) for i in range(group_data.index(data)))
                end_idx = start_idx + len(data)
                mean_rank = ranks[start_idx:end_idx].mean()
                
                group_stats.append({
                    'group': str(name),
                    'n': len(data),
                    'median': float(data.median()),
                    'mean_rank': float(mean_rank),
                    'min': float(data.min()),
                    'max': float(data.max())
                })
            
            # Etki büyüklüğü (Epsilon squared)
            k = len(group_data)
            epsilon_squared = (h_stat - k + 1) / (total_n - k)
            
            # Post-hoc testler (Dunn's test)
            post_hoc_results = None
            if post_hoc and p_value < alpha:
                try:
                    dunn_results = pg.pairwise_tests(
                        data=self.data[[dependent_var, independent_var]].dropna(),
                        dv=dependent_var,
                        between=independent_var,
                        parametric=False,
                        padjust='bonf'
                    )
                    
                    post_hoc_results = {
                        'dunn_test': dunn_results.to_dict('records')
                    }
                except:
                    post_hoc_results = {'error': 'Post-hoc testler hesaplanamadı'}
            
            result = {
                'test_type': 'Kruskal-Wallis H Testi',
                'dependent_variable': dependent_var,
                'independent_variable': independent_var,
                'group_count': len(group_data),
                'total_sample_size': total_n,
                'h_statistic': float(h_stat),
                'degrees_of_freedom': len(group_data) - 1,
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'epsilon_squared': float(epsilon_squared),
                'effect_size_interpretation': self._interpret_effect_size(epsilon_squared, "eta_squared"),
                'group_statistics': group_stats,
                'post_hoc_tests': post_hoc_results,
                'interpretation': self._generate_interpretation(
                    'Kruskal-Wallis H Testi', p_value, alpha, epsilon_squared,
                    {'Grup sayısı': len(group_data),
                     'Non-parametrik test': 'Normallik varsayımı gerektirmez',
                     'H istatistiği': f'{h_stat:.3f}'}
                )
            }
            
            self.results['kruskal_wallis_test'] = result
            self.test_history.append(result)
            logger.info(f"Kruskal-Wallis testi tamamlandı: {dependent_var} ~ {independent_var}")
            
            return result
            
        except Exception as e:
            logger.error(f"Kruskal-Wallis testi hatası: {str(e)}")
            return {'error': f'Kruskal-Wallis testi hatası: {str(e)}'}
    
    def correlation_test(self, column1: str, column2: str, method: str = 'pearson',
                        alpha: float = 0.05) -> Dict[str, Any]:
        """Korelasyon testi"""
        try:
            data_clean = self.data[[column1, column2]].dropna()
            
            if len(data_clean) < 3:
                return {'error': 'Korelasyon testi için en az 3 gözlem gerekli'}
            
            x, y = data_clean[column1], data_clean[column2]
            
            if method == 'pearson':
                corr_coef, p_value = pearsonr(x, y)
                test_name = 'Pearson Korelasyon Testi'
            elif method == 'spearman':
                corr_coef, p_value = spearmanr(x, y)
                test_name = 'Spearman Korelasyon Testi'
            elif method == 'kendall':
                corr_coef, p_value = kendalltau(x, y)
                test_name = 'Kendall Tau Korelasyon Testi'
            else:
                return {'error': 'Desteklenmeyen korelasyon yöntemi'}
            
            # Güven aralığı (Fisher z-transform için)
            if method == 'pearson':
                n = len(data_clean)
                z_transform = 0.5 * np.log((1 + corr_coef) / (1 - corr_coef))
                se_z = 1 / np.sqrt(n - 3)
                z_critical = stats.norm.ppf(1 - alpha/2)
                ci_z_lower = z_transform - z_critical * se_z
                ci_z_upper = z_transform + z_critical * se_z
                
                ci_lower = (np.exp(2 * ci_z_lower) - 1) / (np.exp(2 * ci_z_lower) + 1)
                ci_upper = (np.exp(2 * ci_z_upper) - 1) / (np.exp(2 * ci_z_upper) + 1)
            else:
                ci_lower, ci_upper = None, None
            
            # Korelasyon gücü yorumu
            abs_corr = abs(corr_coef)
            if abs_corr < 0.1:
                strength = "Çok zayıf"
            elif abs_corr < 0.3:
                strength = "Zayıf"
            elif abs_corr < 0.5:
                strength = "Orta"
            elif abs_corr < 0.7:
                strength = "Güçlü"
            else:
                strength = "Çok güçlü"
            
            direction = "Pozitif" if corr_coef > 0 else "Negatif"
            
            result = {
                'test_type': test_name,
                'column1': column1,
                'column2': column2,
                'method': method,
                'sample_size': len(data_clean),
                'correlation_coefficient': float(corr_coef),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'correlation_strength': strength,
                'correlation_direction': direction,
                'confidence_interval': (float(ci_lower), float(ci_upper)) if ci_lower is not None else None,
                'interpretation': self._generate_interpretation(
                    test_name, p_value, alpha, abs_corr,
                    {'Korelasyon katsayısı': f'{corr_coef:.3f}',
                     'Korelasyon gücü': f'{strength} {direction.lower()}',
                     'Yöntem': method.title()}
                )
            }
            
            self.results['correlation_test'] = result
            self.test_history.append(result)
            logger.info(f"Korelasyon testi tamamlandı: {column1} vs {column2} ({method})")
            
            return result
            
        except Exception as e:
            logger.error(f"Korelasyon testi hatası: {str(e)}")
            return {'error': f'Korelasyon testi hatası: {str(e)}'}
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Yapılan testlerin özeti"""
        return {
            'total_tests_performed': len(self.test_history),
            'test_types': list(set([test.get('test_type', 'Unknown') for test in self.test_history])),
            'significant_results': len([test for test in self.test_history if test.get('is_significant', False)]),
            'test_history': self.test_history,
            'available_results': list(self.results.keys())
        }
    
    def multiple_testing_correction(self, p_values: List[float], method: str = 'bonferroni',
                                   alpha: float = 0.05) -> Dict[str, Any]:
        """Çoklu test düzeltmesi"""
        try:
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method=method
            )
            
            return {
                'method': method,
                'original_alpha': alpha,
                'corrected_alpha_bonferroni': float(alpha_bonf),
                'corrected_alpha_sidak': float(alpha_sidak),
                'original_p_values': [float(p) for p in p_values],
                'corrected_p_values': [float(p) for p in p_corrected],
                'rejected_hypotheses': rejected.tolist(),
                'number_of_rejections': int(rejected.sum()),
                'interpretation': f'{method.title()} düzeltmesi ile {int(rejected.sum())} hipotez reddedildi'
            }
            
        except Exception as e:
            return {'error': f'Çoklu test düzeltmesi hatası: {str(e)}'}
    
    def paired_t_test(self, col1: str, col2: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Eşleştirilmiş t-testi wrapper"""
        try:
            from .paired_tests import run_paired_ttest
            result = run_paired_ttest(self.data, col1, col2)
            result['is_significant'] = result['p_value'] < alpha
            self.test_history.append(result)
            return result
        except Exception as e:
            return {'error': f'Paired t-test hatası: {str(e)}'}
    
    def manova_test(self, dependent_vars: List[str], independent_formula: str, alpha: float = 0.05) -> Dict[str, Any]:
        """MANOVA testi wrapper"""
        try:
            from .manova import run_manova
            result = run_manova(self.data, dependent_vars, independent_formula)
            # MANOVA için anlamlılık kontrolü (Wilks' Lambda kullanarak)
            if 'test_results' in result and 'Wilks\' lambda' in result['test_results']:
                result['is_significant'] = result['test_results']['Wilks\' lambda']['p_value'] < alpha
            self.test_history.append(result)
            return result
        except Exception as e:
            return {'error': f'MANOVA hatası: {str(e)}'}
    
    def mixed_anova_test(self, dv: str, within: str, between: str, subject: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Mixed ANOVA testi wrapper"""
        try:
            from .mixed_anova import run_mixed_anova
            result = run_mixed_anova(self.data, dv, within, between, subject)
            # Mixed ANOVA için anlamlılık kontrolü yapılabilir
            result['is_significant'] = True  # Bu kısım anova_table'dan p değerlerine bakılarak güncellenebilir
            self.test_history.append(result)
            return result
        except Exception as e:
            return {'error': f'Mixed ANOVA hatası: {str(e)}'}
    
    def wilcoxon_signed_rank_test(self, col1: str, col2: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Wilcoxon signed-rank testi wrapper"""
        try:
            from .paired_tests import run_wilcoxon_signed_rank
            result = run_wilcoxon_signed_rank(self.data, col1, col2)
            result['is_significant'] = result['p_value'] < alpha
            self.test_history.append(result)
            return result
        except Exception as e:
            return {'error': f'Wilcoxon signed-rank test hatası: {str(e)}'}