"""
İleri Seviye Hipotez Testleri Sınıfı

Bu modül daha karmaşık ve özelleşmiş hipotez testlerini gerçekleştirmek için kullanılır.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    bartlett, fligner, anderson, jarque_bera, normaltest,
    pearsonr, spearmanr, kendalltau, fisher_exact,
    mcnemar, cochran, friedmanchisquare
)
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class AdvancedHypothesisTester:
    """
    İleri seviye hipotez testlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        AdvancedHypothesisTester sınıfını başlatır
        
        Args:
            data: Test edilecek veri seti
        """
        self.data = data.copy()
        self.results = {}
    
    def normality_tests(self, column: str, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Çoklu normallik testleri gerçekleştirir
        
        Args:
            column: Test edilecek sütun
            alpha: Anlamlılık düzeyi
            
        Returns:
            Normallik test sonuçları
        """
        try:
            data_col = self.data[column].dropna()
            
            if len(data_col) < 3:
                return {'error': 'Normallik testleri için en az 3 gözlem gereklidir'}
            
            results = {}
            
            # Shapiro-Wilk testi
            if len(data_col) <= 5000:  # Shapiro-Wilk 5000'den fazla için çalışmaz
                shapiro_stat, shapiro_p = stats.shapiro(data_col)
                results['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > alpha,
                    'interpretation': 'Normal dağılım' if shapiro_p > alpha else 'Normal dağılım değil'
                }
            
            # Kolmogorov-Smirnov testi
            ks_stat, ks_p = stats.kstest(data_col, 'norm', args=(data_col.mean(), data_col.std()))
            results['kolmogorov_smirnov'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'is_normal': ks_p > alpha,
                'interpretation': 'Normal dağılım' if ks_p > alpha else 'Normal dağılım değil'
            }
            
            # Anderson-Darling testi
            ad_stat, ad_critical, ad_significance = anderson(data_col, dist='norm')
            ad_p_approx = 1 - stats.norm.cdf(ad_stat)  # Yaklaşık p-değeri
            results['anderson_darling'] = {
                'statistic': float(ad_stat),
                'critical_values': ad_critical.tolist(),
                'significance_levels': ad_significance.tolist(),
                'p_value_approx': float(ad_p_approx),
                'is_normal': ad_stat < ad_critical[2],  # %5 seviyesi
                'interpretation': 'Normal dağılım' if ad_stat < ad_critical[2] else 'Normal dağılım değil'
            }
            
            # Jarque-Bera testi
            jb_stat, jb_p = jarque_bera(data_col)
            results['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'is_normal': jb_p > alpha,
                'interpretation': 'Normal dağılım' if jb_p > alpha else 'Normal dağılım değil'
            }
            
            # D'Agostino-Pearson testi
            dp_stat, dp_p = normaltest(data_col)
            results['dagostino_pearson'] = {
                'statistic': float(dp_stat),
                'p_value': float(dp_p),
                'is_normal': dp_p > alpha,
                'interpretation': 'Normal dağılım' if dp_p > alpha else 'Normal dağılım değil'
            }
            
            # Lilliefors testi
            try:
                lf_stat, lf_p = lilliefors(data_col, dist='norm')
                results['lilliefors'] = {
                    'statistic': float(lf_stat),
                    'p_value': float(lf_p),
                    'is_normal': lf_p > alpha,
                    'interpretation': 'Normal dağılım' if lf_p > alpha else 'Normal dağılım değil'
                }
            except:
                results['lilliefors'] = {'error': 'Lilliefors testi hesaplanamadı'}
            
            # Genel değerlendirme
            normal_count = sum([1 for test in results.values() 
                              if isinstance(test, dict) and test.get('is_normal', False)])
            total_tests = len([test for test in results.values() if isinstance(test, dict) and 'is_normal' in test])
            
            overall_result = {
                'test_type': 'Multiple Normality Tests',
                'column': column,
                'sample_size': len(data_col),
                'alpha': alpha,
                'individual_tests': results,
                'summary': {
                    'tests_indicating_normal': normal_count,
                    'total_valid_tests': total_tests,
                    'percentage_normal': (normal_count / total_tests * 100) if total_tests > 0 else 0,
                    'overall_conclusion': 'Normal dağılım' if normal_count >= total_tests/2 else 'Normal dağılım değil'
                },
                'interpretation': self._interpret_normality_tests(normal_count, total_tests)
            }
            
            self.results['normality_tests'] = overall_result
            return overall_result
            
        except Exception as e:
            return {'error': f'Normallik testleri hatası: {str(e)}'}
    
    def variance_homogeneity_tests(self, column: str, group_column: str, 
                                  alpha: float = 0.05) -> Dict[str, Any]:
        """
        Varyans homojenliği testleri gerçekleştirir
        
        Args:
            column: Test edilecek sürekli değişken
            group_column: Grup değişkeni
            alpha: Anlamlılık düzeyi
            
        Returns:
            Varyans homojenliği test sonuçları
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
            
            results = {}
            
            # Levene testi
            levene_stat, levene_p = stats.levene(*group_data)
            results['levene'] = {
                'statistic': float(levene_stat),
                'p_value': float(levene_p),
                'equal_variances': levene_p > alpha,
                'interpretation': 'Varyanslar homojen' if levene_p > alpha else 'Varyanslar homojen değil'
            }
            
            # Bartlett testi
            bartlett_stat, bartlett_p = bartlett(*group_data)
            results['bartlett'] = {
                'statistic': float(bartlett_stat),
                'p_value': float(bartlett_p),
                'equal_variances': bartlett_p > alpha,
                'interpretation': 'Varyanslar homojen' if bartlett_p > alpha else 'Varyanslar homojen değil'
            }
            
            # Fligner-Killeen testi
            fligner_stat, fligner_p = fligner(*group_data)
            results['fligner_killeen'] = {
                'statistic': float(fligner_stat),
                'p_value': float(fligner_p),
                'equal_variances': fligner_p > alpha,
                'interpretation': 'Varyanslar homojen' if fligner_p > alpha else 'Varyanslar homojen değil'
            }
            
            # Grup varyansları
            group_variances = {}
            for i, group in enumerate(groups):
                group_variances[str(group)] = {
                    'variance': float(np.var(group_data[i], ddof=1)),
                    'std': float(np.std(group_data[i], ddof=1)),
                    'size': len(group_data[i])
                }
            
            # Genel değerlendirme
            homogeneous_count = sum([1 for test in results.values() if test['equal_variances']])
            total_tests = len(results)
            
            overall_result = {
                'test_type': 'Variance Homogeneity Tests',
                'column': column,
                'group_column': group_column,
                'num_groups': len(groups),
                'alpha': alpha,
                'individual_tests': results,
                'group_variances': group_variances,
                'summary': {
                    'tests_indicating_homogeneous': homogeneous_count,
                    'total_tests': total_tests,
                    'percentage_homogeneous': (homogeneous_count / total_tests * 100),
                    'overall_conclusion': 'Varyanslar homojen' if homogeneous_count >= total_tests/2 else 'Varyanslar homojen değil'
                },
                'interpretation': self._interpret_variance_tests(homogeneous_count, total_tests)
            }
            
            self.results['variance_homogeneity_tests'] = overall_result
            return overall_result
            
        except Exception as e:
            return {'error': f'Varyans homojenliği testleri hatası: {str(e)}'}
    
    def correlation_tests(self, column1: str, column2: str, 
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Çoklu korelasyon testleri gerçekleştirir
        
        Args:
            column1: İlk değişken
            column2: İkinci değişken
            alpha: Anlamlılık düzeyi
            
        Returns:
            Korelasyon test sonuçları
        """
        try:
            # Eksik değerleri çıkar
            data_clean = self.data[[column1, column2]].dropna()
            x = data_clean[column1]
            y = data_clean[column2]
            
            if len(data_clean) < 3:
                return {'error': 'Korelasyon testleri için en az 3 gözlem gereklidir'}
            
            results = {}
            
            # Pearson korelasyonu
            pearson_r, pearson_p = pearsonr(x, y)
            results['pearson'] = {
                'correlation': float(pearson_r),
                'p_value': float(pearson_p),
                'is_significant': pearson_p < alpha,
                'interpretation': self._interpret_correlation(pearson_r, pearson_p, alpha, 'Pearson')
            }
            
            # Spearman korelasyonu
            spearman_r, spearman_p = spearmanr(x, y)
            results['spearman'] = {
                'correlation': float(spearman_r),
                'p_value': float(spearman_p),
                'is_significant': spearman_p < alpha,
                'interpretation': self._interpret_correlation(spearman_r, spearman_p, alpha, 'Spearman')
            }
            
            # Kendall's tau
            kendall_tau, kendall_p = kendalltau(x, y)
            results['kendall'] = {
                'correlation': float(kendall_tau),
                'p_value': float(kendall_p),
                'is_significant': kendall_p < alpha,
                'interpretation': self._interpret_correlation(kendall_tau, kendall_p, alpha, 'Kendall')
            }
            
            overall_result = {
                'test_type': 'Multiple Correlation Tests',
                'column1': column1,
                'column2': column2,
                'sample_size': len(data_clean),
                'alpha': alpha,
                'correlation_tests': results,
                'summary': {
                    'strongest_correlation': max(results.keys(), 
                                               key=lambda k: abs(results[k]['correlation'])),
                    'significant_correlations': [k for k, v in results.items() if v['is_significant']],
                    'average_correlation': float(np.mean([v['correlation'] for v in results.values()]))
                },
                'interpretation': self._interpret_multiple_correlations(results)
            }
            
            self.results['correlation_tests'] = overall_result
            return overall_result
            
        except Exception as e:
            return {'error': f'Korelasyon testleri hatası: {str(e)}'}
    
    def fisher_exact_test(self, column1: str, column2: str, 
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Fisher's exact test gerçekleştirir (2x2 tablolar için)
        
        Args:
            column1: İlk kategorik değişken
            column2: İkinci kategorik değişken
            alpha: Anlamlılık düzeyi
            
        Returns:
            Fisher's exact test sonuçları
        """
        try:
            # Çapraz tablo oluştur
            contingency_table = pd.crosstab(self.data[column1], self.data[column2])
            
            if contingency_table.shape != (2, 2):
                return {'error': "Fisher's exact test sadece 2x2 tablolar için kullanılabilir"}
            
            # Fisher's exact test
            odds_ratio, p_value = fisher_exact(contingency_table)
            
            # Güven aralığı hesaplama (log odds ratio için)
            a, b, c, d = contingency_table.iloc[0, 0], contingency_table.iloc[0, 1], \
                        contingency_table.iloc[1, 0], contingency_table.iloc[1, 1]
            
            log_or = np.log(odds_ratio) if odds_ratio > 0 else np.nan
            se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if all([a, b, c, d]) else np.nan
            
            if not np.isnan(se_log_or):
                ci_lower = np.exp(log_or - 1.96 * se_log_or)
                ci_upper = np.exp(log_or + 1.96 * se_log_or)
            else:
                ci_lower = ci_upper = np.nan
            
            result = {
                'test_type': "Fisher's Exact Test",
                'column1': column1,
                'column2': column2,
                'contingency_table': contingency_table.to_dict(),
                'odds_ratio': float(odds_ratio) if not np.isnan(odds_ratio) else None,
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'confidence_interval_or': (float(ci_lower), float(ci_upper)) if not np.isnan(ci_lower) else None,
                'interpretation': self._interpret_fisher_exact(odds_ratio, p_value, alpha)
            }
            
            self.results['fisher_exact_test'] = result
            return result
            
        except Exception as e:
            return {'error': f"Fisher's exact test hatası: {str(e)}"}
    
    def mcnemar_test(self, column1: str, column2: str, 
                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        McNemar testi gerçekleştirir (eşleştirilmiş kategorik veriler için)
        
        Args:
            column1: İlk kategorik değişken
            column2: İkinci kategorik değişken
            alpha: Anlamlılık düzeyi
            
        Returns:
            McNemar test sonuçları
        """
        try:
            # Çapraz tablo oluştur
            contingency_table = pd.crosstab(self.data[column1], self.data[column2])
            
            if contingency_table.shape != (2, 2):
                return {'error': 'McNemar testi sadece 2x2 tablolar için kullanılabilir'}
            
            # McNemar testi
            result_table = mcnemar_test(contingency_table, exact=True)
            
            # Uyumsuz çiftler
            b = contingency_table.iloc[0, 1]  # (0,1) hücresi
            c = contingency_table.iloc[1, 0]  # (1,0) hücresi
            
            result = {
                'test_type': 'McNemar Test',
                'column1': column1,
                'column2': column2,
                'contingency_table': contingency_table.to_dict(),
                'discordant_pairs': {
                    'b_01': int(b),
                    'c_10': int(c),
                    'total_discordant': int(b + c)
                },
                'statistic': float(result_table.statistic),
                'p_value': float(result_table.pvalue),
                'alpha': alpha,
                'is_significant': result_table.pvalue < alpha,
                'interpretation': self._interpret_mcnemar(b, c, result_table.pvalue, alpha)
            }
            
            self.results['mcnemar_test'] = result
            return result
            
        except Exception as e:
            return {'error': f'McNemar testi hatası: {str(e)}'}
    
    def friedman_test(self, columns: List[str], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Friedman testi gerçekleştirir (tekrarlı ölçümler için)
        
        Args:
            columns: Test edilecek sütunlar listesi
            alpha: Anlamlılık düzeyi
            
        Returns:
            Friedman test sonuçları
        """
        try:
            if len(columns) < 3:
                return {'error': 'Friedman testi için en az 3 sütun gereklidir'}
            
            # Eksik değerleri çıkar
            data_clean = self.data[columns].dropna()
            
            if len(data_clean) < 3:
                return {'error': 'Friedman testi için en az 3 gözlem gereklidir'}
            
            # Friedman testi
            friedman_stat, p_value = friedmanchisquare(*[data_clean[col] for col in columns])
            
            # Kendall's W (uyum katsayısı)
            n = len(data_clean)
            k = len(columns)
            kendalls_w = friedman_stat / (n * (k - 1))
            
            # Sütun istatistikleri
            column_stats = {}
            for col in columns:
                column_stats[col] = {
                    'mean': float(data_clean[col].mean()),
                    'median': float(data_clean[col].median()),
                    'mean_rank': float(data_clean[col].rank(axis=1).mean())
                }
            
            result = {
                'test_type': 'Friedman Test',
                'columns': columns,
                'sample_size': n,
                'num_conditions': k,
                'friedman_statistic': float(friedman_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_significant': p_value < alpha,
                'kendalls_w': float(kendalls_w),
                'effect_size_interpretation': self._interpret_kendalls_w(kendalls_w),
                'column_statistics': column_stats,
                'interpretation': self._interpret_friedman(p_value, alpha, kendalls_w, k)
            }
            
            self.results['friedman_test'] = result
            return result
            
        except Exception as e:
            return {'error': f'Friedman testi hatası: {str(e)}'}
    
    def multiple_comparisons_correction(self, p_values: List[float], 
                                      method: str = 'holm', 
                                      alpha: float = 0.05) -> Dict[str, Any]:
        """
        Çoklu karşılaştırma düzeltmesi uygular
        
        Args:
            p_values: P-değerleri listesi
            method: Düzeltme yöntemi ('holm', 'bonferroni', 'fdr_bh', vb.)
            alpha: Anlamlılık düzeyi
            
        Returns:
            Düzeltilmiş test sonuçları
        """
        try:
            # Çoklu karşılaştırma düzeltmesi
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method=method
            )
            
            result = {
                'test_type': 'Multiple Comparisons Correction',
                'method': method,
                'original_alpha': alpha,
                'num_tests': len(p_values),
                'original_p_values': [float(p) for p in p_values],
                'corrected_p_values': [float(p) for p in p_corrected],
                'rejected_hypotheses': rejected.tolist(),
                'bonferroni_alpha': float(alpha_bonf),
                'sidak_alpha': float(alpha_sidak),
                'summary': {
                    'significant_before_correction': sum([p < alpha for p in p_values]),
                    'significant_after_correction': sum(rejected),
                    'correction_impact': sum([p < alpha for p in p_values]) - sum(rejected)
                },
                'interpretation': self._interpret_multiple_comparisons(
                    len(p_values), sum([p < alpha for p in p_values]), sum(rejected), method
                )
            }
            
            self.results['multiple_comparisons_correction'] = result
            return result
            
        except Exception as e:
            return {'error': f'Çoklu karşılaştırma düzeltmesi hatası: {str(e)}'}
    
    def _interpret_normality_tests(self, normal_count: int, total_tests: int) -> str:
        """Normallik testleri sonucunu yorumlar"""
        percentage = (normal_count / total_tests * 100) if total_tests > 0 else 0
        
        if percentage >= 80:
            return f"Testlerin %{percentage:.1f}'i normal dağılım gösteriyor. Veri büyük olasılıkla normal dağılımlıdır."
        elif percentage >= 50:
            return f"Testlerin %{percentage:.1f}'i normal dağılım gösteriyor. Normallik konusunda kararsızlık var."
        else:
            return f"Testlerin sadece %{percentage:.1f}'i normal dağılım gösteriyor. Veri normal dağılımlı değildir."
    
    def _interpret_variance_tests(self, homogeneous_count: int, total_tests: int) -> str:
        """Varyans homojenliği testleri sonucunu yorumlar"""
        percentage = (homogeneous_count / total_tests * 100)
        
        if percentage >= 67:
            return f"Testlerin %{percentage:.1f}'i varyans homojenliği gösteriyor. Varyanslar büyük olasılıkla homojendir."
        elif percentage >= 33:
            return f"Testlerin %{percentage:.1f}'i varyans homojenliği gösteriyor. Varyans homojenliği konusunda kararsızlık var."
        else:
            return f"Testlerin sadece %{percentage:.1f}'i varyans homojenliği gösteriyor. Varyanslar homojen değildir."
    
    def _interpret_correlation(self, r: float, p_value: float, alpha: float, method: str) -> str:
        """Korelasyon sonucunu yorumlar"""
        strength = self._correlation_strength(abs(r))
        direction = "pozitif" if r > 0 else "negatif"
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        return f"{method} korelasyonu {direction} yönde {strength} ve istatistiksel olarak {significance}dır (r={r:.3f}, p={p_value:.3f})."
    
    def _correlation_strength(self, abs_r: float) -> str:
        """Korelasyon gücünü yorumlar"""
        if abs_r < 0.1:
            return "çok zayıf"
        elif abs_r < 0.3:
            return "zayıf"
        elif abs_r < 0.5:
            return "orta"
        elif abs_r < 0.7:
            return "güçlü"
        else:
            return "çok güçlü"
    
    def _interpret_multiple_correlations(self, results: Dict) -> str:
        """Çoklu korelasyon sonuçlarını yorumlar"""
        significant = [k for k, v in results.items() if v['is_significant']]
        correlations = [v['correlation'] for v in results.values()]
        avg_corr = np.mean(correlations)
        
        if len(significant) == len(results):
            return f"Tüm korelasyon testleri anlamlıdır. Ortalama korelasyon: {avg_corr:.3f}"
        elif len(significant) > 0:
            return f"{len(significant)}/{len(results)} korelasyon testi anlamlıdır. Ortalama korelasyon: {avg_corr:.3f}"
        else:
            return f"Hiçbir korelasyon testi anlamlı değildir. Ortalama korelasyon: {avg_corr:.3f}"
    
    def _interpret_fisher_exact(self, odds_ratio: float, p_value: float, alpha: float) -> str:
        """Fisher's exact test sonucunu yorumlar"""
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        
        if np.isnan(odds_ratio) or odds_ratio == 0:
            or_text = "hesaplanamıyor"
        elif odds_ratio == 1:
            or_text = "1 (ilişki yok)"
        elif odds_ratio > 1:
            or_text = f"{odds_ratio:.3f} (pozitif ilişki)"
        else:
            or_text = f"{odds_ratio:.3f} (negatif ilişki)"
        
        return f"İki kategorik değişken arasında istatistiksel olarak {significance} bir ilişki vardır (p={p_value:.3f}). Odds ratio: {or_text}"
    
    def _interpret_mcnemar(self, b: int, c: int, p_value: float, alpha: float) -> str:
        """McNemar testi sonucunu yorumlar"""
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        total_discordant = b + c
        
        if total_discordant == 0:
            return "Hiç uyumsuz çift yok, değişiklik tespit edilemez."
        
        return (f"İki ölçüm arasında istatistiksel olarak {significance} bir değişiklik vardır "
                f"(p={p_value:.3f}). Toplam {total_discordant} uyumsuz çift tespit edildi.")
    
    def _interpret_friedman(self, p_value: float, alpha: float, kendalls_w: float, k: int) -> str:
        """Friedman testi sonucunu yorumlar"""
        significance = "anlamlı" if p_value < alpha else "anlamlı değil"
        agreement = self._interpret_kendalls_w(kendalls_w)
        
        return (f"{k} koşul arasında istatistiksel olarak {significance} bir fark vardır "
                f"(p={p_value:.3f}). Uyum derecesi {agreement.lower()}dir (W={kendalls_w:.3f}).")
    
    def _interpret_kendalls_w(self, w: float) -> str:
        """Kendall's W uyum katsayısını yorumlar"""
        if w < 0.1:
            return "Çok zayıf uyum"
        elif w < 0.3:
            return "Zayıf uyum"
        elif w < 0.5:
            return "Orta uyum"
        elif w < 0.7:
            return "Güçlü uyum"
        else:
            return "Çok güçlü uyum"
    
    def _interpret_multiple_comparisons(self, num_tests: int, sig_before: int, 
                                      sig_after: int, method: str) -> str:
        """Çoklu karşılaştırma düzeltmesi sonucunu yorumlar"""
        impact = sig_before - sig_after
        
        interpretation = (f"{num_tests} test için {method} düzeltmesi uygulandı. "
                         f"Düzeltme öncesi {sig_before} anlamlı test varken, "
                         f"düzeltme sonrası {sig_after} anlamlı test kaldı.")
        
        if impact > 0:
            interpretation += f" {impact} test düzeltme nedeniyle anlamlılığını kaybetti."
        elif impact == 0:
            interpretation += " Düzeltme sonuçları etkilemedi."
        
        return interpretation
    
    def get_advanced_test_summary(self) -> Dict[str, Any]:
        """
        Gerçekleştirilen tüm ileri seviye testlerin özetini döndürür
        
        Returns:
            Test özetleri
        """
        if not self.results:
            return {'message': 'Henüz ileri seviye test gerçekleştirilmedi'}
        
        summary = {
            'total_advanced_tests': len(self.results),
            'tests_performed': list(self.results.keys()),
            'test_details': self.results
        }
        
        return summary