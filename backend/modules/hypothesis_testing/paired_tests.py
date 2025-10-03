"""
Paired Tests Analysis Module
Eşleştirilmiş testler için modül (paired t-test, Wilcoxon signed-rank test)
"""

import pandas as pd
import numpy as np
import pingouin as pg
from scipy import stats
from typing import Dict, Any


def run_paired_ttest(df: pd.DataFrame, 
                     col1: str, 
                     col2: str) -> Dict[str, Any]:
    """
    Eşleştirilmiş t-testi yapar
    
    Args:
        df: Veri çerçevesi
        col1: İlk ölçüm sütunu
        col2: İkinci ölçüm sütunu
    
    Returns:
        Paired t-test sonuçları ve yorumları
    """
    # Sütun kontrolü
    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' not found in dataframe")
    if col2 not in df.columns:
        raise ValueError(f"Column '{col2}' not found in dataframe")
    
    # Veri hazırlığı
    df_sub = df[[col1, col2]].dropna()
    
    if len(df_sub) < 3:
        raise ValueError("Insufficient data for paired t-test (minimum 3 pairs required)")
    
    x = df_sub[col1].values
    y = df_sub[col2].values
    
    try:
        # Pingouin ile paired t-test
        ttest_result = pg.ttest(x, y, paired=True)
        
        # Sonuçları çıkar
        t_stat = float(ttest_result['T'].iloc[0])
        p_value = float(ttest_result['p-val'].iloc[0])
        dof = int(ttest_result['dof'].iloc[0])
        cohen_d = float(ttest_result['cohen-d'].iloc[0])
        ci_lower = float(ttest_result['CI95%'].iloc[0][0])
        ci_upper = float(ttest_result['CI95%'].iloc[0][1])
        
        # Normallik testi
        shapiro_stat, shapiro_p = stats.shapiro(x - y)  # Farkların normalliği
        
        # Wilcoxon signed-rank test (non-parametrik alternatif)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxontest(x, y)
        
        # İstatistiksel yorum
        if p_value < 0.001:
            sig_level = "p < 0.001"
            interpretation = f"İki ölçüm arasında çok yüksek düzeyde anlamlı fark vardır ({sig_level})."
        elif p_value < 0.01:
            sig_level = f"p = {p_value:.3f}"
            interpretation = f"İki ölçüm arasında yüksek düzeyde anlamlı fark vardır ({sig_level})."
        elif p_value < 0.05:
            sig_level = f"p = {p_value:.3f}"
            interpretation = f"İki ölçüm arasında anlamlı fark vardır ({sig_level})."
        else:
            sig_level = f"p = {p_value:.3f}"
            interpretation = f"İki ölçüm arasında anlamlı fark yoktur ({sig_level})."
        
        # Etki büyüklüğü yorumu
        if abs(cohen_d) >= 0.8:
            effect_interpretation = f"Büyük etki büyüklüğü (Cohen's d = {cohen_d:.3f})"
        elif abs(cohen_d) >= 0.5:
            effect_interpretation = f"Orta etki büyüklüğü (Cohen's d = {cohen_d:.3f})"
        elif abs(cohen_d) >= 0.2:
            effect_interpretation = f"Küçük etki büyüklüğü (Cohen's d = {cohen_d:.3f})"
        else:
            effect_interpretation = f"Çok küçük etki büyüklüğü (Cohen's d = {cohen_d:.3f})"
        
        # Normallik yorumu
        if shapiro_p > 0.05:
            normality_interpretation = f"Farkların dağılımı normal (Shapiro-Wilk p = {shapiro_p:.3f}), parametrik test uygun."
        else:
            normality_interpretation = f"Farkların dağılımı normal değil (Shapiro-Wilk p = {shapiro_p:.3f}), non-parametrik test önerilir."
        
        # Betimsel istatistikler
        mean_diff = float(np.mean(x - y))
        std_diff = float(np.std(x - y, ddof=1))
        
        return {
            'test_type': 'paired_ttest',
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cohen_d': cohen_d,
            'confidence_interval_95': [ci_lower, ci_upper],
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'statistical_interpretation': interpretation,
            'effect_size_interpretation': effect_interpretation,
            'normality_test': {
                'shapiro_statistic': float(shapiro_stat),
                'shapiro_p_value': float(shapiro_p),
                'interpretation': normality_interpretation
            },
            'non_parametric_alternative': {
                'wilcoxon_statistic': float(wilcoxon_stat),
                'wilcoxon_p_value': float(wilcoxon_p),
                'note': "Normallik varsayımı ihlal edilirse Wilcoxon signed-rank testi kullanılabilir."
            },
            'sample_size': len(df_sub),
            'variable_1': col1,
            'variable_2': col2,
            'descriptive_stats': {
                col1: {
                    'mean': float(np.mean(x)),
                    'std': float(np.std(x, ddof=1)),
                    'median': float(np.median(x))
                },
                col2: {
                    'mean': float(np.mean(y)),
                    'std': float(np.std(y, ddof=1)),
                    'median': float(np.median(y))
                }
            }
        }
        
    except Exception as e:
        raise ValueError(f"Paired t-test analysis failed: {str(e)}")


def run_wilcoxon_signed_rank(df: pd.DataFrame, 
                            col1: str, 
                            col2: str) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank testi yapar (non-parametrik paired test)
    
    Args:
        df: Veri çerçevesi
        col1: İlk ölçüm sütunu
        col2: İkinci ölçüm sütunu
    
    Returns:
        Wilcoxon signed-rank test sonuçları ve yorumları
    """
    # Sütun kontrolü
    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' not found in dataframe")
    if col2 not in df.columns:
        raise ValueError(f"Column '{col2}' not found in dataframe")
    
    # Veri hazırlığı
    df_sub = df[[col1, col2]].dropna()
    
    if len(df_sub) < 3:
        raise ValueError("Insufficient data for Wilcoxon signed-rank test (minimum 3 pairs required)")
    
    x = df_sub[col1].values
    y = df_sub[col2].values
    
    try:
        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxontest(x, y)
        
        # İstatistiksel yorum
        if p_value < 0.001:
            sig_level = "p < 0.001"
            interpretation = f"İki ölçüm arasında çok yüksek düzeyde anlamlı fark vardır ({sig_level})."
        elif p_value < 0.01:
            sig_level = f"p = {p_value:.3f}"
            interpretation = f"İki ölçüm arasında yüksek düzeyde anlamlı fark vardır ({sig_level})."
        elif p_value < 0.05:
            sig_level = f"p = {p_value:.3f}"
            interpretation = f"İki ölçüm arasında anlamlı fark vardır ({sig_level})."
        else:
            sig_level = f"p = {p_value:.3f}"
            interpretation = f"İki ölçüm arasında anlamlı fark yoktur ({sig_level})."
        
        # Betimsel istatistikler
        median_diff = float(np.median(x - y))
        
        return {
            'test_type': 'wilcoxon_signed_rank',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'median_difference': median_diff,
            'statistical_interpretation': interpretation,
            'sample_size': len(df_sub),
            'variable_1': col1,
            'variable_2': col2,
            'descriptive_stats': {
                col1: {
                    'median': float(np.median(x)),
                    'q1': float(np.percentile(x, 25)),
                    'q3': float(np.percentile(x, 75))
                },
                col2: {
                    'median': float(np.median(y)),
                    'q1': float(np.percentile(y, 25)),
                    'q3': float(np.percentile(y, 75))
                }
            },
            'note': "Wilcoxon signed-rank testi, eşleştirilmiş örneklemler için non-parametrik bir testtir."
        }
        
    except Exception as e:
        raise ValueError(f"Wilcoxon signed-rank test failed: {str(e)}")