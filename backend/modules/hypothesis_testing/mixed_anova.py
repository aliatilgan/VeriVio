"""
Mixed ANOVA Analysis Module
Karışık ANOVA analizi için modül
"""

import pandas as pd
import pingouin as pg
import numpy as np
from typing import Dict, Any


def run_mixed_anova(df: pd.DataFrame, 
                    dv: str, 
                    within: str, 
                    between: str, 
                    subject: str) -> Dict[str, Any]:
    """
    Karışık ANOVA analizi yapar
    
    Args:
        df: Veri çerçevesi
        dv: Bağımlı değişken
        within: Grup içi faktör
        between: Gruplar arası faktör
        subject: Özne/katılımcı sütunu
    
    Returns:
        ANOVA sonuçları ve yorumları
    """
    # Sütun kontrolü
    if subject not in df.columns:
        raise ValueError(f"Subject column '{subject}' not found in dataframe")
    
    for col in [dv, within, between]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    # Veri hazırlığı
    df_sub = df[[subject, within, between, dv]].dropna()
    
    if len(df_sub) < 4:
        raise ValueError("Insufficient data for mixed ANOVA analysis")
    
    # Kategorik değişkenlere dönüştür
    df_sub[within] = df_sub[within].astype('category')
    df_sub[between] = df_sub[between].astype('category')
    
    try:
        # Mixed ANOVA analizi
        aov = pg.mixed_anova(dv=dv, within=within, between=between, subject=subject, data=df_sub)
        aov_dict = aov.round(4).to_dict(orient='index')
        
        # Yorumlama
        interpretations = []
        for idx, row in aov.iterrows():
            source = row['Source'] if 'Source' in row.index else idx
            p_value = row.get('p-unc', np.nan)
            
            if not pd.isna(p_value):
                if p_value < 0.001:
                    sig_level = "p < 0.001"
                    interpretation = f"{source} etkisi çok yüksek düzeyde anlamlı ({sig_level})."
                elif p_value < 0.01:
                    sig_level = f"p = {p_value:.3f}"
                    interpretation = f"{source} etkisi yüksek düzeyde anlamlı ({sig_level})."
                elif p_value < 0.05:
                    sig_level = f"p = {p_value:.3f}"
                    interpretation = f"{source} etkisi anlamlı ({sig_level})."
                else:
                    sig_level = f"p = {p_value:.3f}"
                    interpretation = f"{source} etkisi anlamlı değil ({sig_level})."
                
                interpretations.append(interpretation)
        
        # Etki büyüklüğü yorumu
        effect_size_interpretations = []
        for idx, row in aov.iterrows():
            source = row['Source'] if 'Source' in row.index else idx
            eta_sq = row.get('np2', np.nan)  # partial eta squared
            
            if not pd.isna(eta_sq):
                if eta_sq >= 0.14:
                    effect_size_interpretations.append(f"{source} için büyük etki büyüklüğü (η²p = {eta_sq:.3f})")
                elif eta_sq >= 0.06:
                    effect_size_interpretations.append(f"{source} için orta etki büyüklüğü (η²p = {eta_sq:.3f})")
                elif eta_sq >= 0.01:
                    effect_size_interpretations.append(f"{source} için küçük etki büyüklüğü (η²p = {eta_sq:.3f})")
                else:
                    effect_size_interpretations.append(f"{source} için çok küçük etki büyüklüğü (η²p = {eta_sq:.3f})")
        
        return {
            'test_type': 'mixed_anova',
            'anova_table': aov_dict,
            'statistical_interpretation': "\n".join(interpretations),
            'effect_size_interpretation': "\n".join(effect_size_interpretations),
            'sample_size': len(df_sub),
            'within_factor': within,
            'between_factor': between,
            'dependent_variable': dv,
            'subject_variable': subject
        }
        
    except Exception as e:
        raise ValueError(f"Mixed ANOVA analysis failed: {str(e)}")