"""
Sağlık Analizi Sınıfı

Bu modül sağlık verilerinin analizi için özel araçlar içerir.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class HealthcareAnalyzer:
    """
    Sağlık veri analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        HealthcareAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek sağlık veri seti
        """
        self.data = data.copy()
        self.results = {}
        
    def clinical_outcome_analysis(self, outcome_column: str, 
                                treatment_column: Optional[str] = None,
                                demographic_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Klinik sonuç analizi gerçekleştirir
        
        Args:
            outcome_column: Sonuç değişkeni (ölüm, iyileşme, vb.)
            treatment_column: Tedavi türü sütunu (opsiyonel)
            demographic_columns: Demografik değişkenler (opsiyonel)
            
        Returns:
            Klinik sonuç analizi sonuçları
        """
        try:
            if outcome_column not in self.data.columns:
                return {'error': f'{outcome_column} sütunu bulunamadı'}
            
            data_clean = self.data[[outcome_column]].copy()
            
            # Tedavi sütunu varsa ekle
            if treatment_column and treatment_column in self.data.columns:
                data_clean[treatment_column] = self.data[treatment_column]
            
            # Demografik sütunlar varsa ekle
            if demographic_columns:
                available_demo_cols = [col for col in demographic_columns if col in self.data.columns]
                for col in available_demo_cols:
                    data_clean[col] = self.data[col]
            
            data_clean = data_clean.dropna()
            
            if len(data_clean) < 10:
                return {'error': 'Klinik sonuç analizi için en az 10 gözlem gereklidir'}
            
            outcomes = data_clean[outcome_column]
            
            # Sonuç değişkeninin tipini belirle
            is_binary = len(outcomes.unique()) == 2
            is_categorical = outcomes.dtype == 'object' or len(outcomes.unique()) <= 10
            
            # Temel sonuç istatistikleri
            outcome_stats = {
                'total_patients': len(outcomes),
                'outcome_type': 'binary' if is_binary else 'categorical' if is_categorical else 'continuous'
            }
            
            if is_binary or is_categorical:
                outcome_counts = outcomes.value_counts()
                outcome_percentages = outcomes.value_counts(normalize=True) * 100
                
                outcome_stats.update({
                    'outcome_distribution': outcome_counts.to_dict(),
                    'outcome_percentages': outcome_percentages.to_dict(),
                    'most_common_outcome': outcome_counts.index[0],
                    'most_common_percentage': float(outcome_percentages.iloc[0])
                })
                
                # Binary outcome için özel metrikler
                if is_binary:
                    positive_outcome = outcome_counts.index[0]  # En yaygın sonucu pozitif kabul et
                    success_rate = outcome_percentages.iloc[0] / 100
                    
                    # Güven aralığı hesaplama (Wilson score interval)
                    n = len(outcomes)
                    p = success_rate
                    z = 1.96  # %95 güven aralığı
                    
                    denominator = 1 + z**2/n
                    centre_adjusted_probability = p + z**2/(2*n)
                    adjusted_standard_deviation = np.sqrt((p*(1-p) + z**2/(4*n))/n)
                    
                    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
                    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
                    
                    outcome_stats.update({
                        'success_rate': float(success_rate),
                        'success_rate_ci_95': (max(0, lower_bound), min(1, upper_bound)),
                        'positive_outcome_label': positive_outcome
                    })
            else:
                # Sürekli değişken için istatistikler
                outcome_stats.update({
                    'mean': float(outcomes.mean()),
                    'median': float(outcomes.median()),
                    'std': float(outcomes.std()),
                    'min': float(outcomes.min()),
                    'max': float(outcomes.max()),
                    'q25': float(outcomes.quantile(0.25)),
                    'q75': float(outcomes.quantile(0.75)),
                    'skewness': float(outcomes.skew()),
                    'kurtosis': float(outcomes.kurtosis())
                })
            
            # Tedavi analizi
            treatment_analysis = None
            if treatment_column and treatment_column in data_clean.columns:
                treatment_analysis = self._analyze_treatment_outcomes(
                    data_clean[outcome_column], 
                    data_clean[treatment_column],
                    is_binary
                )
            
            # Demografik analiz
            demographic_analysis = {}
            if demographic_columns:
                for demo_col in available_demo_cols:
                    if demo_col in data_clean.columns:
                        demographic_analysis[demo_col] = self._analyze_demographic_outcomes(
                            data_clean[outcome_column],
                            data_clean[demo_col],
                            is_binary
                        )
            
            # Risk faktörü analizi
            risk_factor_analysis = None
            if len(demographic_analysis) > 0:
                risk_factor_analysis = self._identify_risk_factors(demographic_analysis, is_binary)
            
            # Survival analizi (eğer binary outcome varsa)
            survival_analysis = None
            if is_binary and len(data_clean) >= 20:
                survival_analysis = self._basic_survival_analysis(outcomes)
            
            result = {
                'analysis_type': 'Clinical Outcome Analysis',
                'outcome_column': outcome_column,
                'treatment_column': treatment_column,
                'demographic_columns': available_demo_cols if demographic_columns else None,
                'outcome_statistics': outcome_stats,
                'treatment_analysis': treatment_analysis,
                'demographic_analysis': demographic_analysis,
                'risk_factor_analysis': risk_factor_analysis,
                'survival_analysis': survival_analysis,
                'interpretation': self._interpret_clinical_outcomes(outcome_stats, treatment_analysis, demographic_analysis)
            }
            
            self.results['clinical_outcome_analysis'] = result
            return result
            
        except Exception as e:
            return {'error': f'Klinik sonuç analizi hatası: {str(e)}'}
    
    def biomarker_analysis(self, biomarker_columns: List[str],
                          reference_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                          group_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Biyobelirteç analizi gerçekleştirir
        
        Args:
            biomarker_columns: Biyobelirteç sütunları
            reference_ranges: Referans aralıkları (opsiyonel)
            group_column: Grup karşılaştırması için sütun (opsiyonel)
            
        Returns:
            Biyobelirteç analizi sonuçları
        """
        try:
            missing_cols = [col for col in biomarker_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal sütunları filtrele
            numeric_biomarkers = []
            for col in biomarker_columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_biomarkers.append(col)
            
            if len(numeric_biomarkers) < 1:
                return {'error': 'En az 1 sayısal biyobelirteç sütunu gereklidir'}
            
            data_clean = self.data[numeric_biomarkers].copy()
            if group_column and group_column in self.data.columns:
                data_clean[group_column] = self.data[group_column]
            
            data_clean = data_clean.dropna()
            
            if len(data_clean) < 10:
                return {'error': 'Biyobelirteç analizi için en az 10 gözlem gereklidir'}
            
            # Her biyobelirteç için analiz
            biomarker_stats = {}
            for biomarker in numeric_biomarkers:
                values = data_clean[biomarker]
                
                # Temel istatistikler
                stats_dict = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'iqr': float(values.quantile(0.75) - values.quantile(0.25)),
                    'cv': float(values.std() / values.mean()) if values.mean() != 0 else None,
                    'skewness': float(values.skew()),
                    'kurtosis': float(values.kurtosis())
                }
                
                # Normallik testi
                if len(values) >= 8:
                    shapiro_stat, shapiro_p = stats.shapiro(values)
                    stats_dict.update({
                        'shapiro_statistic': float(shapiro_stat),
                        'shapiro_p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    })
                
                # Outlier analizi
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                stats_dict.update({
                    'outlier_count': len(outliers),
                    'outlier_percentage': float(len(outliers) / len(values) * 100),
                    'outlier_lower_bound': float(lower_bound),
                    'outlier_upper_bound': float(upper_bound)
                })
                
                # Referans aralığı analizi
                if reference_ranges and biomarker in reference_ranges:
                    ref_min, ref_max = reference_ranges[biomarker]
                    within_range = values[(values >= ref_min) & (values <= ref_max)]
                    below_range = values[values < ref_min]
                    above_range = values[values > ref_max]
                    
                    stats_dict.update({
                        'reference_range': reference_ranges[biomarker],
                        'within_reference_count': len(within_range),
                        'within_reference_percentage': float(len(within_range) / len(values) * 100),
                        'below_reference_count': len(below_range),
                        'below_reference_percentage': float(len(below_range) / len(values) * 100),
                        'above_reference_count': len(above_range),
                        'above_reference_percentage': float(len(above_range) / len(values) * 100)
                    })
                
                biomarker_stats[biomarker] = stats_dict
            
            # Korelasyon analizi
            correlation_analysis = None
            if len(numeric_biomarkers) > 1:
                corr_matrix = data_clean[numeric_biomarkers].corr()
                
                # En yüksek korelasyonları bul
                corr_pairs = []
                for i in range(len(numeric_biomarkers)):
                    for j in range(i+1, len(numeric_biomarkers)):
                        biomarker1 = numeric_biomarkers[i]
                        biomarker2 = numeric_biomarkers[j]
                        corr_value = corr_matrix.loc[biomarker1, biomarker2]
                        
                        corr_pairs.append({
                            'biomarker1': biomarker1,
                            'biomarker2': biomarker2,
                            'correlation': float(corr_value),
                            'correlation_strength': self._interpret_correlation_strength(abs(corr_value))
                        })
                
                # Korelasyona göre sırala
                corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                correlation_analysis = {
                    'correlation_matrix': corr_matrix.to_dict(),
                    'top_correlations': corr_pairs[:5],  # En yüksek 5 korelasyon
                    'average_correlation': float(np.mean([abs(pair['correlation']) for pair in corr_pairs]))
                }
            
            # Grup karşılaştırması
            group_comparison = None
            if group_column and group_column in data_clean.columns:
                group_comparison = self._compare_biomarkers_by_group(
                    data_clean, numeric_biomarkers, group_column
                )
            
            # Anormal değer analizi
            abnormal_analysis = {}
            for biomarker in numeric_biomarkers:
                values = data_clean[biomarker]
                
                # Z-score bazlı anormal değerler
                z_scores = np.abs(stats.zscore(values))
                abnormal_z = values[z_scores > 2]  # |z| > 2
                extreme_z = values[z_scores > 3]   # |z| > 3
                
                abnormal_analysis[biomarker] = {
                    'abnormal_values_count': len(abnormal_z),
                    'abnormal_values_percentage': float(len(abnormal_z) / len(values) * 100),
                    'extreme_values_count': len(extreme_z),
                    'extreme_values_percentage': float(len(extreme_z) / len(values) * 100),
                    'max_z_score': float(z_scores.max())
                }
            
            result = {
                'analysis_type': 'Biomarker Analysis',
                'biomarker_columns': numeric_biomarkers,
                'reference_ranges': reference_ranges,
                'group_column': group_column,
                'n_observations': len(data_clean),
                'biomarker_statistics': biomarker_stats,
                'correlation_analysis': correlation_analysis,
                'group_comparison': group_comparison,
                'abnormal_value_analysis': abnormal_analysis,
                'interpretation': self._interpret_biomarker_analysis(biomarker_stats, correlation_analysis, abnormal_analysis)
            }
            
            self.results['biomarker_analysis'] = result
            return result
            
        except Exception as e:
            return {'error': f'Biyobelirteç analizi hatası: {str(e)}'}
    
    def epidemiological_analysis(self, disease_column: str,
                                exposure_columns: List[str],
                                demographic_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Epidemiyolojik analiz gerçekleştirir
        
        Args:
            disease_column: Hastalık durumu sütunu (0/1 veya True/False)
            exposure_columns: Maruz kalma faktörleri
            demographic_columns: Demografik değişkenler (opsiyonel)
            
        Returns:
            Epidemiyolojik analiz sonuçları
        """
        try:
            required_cols = [disease_column] + exposure_columns
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            data_clean = self.data[required_cols].copy()
            
            # Demografik sütunlar varsa ekle
            if demographic_columns:
                available_demo_cols = [col for col in demographic_columns if col in self.data.columns]
                for col in available_demo_cols:
                    data_clean[col] = self.data[col]
            
            data_clean = data_clean.dropna()
            
            if len(data_clean) < 20:
                return {'error': 'Epidemiyolojik analiz için en az 20 gözlem gereklidir'}
            
            disease = data_clean[disease_column]
            
            # Hastalık sütununu binary'ye çevir
            if disease.dtype == 'object':
                disease = disease.map({'True': 1, 'False': 0, 'true': 1, 'false': 0, 
                                     'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
            disease = disease.astype(int)
            
            # Hastalık prevalansı
            disease_prevalence = {
                'total_cases': int(disease.sum()),
                'total_population': len(disease),
                'prevalence': float(disease.mean()),
                'prevalence_percentage': float(disease.mean() * 100),
                'prevalence_ci_95': self._calculate_proportion_ci(disease.sum(), len(disease))
            }
            
            # Risk faktörü analizi
            risk_factor_analysis = {}
            for exposure in exposure_columns:
                exposure_data = data_clean[exposure]
                
                # Exposure'ı kategorik hale getir
                if pd.api.types.is_numeric_dtype(exposure_data):
                    # Sürekli değişken için median split
                    median_val = exposure_data.median()
                    exposure_binary = (exposure_data > median_val).astype(int)
                    exposure_labels = {0: f'<= {median_val:.2f}', 1: f'> {median_val:.2f}'}
                else:
                    # Kategorik değişken
                    exposure_binary = exposure_data
                    exposure_labels = {val: str(val) for val in exposure_binary.unique()}
                
                # 2x2 contingency table
                contingency = pd.crosstab(exposure_binary, disease)
                
                if contingency.shape == (2, 2):
                    # Risk hesaplamaları
                    a = contingency.iloc[1, 1]  # Exposed + Disease
                    b = contingency.iloc[1, 0]  # Exposed + No Disease
                    c = contingency.iloc[0, 1]  # Not Exposed + Disease
                    d = contingency.iloc[0, 0]  # Not Exposed + No Disease
                    
                    # Risk oranları
                    risk_exposed = a / (a + b) if (a + b) > 0 else 0
                    risk_not_exposed = c / (c + d) if (c + d) > 0 else 0
                    
                    # Relative Risk (RR)
                    relative_risk = risk_exposed / risk_not_exposed if risk_not_exposed > 0 else float('inf')
                    
                    # Odds Ratio (OR)
                    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')
                    
                    # Attributable Risk
                    attributable_risk = risk_exposed - risk_not_exposed
                    
                    # Chi-square test
                    chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)
                    
                    # Fisher's exact test (küçük örneklemler için)
                    if min(contingency.values.flatten()) < 5:
                        fisher_odds, fisher_p = fisher_exact(contingency)
                    else:
                        fisher_odds, fisher_p = None, None
                    
                    risk_factor_analysis[exposure] = {
                        'contingency_table': contingency.to_dict(),
                        'exposure_labels': exposure_labels,
                        'risk_exposed': float(risk_exposed),
                        'risk_not_exposed': float(risk_not_exposed),
                        'relative_risk': float(relative_risk) if relative_risk != float('inf') else None,
                        'odds_ratio': float(odds_ratio) if odds_ratio != float('inf') else None,
                        'attributable_risk': float(attributable_risk),
                        'attributable_risk_percentage': float(attributable_risk / risk_exposed * 100) if risk_exposed > 0 else 0,
                        'chi2_statistic': float(chi2_stat),
                        'chi2_p_value': float(chi2_p),
                        'fisher_odds_ratio': float(fisher_odds) if fisher_odds else None,
                        'fisher_p_value': float(fisher_p) if fisher_p else None,
                        'significant': chi2_p < 0.05
                    }
            
            # Demografik risk analizi
            demographic_risk_analysis = {}
            if demographic_columns:
                for demo_col in available_demo_cols:
                    if demo_col in data_clean.columns:
                        demo_data = data_clean[demo_col]
                        
                        # Demografik gruplara göre hastalık oranları
                        demo_disease_rates = {}
                        for demo_value in demo_data.unique():
                            demo_mask = demo_data == demo_value
                            demo_disease = disease[demo_mask]
                            
                            if len(demo_disease) > 0:
                                demo_disease_rates[str(demo_value)] = {
                                    'count': len(demo_disease),
                                    'cases': int(demo_disease.sum()),
                                    'rate': float(demo_disease.mean()),
                                    'rate_percentage': float(demo_disease.mean() * 100)
                                }
                        
                        demographic_risk_analysis[demo_col] = demo_disease_rates
            
            # Population Attributable Risk (PAR)
            population_attributable_risk = {}
            for exposure in exposure_columns:
                if exposure in risk_factor_analysis:
                    rr = risk_factor_analysis[exposure]['relative_risk']
                    if rr and rr != float('inf'):
                        # Exposure prevalansını hesapla
                        exposure_data = data_clean[exposure]
                        if pd.api.types.is_numeric_dtype(exposure_data):
                            exposure_prev = (exposure_data > exposure_data.median()).mean()
                        else:
                            # En yaygın kategoriyi exposed kabul et
                            exposure_prev = (exposure_data == exposure_data.mode()[0]).mean()
                        
                        par = (exposure_prev * (rr - 1)) / (1 + exposure_prev * (rr - 1))
                        
                        population_attributable_risk[exposure] = {
                            'exposure_prevalence': float(exposure_prev),
                            'relative_risk': float(rr),
                            'population_attributable_risk': float(par),
                            'population_attributable_risk_percentage': float(par * 100)
                        }
            
            result = {
                'analysis_type': 'Epidemiological Analysis',
                'disease_column': disease_column,
                'exposure_columns': exposure_columns,
                'demographic_columns': available_demo_cols if demographic_columns else None,
                'n_observations': len(data_clean),
                'disease_prevalence': disease_prevalence,
                'risk_factor_analysis': risk_factor_analysis,
                'demographic_risk_analysis': demographic_risk_analysis,
                'population_attributable_risk': population_attributable_risk,
                'interpretation': self._interpret_epidemiological_analysis(disease_prevalence, risk_factor_analysis)
            }
            
            self.results['epidemiological_analysis'] = result
            return result
            
        except Exception as e:
            return {'error': f'Epidemiyolojik analiz hatası: {str(e)}'}
    
    def _analyze_treatment_outcomes(self, outcomes: pd.Series, treatments: pd.Series, is_binary: bool) -> Dict[str, Any]:
        """Tedavi sonuçlarını analiz eder"""
        try:
            treatment_stats = {}
            
            for treatment in treatments.unique():
                treatment_mask = treatments == treatment
                treatment_outcomes = outcomes[treatment_mask]
                
                if is_binary:
                    success_rate = treatment_outcomes.value_counts(normalize=True).iloc[0]
                    treatment_stats[str(treatment)] = {
                        'n_patients': len(treatment_outcomes),
                        'success_rate': float(success_rate),
                        'success_count': int(treatment_outcomes.value_counts().iloc[0]),
                        'outcome_distribution': treatment_outcomes.value_counts().to_dict()
                    }
                else:
                    treatment_stats[str(treatment)] = {
                        'n_patients': len(treatment_outcomes),
                        'mean_outcome': float(treatment_outcomes.mean()),
                        'median_outcome': float(treatment_outcomes.median()),
                        'std_outcome': float(treatment_outcomes.std())
                    }
            
            # Tedaviler arası karşılaştırma
            if len(treatments.unique()) >= 2:
                if is_binary:
                    # Chi-square test
                    contingency = pd.crosstab(treatments, outcomes)
                    chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)
                    
                    comparison = {
                        'test_type': 'Chi-square',
                        'statistic': float(chi2_stat),
                        'p_value': float(chi2_p),
                        'significant_difference': chi2_p < 0.05
                    }
                else:
                    # ANOVA veya Kruskal-Wallis
                    treatment_groups = [outcomes[treatments == t] for t in treatments.unique()]
                    
                    # Normallik kontrolü
                    normal_groups = all(len(group) >= 8 and stats.shapiro(group)[1] > 0.05 for group in treatment_groups)
                    
                    if normal_groups:
                        f_stat, f_p = stats.f_oneway(*treatment_groups)
                        comparison = {
                            'test_type': 'ANOVA',
                            'statistic': float(f_stat),
                            'p_value': float(f_p),
                            'significant_difference': f_p < 0.05
                        }
                    else:
                        h_stat, h_p = kruskal(*treatment_groups)
                        comparison = {
                            'test_type': 'Kruskal-Wallis',
                            'statistic': float(h_stat),
                            'p_value': float(h_p),
                            'significant_difference': h_p < 0.05
                        }
                
                return {
                    'treatment_statistics': treatment_stats,
                    'statistical_comparison': comparison
                }
            else:
                return {'treatment_statistics': treatment_stats}
                
        except Exception as e:
            return {'error': f'Tedavi analizi hatası: {str(e)}'}
    
    def _analyze_demographic_outcomes(self, outcomes: pd.Series, demographics: pd.Series, is_binary: bool) -> Dict[str, Any]:
        """Demografik gruplara göre sonuçları analiz eder"""
        try:
            demo_stats = {}
            
            for demo_value in demographics.unique():
                demo_mask = demographics == demo_value
                demo_outcomes = outcomes[demo_mask]
                
                if len(demo_outcomes) > 0:
                    if is_binary:
                        success_rate = demo_outcomes.value_counts(normalize=True).iloc[0]
                        demo_stats[str(demo_value)] = {
                            'n_patients': len(demo_outcomes),
                            'success_rate': float(success_rate),
                            'success_count': int(demo_outcomes.value_counts().iloc[0])
                        }
                    else:
                        demo_stats[str(demo_value)] = {
                            'n_patients': len(demo_outcomes),
                            'mean_outcome': float(demo_outcomes.mean()),
                            'median_outcome': float(demo_outcomes.median()),
                            'std_outcome': float(demo_outcomes.std())
                        }
            
            return demo_stats
            
        except Exception as e:
            return {'error': f'Demografik analiz hatası: {str(e)}'}
    
    def _identify_risk_factors(self, demographic_analysis: Dict, is_binary: bool) -> Dict[str, Any]:
        """Risk faktörlerini belirler"""
        try:
            risk_factors = {}
            
            for demo_var, demo_stats in demographic_analysis.items():
                if isinstance(demo_stats, dict) and not 'error' in demo_stats:
                    if is_binary:
                        # En yüksek ve en düşük başarı oranlarını bul
                        success_rates = {k: v['success_rate'] for k, v in demo_stats.items() if 'success_rate' in v}
                        if success_rates:
                            max_success = max(success_rates.values())
                            min_success = min(success_rates.values())
                            
                            risk_factors[demo_var] = {
                                'highest_success_rate': max_success,
                                'lowest_success_rate': min_success,
                                'risk_difference': max_success - min_success,
                                'high_risk_group': min(success_rates.keys(), key=success_rates.get),
                                'low_risk_group': max(success_rates.keys(), key=success_rates.get)
                            }
            
            return risk_factors
            
        except Exception as e:
            return {'error': f'Risk faktörü analizi hatası: {str(e)}'}
    
    def _basic_survival_analysis(self, outcomes: pd.Series) -> Dict[str, Any]:
        """Temel survival analizi"""
        try:
            # Binary outcome için basit survival metrikleri
            total_patients = len(outcomes)
            survivors = outcomes.sum() if outcomes.dtype in ['int64', 'float64'] else len(outcomes[outcomes == outcomes.mode()[0]])
            
            survival_rate = survivors / total_patients
            mortality_rate = 1 - survival_rate
            
            return {
                'total_patients': total_patients,
                'survivors': int(survivors),
                'deaths': int(total_patients - survivors),
                'survival_rate': float(survival_rate),
                'mortality_rate': float(mortality_rate),
                'survival_percentage': float(survival_rate * 100),
                'mortality_percentage': float(mortality_rate * 100)
            }
            
        except Exception as e:
            return {'error': f'Survival analizi hatası: {str(e)}'}
    
    def _compare_biomarkers_by_group(self, data: pd.DataFrame, biomarkers: List[str], group_col: str) -> Dict[str, Any]:
        """Gruplara göre biyobelirteçleri karşılaştırır"""
        try:
            group_comparison = {}
            
            for biomarker in biomarkers:
                biomarker_data = data[biomarker]
                groups = data[group_col]
                
                # Grup istatistikleri
                group_stats = {}
                for group in groups.unique():
                    group_mask = groups == group
                    group_biomarker = biomarker_data[group_mask]
                    
                    if len(group_biomarker) > 0:
                        group_stats[str(group)] = {
                            'count': len(group_biomarker),
                            'mean': float(group_biomarker.mean()),
                            'median': float(group_biomarker.median()),
                            'std': float(group_biomarker.std())
                        }
                
                # İstatistiksel test
                if len(groups.unique()) == 2:
                    # İki grup karşılaştırması
                    group1_data = biomarker_data[groups == groups.unique()[0]]
                    group2_data = biomarker_data[groups == groups.unique()[1]]
                    
                    # Normallik kontrolü
                    if len(group1_data) >= 8 and len(group2_data) >= 8:
                        normal1 = stats.shapiro(group1_data)[1] > 0.05
                        normal2 = stats.shapiro(group2_data)[1] > 0.05
                        
                        if normal1 and normal2:
                            # t-test
                            t_stat, t_p = stats.ttest_ind(group1_data, group2_data)
                            test_result = {
                                'test_type': 'Independent t-test',
                                'statistic': float(t_stat),
                                'p_value': float(t_p)
                            }
                        else:
                            # Mann-Whitney U test
                            u_stat, u_p = mannwhitneyu(group1_data, group2_data)
                            test_result = {
                                'test_type': 'Mann-Whitney U',
                                'statistic': float(u_stat),
                                'p_value': float(u_p)
                            }
                    else:
                        test_result = {'test_type': 'Insufficient data for testing'}
                
                elif len(groups.unique()) > 2:
                    # Çoklu grup karşılaştırması
                    group_data_list = [biomarker_data[groups == group] for group in groups.unique()]
                    
                    # Kruskal-Wallis test
                    h_stat, h_p = kruskal(*group_data_list)
                    test_result = {
                        'test_type': 'Kruskal-Wallis',
                        'statistic': float(h_stat),
                        'p_value': float(h_p)
                    }
                else:
                    test_result = {'test_type': 'Single group, no comparison possible'}
                
                test_result['significant_difference'] = test_result.get('p_value', 1) < 0.05
                
                group_comparison[biomarker] = {
                    'group_statistics': group_stats,
                    'statistical_test': test_result
                }
            
            return group_comparison
            
        except Exception as e:
            return {'error': f'Grup karşılaştırması hatası: {str(e)}'}
    
    def _calculate_proportion_ci(self, successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Oran için güven aralığı hesaplar (Wilson score interval)"""
        if total == 0:
            return (0.0, 0.0)
        
        p = successes / total
        z = stats.norm.ppf((1 + confidence) / 2)
        
        denominator = 1 + z**2/total
        centre_adjusted_probability = p + z**2/(2*total)
        adjusted_standard_deviation = np.sqrt((p*(1-p) + z**2/(4*total))/total)
        
        lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
        
        return (max(0, lower_bound), min(1, upper_bound))
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Korelasyon gücünü yorumlar"""
        if correlation >= 0.8:
            return 'Çok güçlü'
        elif correlation >= 0.6:
            return 'Güçlü'
        elif correlation >= 0.4:
            return 'Orta'
        elif correlation >= 0.2:
            return 'Zayıf'
        else:
            return 'Çok zayıf'
    
    def _interpret_clinical_outcomes(self, outcome_stats: Dict, treatment_analysis: Dict, demographic_analysis: Dict) -> str:
        """Klinik sonuç analizi sonuçlarını yorumlar"""
        interpretation = "Klinik sonuç analizi: "
        
        if outcome_stats['outcome_type'] == 'binary':
            success_rate = outcome_stats.get('success_rate', 0) * 100
            if success_rate > 80:
                interpretation += f"Yüksek başarı oranı (%{success_rate:.1f}). "
            elif success_rate > 60:
                interpretation += f"Orta başarı oranı (%{success_rate:.1f}). "
            else:
                interpretation += f"Düşük başarı oranı (%{success_rate:.1f}). "
        
        if treatment_analysis and 'statistical_comparison' in treatment_analysis:
            if treatment_analysis['statistical_comparison']['significant_difference']:
                interpretation += "Tedaviler arasında anlamlı fark var. "
            else:
                interpretation += "Tedaviler arasında anlamlı fark yok. "
        
        return interpretation
    
    def _interpret_biomarker_analysis(self, biomarker_stats: Dict, correlation_analysis: Dict, abnormal_analysis: Dict) -> str:
        """Biyobelirteç analizi sonuçlarını yorumlar"""
        interpretation = "Biyobelirteç analizi: "
        
        # En anormal değerlere sahip biyobelirteç
        max_abnormal_biomarker = max(abnormal_analysis.keys(), 
                                   key=lambda x: abnormal_analysis[x]['abnormal_values_percentage'])
        max_abnormal_pct = abnormal_analysis[max_abnormal_biomarker]['abnormal_values_percentage']
        
        if max_abnormal_pct > 20:
            interpretation += f"{max_abnormal_biomarker} yüksek anormal değer oranına sahip (%{max_abnormal_pct:.1f}). "
        
        # Korelasyon yorumu
        if correlation_analysis and 'top_correlations' in correlation_analysis:
            top_corr = correlation_analysis['top_correlations'][0]
            if abs(top_corr['correlation']) > 0.7:
                interpretation += f"Güçlü korelasyon: {top_corr['biomarker1']} - {top_corr['biomarker2']}. "
        
        return interpretation
    
    def _interpret_epidemiological_analysis(self, disease_prevalence: Dict, risk_factor_analysis: Dict) -> str:
        """Epidemiyolojik analiz sonuçlarını yorumlar"""
        interpretation = "Epidemiyolojik analiz: "
        
        prevalence_pct = disease_prevalence['prevalence_percentage']
        if prevalence_pct > 20:
            interpretation += f"Yüksek hastalık prevalansı (%{prevalence_pct:.1f}). "
        elif prevalence_pct > 10:
            interpretation += f"Orta hastalık prevalansı (%{prevalence_pct:.1f}). "
        else:
            interpretation += f"Düşük hastalık prevalansı (%{prevalence_pct:.1f}). "
        
        # En yüksek risk faktörü
        significant_risks = {k: v for k, v in risk_factor_analysis.items() if v.get('significant', False)}
        if significant_risks:
            max_risk_factor = max(significant_risks.keys(), 
                                key=lambda x: significant_risks[x].get('relative_risk', 1))
            max_rr = significant_risks[max_risk_factor].get('relative_risk', 1)
            interpretation += f"En yüksek risk faktörü: {max_risk_factor} (RR: {max_rr:.2f})."
        
        return interpretation
    
    def get_healthcare_summary(self) -> Dict[str, Any]:
        """
        Gerçekleştirilen tüm sağlık analizlerinin özetini döndürür
        
        Returns:
            Sağlık analiz özetleri
        """
        if not self.results:
            return {'message': 'Henüz sağlık analizi gerçekleştirilmedi'}
        
        summary = {
            'total_analyses': len(self.results),
            'analyses_performed': list(self.results.keys()),
            'analysis_details': self.results
        }
        
        # En önemli bulgular
        key_findings = []
        
        if 'clinical_outcome_analysis' in self.results:
            clinical = self.results['clinical_outcome_analysis']
            if 'outcome_statistics' in clinical:
                if clinical['outcome_statistics']['outcome_type'] == 'binary':
                    success_rate = clinical['outcome_statistics'].get('success_rate', 0) * 100
                    key_findings.append(f"Başarı oranı: %{success_rate:.1f}")
        
        if 'biomarker_analysis' in self.results:
            biomarker = self.results['biomarker_analysis']
            if 'abnormal_value_analysis' in biomarker:
                max_abnormal = max(biomarker['abnormal_value_analysis'].values(), 
                                 key=lambda x: x['abnormal_values_percentage'])
                key_findings.append(f"En yüksek anormal değer oranı: %{max_abnormal['abnormal_values_percentage']:.1f}")
        
        if 'epidemiological_analysis' in self.results:
            epi = self.results['epidemiological_analysis']
            prevalence = epi['disease_prevalence']['prevalence_percentage']
            key_findings.append(f"Hastalık prevalansı: %{prevalence:.1f}")
        
        summary['key_findings'] = key_findings
        
        return summary