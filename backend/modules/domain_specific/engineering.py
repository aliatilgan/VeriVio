"""
Mühendislik Analizi Sınıfı

Bu modül mühendislik verilerinin analizi için özel araçlar içerir.
DOE (Design of Experiments), Six Sigma, risk analizi gibi işlevler sunar.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal
from scipy.stats import pearsonr, spearmanr, kendalltau, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class EngineeringAnalyzer:
    """
    Mühendislik veri analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        EngineeringAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek mühendislik veri seti
        """
        self.data = data.copy()
        self.results = {}
        
    def design_of_experiments_analysis(self, response_variable: str, 
                                     factor_columns: List[str],
                                     block_variable: Optional[str] = None) -> Dict[str, Any]:
        """
        Deney Tasarımı (DOE) analizi gerçekleştirir
        
        Args:
            response_variable: Yanıt değişkeni
            factor_columns: Faktör sütunları
            block_variable: Blok değişkeni (opsiyonel)
            
        Returns:
            DOE analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            all_columns = [response_variable] + factor_columns
            if block_variable:
                all_columns.append(block_variable)
            
            missing_cols = [col for col in all_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Yanıt değişkeninin sayısal olduğunu kontrol et
            if not pd.api.types.is_numeric_dtype(self.data[response_variable]):
                return {'error': f'{response_variable} sayısal bir sütun olmalıdır'}
            
            data_clean = self.data[all_columns].dropna()
            
            if len(data_clean) < 10:
                return {'error': 'DOE analizi için en az 10 gözlem gereklidir'}
            
            response = data_clean[response_variable]
            
            # Faktör etkilerini analiz et
            factor_effects = {}
            main_effects = {}
            
            for factor in factor_columns:
                if factor in data_clean.columns:
                    factor_levels = data_clean[factor].unique()
                    
                    if len(factor_levels) >= 2:
                        # Ana etki analizi
                        groups = [data_clean[data_clean[factor] == level][response_variable].values 
                                for level in factor_levels]
                        
                        # ANOVA testi
                        if len(factor_levels) > 2:
                            f_stat, f_p = f_oneway(*groups)
                            test_type = 'ANOVA'
                        else:
                            f_stat, f_p = stats.ttest_ind(groups[0], groups[1])
                            test_type = 't-Test'
                        
                        # Etki büyüklüğü (Eta squared)
                        ss_between = sum([len(group) * (np.mean(group) - response.mean())**2 
                                        for group in groups])
                        ss_total = sum([(x - response.mean())**2 for x in response])
                        eta_squared = ss_between / ss_total if ss_total > 0 else 0
                        
                        # Grup istatistikleri
                        group_stats = {}
                        for i, level in enumerate(factor_levels):
                            group_data = groups[i]
                            group_stats[str(level)] = {
                                'mean': float(np.mean(group_data)),
                                'std': float(np.std(group_data)),
                                'count': len(group_data)
                            }
                        
                        main_effects[factor] = {
                            'test_type': test_type,
                            'statistic': float(f_stat),
                            'p_value': float(f_p),
                            'significant': f_p < 0.05,
                            'eta_squared': float(eta_squared),
                            'group_statistics': group_stats
                        }
            
            # İki faktörlü etkileşimler
            interaction_effects = {}
            if len(factor_columns) >= 2:
                for i in range(len(factor_columns)):
                    for j in range(i+1, len(factor_columns)):
                        factor1, factor2 = factor_columns[i], factor_columns[j]
                        
                        if factor1 in data_clean.columns and factor2 in data_clean.columns:
                            interaction_analysis = self._analyze_interaction(
                                data_clean, response_variable, factor1, factor2
                            )
                            interaction_effects[f'{factor1}_x_{factor2}'] = interaction_analysis
            
            # Kalıntı analizi
            residual_analysis = self._residual_analysis(data_clean, response_variable, factor_columns)
            
            # Model uyum istatistikleri
            model_fit = self._calculate_model_fit(data_clean, response_variable, factor_columns)
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'response_variable_stats': {
                    'mean': float(response.mean()),
                    'std': float(response.std()),
                    'min': float(response.min()),
                    'max': float(response.max())
                },
                'main_effects': main_effects,
                'interaction_effects': interaction_effects,
                'residual_analysis': residual_analysis,
                'model_fit': model_fit,
                'interpretation': self._interpret_doe_results(main_effects, interaction_effects, model_fit)
            }
            
            return results
            
        except Exception as e:
            return {'error': f'DOE analizi hatası: {str(e)}'}
    
    def six_sigma_analysis(self, measurement_column: str, 
                          specification_limits: Tuple[float, float],
                          target_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Six Sigma analizi gerçekleştirir
        
        Args:
            measurement_column: Ölçüm değerleri sütunu
            specification_limits: Spesifikasyon limitleri (alt, üst)
            target_value: Hedef değer (opsiyonel)
            
        Returns:
            Six Sigma analizi sonuçları
        """
        try:
            if measurement_column not in self.data.columns:
                return {'error': f'{measurement_column} sütunu bulunamadı'}
            
            if not pd.api.types.is_numeric_dtype(self.data[measurement_column]):
                return {'error': f'{measurement_column} sayısal bir sütun olmalıdır'}
            
            data_clean = self.data[measurement_column].dropna()
            
            if len(data_clean) < 30:
                return {'error': 'Six Sigma analizi için en az 30 ölçüm gereklidir'}
            
            lsl, usl = specification_limits  # Lower/Upper Specification Limits
            
            if target_value is None:
                target_value = (lsl + usl) / 2
            
            # Temel istatistikler
            mean_val = data_clean.mean()
            std_val = data_clean.std()
            
            # Process Capability İndeksleri
            # Cp (Process Capability)
            cp = (usl - lsl) / (6 * std_val) if std_val > 0 else 0
            
            # Cpk (Process Capability Index)
            cpu = (usl - mean_val) / (3 * std_val) if std_val > 0 else 0
            cpl = (mean_val - lsl) / (3 * std_val) if std_val > 0 else 0
            cpk = min(cpu, cpl)
            
            # Pp ve Ppk (Performance indices)
            pp = (usl - lsl) / (6 * std_val) if std_val > 0 else 0
            ppu = (usl - mean_val) / (3 * std_val) if std_val > 0 else 0
            ppl = (mean_val - lsl) / (3 * std_val) if std_val > 0 else 0
            ppk = min(ppu, ppl)
            
            # Sigma Level hesaplama
            sigma_level = min(cpu, cpl) * 3 if std_val > 0 else 0
            
            # Defect Rate hesaplama
            out_of_spec = ((data_clean < lsl) | (data_clean > usl)).sum()
            defect_rate = out_of_spec / len(data_clean) * 1000000  # PPM
            
            # DPMO (Defects Per Million Opportunities)
            dpmo = defect_rate
            
            # Yield hesaplama
            yield_rate = (1 - out_of_spec / len(data_clean)) * 100
            
            # Process Performance
            within_spec = ((data_clean >= lsl) & (data_clean <= usl)).sum()
            process_performance = within_spec / len(data_clean) * 100
            
            # Normallik testi
            shapiro_stat, shapiro_p = stats.shapiro(data_clean)
            
            # Control Chart verisi
            control_limits = {
                'ucl': mean_val + 3 * std_val,  # Upper Control Limit
                'lcl': mean_val - 3 * std_val,  # Lower Control Limit
                'center_line': mean_val
            }
            
            # Capability yorumu
            capability_interpretation = self._interpret_capability(cp, cpk)
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'basic_statistics': {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(data_clean.min()),
                    'max': float(data_clean.max())
                },
                'specification_limits': {
                    'lsl': lsl,
                    'usl': usl,
                    'target': target_value
                },
                'capability_indices': {
                    'cp': float(cp),
                    'cpk': float(cpk),
                    'cpu': float(cpu),
                    'cpl': float(cpl),
                    'pp': float(pp),
                    'ppk': float(ppk)
                },
                'performance_metrics': {
                    'sigma_level': float(sigma_level),
                    'defect_rate_ppm': float(dpmo),
                    'yield_percentage': float(yield_rate),
                    'process_performance': float(process_performance)
                },
                'control_limits': control_limits,
                'normality_test': {
                    'shapiro_statistic': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                },
                'capability_interpretation': capability_interpretation,
                'interpretation': self._interpret_six_sigma_results(
                    cp, cpk, sigma_level, yield_rate, dpmo
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Six Sigma analizi hatası: {str(e)}'}
    
    def risk_analysis(self, risk_factors: List[str], 
                     impact_column: str,
                     probability_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Risk analizi gerçekleştirir
        
        Args:
            risk_factors: Risk faktörü sütunları
            impact_column: Etki/sonuç sütunu
            probability_column: Olasılık sütunu (opsiyonel)
            
        Returns:
            Risk analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            all_columns = risk_factors + [impact_column]
            if probability_column:
                all_columns.append(probability_column)
            
            missing_cols = [col for col in all_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Etki sütununun sayısal olduğunu kontrol et
            if not pd.api.types.is_numeric_dtype(self.data[impact_column]):
                return {'error': f'{impact_column} sayısal bir sütun olmalıdır'}
            
            data_clean = self.data[all_columns].dropna()
            
            if len(data_clean) < 20:
                return {'error': 'Risk analizi için en az 20 gözlem gereklidir'}
            
            impact_values = data_clean[impact_column]
            
            # Risk faktörlerinin etkisini analiz et
            risk_factor_analysis = {}
            
            for factor in risk_factors:
                if factor in data_clean.columns:
                    factor_analysis = self._analyze_risk_factor(
                        data_clean, factor, impact_column
                    )
                    risk_factor_analysis[factor] = factor_analysis
            
            # Risk skorları hesaplama
            if probability_column and probability_column in data_clean.columns:
                probability_values = data_clean[probability_column]
                risk_scores = impact_values * probability_values
                
                risk_score_analysis = {
                    'mean_risk_score': float(risk_scores.mean()),
                    'std_risk_score': float(risk_scores.std()),
                    'max_risk_score': float(risk_scores.max()),
                    'min_risk_score': float(risk_scores.min()),
                    'high_risk_threshold': float(risk_scores.quantile(0.9)),
                    'high_risk_count': int((risk_scores > risk_scores.quantile(0.9)).sum())
                }
            else:
                risk_score_analysis = {'message': 'Olasılık sütunu olmadığı için risk skoru hesaplanamadı'}
            
            # Risk kategorileri
            risk_categories = self._categorize_risks(impact_values)
            category_distribution = risk_categories.value_counts()
            category_percentages = risk_categories.value_counts(normalize=True) * 100
            
            # Monte Carlo simülasyonu (basit)
            monte_carlo_results = self._monte_carlo_risk_simulation(
                data_clean, impact_column, probability_column
            )
            
            # Risk matrisi
            risk_matrix = self._create_risk_matrix(data_clean, impact_column, probability_column)
            
            # Korelasyon analizi
            numeric_factors = [col for col in risk_factors 
                             if pd.api.types.is_numeric_dtype(data_clean[col])]
            
            correlation_analysis = {}
            if len(numeric_factors) > 0:
                correlation_matrix = data_clean[numeric_factors + [impact_column]].corr()
                correlation_analysis = {
                    'correlation_matrix': correlation_matrix.to_dict(),
                    'strongest_correlations': self._find_strongest_correlations(
                        correlation_matrix, impact_column
                    )
                }
            
            # Sonuçları birleştir
            results = {
                'sample_size': len(data_clean),
                'impact_statistics': {
                    'mean': float(impact_values.mean()),
                    'std': float(impact_values.std()),
                    'min': float(impact_values.min()),
                    'max': float(impact_values.max()),
                    'q95': float(impact_values.quantile(0.95))
                },
                'risk_factor_analysis': risk_factor_analysis,
                'risk_score_analysis': risk_score_analysis,
                'risk_categories': {
                    'distribution': category_distribution.to_dict(),
                    'percentages': category_percentages.to_dict()
                },
                'monte_carlo_results': monte_carlo_results,
                'risk_matrix': risk_matrix,
                'correlation_analysis': correlation_analysis,
                'interpretation': self._interpret_risk_results(
                    risk_factor_analysis, risk_score_analysis, category_distribution
                )
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Risk analizi hatası: {str(e)}'}
    
    def _analyze_interaction(self, data: pd.DataFrame, 
                           response_var: str, 
                           factor1: str, 
                           factor2: str) -> Dict[str, Any]:
        """
        İki faktör arasındaki etkileşimi analiz eder
        """
        try:
            # İki yönlü ANOVA için grupları oluştur
            groups = []
            group_labels = []
            
            for level1 in data[factor1].unique():
                for level2 in data[factor2].unique():
                    group_data = data[
                        (data[factor1] == level1) & (data[factor2] == level2)
                    ][response_var]
                    
                    if len(group_data) > 0:
                        groups.append(group_data.values)
                        group_labels.append(f'{level1}_{level2}')
            
            if len(groups) < 2:
                return {'error': 'Etkileşim analizi için yeterli grup yok'}
            
            # ANOVA testi
            f_stat, f_p = f_oneway(*groups)
            
            # Grup istatistikleri
            group_stats = {}
            for i, label in enumerate(group_labels):
                group_stats[label] = {
                    'mean': float(np.mean(groups[i])),
                    'std': float(np.std(groups[i])),
                    'count': len(groups[i])
                }
            
            return {
                'f_statistic': float(f_stat),
                'p_value': float(f_p),
                'significant': f_p < 0.05,
                'group_statistics': group_stats
            }
            
        except Exception as e:
            return {'error': f'Etkileşim analizi hatası: {str(e)}'}
    
    def _residual_analysis(self, data: pd.DataFrame, 
                          response_var: str, 
                          factors: List[str]) -> Dict[str, Any]:
        """
        Kalıntı analizi yapar
        """
        try:
            # Basit doğrusal model için kalıntıları hesapla
            response = data[response_var]
            
            # Faktörlerin ortalama etkisini hesapla
            predicted = np.full(len(response), response.mean())
            
            for factor in factors:
                if factor in data.columns:
                    factor_means = data.groupby(factor)[response_var].mean()
                    for level, mean_val in factor_means.items():
                        mask = data[factor] == level
                        predicted[mask] = mean_val
            
            residuals = response - predicted
            
            # Kalıntı istatistikleri
            residual_stats = {
                'mean': float(residuals.mean()),
                'std': float(residuals.std()),
                'min': float(residuals.min()),
                'max': float(residuals.max()),
                'skewness': float(residuals.skew()),
                'kurtosis': float(residuals.kurtosis())
            }
            
            # Normallik testi
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            
            return {
                'residual_statistics': residual_stats,
                'normality_test': {
                    'shapiro_statistic': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            }
            
        except Exception as e:
            return {'error': f'Kalıntı analizi hatası: {str(e)}'}
    
    def _calculate_model_fit(self, data: pd.DataFrame, 
                           response_var: str, 
                           factors: List[str]) -> Dict[str, Any]:
        """
        Model uyum istatistiklerini hesaplar
        """
        try:
            response = data[response_var]
            
            # Basit R-squared hesaplama
            ss_total = ((response - response.mean()) ** 2).sum()
            
            # Faktör etkilerini hesapla
            ss_explained = 0
            for factor in factors:
                if factor in data.columns:
                    factor_means = data.groupby(factor)[response_var].mean()
                    factor_ss = 0
                    for level, mean_val in factor_means.items():
                        group_size = (data[factor] == level).sum()
                        factor_ss += group_size * (mean_val - response.mean()) ** 2
                    ss_explained += factor_ss
            
            r_squared = ss_explained / ss_total if ss_total > 0 else 0
            
            # Adjusted R-squared
            n = len(data)
            p = len(factors)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else 0
            
            return {
                'r_squared': float(r_squared),
                'adjusted_r_squared': float(adj_r_squared),
                'ss_total': float(ss_total),
                'ss_explained': float(ss_explained)
            }
            
        except Exception as e:
            return {'error': f'Model uyum hesaplama hatası: {str(e)}'}
    
    def _analyze_risk_factor(self, data: pd.DataFrame, 
                           factor: str, 
                           impact_column: str) -> Dict[str, Any]:
        """
        Tek bir risk faktörünü analiz eder
        """
        try:
            if pd.api.types.is_numeric_dtype(data[factor]):
                # Sayısal faktör için korelasyon
                correlation, p_value = pearsonr(data[factor], data[impact_column])
                
                return {
                    'type': 'numeric',
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': f"{'Güçlü' if abs(correlation) > 0.7 else 'Orta' if abs(correlation) > 0.3 else 'Zayıf'} korelasyon"
                }
            else:
                # Kategorik faktör için grup karşılaştırması
                groups = [data[data[factor] == level][impact_column].values 
                         for level in data[factor].unique()]
                
                if len(groups) > 2:
                    f_stat, f_p = f_oneway(*groups)
                    test_type = 'ANOVA'
                    statistic = f_stat
                else:
                    t_stat, t_p = stats.ttest_ind(groups[0], groups[1])
                    test_type = 't-Test'
                    statistic = t_stat
                    f_p = t_p
                
                # Grup istatistikleri
                group_stats = {}
                for level in data[factor].unique():
                    group_data = data[data[factor] == level][impact_column]
                    group_stats[str(level)] = {
                        'mean': float(group_data.mean()),
                        'std': float(group_data.std()),
                        'count': len(group_data)
                    }
                
                return {
                    'type': 'categorical',
                    'test_type': test_type,
                    'statistic': float(statistic),
                    'p_value': float(f_p),
                    'significant': f_p < 0.05,
                    'group_statistics': group_stats
                }
                
        except Exception as e:
            return {'error': f'Risk faktörü analizi hatası: {str(e)}'}
    
    def _categorize_risks(self, impact_values: pd.Series) -> pd.Series:
        """
        Risk değerlerini kategorilere ayırır
        """
        # Percentile bazlı kategorileme
        q1, q2, q3 = impact_values.quantile([0.33, 0.67, 0.9])
        
        categories = pd.cut(
            impact_values,
            bins=[-np.inf, q1, q2, q3, np.inf],
            labels=['Düşük Risk', 'Orta Risk', 'Yüksek Risk', 'Kritik Risk']
        )
        
        return categories
    
    def _monte_carlo_risk_simulation(self, data: pd.DataFrame, 
                                   impact_column: str, 
                                   probability_column: Optional[str],
                                   n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Monte Carlo risk simülasyonu yapar
        """
        try:
            if not probability_column or probability_column not in data.columns:
                return {'message': 'Monte Carlo simülasyonu için olasılık sütunu gereklidir'}
            
            impact_values = data[impact_column]
            prob_values = data[probability_column]
            
            # Simülasyon parametreleri
            impact_mean, impact_std = impact_values.mean(), impact_values.std()
            prob_mean, prob_std = prob_values.mean(), prob_values.std()
            
            # Monte Carlo simülasyonu
            simulated_risks = []
            for _ in range(n_simulations):
                sim_impact = np.random.normal(impact_mean, impact_std)
                sim_prob = np.random.normal(prob_mean, prob_std)
                sim_prob = max(0, min(1, sim_prob))  # 0-1 arasında sınırla
                
                risk_score = sim_impact * sim_prob
                simulated_risks.append(risk_score)
            
            simulated_risks = np.array(simulated_risks)
            
            return {
                'mean_simulated_risk': float(simulated_risks.mean()),
                'std_simulated_risk': float(simulated_risks.std()),
                'percentile_95': float(np.percentile(simulated_risks, 95)),
                'percentile_99': float(np.percentile(simulated_risks, 99)),
                'max_simulated_risk': float(simulated_risks.max()),
                'simulation_count': n_simulations
            }
            
        except Exception as e:
            return {'error': f'Monte Carlo simülasyonu hatası: {str(e)}'}
    
    def _create_risk_matrix(self, data: pd.DataFrame, 
                          impact_column: str, 
                          probability_column: Optional[str]) -> Dict[str, Any]:
        """
        Risk matrisi oluşturur
        """
        try:
            if not probability_column or probability_column not in data.columns:
                return {'message': 'Risk matrisi için olasılık sütunu gereklidir'}
            
            impact_values = data[impact_column]
            prob_values = data[probability_column]
            
            # Kategorilere ayır
            impact_categories = pd.cut(impact_values, bins=3, labels=['Düşük', 'Orta', 'Yüksek'])
            prob_categories = pd.cut(prob_values, bins=3, labels=['Düşük', 'Orta', 'Yüksek'])
            
            # Risk matrisi oluştur
            risk_matrix = pd.crosstab(impact_categories, prob_categories, margins=True)
            
            return {
                'risk_matrix': risk_matrix.to_dict(),
                'total_observations': len(data)
            }
            
        except Exception as e:
            return {'error': f'Risk matrisi oluşturma hatası: {str(e)}'}
    
    def _find_strongest_correlations(self, correlation_matrix: pd.DataFrame, 
                                   target_column: str) -> Dict[str, float]:
        """
        En güçlü korelasyonları bulur
        """
        target_correlations = correlation_matrix[target_column].drop(target_column)
        target_correlations = target_correlations.abs().sort_values(ascending=False)
        
        return target_correlations.head(5).to_dict()
    
    def _interpret_capability(self, cp: float, cpk: float) -> str:
        """
        Process capability sonuçlarını yorumlar
        """
        if cpk >= 2.0:
            return "Mükemmel process capability"
        elif cpk >= 1.33:
            return "İyi process capability"
        elif cpk >= 1.0:
            return "Kabul edilebilir process capability"
        elif cpk >= 0.67:
            return "Zayıf process capability"
        else:
            return "Yetersiz process capability"
    
    def _interpret_doe_results(self, main_effects: Dict, 
                             interaction_effects: Dict, 
                             model_fit: Dict) -> str:
        """
        DOE sonuçlarını yorumlar
        """
        interpretation = []
        
        # Ana etkiler
        significant_factors = [factor for factor, effect in main_effects.items() 
                             if effect.get('significant', False)]
        
        if significant_factors:
            interpretation.append(f"Anlamlı ana etkiler: {', '.join(significant_factors)}")
        else:
            interpretation.append("Anlamlı ana etki bulunamadı")
        
        # Etkileşimler
        significant_interactions = [interaction for interaction, effect in interaction_effects.items() 
                                  if effect.get('significant', False)]
        
        if significant_interactions:
            interpretation.append(f"Anlamlı etkileşimler: {', '.join(significant_interactions)}")
        
        # Model uyumu
        r_squared = model_fit.get('r_squared', 0)
        interpretation.append(f"Model açıklama gücü: %{r_squared*100:.1f}")
        
        return ". ".join(interpretation)
    
    def _interpret_six_sigma_results(self, cp: float, cpk: float, 
                                   sigma_level: float, yield_rate: float, 
                                   dpmo: float) -> str:
        """
        Six Sigma sonuçlarını yorumlar
        """
        interpretation = []
        
        # Capability yorumu
        interpretation.append(f"Process capability (Cpk): {cpk:.3f}")
        interpretation.append(self._interpret_capability(cp, cpk))
        
        # Sigma level
        interpretation.append(f"Sigma seviyesi: {sigma_level:.2f}")
        
        # Yield ve defect rate
        interpretation.append(f"Verim oranı: %{yield_rate:.2f}")
        interpretation.append(f"Hata oranı: {dpmo:.0f} PPM")
        
        return ". ".join(interpretation)
    
    def _interpret_risk_results(self, risk_factor_analysis: Dict, 
                              risk_score_analysis: Dict, 
                              category_distribution: pd.Series) -> str:
        """
        Risk analizi sonuçlarını yorumlar
        """
        interpretation = []
        
        # Risk faktörleri
        significant_factors = [factor for factor, analysis in risk_factor_analysis.items() 
                             if analysis.get('significant', False)]
        
        if significant_factors:
            interpretation.append(f"Anlamlı risk faktörleri: {', '.join(significant_factors)}")
        else:
            interpretation.append("Anlamlı risk faktörü bulunamadı")
        
        # Risk dağılımı
        if len(category_distribution) > 0:
            highest_risk_category = category_distribution.index[0]
            highest_percentage = category_distribution.iloc[0] / category_distribution.sum() * 100
            interpretation.append(f"En yaygın risk kategorisi: {highest_risk_category} (%{highest_percentage:.1f})")
        
        # Risk skoru
        if 'mean_risk_score' in risk_score_analysis:
            mean_risk = risk_score_analysis['mean_risk_score']
            interpretation.append(f"Ortalama risk skoru: {mean_risk:.2f}")
        
        return ". ".join(interpretation)