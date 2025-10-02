"""
Pazarlama Analizi Sınıfı

Bu modül pazarlama verilerinin analizi için özel araçlar içerir.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class MarketingAnalyzer:
    """
    Pazarlama veri analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        MarketingAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek pazarlama veri seti
        """
        self.data = data.copy()
        self.results = {}
        
    def customer_segmentation(self, features: List[str], 
                            customer_id_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Müşteri segmentasyonu analizi gerçekleştirir
        
        Args:
            features: Segmentasyon için kullanılacak özellikler
            customer_id_column: Müşteri ID sütunu (opsiyonel)
            
        Returns:
            Müşteri segmentasyonu sonuçları
        """
        try:
            # Veriyi kontrol et
            missing_cols = [col for col in features if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal sütunları filtrele
            numeric_features = []
            for col in features:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_features.append(col)
            
            if len(numeric_features) < 2:
                return {'error': 'En az 2 sayısal özellik gereklidir'}
            
            data_clean = self.data[numeric_features].dropna()
            
            if len(data_clean) < 10:
                return {'error': 'Segmentasyon için en az 10 müşteri gereklidir'}
            
            # RFM analizi (eğer uygun sütunlar varsa)
            rfm_analysis = None
            rfm_columns = {
                'recency': None,
                'frequency': None, 
                'monetary': None
            }
            
            # Sütun isimlerinden RFM bileşenlerini tahmin et
            for col in numeric_features:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['recency', 'son', 'last', 'recent']):
                    rfm_columns['recency'] = col
                elif any(keyword in col_lower for keyword in ['frequency', 'freq', 'count', 'adet', 'miktar']):
                    rfm_columns['frequency'] = col
                elif any(keyword in col_lower for keyword in ['monetary', 'amount', 'value', 'tutar', 'gelir', 'revenue']):
                    rfm_columns['monetary'] = col
            
            if all(v is not None for v in rfm_columns.values()):
                rfm_analysis = self._perform_rfm_analysis(data_clean, rfm_columns)
            
            # Temel istatistiksel segmentasyon
            segment_stats = {}
            for feature in numeric_features:
                values = data_clean[feature]
                
                # Quartile bazlı segmentasyon
                q1, q2, q3 = values.quantile([0.25, 0.5, 0.75])
                
                segments = {
                    'low': values[values <= q1],
                    'medium_low': values[(values > q1) & (values <= q2)],
                    'medium_high': values[(values > q2) & (values <= q3)],
                    'high': values[values > q3]
                }
                
                segment_stats[feature] = {
                    'quartiles': {'q1': float(q1), 'q2': float(q2), 'q3': float(q3)},
                    'segment_sizes': {k: len(v) for k, v in segments.items()},
                    'segment_percentages': {k: len(v)/len(values)*100 for k, v in segments.items()},
                    'segment_means': {k: float(v.mean()) for k, v in segments.items()},
                    'segment_stds': {k: float(v.std()) for k, v in segments.items()}
                }
            
            # Korelasyon analizi
            correlation_matrix = data_clean.corr()
            
            # Müşteri değer analizi
            customer_value_analysis = None
            if len(numeric_features) >= 2:
                # İlk iki özelliği kullanarak basit değer skoru hesapla
                feature1, feature2 = numeric_features[0], numeric_features[1]
                
                # Normalize et
                norm_f1 = (data_clean[feature1] - data_clean[feature1].min()) / (data_clean[feature1].max() - data_clean[feature1].min())
                norm_f2 = (data_clean[feature2] - data_clean[feature2].min()) / (data_clean[feature2].max() - data_clean[feature2].min())
                
                # Değer skoru (eşit ağırlık)
                value_score = (norm_f1 + norm_f2) / 2
                
                # Değer segmentleri
                value_segments = pd.cut(value_score, bins=3, labels=['Düşük Değer', 'Orta Değer', 'Yüksek Değer'])
                
                customer_value_analysis = {
                    'value_score_stats': {
                        'mean': float(value_score.mean()),
                        'std': float(value_score.std()),
                        'min': float(value_score.min()),
                        'max': float(value_score.max())
                    },
                    'value_segments': {
                        'segment_counts': value_segments.value_counts().to_dict(),
                        'segment_percentages': (value_segments.value_counts() / len(value_segments) * 100).to_dict()
                    },
                    'high_value_threshold': float(value_score.quantile(0.8)),
                    'low_value_threshold': float(value_score.quantile(0.2))
                }
            
            # Outlier analizi (potansiyel VIP müşteriler)
            outlier_analysis = {}
            for feature in numeric_features:
                values = data_clean[feature]
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                high_outliers = values[values > upper_bound]  # Potansiyel VIP'ler
                
                outlier_analysis[feature] = {
                    'total_outliers': len(outliers),
                    'outlier_percentage': float(len(outliers) / len(values) * 100),
                    'high_value_outliers': len(high_outliers),
                    'high_value_percentage': float(len(high_outliers) / len(values) * 100),
                    'outlier_threshold_upper': float(upper_bound),
                    'outlier_threshold_lower': float(lower_bound)
                }
            
            result = {
                'analysis_type': 'Customer Segmentation',
                'features_used': numeric_features,
                'n_customers': len(data_clean),
                'customer_id_column': customer_id_column,
                'segment_statistics': segment_stats,
                'correlation_matrix': correlation_matrix.to_dict(),
                'customer_value_analysis': customer_value_analysis,
                'outlier_analysis': outlier_analysis,
                'rfm_analysis': rfm_analysis,
                'interpretation': self._interpret_customer_segmentation(segment_stats, customer_value_analysis, outlier_analysis)
            }
            
            self.results['customer_segmentation'] = result
            return result
            
        except Exception as e:
            return {'error': f'Müşteri segmentasyonu hatası: {str(e)}'}
    
    def campaign_analysis(self, campaign_column: str, response_column: str,
                         cost_column: Optional[str] = None,
                         revenue_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Kampanya etkinlik analizi gerçekleştirir
        
        Args:
            campaign_column: Kampanya türü sütunu
            response_column: Yanıt sütunu (0/1 veya True/False)
            cost_column: Maliyet sütunu (opsiyonel)
            revenue_column: Gelir sütunu (opsiyonel)
            
        Returns:
            Kampanya analizi sonuçları
        """
        try:
            required_cols = [campaign_column, response_column]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            data_clean = self.data[required_cols].dropna()
            
            if len(data_clean) < 10:
                return {'error': 'Kampanya analizi için en az 10 gözlem gereklidir'}
            
            campaigns = data_clean[campaign_column]
            responses = data_clean[response_column]
            
            # Yanıt sütununu binary'ye çevir
            if responses.dtype == 'object':
                responses = responses.map({'True': 1, 'False': 0, 'true': 1, 'false': 0, 'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
            responses = responses.astype(int)
            
            # Kampanya türleri
            campaign_types = campaigns.unique()
            
            # Her kampanya için analiz
            campaign_performance = {}
            for campaign in campaign_types:
                campaign_mask = campaigns == campaign
                campaign_responses = responses[campaign_mask]
                
                # Temel metrikler
                total_customers = len(campaign_responses)
                total_responses = campaign_responses.sum()
                response_rate = total_responses / total_customers if total_customers > 0 else 0
                
                # Güven aralığı (Wilson score interval)
                if total_customers > 0:
                    p = response_rate
                    n = total_customers
                    z = 1.96  # %95 güven aralığı
                    
                    denominator = 1 + z**2/n
                    centre_adjusted_probability = p + z**2/(2*n)
                    adjusted_standard_deviation = np.sqrt((p*(1-p) + z**2/(4*n))/n)
                    
                    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
                    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
                    
                    confidence_interval = (max(0, lower_bound), min(1, upper_bound))
                else:
                    confidence_interval = (0, 0)
                
                campaign_performance[campaign] = {
                    'total_customers': total_customers,
                    'total_responses': int(total_responses),
                    'response_rate': float(response_rate),
                    'response_rate_percentage': float(response_rate * 100),
                    'confidence_interval_95': confidence_interval,
                    'non_response_rate': float(1 - response_rate)
                }
            
            # Kampanyalar arası karşılaştırma
            campaign_comparison = {
                'best_campaign': max(campaign_performance.keys(), key=lambda x: campaign_performance[x]['response_rate']),
                'worst_campaign': min(campaign_performance.keys(), key=lambda x: campaign_performance[x]['response_rate']),
                'average_response_rate': float(np.mean([v['response_rate'] for v in campaign_performance.values()])),
                'response_rate_variance': float(np.var([v['response_rate'] for v in campaign_performance.values()]))
            }
            
            # İstatistiksel test (Chi-square)
            contingency_table = pd.crosstab(campaigns, responses)
            chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(contingency_table)
            
            statistical_test = {
                'chi2_statistic': float(chi2_stat),
                'chi2_p_value': float(chi2_p),
                'chi2_degrees_of_freedom': int(chi2_dof),
                'significant_difference': chi2_p < 0.05,
                'contingency_table': contingency_table.to_dict()
            }
            
            # ROI analizi (eğer maliyet ve gelir verileri varsa)
            roi_analysis = None
            if cost_column and cost_column in self.data.columns and revenue_column and revenue_column in self.data.columns:
                roi_analysis = self._calculate_campaign_roi(campaigns, responses, 
                                                          self.data[cost_column], 
                                                          self.data[revenue_column])
            
            # Lift analizi
            lift_analysis = {}
            overall_response_rate = responses.mean()
            
            for campaign in campaign_types:
                campaign_response_rate = campaign_performance[campaign]['response_rate']
                lift = campaign_response_rate / overall_response_rate if overall_response_rate > 0 else 0
                
                lift_analysis[campaign] = {
                    'lift': float(lift),
                    'lift_percentage': float((lift - 1) * 100),
                    'performance_vs_average': 'Above Average' if lift > 1 else 'Below Average' if lift < 1 else 'Average'
                }
            
            result = {
                'analysis_type': 'Campaign Analysis',
                'campaign_column': campaign_column,
                'response_column': response_column,
                'n_observations': len(data_clean),
                'campaign_types': list(campaign_types),
                'overall_response_rate': float(overall_response_rate),
                'campaign_performance': campaign_performance,
                'campaign_comparison': campaign_comparison,
                'statistical_test': statistical_test,
                'lift_analysis': lift_analysis,
                'roi_analysis': roi_analysis,
                'interpretation': self._interpret_campaign_analysis(campaign_performance, campaign_comparison, statistical_test)
            }
            
            self.results['campaign_analysis'] = result
            return result
            
        except Exception as e:
            return {'error': f'Kampanya analizi hatası: {str(e)}'}
    
    def cohort_analysis(self, customer_id_column: str, date_column: str,
                       revenue_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Kohort analizi gerçekleştirir
        
        Args:
            customer_id_column: Müşteri ID sütunu
            date_column: Tarih sütunu
            revenue_column: Gelir sütunu (opsiyonel)
            
        Returns:
            Kohort analizi sonuçları
        """
        try:
            required_cols = [customer_id_column, date_column]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            data_clean = self.data[required_cols].copy()
            if revenue_column and revenue_column in self.data.columns:
                data_clean[revenue_column] = self.data[revenue_column]
            
            data_clean = data_clean.dropna()
            
            if len(data_clean) < 20:
                return {'error': 'Kohort analizi için en az 20 gözlem gereklidir'}
            
            # Tarih sütununu datetime'a çevir
            try:
                data_clean[date_column] = pd.to_datetime(data_clean[date_column])
            except:
                return {'error': f'{date_column} sütunu tarih formatına çevrilemedi'}
            
            # Her müşterinin ilk satın alma tarihini bul
            customer_first_purchase = data_clean.groupby(customer_id_column)[date_column].min().reset_index()
            customer_first_purchase.columns = [customer_id_column, 'first_purchase_date']
            
            # Kohort ayını hesapla (ilk satın alma ayı)
            customer_first_purchase['cohort_month'] = customer_first_purchase['first_purchase_date'].dt.to_period('M')
            
            # Ana veriyi kohort bilgisiyle birleştir
            data_with_cohort = data_clean.merge(customer_first_purchase, on=customer_id_column)
            
            # Her işlem için period hesapla
            data_with_cohort['transaction_month'] = data_with_cohort[date_column].dt.to_period('M')
            
            # Kohort'tan bu yana geçen ay sayısını hesapla
            data_with_cohort['months_since_first_purchase'] = (
                data_with_cohort['transaction_month'] - data_with_cohort['cohort_month']
            ).apply(attrgetter('n'))
            
            # Retention analizi
            cohort_table = data_with_cohort.groupby(['cohort_month', 'months_since_first_purchase'])[customer_id_column].nunique().reset_index()
            cohort_table = cohort_table.pivot(index='cohort_month', columns='months_since_first_purchase', values=customer_id_column)
            
            # Kohort boyutları
            cohort_sizes = customer_first_purchase.groupby('cohort_month')[customer_id_column].nunique()
            
            # Retention oranları
            retention_table = cohort_table.divide(cohort_sizes, axis=0)
            
            # Ortalama retention oranları
            avg_retention_by_month = retention_table.mean()
            
            # Revenue kohort analizi (eğer gelir verisi varsa)
            revenue_cohort_analysis = None
            if revenue_column and revenue_column in data_with_cohort.columns:
                revenue_cohort_table = data_with_cohort.groupby(['cohort_month', 'months_since_first_purchase'])[revenue_column].sum().reset_index()
                revenue_cohort_table = revenue_cohort_table.pivot(index='cohort_month', columns='months_since_first_purchase', values=revenue_column)
                
                # Ortalama gelir per müşteri
                avg_revenue_per_customer = revenue_cohort_table.divide(cohort_table)
                
                revenue_cohort_analysis = {
                    'total_revenue_by_cohort': revenue_cohort_table.to_dict(),
                    'avg_revenue_per_customer': avg_revenue_per_customer.to_dict(),
                    'cohort_ltv': revenue_cohort_table.sum(axis=1).to_dict()  # Lifetime Value by cohort
                }
            
            # Churn analizi
            # Son 3 ayda işlem yapmayan müşteriler
            latest_date = data_clean[date_column].max()
            three_months_ago = latest_date - pd.DateOffset(months=3)
            
            recent_customers = data_clean[data_clean[date_column] >= three_months_ago][customer_id_column].unique()
            all_customers = data_clean[customer_id_column].unique()
            churned_customers = set(all_customers) - set(recent_customers)
            
            churn_analysis = {
                'total_customers': len(all_customers),
                'active_customers': len(recent_customers),
                'churned_customers': len(churned_customers),
                'churn_rate': float(len(churned_customers) / len(all_customers)),
                'retention_rate': float(len(recent_customers) / len(all_customers))
            }
            
            # Kohort performans metrikleri
            cohort_metrics = {}
            for cohort in cohort_sizes.index:
                cohort_data = retention_table.loc[cohort].dropna()
                
                cohort_metrics[str(cohort)] = {
                    'initial_size': int(cohort_sizes[cohort]),
                    'month_1_retention': float(cohort_data.iloc[1]) if len(cohort_data) > 1 else None,
                    'month_3_retention': float(cohort_data.iloc[3]) if len(cohort_data) > 3 else None,
                    'month_6_retention': float(cohort_data.iloc[6]) if len(cohort_data) > 6 else None,
                    'month_12_retention': float(cohort_data.iloc[12]) if len(cohort_data) > 12 else None,
                    'avg_retention_first_6_months': float(cohort_data.iloc[1:7].mean()) if len(cohort_data) > 6 else None
                }
            
            result = {
                'analysis_type': 'Cohort Analysis',
                'customer_id_column': customer_id_column,
                'date_column': date_column,
                'revenue_column': revenue_column,
                'analysis_period': {
                    'start_date': data_clean[date_column].min().strftime('%Y-%m-%d'),
                    'end_date': data_clean[date_column].max().strftime('%Y-%m-%d')
                },
                'n_customers': len(all_customers),
                'n_cohorts': len(cohort_sizes),
                'cohort_sizes': cohort_sizes.to_dict(),
                'retention_table': retention_table.to_dict(),
                'avg_retention_by_month': avg_retention_by_month.to_dict(),
                'cohort_metrics': cohort_metrics,
                'churn_analysis': churn_analysis,
                'revenue_cohort_analysis': revenue_cohort_analysis,
                'interpretation': self._interpret_cohort_analysis(avg_retention_by_month, churn_analysis, cohort_metrics)
            }
            
            self.results['cohort_analysis'] = result
            return result
            
        except Exception as e:
            return {'error': f'Kohort analizi hatası: {str(e)}'}
    
    def _perform_rfm_analysis(self, data: pd.DataFrame, rfm_columns: Dict[str, str]) -> Dict[str, Any]:
        """RFM analizi gerçekleştirir"""
        try:
            recency = data[rfm_columns['recency']]
            frequency = data[rfm_columns['frequency']]
            monetary = data[rfm_columns['monetary']]
            
            # RFM skorları (1-5 arası)
            r_score = pd.qcut(recency.rank(method='first'), 5, labels=[5,4,3,2,1])  # Düşük recency = yüksek skor
            f_score = pd.qcut(frequency.rank(method='first'), 5, labels=[1,2,3,4,5])
            m_score = pd.qcut(monetary.rank(method='first'), 5, labels=[1,2,3,4,5])
            
            # RFM segmentleri
            rfm_segments = r_score.astype(str) + f_score.astype(str) + m_score.astype(str)
            
            # Segment yorumları
            def segment_customers(rfm):
                if rfm in ['555', '554', '544', '545', '454', '455', '445']:
                    return 'Champions'
                elif rfm in ['543', '444', '435', '355', '354', '345', '344', '335']:
                    return 'Loyal Customers'
                elif rfm in ['512', '511', '422', '421', '412', '411', '311']:
                    return 'New Customers'
                elif rfm in ['533', '532', '531', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
                    return 'Potential Loyalists'
                elif rfm in ['155', '154', '144', '214', '215', '115', '114']:
                    return 'At Risk'
                elif rfm in ['155', '154', '144', '214', '215', '115', '114']:
                    return 'Cannot Lose Them'
                else:
                    return 'Others'
            
            segment_labels = rfm_segments.apply(segment_customers)
            
            # Segment istatistikleri
            segment_stats = {}
            for segment in segment_labels.unique():
                segment_mask = segment_labels == segment
                segment_data = data[segment_mask]
                
                segment_stats[segment] = {
                    'count': int(segment_mask.sum()),
                    'percentage': float(segment_mask.sum() / len(data) * 100),
                    'avg_recency': float(segment_data[rfm_columns['recency']].mean()),
                    'avg_frequency': float(segment_data[rfm_columns['frequency']].mean()),
                    'avg_monetary': float(segment_data[rfm_columns['monetary']].mean())
                }
            
            return {
                'rfm_columns': rfm_columns,
                'segment_distribution': segment_stats,
                'total_customers': len(data),
                'rfm_summary': {
                    'avg_recency': float(recency.mean()),
                    'avg_frequency': float(frequency.mean()),
                    'avg_monetary': float(monetary.mean())
                }
            }
            
        except Exception as e:
            return {'error': f'RFM analizi hatası: {str(e)}'}
    
    def _calculate_campaign_roi(self, campaigns: pd.Series, responses: pd.Series,
                               costs: pd.Series, revenues: pd.Series) -> Dict[str, Any]:
        """Kampanya ROI hesaplar"""
        try:
            roi_analysis = {}
            
            for campaign in campaigns.unique():
                campaign_mask = campaigns == campaign
                campaign_responses = responses[campaign_mask]
                campaign_costs = costs[campaign_mask]
                campaign_revenues = revenues[campaign_mask]
                
                # Sadece yanıt veren müşterilerin geliri
                responding_customers = campaign_responses == 1
                total_cost = campaign_costs.sum()
                total_revenue = campaign_revenues[responding_customers].sum() if responding_customers.any() else 0
                
                roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
                
                roi_analysis[campaign] = {
                    'total_cost': float(total_cost),
                    'total_revenue': float(total_revenue),
                    'net_profit': float(total_revenue - total_cost),
                    'roi': float(roi),
                    'roi_percentage': float(roi * 100),
                    'cost_per_response': float(total_cost / campaign_responses.sum()) if campaign_responses.sum() > 0 else 0,
                    'revenue_per_response': float(total_revenue / campaign_responses.sum()) if campaign_responses.sum() > 0 else 0
                }
            
            return roi_analysis
            
        except Exception as e:
            return {'error': f'ROI hesaplama hatası: {str(e)}'}
    
    def _interpret_customer_segmentation(self, segment_stats: Dict, value_analysis: Dict, outlier_analysis: Dict) -> str:
        """Müşteri segmentasyonu sonuçlarını yorumlar"""
        interpretation = "Müşteri segmentasyonu: "
        
        # En değişken özelliği bul
        max_variation_feature = max(segment_stats.keys(), 
                                  key=lambda x: max(segment_stats[x]['segment_stds'].values()))
        
        interpretation += f"En değişken özellik: {max_variation_feature}. "
        
        # Değer analizi yorumu
        if value_analysis and 'value_segments' in value_analysis:
            high_value_pct = value_analysis['value_segments']['segment_percentages'].get('Yüksek Değer', 0)
            if high_value_pct > 25:
                interpretation += "Yüksek değerli müşteri oranı iyi. "
            else:
                interpretation += "Yüksek değerli müşteri oranı düşük. "
        
        # Outlier yorumu
        total_outliers = sum(outlier_analysis[feature]['high_value_outliers'] for feature in outlier_analysis)
        if total_outliers > 0:
            interpretation += f"{total_outliers} potansiyel VIP müşteri tespit edildi."
        
        return interpretation
    
    def _interpret_campaign_analysis(self, performance: Dict, comparison: Dict, statistical_test: Dict) -> str:
        """Kampanya analizi sonuçlarını yorumlar"""
        interpretation = "Kampanya analizi: "
        
        best_campaign = comparison['best_campaign']
        best_rate = performance[best_campaign]['response_rate_percentage']
        
        interpretation += f"En başarılı kampanya: {best_campaign} (%{best_rate:.1f} yanıt oranı). "
        
        if statistical_test['significant_difference']:
            interpretation += "Kampanyalar arasında istatistiksel olarak anlamlı fark var. "
        else:
            interpretation += "Kampanyalar arasında anlamlı fark yok. "
        
        avg_rate = comparison['average_response_rate'] * 100
        if avg_rate > 10:
            interpretation += "Genel yanıt oranı yüksek."
        elif avg_rate > 5:
            interpretation += "Genel yanıt oranı orta."
        else:
            interpretation += "Genel yanıt oranı düşük."
        
        return interpretation
    
    def _interpret_cohort_analysis(self, avg_retention: pd.Series, churn_analysis: Dict, cohort_metrics: Dict) -> str:
        """Kohort analizi sonuçlarını yorumlar"""
        interpretation = "Kohort analizi: "
        
        # 1. ay retention
        month_1_retention = avg_retention.iloc[1] if len(avg_retention) > 1 else None
        if month_1_retention:
            if month_1_retention > 0.8:
                interpretation += "Mükemmel 1. ay tutma oranı. "
            elif month_1_retention > 0.6:
                interpretation += "İyi 1. ay tutma oranı. "
            else:
                interpretation += "Düşük 1. ay tutma oranı. "
        
        # Churn oranı
        churn_rate = churn_analysis['churn_rate']
        if churn_rate < 0.1:
            interpretation += "Düşük churn oranı, iyi müşteri tutma."
        elif churn_rate < 0.3:
            interpretation += "Orta churn oranı."
        else:
            interpretation += "Yüksek churn oranı, müşteri tutma stratejileri gerekli."
        
        return interpretation
    
    def get_marketing_summary(self) -> Dict[str, Any]:
        """
        Gerçekleştirilen tüm pazarlama analizlerinin özetini döndürür
        
        Returns:
            Pazarlama analiz özetleri
        """
        if not self.results:
            return {'message': 'Henüz pazarlama analizi gerçekleştirilmedi'}
        
        summary = {
            'total_analyses': len(self.results),
            'analyses_performed': list(self.results.keys()),
            'analysis_details': self.results
        }
        
        # En önemli bulgular
        key_findings = []
        
        if 'customer_segmentation' in self.results:
            segmentation = self.results['customer_segmentation']
            if 'customer_value_analysis' in segmentation and segmentation['customer_value_analysis']:
                high_value_pct = list(segmentation['customer_value_analysis']['value_segments']['segment_percentages'].values())[2]  # Yüksek değer
                key_findings.append(f"Yüksek değerli müşteri oranı: %{high_value_pct:.1f}")
        
        if 'campaign_analysis' in self.results:
            campaign = self.results['campaign_analysis']
            best_campaign = campaign['campaign_comparison']['best_campaign']
            best_rate = campaign['campaign_performance'][best_campaign]['response_rate_percentage']
            key_findings.append(f"En başarılı kampanya: {best_campaign} (%{best_rate:.1f})")
        
        if 'cohort_analysis' in self.results:
            cohort = self.results['cohort_analysis']
            churn_rate = cohort['churn_analysis']['churn_rate']
            key_findings.append(f"Churn oranı: %{churn_rate*100:.1f}")
        
        summary['key_findings'] = key_findings
        
        return summary


# Yardımcı import
from operator import attrgetter