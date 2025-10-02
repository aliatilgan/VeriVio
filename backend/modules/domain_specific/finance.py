"""
Finans Analizi Sınıfı

Bu modül finansal verilerin analizi için özel araçlar içerir.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import jarque_bera, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class FinanceAnalyzer:
    """
    Finansal veri analizlerini gerçekleştiren sınıf
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        FinanceAnalyzer sınıfını başlatır
        
        Args:
            data: Analiz edilecek finansal veri seti
        """
        self.data = data.copy()
        self.results = {}
        
    def portfolio_analysis(self, return_columns: List[str], 
                          risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Portföy analizi gerçekleştirir
        
        Args:
            return_columns: Getiri sütunları
            risk_free_rate: Risksiz faiz oranı (yıllık)
            
        Returns:
            Portföy analizi sonuçları
        """
        try:
            # Veriyi kontrol et
            missing_cols = [col for col in return_columns if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal olmayan sütunları filtrele
            numeric_cols = []
            for col in return_columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    numeric_cols.append(col)
            
            if len(numeric_cols) < 1:
                return {'error': 'En az 1 sayısal getiri sütunu gereklidir'}
            
            returns_data = self.data[numeric_cols].dropna()
            
            if len(returns_data) < 10:
                return {'error': 'Portföy analizi için en az 10 gözlem gereklidir'}
            
            # Temel istatistikler
            portfolio_stats = {}
            for col in numeric_cols:
                returns = returns_data[col]
                
                # Getiri istatistikleri
                mean_return = returns.mean()
                std_return = returns.std()
                
                # Yıllık getiri ve volatilite (günlük veriler varsayımı)
                annual_return = mean_return * 252
                annual_volatility = std_return * np.sqrt(252)
                
                # Sharpe oranı
                sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
                
                # Downside deviation
                negative_returns = returns[returns < 0]
                downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
                
                # Sortino oranı
                sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
                
                # Maximum Drawdown
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.min()
                
                # Value at Risk (VaR) %5 ve %1
                var_5 = np.percentile(returns, 5)
                var_1 = np.percentile(returns, 1)
                
                # Conditional VaR (Expected Shortfall)
                cvar_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else var_5
                cvar_1 = returns[returns <= var_1].mean() if len(returns[returns <= var_1]) > 0 else var_1
                
                # Skewness ve Kurtosis
                skewness = returns.skew()
                kurtosis = returns.kurtosis()
                
                # Normallik testi
                _, normality_p = normaltest(returns)
                
                # Beta hesaplama (eğer benchmark varsa)
                beta = None
                if len(numeric_cols) > 1:
                    # İlk sütunu benchmark olarak kabul et
                    if col != numeric_cols[0]:
                        benchmark_returns = returns_data[numeric_cols[0]]
                        covariance = np.cov(returns, benchmark_returns)[0, 1]
                        benchmark_variance = benchmark_returns.var()
                        beta = covariance / benchmark_variance if benchmark_variance > 0 else None
                
                portfolio_stats[col] = {
                    'daily_return_mean': float(mean_return),
                    'daily_return_std': float(std_return),
                    'annual_return': float(annual_return),
                    'annual_volatility': float(annual_volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'sortino_ratio': float(sortino_ratio),
                    'max_drawdown': float(max_drawdown),
                    'var_5_percent': float(var_5),
                    'var_1_percent': float(var_1),
                    'cvar_5_percent': float(cvar_5),
                    'cvar_1_percent': float(cvar_1),
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'is_normal': normality_p > 0.05,
                    'normality_p_value': float(normality_p),
                    'beta': float(beta) if beta is not None else None,
                    'downside_deviation': float(downside_deviation)
                }
            
            # Korelasyon matrisi
            correlation_matrix = returns_data.corr()
            
            # Portföy optimizasyonu (eşit ağırlıklı)
            if len(numeric_cols) > 1:
                weights = np.array([1/len(numeric_cols)] * len(numeric_cols))
                portfolio_return = np.sum(weights * returns_data.mean()) * 252
                portfolio_variance = np.dot(weights.T, np.dot(returns_data.cov() * 252, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                portfolio_optimization = {
                    'equal_weighted_return': float(portfolio_return),
                    'equal_weighted_volatility': float(portfolio_volatility),
                    'equal_weighted_sharpe': float(portfolio_sharpe),
                    'weights': weights.tolist(),
                    'asset_names': numeric_cols
                }
            else:
                portfolio_optimization = None
            
            result = {
                'analysis_type': 'Portfolio Analysis',
                'n_assets': len(numeric_cols),
                'n_observations': len(returns_data),
                'asset_columns': numeric_cols,
                'risk_free_rate': risk_free_rate,
                'individual_asset_stats': portfolio_stats,
                'correlation_matrix': correlation_matrix.to_dict(),
                'portfolio_optimization': portfolio_optimization,
                'market_analysis': {
                    'average_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()),
                    'max_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()),
                    'min_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min())
                },
                'interpretation': self._interpret_portfolio_analysis(portfolio_stats, correlation_matrix)
            }
            
            self.results['portfolio_analysis'] = result
            return result
            
        except Exception as e:
            return {'error': f'Portföy analizi hatası: {str(e)}'}
    
    def risk_analysis(self, price_column: str, return_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Risk analizi gerçekleştirir
        
        Args:
            price_column: Fiyat sütunu
            return_column: Getiri sütunu (opsiyonel, hesaplanabilir)
            
        Returns:
            Risk analizi sonuçları
        """
        try:
            if price_column not in self.data.columns:
                return {'error': f'{price_column} sütunu bulunamadı'}
            
            if not pd.api.types.is_numeric_dtype(self.data[price_column]):
                return {'error': f'{price_column} sütunu sayısal değil'}
            
            prices = self.data[price_column].dropna()
            
            if len(prices) < 20:
                return {'error': 'Risk analizi için en az 20 gözlem gereklidir'}
            
            # Getiri hesaplama
            if return_column and return_column in self.data.columns:
                returns = self.data[return_column].dropna()
            else:
                returns = prices.pct_change().dropna()
            
            if len(returns) < 10:
                return {'error': 'Yeterli getiri verisi yok'}
            
            # Volatilite analizi
            volatility_analysis = {
                'daily_volatility': float(returns.std()),
                'annual_volatility': float(returns.std() * np.sqrt(252)),
                'volatility_of_volatility': float(returns.rolling(window=30).std().std()) if len(returns) >= 30 else None
            }
            
            # VaR analizi (farklı güven seviyeleri)
            var_analysis = {}
            confidence_levels = [0.01, 0.05, 0.10]
            
            for conf in confidence_levels:
                var_value = np.percentile(returns, conf * 100)
                cvar_value = returns[returns <= var_value].mean() if len(returns[returns <= var_value]) > 0 else var_value
                
                var_analysis[f'var_{int(conf*100)}'] = {
                    'value': float(var_value),
                    'conditional_var': float(cvar_value),
                    'confidence_level': conf
                }
            
            # Extreme Value Analysis
            # En büyük ve en küçük getiriler
            extreme_returns = {
                'worst_return': float(returns.min()),
                'best_return': float(returns.max()),
                'worst_return_date': returns.idxmin().strftime('%Y-%m-%d') if hasattr(returns.idxmin(), 'strftime') else str(returns.idxmin()),
                'best_return_date': returns.idxmax().strftime('%Y-%m-%d') if hasattr(returns.idxmax(), 'strftime') else str(returns.idxmax())
            }
            
            # Tail risk analizi
            tail_threshold = 0.05
            left_tail = returns[returns <= np.percentile(returns, tail_threshold * 100)]
            right_tail = returns[returns >= np.percentile(returns, (1 - tail_threshold) * 100)]
            
            tail_analysis = {
                'left_tail_mean': float(left_tail.mean()) if len(left_tail) > 0 else None,
                'right_tail_mean': float(right_tail.mean()) if len(right_tail) > 0 else None,
                'tail_ratio': float(len(left_tail) / len(right_tail)) if len(right_tail) > 0 else None,
                'left_tail_count': len(left_tail),
                'right_tail_count': len(right_tail)
            }
            
            # Drawdown analizi
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            
            drawdown_analysis = {
                'max_drawdown': float(drawdown.min()),
                'current_drawdown': float(drawdown.iloc[-1]),
                'avg_drawdown': float(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0,
                'drawdown_periods': int(len(drawdown[drawdown < 0])),
                'recovery_time': self._calculate_recovery_time(drawdown)
            }
            
            # Risk metrikleri
            risk_metrics = {
                'downside_deviation': float(returns[returns < 0].std()) if len(returns[returns < 0]) > 0 else 0,
                'upside_deviation': float(returns[returns > 0].std()) if len(returns[returns > 0]) > 0 else 0,
                'gain_loss_ratio': float(returns[returns > 0].mean() / abs(returns[returns < 0].mean())) if len(returns[returns < 0]) > 0 and returns[returns < 0].mean() != 0 else None,
                'hit_ratio': float(len(returns[returns > 0]) / len(returns)),
                'pain_index': float(abs(drawdown.mean()))
            }
            
            # Stress testing
            stress_scenarios = {
                'scenario_1_sigma': float(returns.mean() - returns.std()),
                'scenario_2_sigma': float(returns.mean() - 2 * returns.std()),
                'scenario_3_sigma': float(returns.mean() - 3 * returns.std()),
                'historical_worst_month': float(returns.rolling(window=21).sum().min()) if len(returns) >= 21 else None,
                'historical_worst_quarter': float(returns.rolling(window=63).sum().min()) if len(returns) >= 63 else None
            }
            
            result = {
                'analysis_type': 'Risk Analysis',
                'price_column': price_column,
                'return_column': return_column,
                'n_observations': len(returns),
                'analysis_period': {
                    'start_date': returns.index[0].strftime('%Y-%m-%d') if hasattr(returns.index[0], 'strftime') else str(returns.index[0]),
                    'end_date': returns.index[-1].strftime('%Y-%m-%d') if hasattr(returns.index[-1], 'strftime') else str(returns.index[-1])
                },
                'volatility_analysis': volatility_analysis,
                'var_analysis': var_analysis,
                'extreme_returns': extreme_returns,
                'tail_analysis': tail_analysis,
                'drawdown_analysis': drawdown_analysis,
                'risk_metrics': risk_metrics,
                'stress_scenarios': stress_scenarios,
                'interpretation': self._interpret_risk_analysis(volatility_analysis, var_analysis, drawdown_analysis)
            }
            
            self.results['risk_analysis'] = result
            return result
            
        except Exception as e:
            return {'error': f'Risk analizi hatası: {str(e)}'}
    
    def financial_ratios(self, revenue_column: str, profit_column: str,
                        assets_column: Optional[str] = None,
                        equity_column: Optional[str] = None,
                        debt_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Finansal oran analizi gerçekleştirir
        
        Args:
            revenue_column: Gelir sütunu
            profit_column: Kar sütunu
            assets_column: Varlık sütunu (opsiyonel)
            equity_column: Özkaynak sütunu (opsiyonel)
            debt_column: Borç sütunu (opsiyonel)
            
        Returns:
            Finansal oran analizi sonuçları
        """
        try:
            required_cols = [revenue_column, profit_column]
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                return {'error': f'Şu sütunlar bulunamadı: {missing_cols}'}
            
            # Sayısal kontrol
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    return {'error': f'{col} sütunu sayısal değil'}
            
            data_clean = self.data[required_cols].dropna()
            
            if len(data_clean) < 3:
                return {'error': 'Finansal oran analizi için en az 3 gözlem gereklidir'}
            
            revenue = data_clean[revenue_column]
            profit = data_clean[profit_column]
            
            # Temel karlılık oranları
            profitability_ratios = {
                'profit_margin': (profit / revenue).mean() if revenue.mean() != 0 else None,
                'profit_margin_trend': self._calculate_trend(profit / revenue),
                'revenue_growth': self._calculate_growth_rate(revenue),
                'profit_growth': self._calculate_growth_rate(profit)
            }
            
            # Ek oranlar (eğer veriler varsa)
            additional_ratios = {}
            
            if assets_column and assets_column in self.data.columns:
                assets = self.data[assets_column].dropna()
                if len(assets) >= len(data_clean):
                    assets = assets[:len(data_clean)]
                    additional_ratios['roa'] = (profit / assets).mean() if assets.mean() != 0 else None
                    additional_ratios['asset_turnover'] = (revenue / assets).mean() if assets.mean() != 0 else None
            
            if equity_column and equity_column in self.data.columns:
                equity = self.data[equity_column].dropna()
                if len(equity) >= len(data_clean):
                    equity = equity[:len(data_clean)]
                    additional_ratios['roe'] = (profit / equity).mean() if equity.mean() != 0 else None
            
            if debt_column and debt_column in self.data.columns:
                debt = self.data[debt_column].dropna()
                if len(debt) >= len(data_clean):
                    debt = debt[:len(data_clean)]
                    if equity_column and equity_column in self.data.columns:
                        equity = self.data[equity_column].dropna()[:len(data_clean)]
                        additional_ratios['debt_to_equity'] = (debt / equity).mean() if equity.mean() != 0 else None
            
            # Performans analizi
            performance_metrics = {
                'revenue_volatility': float(revenue.std() / revenue.mean()) if revenue.mean() != 0 else None,
                'profit_volatility': float(profit.std() / profit.mean()) if profit.mean() != 0 else None,
                'profit_consistency': float(len(profit[profit > 0]) / len(profit)),
                'revenue_trend_strength': float(abs(stats.pearsonr(range(len(revenue)), revenue)[0])) if len(revenue) > 1 else None
            }
            
            # Benchmark karşılaştırması (sektör ortalamaları - örnek değerler)
            industry_benchmarks = {
                'typical_profit_margin': 0.10,  # %10
                'typical_roa': 0.05,           # %5
                'typical_roe': 0.15,           # %15
                'typical_debt_to_equity': 0.5   # 0.5
            }
            
            benchmark_comparison = {}
            if profitability_ratios['profit_margin'] is not None:
                benchmark_comparison['profit_margin_vs_benchmark'] = profitability_ratios['profit_margin'] - industry_benchmarks['typical_profit_margin']
            
            if 'roa' in additional_ratios and additional_ratios['roa'] is not None:
                benchmark_comparison['roa_vs_benchmark'] = additional_ratios['roa'] - industry_benchmarks['typical_roa']
            
            if 'roe' in additional_ratios and additional_ratios['roe'] is not None:
                benchmark_comparison['roe_vs_benchmark'] = additional_ratios['roe'] - industry_benchmarks['typical_roe']
            
            result = {
                'analysis_type': 'Financial Ratios Analysis',
                'revenue_column': revenue_column,
                'profit_column': profit_column,
                'n_observations': len(data_clean),
                'profitability_ratios': profitability_ratios,
                'additional_ratios': additional_ratios,
                'performance_metrics': performance_metrics,
                'benchmark_comparison': benchmark_comparison,
                'industry_benchmarks': industry_benchmarks,
                'interpretation': self._interpret_financial_ratios(profitability_ratios, additional_ratios, benchmark_comparison)
            }
            
            self.results['financial_ratios'] = result
            return result
            
        except Exception as e:
            return {'error': f'Finansal oran analizi hatası: {str(e)}'}
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> Optional[int]:
        """Drawdown'dan toparlanma süresini hesaplar"""
        try:
            if drawdown.min() >= 0:
                return 0
            
            # En büyük drawdown noktasını bul
            max_dd_idx = drawdown.idxmin()
            
            # Bu noktadan sonra sıfıra ulaşma süresini bul
            after_max_dd = drawdown.loc[max_dd_idx:]
            recovery_idx = after_max_dd[after_max_dd >= 0].index
            
            if len(recovery_idx) > 0:
                return int((recovery_idx[0] - max_dd_idx).days) if hasattr((recovery_idx[0] - max_dd_idx), 'days') else int(recovery_idx[0] - max_dd_idx)
            else:
                return None  # Henüz toparlanmamış
        except:
            return None
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Seri için trend hesaplar"""
        if len(series) < 2:
            return 'Yetersiz veri'
        
        slope, _, r_value, p_value, _ = stats.linregress(range(len(series)), series)
        
        if p_value > 0.05:
            return 'Trend yok'
        elif slope > 0:
            return 'Artan' if abs(r_value) > 0.5 else 'Hafif artan'
        else:
            return 'Azalan' if abs(r_value) > 0.5 else 'Hafif azalan'
    
    def _calculate_growth_rate(self, series: pd.Series) -> Optional[float]:
        """Büyüme oranını hesaplar"""
        if len(series) < 2:
            return None
        
        first_value = series.iloc[0]
        last_value = series.iloc[-1]
        periods = len(series) - 1
        
        if first_value <= 0:
            return None
        
        growth_rate = (last_value / first_value) ** (1/periods) - 1
        return float(growth_rate)
    
    def _interpret_portfolio_analysis(self, portfolio_stats: Dict, correlation_matrix: pd.DataFrame) -> str:
        """Portföy analizi sonuçlarını yorumlar"""
        interpretation = "Portföy analizi: "
        
        # En iyi Sharpe oranına sahip varlığı bul
        best_sharpe_asset = max(portfolio_stats.keys(), key=lambda x: portfolio_stats[x]['sharpe_ratio'])
        best_sharpe = portfolio_stats[best_sharpe_asset]['sharpe_ratio']
        
        if best_sharpe > 1.0:
            interpretation += f"Mükemmel risk-getiri profili ({best_sharpe_asset}). "
        elif best_sharpe > 0.5:
            interpretation += f"İyi risk-getiri profili ({best_sharpe_asset}). "
        else:
            interpretation += "Risk-getiri profili geliştirilmeli. "
        
        # Korelasyon analizi
        avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        if avg_corr > 0.8:
            interpretation += "Yüksek korelasyon, diversifikasyon sınırlı. "
        elif avg_corr > 0.5:
            interpretation += "Orta korelasyon, makul diversifikasyon. "
        else:
            interpretation += "Düşük korelasyon, iyi diversifikasyon."
        
        return interpretation
    
    def _interpret_risk_analysis(self, volatility: Dict, var: Dict, drawdown: Dict) -> str:
        """Risk analizi sonuçlarını yorumlar"""
        interpretation = "Risk analizi: "
        
        # Volatilite yorumu
        annual_vol = volatility['annual_volatility']
        if annual_vol > 0.3:
            interpretation += "Yüksek volatilite, riskli yatırım. "
        elif annual_vol > 0.15:
            interpretation += "Orta volatilite, makul risk. "
        else:
            interpretation += "Düşük volatilite, muhafazakar yatırım. "
        
        # Drawdown yorumu
        max_dd = abs(drawdown['max_drawdown'])
        if max_dd > 0.3:
            interpretation += "Yüksek maksimum kayıp riski. "
        elif max_dd > 0.15:
            interpretation += "Orta düzeyde maksimum kayıp riski. "
        else:
            interpretation += "Düşük maksimum kayıp riski."
        
        return interpretation
    
    def _interpret_financial_ratios(self, profitability: Dict, additional: Dict, benchmark: Dict) -> str:
        """Finansal oran analizi sonuçlarını yorumlar"""
        interpretation = "Finansal oran analizi: "
        
        # Kar marjı yorumu
        profit_margin = profitability.get('profit_margin')
        if profit_margin is not None:
            if profit_margin > 0.15:
                interpretation += "Yüksek kar marjı, güçlü karlılık. "
            elif profit_margin > 0.05:
                interpretation += "Makul kar marjı. "
            elif profit_margin > 0:
                interpretation += "Düşük kar marjı. "
            else:
                interpretation += "Negatif kar marjı, zarar durumu. "
        
        # ROE yorumu
        roe = additional.get('roe')
        if roe is not None:
            if roe > 0.2:
                interpretation += "Mükemmel özkaynak getirisi. "
            elif roe > 0.1:
                interpretation += "İyi özkaynak getirisi. "
            elif roe > 0:
                interpretation += "Düşük özkaynak getirisi. "
            else:
                interpretation += "Negatif özkaynak getirisi."
        
        return interpretation
    
    def get_finance_summary(self) -> Dict[str, Any]:
        """
        Gerçekleştirilen tüm finansal analizlerin özetini döndürür
        
        Returns:
            Finansal analiz özetleri
        """
        if not self.results:
            return {'message': 'Henüz finansal analiz gerçekleştirilmedi'}
        
        summary = {
            'total_analyses': len(self.results),
            'analyses_performed': list(self.results.keys()),
            'analysis_details': self.results
        }
        
        # En önemli bulgular
        key_findings = []
        
        if 'portfolio_analysis' in self.results:
            portfolio = self.results['portfolio_analysis']
            if 'individual_asset_stats' in portfolio:
                best_asset = max(portfolio['individual_asset_stats'].keys(), 
                               key=lambda x: portfolio['individual_asset_stats'][x]['sharpe_ratio'])
                key_findings.append(f"En iyi Sharpe oranı: {best_asset}")
        
        if 'risk_analysis' in self.results:
            risk = self.results['risk_analysis']
            if 'drawdown_analysis' in risk:
                max_dd = abs(risk['drawdown_analysis']['max_drawdown'])
                key_findings.append(f"Maksimum drawdown: %{max_dd*100:.1f}")
        
        if 'financial_ratios' in self.results:
            ratios = self.results['financial_ratios']
            if 'profitability_ratios' in ratios and ratios['profitability_ratios']['profit_margin']:
                profit_margin = ratios['profitability_ratios']['profit_margin']
                key_findings.append(f"Kar marjı: %{profit_margin*100:.1f}")
        
        summary['key_findings'] = key_findings
        
        return summary