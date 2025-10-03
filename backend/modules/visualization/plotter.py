"""
VeriVio Gelişmiş Görselleştirme Modülü
Kapsamlı grafik türleri ve interaktif görselleştirmeler
Otomatik analiz ve yorum desteği
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import plotly.figure_factory as ff
import base64
import io
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import gaussian_kde
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Matplotlib ve Seaborn stil ayarları
plt.style.use('default')
sns.set_palette("husl")


class DataPlotter:
    """Gelişmiş veri görselleştirme sınıfı"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 10)
        self.plots_created = []
        self.plot_counter = 0
        
        # Plotly tema ayarları
        self.plotly_theme = "plotly_white"
        
    def create_comprehensive_visualization_suite(self, df: pd.DataFrame, 
                                               analysis_type: str = "exploratory") -> Dict[str, Any]:
        """Kapsamlı görselleştirme paketi oluştur"""
        
        logger.info(f"Kapsamlı görselleştirme paketi oluşturuluyor: {analysis_type}")
        
        results = {
            'basic_plots': self._create_basic_plots(df),
            'distribution_plots': self._create_distribution_plots(df),
            'correlation_plots': self._create_correlation_plots(df),
            'advanced_plots': self._create_advanced_plots(df),
            'interactive_dashboard': self._create_interactive_dashboard(df),
            'statistical_plots': self._create_statistical_plots(df)
        }
        
        if analysis_type == "detailed":
            results['specialized_plots'] = self._create_specialized_plots(df)
            results['comparative_analysis'] = self._create_comparative_plots(df)
        
        logger.info("Kapsamlı görselleştirme paketi tamamlandı")
        return results
    
    def create_plot(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Tek bir grafik oluştur"""
        
        logger.info(f"Grafik oluşturuluyor: {params.get('visualization_type', 'default')}")
        
        visualization_type = params.get('visualization_type', 'histogram')
        columns = params.get('columns', [])
        
        try:
            if visualization_type == 'histogram':
                return self._create_histogram(df, columns, params)
            elif visualization_type == 'scatter':
                return self._create_scatter_plot(df, columns, params)
            elif visualization_type == 'box':
                return self._create_box_plot(df, columns, params)
            elif visualization_type == 'correlation':
                return self._create_correlation_heatmap(df, params)
            elif visualization_type == 'distribution':
                return self._create_distribution_plots(df)
            elif visualization_type == 'comprehensive':
                return self.create_comprehensive_visualization_suite(df, params.get('analysis_type', 'exploratory'))
            else:
                # Default to comprehensive visualization
                return self.create_comprehensive_visualization_suite(df, 'exploratory')
                
        except Exception as e:
            logger.error(f"Grafik oluşturma hatası: {str(e)}")
            return {
                'error': f"Grafik oluşturulamadı: {str(e)}",
                'success': False
            }
    
    def _create_histogram(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Histogram oluştur"""
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        
        histograms = {}
        for col in columns:
            if col in df.columns and df[col].dtype in [np.number]:
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.hist(df[col].dropna(), bins='auto', alpha=0.7, color=self.color_palette[0])
                ax.set_title(f'{col} - Histogram')
                ax.set_xlabel(col)
                ax.set_ylabel('Frekans')
                ax.grid(True, alpha=0.3)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
                buffer.seek(0)
                histograms[f'{col}_histogram'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
        
        return {
            'plots': histograms,
            'success': True,
            'plot_type': 'histogram'
        }
    
    def _create_scatter_plot(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Scatter plot oluştur"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
        elif len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
        else:
            return {'error': 'Scatter plot için en az 2 sayısal sütun gerekli', 'success': False}
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(df[x_col], df[y_col], alpha=0.6, color=self.color_palette[0])
        ax.set_title(f'{x_col} vs {y_col} - Scatter Plot')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        scatter_plot = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'plots': {'scatter_plot': scatter_plot},
            'success': True,
            'plot_type': 'scatter'
        }
    
    def _create_box_plot(self, df: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Box plot oluştur"""
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        
        box_plots = {}
        for col in columns:
            if col in df.columns and df[col].dtype in [np.number]:
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.boxplot(df[col].dropna(), labels=[col])
                ax.set_title(f'{col} - Box Plot')
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
                buffer.seek(0)
                box_plots[f'{col}_boxplot'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
        
        return {
            'plots': box_plots,
            'success': True,
            'plot_type': 'box'
        }
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Korelasyon heatmap oluştur"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'error': 'Korelasyon analizi için en az 2 sayısal sütun gerekli', 'success': False}
        
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Korelasyon Matrisi')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        heatmap = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'plots': {'correlation_heatmap': heatmap},
            'success': True,
            'plot_type': 'correlation'
        }
    
    def _create_basic_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Temel grafikler"""
        basic_plots = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Histogram ve yoğunluk grafikleri
        if len(numeric_cols) > 0:
            basic_plots['histograms'] = self._create_enhanced_histograms(df, numeric_cols)
            basic_plots['density_plots'] = self._create_density_plots(df, numeric_cols)
            basic_plots['box_plots'] = self._create_enhanced_boxplots(df, numeric_cols)
        
        # Kategorik değişken grafikleri
        if len(categorical_cols) > 0:
            basic_plots['bar_charts'] = self._create_enhanced_bar_charts(df, categorical_cols)
            basic_plots['pie_charts'] = self._create_pie_charts(df, categorical_cols)
        
        return basic_plots
    
    def _create_enhanced_histograms(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Gelişmiş histogram grafikleri"""
        histograms = {}
        
        for col in columns[:6]:  # İlk 6 sütun
            series = df[col].dropna()
            if len(series) < 3:
                continue
            
            # Matplotlib histogram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram + KDE
            ax1.hist(series, bins='auto', density=True, alpha=0.7, color=self.color_palette[0])
            
            # KDE eğrisi ekle
            try:
                kde = gaussian_kde(series)
                x_range = np.linspace(series.min(), series.max(), 100)
                ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                ax1.legend()
            except:
                pass
            
            ax1.set_title(f'{col} - Histogram + KDE')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Yoğunluk')
            ax1.grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(series, dist="norm", plot=ax2)
            ax2.set_title(f'{col} - Q-Q Plot (Normallik)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Base64 encode
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Plotly interaktif histogram
            plotly_fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[f'{col} - Histogram', f'{col} - Box Plot'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Histogram
            plotly_fig.add_trace(
                go.Histogram(x=series, name='Histogram', nbinsx=30),
                row=1, col=1
            )
            
            # Box plot
            plotly_fig.add_trace(
                go.Box(y=series, name='Box Plot'),
                row=1, col=2
            )
            
            plotly_fig.update_layout(
                title=f'{col} - Dağılım Analizi',
                template=self.plotly_theme,
                height=400
            )
            
            plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
            
            # İstatistiksel özellikler
            stats_info = {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'skewness': float(stats.skew(series)),
                'kurtosis': float(stats.kurtosis(series)),
                'normality_test': {
                    'shapiro_p': float(stats.shapiro(series)[1]) if len(series) <= 5000 else None,
                    'is_normal': bool(stats.shapiro(series)[1] > 0.05) if len(series) <= 5000 else None
                }
            }
            
            histograms[col] = {
                'image_base64': image_base64,
                'plotly_html': plotly_html,
                'statistics': stats_info
            }
        
        return histograms
    
    def _create_correlation_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Korelasyon grafikleri"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Korelasyon analizi için en az 2 sayısal sütun gerekli"}
        
        correlation_plots = {}
        
        # Korelasyon matrisi heatmap
        corr_matrix = df[numeric_cols].corr()
        
        # Matplotlib heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Korelasyon Matrisi Heatmap')
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Plotly interaktif heatmap
        plotly_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        plotly_heatmap.update_layout(
            title='İnteraktif Korelasyon Matrisi',
            template=self.plotly_theme,
            width=800,
            height=600
        )
        
        heatmap_html = plotly_heatmap.to_html(include_plotlyjs='cdn')
        
        # Scatter plot matrisi
        if len(numeric_cols) <= 6:  # Çok fazla sütun varsa performans sorunu
            scatter_matrix_fig = px.scatter_matrix(
                df[numeric_cols].dropna(),
                dimensions=numeric_cols.tolist(),
                title="Scatter Plot Matrisi"
            )
            scatter_matrix_fig.update_layout(template=self.plotly_theme)
            scatter_matrix_html = scatter_matrix_fig.to_html(include_plotlyjs='cdn')
        else:
            scatter_matrix_html = None
        
        # En güçlü korelasyonları bul
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': float(corr_val),
                        'strength': 'very_strong' if abs(corr_val) > 0.8 else 'strong'
                    })
        
        correlation_plots = {
            'heatmap_base64': heatmap_base64,
            'heatmap_html': heatmap_html,
            'scatter_matrix_html': scatter_matrix_html,
            'strong_correlations': strong_correlations,
            'correlation_matrix': corr_matrix.to_dict()
        }
        
        return correlation_plots
    
    def _create_distribution_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Dağılım grafikleri"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_plots = {}
        
        for col in numeric_cols[:4]:  # İlk 4 sütun
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            # Violin plot + Box plot kombinasyonu
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Violin plot
            sns.violinplot(y=series, ax=ax1, color=self.color_palette[0])
            ax1.set_title(f'{col} - Violin Plot')
            ax1.grid(True, alpha=0.3)
            
            # Histogram + Normal dağılım karşılaştırması
            ax2.hist(series, bins='auto', density=True, alpha=0.7, label='Veri')
            
            # Normal dağılım overlay
            mu, sigma = series.mean(), series.std()
            x = np.linspace(series.min(), series.max(), 100)
            normal_dist = stats.norm.pdf(x, mu, sigma)
            ax2.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Dağılım')
            ax2.set_title(f'{col} - Normal Dağılım Karşılaştırması')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # ECDF (Empirical Cumulative Distribution Function)
            sorted_data = np.sort(series)
            y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax3.plot(sorted_data, y_vals, marker='.', linestyle='none', alpha=0.7)
            ax3.set_title(f'{col} - ECDF')
            ax3.set_xlabel(col)
            ax3.set_ylabel('Kümülatif Olasılık')
            ax3.grid(True, alpha=0.3)
            
            # Percentile plot
            percentiles = np.percentile(series, np.arange(0, 101, 5))
            ax4.plot(np.arange(0, 101, 5), percentiles, 'o-')
            ax4.set_title(f'{col} - Percentile Plot')
            ax4.set_xlabel('Percentile')
            ax4.set_ylabel('Değer')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Plotly distplot
            plotly_fig = ff.create_distplot(
                [series], [col],
                bin_size=0.2,
                show_hist=True,
                show_curve=True,
                show_rug=True
            )
            plotly_fig.update_layout(
                title=f'{col} - Dağılım Analizi',
                template=self.plotly_theme
            )
            plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
            
            distribution_plots[col] = {
                'image_base64': image_base64,
                'plotly_html': plotly_html
            }
        
        return distribution_plots
    
    def _create_advanced_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gelişmiş grafikler"""
        advanced_plots = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Pair plot
            if len(numeric_cols) <= 5:  # Performans için sınırla
                pair_plot_data = df[numeric_cols].dropna()
                
                # Seaborn pair plot
                fig = plt.figure(figsize=(15, 12))
                pair_plot = sns.pairplot(pair_plot_data, diag_kind='kde')
                pair_plot.fig.suptitle('Pair Plot - Değişkenler Arası İlişkiler', y=1.02)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
                buffer.seek(0)
                pairplot_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                advanced_plots['pair_plot'] = pairplot_base64
            
            # 3D Scatter plot (ilk 3 sütun)
            if len(numeric_cols) >= 3:
                cols_3d = numeric_cols[:3].tolist()
                plot_data = df[cols_3d].dropna()
                
                plotly_3d = go.Figure(data=[go.Scatter3d(
                    x=plot_data[cols_3d[0]],
                    y=plot_data[cols_3d[1]],
                    z=plot_data[cols_3d[2]],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=plot_data[cols_3d[2]],
                        colorscale='Viridis',
                        showscale=True
                    )
                )])
                
                plotly_3d.update_layout(
                    title=f'3D Scatter Plot: {" vs ".join(cols_3d)}',
                    scene=dict(
                        xaxis_title=cols_3d[0],
                        yaxis_title=cols_3d[1],
                        zaxis_title=cols_3d[2]
                    ),
                    template=self.plotly_theme
                )
                
                advanced_plots['scatter_3d_html'] = plotly_3d.to_html(include_plotlyjs='cdn')
        
        # Parallel coordinates plot
        if len(numeric_cols) >= 3:
            parallel_data = df[numeric_cols].dropna()
            
            # Normalize data for better visualization
            normalized_data = (parallel_data - parallel_data.min()) / (parallel_data.max() - parallel_data.min())
            
            plotly_parallel = go.Figure(data=go.Parcoords(
                line=dict(color=normalized_data.iloc[:, 0],
                         colorscale='Viridis',
                         showscale=True),
                dimensions=list([
                    dict(range=[normalized_data[col].min(), normalized_data[col].max()],
                         label=col, values=normalized_data[col])
                    for col in normalized_data.columns
                ])
            ))
            
            plotly_parallel.update_layout(
                title='Parallel Coordinates Plot',
                template=self.plotly_theme
            )
            
            advanced_plots['parallel_coordinates_html'] = plotly_parallel.to_html(include_plotlyjs='cdn')
        
        return advanced_plots
    
    def _create_statistical_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """İstatistiksel grafikler"""
        statistical_plots = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # İlk 3 sütun
            series = df[col].dropna()
            if len(series) < 10:
                continue
            
            # Normallik testleri görselleştirmesi
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Q-Q plot
            stats.probplot(series, dist="norm", plot=ax1)
            ax1.set_title(f'{col} - Q-Q Plot (Normal)')
            ax1.grid(True, alpha=0.3)
            
            # P-P plot
            sorted_data = np.sort(series)
            theoretical_quantiles = stats.norm.cdf(sorted_data, series.mean(), series.std())
            empirical_quantiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax2.plot(theoretical_quantiles, empirical_quantiles, 'o', alpha=0.7)
            ax2.plot([0, 1], [0, 1], 'r-', linewidth=2)
            ax2.set_title(f'{col} - P-P Plot (Normal)')
            ax2.set_xlabel('Teorik Kümülatif Olasılık')
            ax2.set_ylabel('Gözlenen Kümülatif Olasılık')
            ax2.grid(True, alpha=0.3)
            
            # Residual plot (detrended)
            detrended = series - np.mean(series)
            ax3.scatter(range(len(detrended)), detrended, alpha=0.7)
            ax3.axhline(y=0, color='r', linestyle='--')
            ax3.set_title(f'{col} - Residual Plot')
            ax3.set_xlabel('Gözlem Sırası')
            ax3.set_ylabel('Residual')
            ax3.grid(True, alpha=0.3)
            
            # Lag plot (autocorrelation)
            if len(series) > 1:
                ax4.scatter(series[:-1], series[1:], alpha=0.7)
                ax4.set_title(f'{col} - Lag Plot')
                ax4.set_xlabel(f'{col}(t)')
                ax4.set_ylabel(f'{col}(t+1)')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            statistical_plots[col] = {
                'statistical_tests_plot': image_base64
            }
        
        return statistical_plots
    
    def _create_interactive_dashboard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """İnteraktif dashboard"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) == 0:
            return {"message": "İnteraktif dashboard için sayısal sütun gerekli"}
        
        # Ana dashboard
        dashboard_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Dağılım Özeti', 'Korelasyon Özeti', 'Outlier Analizi', 'Trend Analizi'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # İlk sayısal sütun için histogram
        if len(numeric_cols) > 0:
            col1 = numeric_cols[0]
            dashboard_fig.add_trace(
                go.Histogram(x=df[col1].dropna(), name=f'{col1} Dağılımı'),
                row=1, col=1
            )
        
        # Korelasyon için scatter plot
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            dashboard_fig.add_trace(
                go.Scatter(x=df[col1], y=df[col2], mode='markers', 
                          name=f'{col1} vs {col2}'),
                row=1, col=2
            )
        
        # Box plot için outlier analizi
        if len(numeric_cols) > 0:
            col1 = numeric_cols[0]
            dashboard_fig.add_trace(
                go.Box(y=df[col1].dropna(), name=f'{col1} Outliers'),
                row=2, col=1
            )
        
        # Trend analizi (zaman serisi benzeri)
        if len(numeric_cols) > 0:
            col1 = numeric_cols[0]
            dashboard_fig.add_trace(
                go.Scatter(y=df[col1].dropna(), mode='lines', 
                          name=f'{col1} Trend'),
                row=2, col=2
            )
        
        dashboard_fig.update_layout(
            title='VeriVio İnteraktif Veri Analizi Dashboard',
            template=self.plotly_theme,
            height=800,
            showlegend=True
        )
        
        dashboard_html = dashboard_fig.to_html(include_plotlyjs='cdn')
        
        return {
            'dashboard_html': dashboard_html,
            'description': 'İnteraktif veri analizi dashboard\'u'
        }
    
    def create_custom_plot(self, df: pd.DataFrame, plot_config: Dict[str, Any]) -> Dict[str, Any]:
        """Özel grafik oluştur"""
        plot_type = plot_config.get('type', 'scatter')
        
        if plot_type == 'advanced_scatter':
            return self._create_advanced_scatter(df, plot_config)
        elif plot_type == 'multi_axis':
            return self._create_multi_axis_plot(df, plot_config)
        elif plot_type == 'animated':
            return self._create_animated_plot(df, plot_config)
        else:
            raise ValueError(f"Desteklenmeyen grafik türü: {plot_type}")
    
    def generate_plot_summary(self) -> Dict[str, Any]:
        """Oluşturulan grafiklerin özeti"""
        return {
            'total_plots_created': len(self.plots_created),
            'plot_types': list(set([plot.get('type', 'unknown') for plot in self.plots_created])),
            'plots_summary': self.plots_created
        }
    
    def create_histogram(self, df: pd.DataFrame, columns: list) -> Dict[str, Any]:
        """Basit histogram oluştur (main.py uyumluluğu için)"""
        plots = {}
        for col in columns:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                plt.figure(figsize=self.figsize, dpi=self.dpi)
                sns.histplot(df[col], kde=True)
                plt.title(f"{col} Histogram")
                plt.xlabel(col)
                plt.ylabel("Frekans")
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                plots[f"{col}_histogram"] = base64.b64encode(buf.read()).decode('utf-8')
        
        return {'plot_type': 'histogram', 'plots': plots}