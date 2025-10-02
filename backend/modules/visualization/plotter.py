"""
VeriVio Ana Veri Görselleştirme Sınıfı
Temel grafik türleri ve görselleştirme işlemleri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
import logging

logger = logging.getLogger(__name__)

# Matplotlib ve Seaborn ayarları
plt.style.use('default')
sns.set_palette("husl")


class DataPlotter:
    """Ana veri görselleştirme sınıfı"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 10)
        self.plots_created = []
    
    def create_histogram(self, df: pd.DataFrame, column: str, 
                        bins: Union[int, str] = 'auto',
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """Histogram oluştur"""
        
        if column not in df.columns:
            raise ValueError(f"Sütun '{column}' veri setinde bulunamadı")
        
        series = df[column].dropna()
        
        if len(series) == 0:
            raise ValueError(f"'{column}' sütununda görselleştirilebilir veri yok")
        
        # Matplotlib histogram
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        n, bins_array, patches = ax.hist(series, bins=bins, alpha=0.7, 
                                        color=self.color_palette[0], edgecolor='black')
        
        ax.set_xlabel(column)
        ax.set_ylabel('Frekans')
        ax.set_title(title or f'{column} Histogramı')
        ax.grid(True, alpha=0.3)
        
        # İstatistikler ekle
        mean_val = series.mean()
        median_val = series.median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Ortalama: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Medyan: {median_val:.2f}')
        ax.legend()
        
        plt.tight_layout()
        
        # Kaydet
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Base64 encode
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()
        
        # Plotly versiyonu
        plotly_fig = px.histogram(df, x=column, title=title or f'{column} Histogramı',
                                 nbins=len(bins_array)-1 if isinstance(bins_array, np.ndarray) else bins)
        plotly_fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                            annotation_text=f"Ortalama: {mean_val:.2f}")
        plotly_fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                            annotation_text=f"Medyan: {median_val:.2f}")
        
        plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
        
        result = {
            'type': 'histogram',
            'column': column,
            'image_base64': image_base64,
            'plotly_html': plotly_html,
            'statistics': {
                'mean': float(mean_val),
                'median': float(median_val),
                'std': float(series.std()),
                'count': len(series),
                'bins_count': len(bins_array)-1 if isinstance(bins_array, np.ndarray) else bins
            },
            'save_path': save_path
        }
        
        self.plots_created.append(result)
        logger.info(f"{column} için histogram oluşturuldu")
        
        return result
    
    def create_boxplot(self, df: pd.DataFrame, columns: List[str],
                      title: Optional[str] = None,
                      save_path: Optional[str] = None) -> Dict[str, Any]:
        """Box plot oluştur"""
        
        numeric_cols = [col for col in columns if col in df.columns and 
                       df[col].dtype in [np.number, 'int64', 'float64']]
        
        if not numeric_cols:
            raise ValueError("Box plot için sayısal sütun bulunamadı")
        
        # Matplotlib box plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        data_to_plot = [df[col].dropna() for col in numeric_cols]
        box_plot = ax.boxplot(data_to_plot, labels=numeric_cols, patch_artist=True)
        
        # Renklendirme
        for patch, color in zip(box_plot['boxes'], self.color_palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(title or 'Box Plot')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Base64 encode
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()
        
        # Plotly versiyonu
        plotly_fig = go.Figure()
        for i, col in enumerate(numeric_cols):
            plotly_fig.add_trace(go.Box(y=df[col].dropna(), name=col))
        
        plotly_fig.update_layout(title=title or 'Box Plot')
        plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
        
        # Outlier istatistikleri
        outlier_stats = {}
        for col in numeric_cols:
            series = df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            outlier_stats[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': float((len(outliers) / len(series)) * 100),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr)
            }
        
        result = {
            'type': 'boxplot',
            'columns': numeric_cols,
            'image_base64': image_base64,
            'plotly_html': plotly_html,
            'outlier_statistics': outlier_stats,
            'save_path': save_path
        }
        
        self.plots_created.append(result)
        logger.info(f"{len(numeric_cols)} sütun için box plot oluşturuldu")
        
        return result
    
    def create_scatter_plot(self, df: pd.DataFrame, x_column: str, y_column: str,
                           color_column: Optional[str] = None,
                           size_column: Optional[str] = None,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """Scatter plot oluştur"""
        
        if x_column not in df.columns or y_column not in df.columns:
            raise ValueError("Belirtilen sütunlar veri setinde bulunamadı")
        
        # Veriyi temizle
        plot_data = df[[x_column, y_column]].dropna()
        
        if color_column and color_column in df.columns:
            plot_data = df[[x_column, y_column, color_column]].dropna()
        
        if size_column and size_column in df.columns:
            plot_data = df[[x_column, y_column, size_column]].dropna()
            if color_column:
                plot_data = df[[x_column, y_column, color_column, size_column]].dropna()
        
        if len(plot_data) == 0:
            raise ValueError("Görselleştirilebilir veri yok")
        
        # Matplotlib scatter plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if color_column and color_column in plot_data.columns:
            scatter = ax.scatter(plot_data[x_column], plot_data[y_column], 
                               c=plot_data[color_column], alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, label=color_column)
        else:
            ax.scatter(plot_data[x_column], plot_data[y_column], 
                      alpha=0.6, color=self.color_palette[0])
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(title or f'{x_column} vs {y_column}')
        ax.grid(True, alpha=0.3)
        
        # Korelasyon hesapla ve ekle
        correlation = plot_data[x_column].corr(plot_data[y_column])
        ax.text(0.05, 0.95, f'Korelasyon: {correlation:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Base64 encode
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()
        
        # Plotly versiyonu
        plotly_fig = px.scatter(plot_data, x=x_column, y=y_column, 
                               color=color_column, size=size_column,
                               title=title or f'{x_column} vs {y_column}')
        
        # Trend line ekle
        plotly_fig.add_trace(go.Scatter(
            x=plot_data[x_column], 
            y=np.poly1d(np.polyfit(plot_data[x_column], plot_data[y_column], 1))(plot_data[x_column]),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        ))
        
        plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
        
        result = {
            'type': 'scatter',
            'x_column': x_column,
            'y_column': y_column,
            'color_column': color_column,
            'size_column': size_column,
            'image_base64': image_base64,
            'plotly_html': plotly_html,
            'correlation': float(correlation),
            'data_points': len(plot_data),
            'save_path': save_path
        }
        
        self.plots_created.append(result)
        logger.info(f"{x_column} vs {y_column} scatter plot oluşturuldu")
        
        return result
    
    def create_correlation_heatmap(self, df: pd.DataFrame, 
                                  columns: Optional[List[str]] = None,
                                  method: str = 'pearson',
                                  title: Optional[str] = None,
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """Korelasyon ısı haritası oluştur"""
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.columns and 
                           df[col].dtype in [np.number, 'int64', 'float64']]
        
        if len(numeric_cols) < 2:
            raise ValueError("Korelasyon analizi için en az 2 sayısal sütun gerekli")
        
        # Korelasyon matrisi
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # Matplotlib heatmap
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Üst üçgeni gizle
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title(title or f'Korelasyon Matrisi ({method.title()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Base64 encode
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()
        
        # Plotly versiyonu
        plotly_fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title=title or f'Korelasyon Matrisi ({method.title()})',
                              color_continuous_scale='RdBu_r')
        
        plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
        
        # En yüksek korelasyonları bul
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if not np.isnan(corr_val):
                    corr_pairs.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': float(corr_val)
                    })
        
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        result = {
            'type': 'correlation_heatmap',
            'method': method,
            'columns': numeric_cols,
            'image_base64': image_base64,
            'plotly_html': plotly_html,
            'correlation_matrix': corr_matrix.to_dict(),
            'top_correlations': corr_pairs[:10],
            'save_path': save_path
        }
        
        self.plots_created.append(result)
        logger.info(f"Korelasyon ısı haritası oluşturuldu ({method} yöntemi)")
        
        return result
    
    def create_bar_chart(self, df: pd.DataFrame, column: str,
                        value_column: Optional[str] = None,
                        top_n: int = 20,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """Bar chart oluştur"""
        
        if column not in df.columns:
            raise ValueError(f"Sütun '{column}' veri setinde bulunamadı")
        
        if value_column and value_column not in df.columns:
            raise ValueError(f"Değer sütunu '{value_column}' veri setinde bulunamadı")
        
        # Veri hazırlama
        if value_column:
            # Gruplandırılmış bar chart
            plot_data = df.groupby(column)[value_column].sum().sort_values(ascending=False).head(top_n)
        else:
            # Frekans bar chart
            plot_data = df[column].value_counts().head(top_n)
        
        if len(plot_data) == 0:
            raise ValueError("Görselleştirilebilir veri yok")
        
        # Matplotlib bar chart
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        bars = ax.bar(range(len(plot_data)), plot_data.values, 
                     color=self.color_palette[:len(plot_data)])
        
        ax.set_xlabel(column)
        ax.set_ylabel(value_column or 'Frekans')
        ax.set_title(title or f'{column} Bar Chart')
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Değerleri bar'ların üzerine ekle
        for bar, value in zip(bars, plot_data.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Base64 encode
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()
        
        # Plotly versiyonu
        plotly_fig = px.bar(x=plot_data.index, y=plot_data.values,
                           title=title or f'{column} Bar Chart',
                           labels={'x': column, 'y': value_column or 'Frekans'})
        
        plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
        
        result = {
            'type': 'bar_chart',
            'column': column,
            'value_column': value_column,
            'top_n': top_n,
            'image_base64': image_base64,
            'plotly_html': plotly_html,
            'data_summary': {
                'categories_shown': len(plot_data),
                'total_categories': df[column].nunique(),
                'top_category': str(plot_data.index[0]),
                'top_value': float(plot_data.iloc[0])
            },
            'save_path': save_path
        }
        
        self.plots_created.append(result)
        logger.info(f"{column} için bar chart oluşturuldu")
        
        return result
    
    def create_line_plot(self, df: pd.DataFrame, x_column: str, y_columns: List[str],
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """Line plot oluştur"""
        
        if x_column not in df.columns:
            raise ValueError(f"X sütunu '{x_column}' veri setinde bulunamadı")
        
        valid_y_columns = [col for col in y_columns if col in df.columns]
        
        if not valid_y_columns:
            raise ValueError("Geçerli Y sütunu bulunamadı")
        
        # Veriyi sırala
        plot_data = df[[x_column] + valid_y_columns].dropna().sort_values(x_column)
        
        if len(plot_data) == 0:
            raise ValueError("Görselleştirilebilir veri yok")
        
        # Matplotlib line plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for i, col in enumerate(valid_y_columns):
            ax.plot(plot_data[x_column], plot_data[col], 
                   label=col, color=self.color_palette[i % len(self.color_palette)],
                   linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel(x_column)
        ax.set_ylabel('Değer')
        ax.set_title(title or f'{x_column} Zaman Serisi')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Base64 encode
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()
        
        # Plotly versiyonu
        plotly_fig = go.Figure()
        
        for col in valid_y_columns:
            plotly_fig.add_trace(go.Scatter(
                x=plot_data[x_column],
                y=plot_data[col],
                mode='lines+markers',
                name=col,
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        plotly_fig.update_layout(
            title=title or f'{x_column} Zaman Serisi',
            xaxis_title=x_column,
            yaxis_title='Değer'
        )
        
        plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
        
        result = {
            'type': 'line_plot',
            'x_column': x_column,
            'y_columns': valid_y_columns,
            'image_base64': image_base64,
            'plotly_html': plotly_html,
            'data_points': len(plot_data),
            'save_path': save_path
        }
        
        self.plots_created.append(result)
        logger.info(f"{x_column} için line plot oluşturuldu")
        
        return result
    
    def create_pie_chart(self, df: pd.DataFrame, column: str,
                        top_n: int = 10,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """Pie chart oluştur"""
        
        if column not in df.columns:
            raise ValueError(f"Sütun '{column}' veri setinde bulunamadı")
        
        # Veri hazırlama
        value_counts = df[column].value_counts().head(top_n)
        
        if len(value_counts) == 0:
            raise ValueError("Görselleştirilebilir veri yok")
        
        # Diğer kategorileri birleştir
        if len(df[column].value_counts()) > top_n:
            others_count = df[column].value_counts().iloc[top_n:].sum()
            value_counts['Diğer'] = others_count
        
        # Matplotlib pie chart
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index,
                                         autopct='%1.1f%%', startangle=90,
                                         colors=self.color_palette[:len(value_counts)])
        
        ax.set_title(title or f'{column} Dağılımı')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Base64 encode
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()
        
        # Plotly versiyonu
        plotly_fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=title or f'{column} Dağılımı')
        
        plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
        
        result = {
            'type': 'pie_chart',
            'column': column,
            'top_n': top_n,
            'image_base64': image_base64,
            'plotly_html': plotly_html,
            'categories_data': value_counts.to_dict(),
            'total_categories': df[column].nunique(),
            'save_path': save_path
        }
        
        self.plots_created.append(result)
        logger.info(f"{column} için pie chart oluşturuldu")
        
        return result
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str,
                               plot_type: str = 'both',
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """Dağılım grafiği oluştur (histogram + KDE)"""
        
        if column not in df.columns:
            raise ValueError(f"Sütun '{column}' veri setinde bulunamadı")
        
        series = df[column].dropna()
        
        if len(series) == 0:
            raise ValueError(f"'{column}' sütununda görselleştirilebilir veri yok")
        
        # Matplotlib distribution plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if plot_type in ['histogram', 'both']:
            ax.hist(series, bins='auto', alpha=0.7, density=True, 
                   color=self.color_palette[0], edgecolor='black', label='Histogram')
        
        if plot_type in ['kde', 'both']:
            series.plot.kde(ax=ax, color='red', linewidth=2, label='KDE')
        
        ax.set_xlabel(column)
        ax.set_ylabel('Yoğunluk')
        ax.set_title(title or f'{column} Dağılımı')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        # Base64 encode
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()
        
        # Plotly versiyonu
        plotly_fig = go.Figure()
        
        if plot_type in ['histogram', 'both']:
            plotly_fig.add_trace(go.Histogram(
                x=series, 
                histnorm='probability density',
                name='Histogram',
                opacity=0.7
            ))
        
        if plot_type in ['kde', 'both']:
            # KDE hesapla
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series)
            x_range = np.linspace(series.min(), series.max(), 100)
            kde_values = kde(x_range)
            
            plotly_fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_values,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2)
            ))
        
        plotly_fig.update_layout(
            title=title or f'{column} Dağılımı',
            xaxis_title=column,
            yaxis_title='Yoğunluk'
        )
        
        plotly_html = plotly_fig.to_html(include_plotlyjs='cdn')
        
        result = {
            'type': 'distribution_plot',
            'column': column,
            'plot_type': plot_type,
            'image_base64': image_base64,
            'plotly_html': plotly_html,
            'statistics': {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis())
            },
            'save_path': save_path
        }
        
        self.plots_created.append(result)
        logger.info(f"{column} için dağılım grafiği oluşturuldu")
        
        return result
    
    def get_plot_summary(self) -> Dict[str, Any]:
        """Oluşturulan grafiklerin özeti"""
        
        plot_types = {}
        for plot in self.plots_created:
            plot_type = plot['type']
            if plot_type in plot_types:
                plot_types[plot_type] += 1
            else:
                plot_types[plot_type] = 1
        
        return {
            'total_plots': len(self.plots_created),
            'plot_types': plot_types,
            'plots_created': self.plots_created
        }