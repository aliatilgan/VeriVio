"""
Regresyon Analizi Modülü

Bu modül çeşitli regresyon analizi türlerini gerçekleştirmek için kullanılır.
Linear, logistic, polynomial ve diğer regresyon türlerini destekler.
"""

from .analyzer import RegressionAnalyzer
from .advanced_regression import AdvancedRegressionAnalyzer

__all__ = ['RegressionAnalyzer', 'AdvancedRegressionAnalyzer']
__version__ = "1.0.0"