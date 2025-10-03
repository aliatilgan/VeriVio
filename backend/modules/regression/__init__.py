"""
Regresyon Analizi Modülü

Bu modül çeşitli regresyon analizi türlerini gerçekleştirmek için kullanılır.
Linear, logistic, polynomial ve diğer regresyon türlerini destekler.
"""

from .analyzer import ComprehensiveRegressionAnalyzer
from .advanced_regression import AdvancedRegressionAnalyzer

__all__ = ['ComprehensiveRegressionAnalyzer', 'AdvancedRegressionAnalyzer']
__version__ = "1.0.0"