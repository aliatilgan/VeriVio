"""
Alan Uygulamaları Modülü

Bu modül finans, pazarlama ve sağlık alanlarına özel analiz araçları içerir.
"""

from .finance import FinanceAnalyzer
from .marketing import MarketingAnalyzer
from .healthcare import HealthcareAnalyzer

__version__ = "1.0.0"
__all__ = ["FinanceAnalyzer", "MarketingAnalyzer", "HealthcareAnalyzer"]