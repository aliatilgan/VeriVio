"""
Hipotez Testleri Modülü

Bu modül çeşitli istatistiksel hipotez testlerini gerçekleştirmek için kullanılır.
Parametrik ve non-parametrik testleri destekler.
"""

from .tester import ComprehensiveHypothesisTester
from .advanced_tests import AdvancedHypothesisTester

__all__ = ['ComprehensiveHypothesisTester', 'AdvancedHypothesisTester']
__version__ = "1.0.0"