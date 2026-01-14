"""
因子研究模块
Factor Research Module

包含：
- 行为偏差分析
- 信息扩散分析
- 因子相关性分析
- 因子稳定性分析
- 因子组合优化
"""

from .behavioral_bias import BehavioralBiasAnalyzer
from .information_diffusion import InformationDiffusionAnalyzer
from .factor_correlation import FactorCorrelationAnalyzer
from .factor_stability import FactorStabilityAnalyzer
from .factor_combination import FactorCombinationOptimizer

__all__ = [
    'BehavioralBiasAnalyzer',
    'InformationDiffusionAnalyzer',
    'FactorCorrelationAnalyzer',
    'FactorStabilityAnalyzer',
    'FactorCombinationOptimizer'
]
