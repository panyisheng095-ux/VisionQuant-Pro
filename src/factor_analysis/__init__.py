"""
因子有效性分析模块
Factor Effectiveness Analysis Module

包含：
- IC/Sharpe分析
- Regime识别
- 因子衰减分析
- 拥挤交易检测
- 风险补偿分析
- 行业分层分析
- 因子失效检测
- 报告生成
"""

from .ic_analysis import ICAnalyzer, calculate_rolling_ic, calculate_rolling_sharpe
from .regime_detector import RegimeDetector, MarketRegime
from .decay_analysis import DecayAnalyzer
from .crowding_detector import CrowdingDetector
from .risk_compensation import RiskCompensationAnalyzer
from .industry_stratification import IndustryStratifier
from .factor_invalidation import FactorInvalidationDetector, InvalidationConfig
from .report_generator import FactorReportGenerator

__all__ = [
    'ICAnalyzer',
    'calculate_rolling_ic',
    'calculate_rolling_sharpe',
    'RegimeDetector',
    'MarketRegime',
    'DecayAnalyzer',
    'CrowdingDetector',
    'RiskCompensationAnalyzer',
    'IndustryStratifier',
    'FactorInvalidationDetector',
    'InvalidationConfig',
    'FactorReportGenerator'
]
