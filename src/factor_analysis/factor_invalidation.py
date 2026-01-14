"""
因子失效多维度检测模块
Multi-dimensional Factor Invalidation Detection Module

综合IC、Sharpe、衰减、拥挤度等多维度指标判断因子是否失效

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass

from .ic_analysis import ICAnalyzer
from .decay_analysis import DecayAnalyzer
from .crowding_detector import CrowdingDetector


@dataclass
class InvalidationConfig:
    """因子失效检测配置"""
    ic_threshold: float = 0.02          # IC阈值（低于此值视为失效）
    ic_duration: int = 60               # IC持续低于阈值的天数
    sharpe_threshold: float = 0.5       # Sharpe阈值
    decay_threshold: float = 0.02       # 衰减阈值
    crowding_threshold: float = 0.5     # 拥挤度阈值
    significance_level: float = 0.05    # 显著性水平


class FactorInvalidationDetector:
    """
    因子失效多维度检测器
    
    检测维度：
    1. IC衰减
    2. Sharpe下降
    3. 因子衰减
    4. 拥挤交易
    5. 统计显著性
    """
    
    def __init__(self, config: InvalidationConfig = None):
        """
        初始化失效检测器
        
        Args:
            config: 配置参数
        """
        self.config = config or InvalidationConfig()
        
        # 子分析器
        self.ic_analyzer = ICAnalyzer(window=252)
        self.decay_analyzer = DecayAnalyzer(
            decay_threshold=self.config.decay_threshold
        )
        self.crowding_detector = CrowdingDetector()
    
    def detect_invalidation(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        factor_exposures: pd.DataFrame = None
    ) -> Dict:
        """
        多维度检测因子失效
        
        Args:
            factor_values: 因子值序列
            returns: 未来收益率序列
            factor_exposures: 因子暴露度DataFrame（可选，用于拥挤检测）
            
        Returns:
            失效检测结果
        """
        # 1. IC检测
        ic_result = self._detect_ic_invalidation(factor_values, returns)
        
        # 2. Sharpe检测
        sharpe_result = self._detect_sharpe_invalidation(factor_values, returns)
        
        # 3. 衰减检测
        decay_result = self._detect_decay_invalidation(factor_values, returns)
        
        # 4. 拥挤检测（如果提供暴露度）
        crowding_result = None
        if factor_exposures is not None:
            crowding_result = self._detect_crowding_invalidation(factor_exposures)
        
        # 5. 统计显著性检测
        significance_result = self._detect_significance(factor_values, returns)
        
        # 6. 综合判断
        invalidation_score = self._calculate_invalidation_score(
            ic_result, sharpe_result, decay_result, crowding_result, significance_result
        )
        
        is_invalidated = invalidation_score >= 0.6  # 综合得分>=0.6视为失效
        
        return {
            'is_invalidated': is_invalidated,
            'invalidation_score': invalidation_score,
            'ic_result': ic_result,
            'sharpe_result': sharpe_result,
            'decay_result': decay_result,
            'crowding_result': crowding_result,
            'significance_result': significance_result,
            'dimensions': {
                'ic_failed': ic_result.get('is_failed', False),
                'sharpe_failed': sharpe_result.get('is_failed', False),
                'decay_failed': decay_result.get('is_decaying', False),
                'crowding_failed': crowding_result.get('is_crowded', False) if crowding_result else False,
                'not_significant': not significance_result.get('is_significant', True)
            }
        }
    
    def _detect_ic_invalidation(
        self,
        factor_values: pd.Series,
        returns: pd.Series
    ) -> Dict:
        """
        检测IC失效
        """
        # 计算Rolling IC
        ic_series = self.ic_analyzer.analyze(factor_values, returns)['ic_series']
        
        # 检查最近窗口的IC
        recent_ic = ic_series.tail(self.config.ic_duration)
        recent_mean_ic = recent_ic.mean()
        
        # 检查IC是否持续低于阈值
        below_threshold = (recent_ic < self.config.ic_threshold).sum()
        below_ratio = below_threshold / len(recent_ic) if len(recent_ic) > 0 else 0
        
        is_failed = recent_mean_ic < self.config.ic_threshold and below_ratio > 0.7
        
        return {
            'is_failed': is_failed,
            'recent_mean_ic': float(recent_mean_ic),
            'below_threshold_ratio': float(below_ratio),
            'ic_series': ic_series
        }
    
    def _detect_sharpe_invalidation(
        self,
        factor_values: pd.Series,
        returns: pd.Series
    ) -> Dict:
        """
        检测Sharpe失效
        """
        from .ic_analysis import calculate_rolling_sharpe
        
        sharpe_series = calculate_rolling_sharpe(factor_values, returns, window=252)
        
        recent_sharpe = sharpe_series.tail(60).mean()
        is_failed = recent_sharpe < self.config.sharpe_threshold
        
        return {
            'is_failed': is_failed,
            'recent_mean_sharpe': float(recent_sharpe),
            'sharpe_series': sharpe_series
        }
    
    def _detect_decay_invalidation(
        self,
        factor_values: pd.Series,
        returns: pd.Series
    ) -> Dict:
        """
        检测衰减失效
        """
        from .ic_analysis import calculate_rolling_ic
        
        ic_series = calculate_rolling_ic(factor_values, returns, window=252)
        decay_result = self.decay_analyzer.analyze_decay(ic_series)
        
        return {
            'is_decaying': decay_result['is_decaying'],
            'decay_start_date': decay_result.get('decay_start_date'),
            'decay_rate': decay_result.get('decay_rate', 0.0)
        }
    
    def _detect_crowding_invalidation(
        self,
        factor_exposures: pd.DataFrame
    ) -> Dict:
        """
        检测拥挤失效
        """
        crowding_result = self.crowding_detector.detect_crowding(factor_exposures)
        
        return {
            'is_crowded': crowding_result['is_crowded'],
            'crowding_score': crowding_result['crowding_score']
        }
    
    def _detect_significance(
        self,
        factor_values: pd.Series,
        returns: pd.Series
    ) -> Dict:
        """
        检测统计显著性
        """
        # 计算IC
        ic_series = self.ic_analyzer.analyze(factor_values, returns)['ic_series']
        ic_clean = ic_series.dropna()
        
        if len(ic_clean) < 10:
            return {
                'is_significant': False,
                'p_value': 1.0
            }
        
        # t检验：IC是否显著不为0
        t_stat, p_value = stats.ttest_1samp(ic_clean, 0)
        is_significant = p_value < self.config.significance_level
        
        return {
            'is_significant': is_significant,
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        }
    
    def _calculate_invalidation_score(
        self,
        ic_result: Dict,
        sharpe_result: Dict,
        decay_result: Dict,
        crowding_result: Optional[Dict],
        significance_result: Dict
    ) -> float:
        """
        计算综合失效得分（0-1之间，越高越可能失效）
        """
        scores = []
        weights = []
        
        # IC失效（权重0.3）
        if ic_result.get('is_failed', False):
            scores.append(1.0)
        else:
            # 根据IC接近阈值的程度给分
            recent_ic = ic_result.get('recent_mean_ic', 0.0)
            if recent_ic < self.config.ic_threshold:
                scores.append(1.0)
            else:
                # 线性映射：IC从阈值到0.1，得分从1到0
                score = max(0, 1 - (recent_ic - self.config.ic_threshold) / 0.08)
                scores.append(score)
        weights.append(0.3)
        
        # Sharpe失效（权重0.2）
        if sharpe_result.get('is_failed', False):
            scores.append(1.0)
        else:
            recent_sharpe = sharpe_result.get('recent_mean_sharpe', 0.0)
            score = max(0, 1 - recent_sharpe / self.config.sharpe_threshold)
            scores.append(score)
        weights.append(0.2)
        
        # 衰减失效（权重0.2）
        if decay_result.get('is_decaying', False):
            scores.append(1.0)
        else:
            scores.append(0.0)
        weights.append(0.2)
        
        # 拥挤失效（权重0.15，可选）
        if crowding_result:
            if crowding_result.get('is_crowded', False):
                scores.append(1.0)
            else:
                crowding_score = crowding_result.get('crowding_score', 0.0)
                scores.append(min(1.0, crowding_score / self.config.crowding_threshold))
            weights.append(0.15)
        else:
            weights.append(0.0)
        
        # 显著性失效（权重0.15）
        if not significance_result.get('is_significant', True):
            scores.append(1.0)
        else:
            p_value = significance_result.get('p_value', 1.0)
            scores.append(min(1.0, p_value / self.config.significance_level))
        weights.append(0.15)
        
        # 加权平均
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        invalidation_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return float(invalidation_score)
    
    def get_invalidation_warning(
        self,
        invalidation_result: Dict
    ) -> str:
        """
        生成失效预警信息
        """
        if not invalidation_result['is_invalidated']:
            return "因子状态正常"
        
        warnings = []
        dims = invalidation_result['dimensions']
        
        if dims['ic_failed']:
            warnings.append("IC持续低于阈值")
        if dims['sharpe_failed']:
            warnings.append("Sharpe比率下降")
        if dims['decay_failed']:
            warnings.append("因子出现衰减")
        if dims['crowding_failed']:
            warnings.append("检测到拥挤交易")
        if dims['not_significant']:
            warnings.append("统计显著性不足")
        
        score = invalidation_result['invalidation_score']
        
        return f"⚠️ 因子可能失效（得分: {score:.2f}）\n原因: {', '.join(warnings)}"


if __name__ == "__main__":
    print("=== 因子失效检测器测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 模拟失效的因子（IC逐渐下降）
    factor_values = pd.Series(np.random.randn(500).cumsum(), index=dates)
    returns = pd.Series(np.random.randn(500) * 0.001, index=dates)  # 低相关性，模拟失效
    
    # 创建检测器
    detector = FactorInvalidationDetector()
    result = detector.detect_invalidation(factor_values, returns)
    
    print(f"\n失效检测结果:")
    print(f"  是否失效: {result['is_invalidated']}")
    print(f"  失效得分: {result['invalidation_score']:.4f}")
    print(f"  预警信息: {detector.get_invalidation_warning(result)}")
