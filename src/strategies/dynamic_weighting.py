"""
动态权重管理系统
Dynamic Weighting Management System

根据因子失效程度动态调整权重

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.factor_analysis.factor_invalidation import FactorInvalidationDetector, InvalidationConfig


@dataclass
class WeightConfig:
    """权重配置"""
    base_weight: float = 0.6          # 基础权重（K线因子）
    min_weight: float = 0.1           # 最小权重
    max_weight: float = 0.8           # 最大权重
    invalidation_threshold: float = 0.6  # 失效阈值
    decay_penalty: float = 0.2       # 衰减惩罚系数


class DynamicWeightManager:
    """
    动态权重管理器
    
    功能：
    1. 根据因子失效程度调整权重
    2. 平滑权重变化（避免剧烈波动）
    3. 权重历史记录
    """
    
    def __init__(
        self,
        weight_config: WeightConfig = None,
        invalidation_config: InvalidationConfig = None
    ):
        """
        初始化动态权重管理器
        
        Args:
            weight_config: 权重配置
            invalidation_config: 失效检测配置
        """
        self.weight_config = weight_config or WeightConfig()
        self.invalidation_detector = FactorInvalidationDetector(invalidation_config)
        
        # 权重历史
        self.weight_history = []
    
    def calculate_dynamic_weight(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        factor_exposures: pd.DataFrame = None,
        current_date: pd.Timestamp = None
    ) -> Dict:
        """
        计算动态权重
        
        Args:
            factor_values: 因子值序列
            returns: 未来收益率序列
            factor_exposures: 因子暴露度（可选）
            current_date: 当前日期（可选，用于时间序列权重）
            
        Returns:
            权重计算结果
        """
        # 1. 检测因子失效
        invalidation_result = self.invalidation_detector.detect_invalidation(
            factor_values, returns, factor_exposures
        )
        
        # 2. 计算权重调整
        invalidation_score = invalidation_result['invalidation_score']
        
        # 权重调整公式：
        # new_weight = base_weight * (1 - invalidation_score * decay_penalty)
        weight_adjustment = 1 - invalidation_score * self.weight_config.decay_penalty
        adjusted_weight = self.weight_config.base_weight * weight_adjustment
        
        # 3. 限制权重范围
        final_weight = max(
            self.weight_config.min_weight,
            min(self.weight_config.max_weight, adjusted_weight)
        )
        
        # 4. 平滑处理（如果有权重历史）
        if len(self.weight_history) > 0:
            last_weight = self.weight_history[-1]['weight']
            # 指数移动平均平滑
            smoothing_factor = 0.3
            final_weight = smoothing_factor * final_weight + (1 - smoothing_factor) * last_weight
        
        # 5. 记录权重历史
        weight_record = {
            'date': current_date if current_date else pd.Timestamp.now(),
            'weight': final_weight,
            'invalidation_score': invalidation_score,
            'is_invalidated': invalidation_result['is_invalidated'],
            'adjustment_reason': self._get_adjustment_reason(invalidation_result)
        }
        self.weight_history.append(weight_record)
        
        return {
            'weight': final_weight,
            'base_weight': self.weight_config.base_weight,
            'adjustment': final_weight - self.weight_config.base_weight,
            'adjustment_pct': (final_weight - self.weight_config.base_weight) / self.weight_config.base_weight * 100,
            'invalidation_score': invalidation_score,
            'is_invalidated': invalidation_result['is_invalidated'],
            'reason': weight_record['adjustment_reason']
        }
    
    def _get_adjustment_reason(self, invalidation_result: Dict) -> str:
        """
        获取权重调整原因
        """
        reasons = []
        dims = invalidation_result['dimensions']
        
        if dims['ic_failed']:
            reasons.append('IC失效')
        if dims['sharpe_failed']:
            reasons.append('Sharpe下降')
        if dims['decay_failed']:
            reasons.append('因子衰减')
        if dims['crowding_failed']:
            reasons.append('拥挤交易')
        if dims['not_significant']:
            reasons.append('统计不显著')
        
        return ', '.join(reasons) if reasons else '正常'
    
    def get_weight_history(self) -> pd.DataFrame:
        """
        获取权重历史记录
        """
        if not self.weight_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.weight_history)
    
    def reset_history(self):
        """
        重置权重历史
        """
        self.weight_history = []
    
    def calculate_regime_adjusted_weight(
        self,
        base_weight: float,
        current_regime: str,
        factor_ic: float
    ) -> float:
        """
        根据市场Regime和因子IC计算调整后的权重
        
        Args:
            base_weight: 基础权重
            current_regime: 当前市场状态 ('bull_market', 'bear_market', 'oscillating')
            factor_ic: 当前因子IC值
            
        Returns:
            调整后的权重
        """
        # Regime权重调整系数
        regime_multipliers = {
            'bull_market': 1.2,      # 牛市：K线因子权重增加20%
            'bear_market': 0.8,      # 熊市：K线因子权重降低20%
            'oscillating': 1.0       # 震荡市：不变
        }
        
        regime_mult = regime_multipliers.get(current_regime, 1.0)
        
        # IC调整系数（IC越高，权重越高）
        ic_mult = 1 + factor_ic * 2  # IC每增加0.01，权重增加2%
        ic_mult = max(0.5, min(1.5, ic_mult))  # 限制在0.5-1.5倍
        
        # 综合调整
        adjusted_weight = base_weight * regime_mult * ic_mult
        
        # 限制范围
        return max(
            self.weight_config.min_weight,
            min(self.weight_config.max_weight, adjusted_weight)
        )


if __name__ == "__main__":
    print("=== 动态权重管理器测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    factor_values = pd.Series(np.random.randn(500).cumsum(), index=dates)
    returns = pd.Series(np.random.randn(500) * 0.001, index=dates)
    
    # 创建管理器
    manager = DynamicWeightManager()
    result = manager.calculate_dynamic_weight(factor_values, returns)
    
    print(f"\n动态权重计算结果:")
    print(f"  最终权重: {result['weight']:.4f}")
    print(f"  基础权重: {result['base_weight']:.4f}")
    print(f"  调整幅度: {result['adjustment_pct']:.2f}%")
    print(f"  失效得分: {result['invalidation_score']:.4f}")
    print(f"  调整原因: {result['reason']}")
