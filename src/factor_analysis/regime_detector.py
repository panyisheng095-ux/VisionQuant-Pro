"""
市场Regime识别模块
Market Regime Detection Module

识别市场状态：牛市、熊市、震荡市

基于：
- 滚动收益率
- 滚动波动率
- 趋势强度
- 市场情绪指标（可选）

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from enum import Enum
from dataclasses import dataclass


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL = "bull_market"      # 牛市
    BEAR = "bear_market"      # 熊市
    OSCILLATING = "oscillating"  # 震荡市
    UNKNOWN = "unknown"       # 未知


@dataclass
class RegimeConfig:
    """Regime识别配置"""
    return_window: int = 60        # 收益率滚动窗口（交易日）
    vol_window: int = 60          # 波动率滚动窗口
    bull_return_threshold: float = 0.05    # 牛市收益率阈值（5%）
    bear_return_threshold: float = -0.05   # 熊市收益率阈值（-5%）
    low_vol_threshold: float = 0.20       # 低波动率阈值（20%）
    high_vol_threshold: float = 0.30      # 高波动率阈值（30%）
    trend_strength_window: int = 20        # 趋势强度窗口


class RegimeDetector:
    """
    市场Regime识别器
    
    识别规则：
    1. 牛市：收益率>5% 且 波动率<30%
    2. 熊市：收益率<-5%
    3. 震荡市：其他情况
    """
    
    def __init__(self, config: RegimeConfig = None):
        """
        初始化Regime识别器
        
        Args:
            config: 配置参数
        """
        self.config = config or RegimeConfig()
    
    def detect_regime(
        self,
        returns: pd.Series,
        prices: pd.Series = None
    ) -> pd.Series:
        """
        识别市场Regime
        
        Args:
            returns: 收益率序列
            prices: 价格序列（可选，用于计算趋势强度）
            
        Returns:
            Regime序列（MarketRegime枚举值）
        """
        if len(returns) < self.config.return_window:
            return pd.Series([MarketRegime.UNKNOWN] * len(returns), index=returns.index)
        
        # 1. 计算滚动收益率
        rolling_return = returns.rolling(self.config.return_window).mean()
        
        # 2. 计算滚动波动率（年化）
        rolling_vol = returns.rolling(self.config.vol_window).std() * np.sqrt(252)
        
        # 3. 计算趋势强度（如果提供价格序列）
        trend_strength = None
        if prices is not None and len(prices) >= self.config.trend_strength_window:
            trend_strength = self._calculate_trend_strength(prices)
        
        # 4. 识别Regime
        regimes = []
        
        for i in range(len(returns)):
            if i < self.config.return_window:
                regimes.append(MarketRegime.UNKNOWN)
                continue
            
            ret = rolling_return.iloc[i]
            vol = rolling_vol.iloc[i]
            
            # 牛市判断
            if ret >= self.config.bull_return_threshold and vol <= self.config.high_vol_threshold:
                regimes.append(MarketRegime.BULL)
            # 熊市判断
            elif ret <= self.config.bear_return_threshold:
                regimes.append(MarketRegime.BEAR)
            # 震荡市
            else:
                regimes.append(MarketRegime.OSCILLATING)
        
        return pd.Series(regimes, index=returns.index)
    
    def _calculate_trend_strength(self, prices: pd.Series) -> pd.Series:
        """
        计算趋势强度
        
        方法：计算短期均线与长期均线的距离
        """
        short_ma = prices.rolling(self.config.trend_strength_window).mean()
        long_ma = prices.rolling(self.config.trend_strength_window * 2).mean()
        
        # 趋势强度 = (短期均线 - 长期均线) / 长期均线
        trend_strength = (short_ma - long_ma) / long_ma
        
        return trend_strength
    
    def get_regime_statistics(
        self,
        regimes: pd.Series,
        returns: pd.Series = None
    ) -> Dict:
        """
        获取Regime统计信息
        
        Args:
            regimes: Regime序列
            returns: 收益率序列（可选，用于计算各Regime下的收益统计）
            
        Returns:
            统计信息字典
        """
        # Regime分布
        regime_counts = regimes.value_counts()
        total = len(regimes)
        
        stats = {
            'regime_distribution': {
                regime.value: {
                    'count': int(count),
                    'percentage': round(count / total * 100, 2)
                }
                for regime, count in regime_counts.items()
            },
            'total_periods': total
        }
        
        # 各Regime下的收益统计（如果提供returns）
        if returns is not None and len(returns) == len(regimes):
            returns_aligned = returns.reindex(regimes.index)
            
            for regime in MarketRegime:
                regime_mask = regimes == regime
                if regime_mask.sum() > 0:
                    regime_returns = returns_aligned[regime_mask]
                    stats[f'{regime.value}_returns'] = {
                        'mean': float(regime_returns.mean()),
                        'std': float(regime_returns.std()),
                        'sharpe': float(np.sqrt(252) * regime_returns.mean() / regime_returns.std())
                        if regime_returns.std() > 0 else 0.0,
                        'count': int(regime_mask.sum())
                    }
        
        return stats
    
    def detect_regime_transitions(
        self,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        检测Regime转换点
        
        Args:
            regimes: Regime序列
            
        Returns:
            转换点DataFrame（包含日期、从、到）
        """
        transitions = []
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] != regimes.iloc[i-1]:
                transitions.append({
                    'date': regimes.index[i],
                    'from_regime': regimes.iloc[i-1].value,
                    'to_regime': regimes.iloc[i].value
                })
        
        return pd.DataFrame(transitions)


if __name__ == "__main__":
    print("=== Regime识别器测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 模拟牛市（前200天）
    bull_returns = np.random.normal(0.001, 0.01, 200)
    # 模拟熊市（中间100天）
    bear_returns = np.random.normal(-0.002, 0.015, 100)
    # 模拟震荡市（后200天）
    oscillating_returns = np.random.normal(0.0001, 0.008, 200)
    
    returns = pd.Series(
        np.concatenate([bull_returns, bear_returns, oscillating_returns]),
        index=dates
    )
    
    # 创建识别器
    detector = RegimeDetector()
    regimes = detector.detect_regime(returns)
    
    # 统计
    stats = detector.get_regime_statistics(regimes, returns)
    print(f"\nRegime分布:")
    for regime, info in stats['regime_distribution'].items():
        print(f"  {regime}: {info['count']}天 ({info['percentage']}%)")
    
    # 转换点
    transitions = detector.detect_regime_transitions(regimes)
    print(f"\nRegime转换点（前5个）:")
    print(transitions.head())
