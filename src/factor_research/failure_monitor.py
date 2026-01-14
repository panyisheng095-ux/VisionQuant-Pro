"""
因子失效原因量化监测
Factor Failure Monitor

实时监测因子失效的原因
Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FactorFailureMonitor:
    """
    因子失效原因量化监测器
    
    功能：
    1. 实时监测因子IC衰减
    2. 监测因子拥挤度变化
    3. 监测市场结构变化
    4. 生成失效预警
    """
    
    def __init__(
        self,
        ic_threshold: float = 0.02,
        sharpe_threshold: float = 0.5,
        crowding_threshold: float = 0.8
    ):
        """
        初始化因子失效监测器
        
        Args:
            ic_threshold: IC阈值（低于此值视为失效）
            sharpe_threshold: Sharpe阈值（低于此值视为失效）
            crowding_threshold: 拥挤度阈值（超过此值视为失效）
        """
        self.ic_threshold = ic_threshold
        self.sharpe_threshold = sharpe_threshold
        self.crowding_threshold = crowding_threshold
    
    def monitor_ic_decay(
        self,
        rolling_ic: pd.Series,
        window: int = 60
    ) -> Dict:
        """
        监测IC衰减
        
        Args:
            rolling_ic: 滚动IC序列
            window: 监测窗口
            
        Returns:
            IC衰减监测结果
        """
        if len(rolling_ic) < window:
            return {'error': '数据不足'}
        
        recent_ic = rolling_ic.tail(window)
        historical_ic = rolling_ic.iloc[:-window] if len(rolling_ic) > window else rolling_ic
        
        recent_mean = recent_ic.mean()
        historical_mean = historical_ic.mean() if len(historical_ic) > 0 else recent_mean
        
        ic_decay = recent_mean - historical_mean
        decay_rate = ic_decay / abs(historical_mean) if historical_mean != 0 else 0
        
        is_decaying = recent_mean < self.ic_threshold or decay_rate < -0.3
        
        return {
            'recent_ic_mean': recent_mean,
            'historical_ic_mean': historical_mean,
            'ic_decay': ic_decay,
            'decay_rate': decay_rate,
            'is_decaying': is_decaying,
            'warning_level': self._get_warning_level(decay_rate, recent_mean)
        }
    
    def monitor_crowding(
        self,
        factor_values: pd.Series,
        window: int = 60
    ) -> Dict:
        """
        监测因子拥挤度
        
        Args:
            factor_values: 因子值序列
            window: 监测窗口
            
        Returns:
            拥挤度监测结果
        """
        if len(factor_values) < window:
            return {'error': '数据不足'}
        
        recent_values = factor_values.tail(window)
        
        # 计算因子值分布
        percentile_90 = recent_values.quantile(0.9)
        percentile_10 = recent_values.quantile(0.1)
        
        # 拥挤度：高因子值占比
        high_factor_ratio = (recent_values > percentile_90).sum() / len(recent_values)
        
        is_crowded = high_factor_ratio > self.crowding_threshold
        
        return {
            'high_factor_ratio': high_factor_ratio,
            'percentile_90': percentile_90,
            'percentile_10': percentile_10,
            'is_crowded': is_crowded,
            'warning_level': 'high' if is_crowded else 'normal'
        }
    
    def monitor_market_structure_change(
        self,
        returns: pd.Series,
        volume: pd.Series,
        window: int = 60
    ) -> Dict:
        """
        监测市场结构变化
        
        Args:
            returns: 收益率序列
            volume: 成交量序列
            window: 监测窗口
            
        Returns:
            市场结构变化监测结果
        """
        if len(returns) < window * 2:
            return {'error': '数据不足'}
        
        recent_returns = returns.tail(window)
        historical_returns = returns.iloc[-window*2:-window]
        
        recent_volatility = recent_returns.std()
        historical_volatility = historical_returns.std()
        
        volatility_change = (recent_volatility - historical_volatility) / historical_volatility if historical_volatility > 0 else 0
        
        # 成交量变化
        if len(volume) >= window * 2:
            recent_volume = volume.tail(window).mean()
            historical_volume = volume.iloc[-window*2:-window].mean()
            volume_change = (recent_volume - historical_volume) / historical_volume if historical_volume > 0 else 0
        else:
            volume_change = 0
        
        structure_changed = abs(volatility_change) > 0.3 or abs(volume_change) > 0.5
        
        return {
            'volatility_change': volatility_change,
            'volume_change': volume_change,
            'structure_changed': structure_changed,
            'warning_level': 'high' if structure_changed else 'normal'
        }
    
    def comprehensive_monitoring(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        rolling_ic: pd.Series,
        volume: Optional[pd.Series] = None
    ) -> Dict:
        """
        综合监测
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益率序列
            rolling_ic: 滚动IC序列
            volume: 成交量序列（可选）
            
        Returns:
            综合监测结果
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'ic_decay': self.monitor_ic_decay(rolling_ic),
            'crowding': self.monitor_crowding(factor_values)
        }
        
        if volume is not None:
            results['market_structure'] = self.monitor_market_structure_change(
                forward_returns, volume
            )
        
        # 综合失效判断
        is_failing = (
            results['ic_decay'].get('is_decaying', False) or
            results['crowding'].get('is_crowded', False) or
            results.get('market_structure', {}).get('structure_changed', False)
        )
        
        results['is_failing'] = is_failing
        results['overall_warning'] = self._get_overall_warning(results)
        
        return results
    
    def _get_warning_level(self, decay_rate: float, recent_ic: float) -> str:
        """获取警告级别"""
        if recent_ic < self.ic_threshold or decay_rate < -0.5:
            return 'critical'
        elif decay_rate < -0.3:
            return 'high'
        elif decay_rate < -0.1:
            return 'medium'
        else:
            return 'low'
    
    def _get_overall_warning(self, results: Dict) -> str:
        """获取综合警告"""
        warnings = []
        
        if results['ic_decay'].get('is_decaying', False):
            warnings.append('IC衰减')
        
        if results['crowding'].get('is_crowded', False):
            warnings.append('因子拥挤')
        
        if results.get('market_structure', {}).get('structure_changed', False):
            warnings.append('市场结构变化')
        
        if warnings:
            return f"因子失效风险: {', '.join(warnings)}"
        else:
            return "因子状态正常"


if __name__ == "__main__":
    print("=== 因子失效监测器测试 ===")
    
    monitor = FactorFailureMonitor()
    
    # 模拟数据
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    factor_values = pd.Series(np.random.randn(500), index=dates)
    forward_returns = pd.Series(np.random.randn(500) * 0.01, index=dates)
    rolling_ic = pd.Series(np.random.randn(500) * 0.05, index=dates)
    
    result = monitor.comprehensive_monitoring(factor_values, forward_returns, rolling_ic)
    print(f"监测结果: {result}")
