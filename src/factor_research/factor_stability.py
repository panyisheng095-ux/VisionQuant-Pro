"""
因子稳定性分析
Factor Stability Analysis

分析因子在不同时间窗口的IC稳定性

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import pearsonr


class FactorStabilityAnalyzer:
    """
    因子稳定性分析器
    
    功能：
    1. 计算不同时间窗口的IC
    2. 分析IC的稳定性
    3. 识别IC波动较大的时期
    """
    
    def __init__(self, window_sizes: List[int] = None):
        """
        初始化因子稳定性分析器
        
        Args:
            window_sizes: 时间窗口大小列表（如[60, 120, 252]）
        """
        self.window_sizes = window_sizes or [60, 120, 252]
    
    def analyze_ic_stability(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        window_size: int = 60
    ) -> Dict:
        """
        分析IC稳定性
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益率序列
            window_size: 滚动窗口大小
            
        Returns:
            IC稳定性分析结果
        """
        from src.factor_analysis.ic_analysis import ICAnalyzer
        
        analyzer = ICAnalyzer(factor_values, forward_returns)
        rolling_ic = analyzer.calculate_rolling_ic(window=window_size)
        
        return {
            'mean_ic': rolling_ic.mean(),
            'std_ic': rolling_ic.std(),
            'ic_ir': analyzer.calculate_ic_ir(rolling_ic),
            'positive_ic_ratio': (rolling_ic > 0).sum() / len(rolling_ic) if len(rolling_ic) > 0 else 0,
            'stability_score': 1 - (rolling_ic.std() / abs(rolling_ic.mean())) if rolling_ic.mean() != 0 else 0
        }
    
    def compare_window_stability(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series
    ) -> pd.DataFrame:
        """
        比较不同窗口大小的IC稳定性
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益率序列
            
        Returns:
            不同窗口的稳定性对比DataFrame
        """
        results = []
        
        for window_size in self.window_sizes:
            stability = self.analyze_ic_stability(factor_values, forward_returns, window_size)
            results.append({
                'window_size': window_size,
                'mean_ic': stability['mean_ic'],
                'std_ic': stability['std_ic'],
                'ic_ir': stability['ic_ir'],
                'stability_score': stability['stability_score']
            })
        
        return pd.DataFrame(results)
    
    def identify_unstable_periods(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        window_size: int = 60,
        threshold: float = 2.0
    ) -> List[Dict]:
        """
        识别IC不稳定的时期
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益率序列
            window_size: 滚动窗口大小
            threshold: 异常阈值（标准差倍数）
            
        Returns:
            不稳定时期列表
        """
        from src.factor_analysis.ic_analysis import ICAnalyzer
        
        analyzer = ICAnalyzer(factor_values, forward_returns)
        rolling_ic = analyzer.calculate_rolling_ic(window=window_size)
        
        mean_ic = rolling_ic.mean()
        std_ic = rolling_ic.std()
        
        # 识别异常值
        unstable_periods = []
        for date, ic_value in rolling_ic.items():
            z_score = abs(ic_value - mean_ic) / std_ic if std_ic > 0 else 0
            if z_score > threshold:
                unstable_periods.append({
                    'date': date,
                    'ic_value': ic_value,
                    'z_score': z_score
                })
        
        return unstable_periods


if __name__ == "__main__":
    print("=== 因子稳定性分析器测试 ===")
    
    analyzer = FactorStabilityAnalyzer()
    
    # 模拟数据
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    factor_values = pd.Series(np.random.randn(500), index=dates)
    forward_returns = pd.Series(np.random.randn(500) * 0.01, index=dates)
    
    stability = analyzer.analyze_ic_stability(factor_values, forward_returns)
    print(f"稳定性分析: {stability}")
