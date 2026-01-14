"""
信息扩散分析
Information Diffusion Analysis

分析因子信号的传播速度

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class InformationDiffusionAnalyzer:
    """
    信息扩散分析器
    
    功能：
    1. 计算因子信号出现到价格反应的时间差
    2. 分析不同股票的信号传播速度
    3. 识别信息扩散的模式
    """
    
    def __init__(self, lag_window: int = 10):
        """
        初始化信息扩散分析器
        
        Args:
            lag_window: 滞后窗口（最大滞后天数）
        """
        self.lag_window = lag_window
    
    def analyze_signal_lag(
        self,
        factor_signals: pd.Series,
        price_movements: pd.Series,
        threshold: float = 0.02
    ) -> Dict:
        """
        分析信号滞后
        
        Args:
            factor_signals: 因子信号序列（1=买入信号，-1=卖出信号，0=无信号）
            price_movements: 价格变动序列（收益率）
            threshold: 价格变动阈值（超过此值视为有反应）
            
        Returns:
            滞后分析结果
        """
        # 识别信号日
        signal_days = factor_signals[factor_signals != 0].index
        
        if len(signal_days) == 0:
            return {'error': '无信号'}
        
        # 计算每个信号后的价格反应时间
        reaction_lags = []
        
        for signal_date in signal_days:
            signal_idx = price_movements.index.get_loc(signal_date)
            
            # 查找未来N天内的价格反应
            for lag in range(1, min(self.lag_window + 1, len(price_movements) - signal_idx)):
                future_idx = signal_idx + lag
                if future_idx >= len(price_movements):
                    break
                
                price_change = price_movements.iloc[future_idx]
                
                # 检查是否有反应（与信号方向一致）
                signal_direction = factor_signals.loc[signal_date]
                if signal_direction * price_change > threshold:
                    reaction_lags.append(lag)
                    break
        
        if len(reaction_lags) == 0:
            return {'error': '未发现价格反应'}
        
        return {
            'mean_lag': np.mean(reaction_lags),
            'median_lag': np.median(reaction_lags),
            'std_lag': np.std(reaction_lags),
            'min_lag': np.min(reaction_lags),
            'max_lag': np.max(reaction_lags),
            'reaction_rate': len(reaction_lags) / len(signal_days)
        }
    
    def analyze_cross_stock_diffusion(
        self,
        factor_signals_by_stock: Dict[str, pd.Series],
        price_movements_by_stock: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        分析跨股票的信息扩散
        
        Args:
            factor_signals_by_stock: {symbol: signals} 字典
            price_movements_by_stock: {symbol: returns} 字典
            
        Returns:
            各股票的信息扩散速度DataFrame
        """
        results = []
        
        for symbol in factor_signals_by_stock.keys():
            if symbol not in price_movements_by_stock:
                continue
            
            signals = factor_signals_by_stock[symbol]
            returns = price_movements_by_stock[symbol]
            
            lag_result = self.analyze_signal_lag(signals, returns)
            
            if 'error' not in lag_result:
                results.append({
                    'symbol': symbol,
                    'mean_lag': lag_result['mean_lag'],
                    'reaction_rate': lag_result['reaction_rate']
                })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== 信息扩散分析器测试 ===")
    
    analyzer = InformationDiffusionAnalyzer()
    
    # 模拟数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    signals = pd.Series(np.random.choice([-1, 0, 1], 100), index=dates)
    returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
    
    result = analyzer.analyze_signal_lag(signals, returns)
    print(f"滞后分析: {result}")
