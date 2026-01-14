"""
因子组合优化
Factor Combination Optimization

优化多个K线因子的组合

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.optimize import minimize
from itertools import combinations


class FactorCombinationOptimizer:
    """
    因子组合优化器
    
    功能：
    1. 优化多个K线因子的权重组合
    2. 最大化IC，最小化相关性
    3. 使用遗传算法或网格搜索
    """
    
    def __init__(self):
        """初始化因子组合优化器"""
        pass
    
    def optimize_weights(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        method: str = 'maximize_ic'
    ) -> Dict:
        """
        优化因子权重
        
        Args:
            factors: 因子DataFrame，列为因子名，行为日期
            forward_returns: 未来收益率序列
            method: 优化方法 ('maximize_ic', 'minimize_correlation')
            
        Returns:
            优化结果（权重字典）
        """
        # 对齐索引
        common_index = factors.index.intersection(forward_returns.index)
        factors_aligned = factors.loc[common_index]
        returns_aligned = forward_returns.loc[common_index]
        
        if len(common_index) < 60:
            return {'error': '数据不足'}
        
        # 计算各因子的IC
        factor_ics = {}
        for col in factors_aligned.columns:
            factor_values = factors_aligned[col].dropna()
            returns_aligned_sub = returns_aligned.reindex(factor_values.index)
            if len(factor_values) > 10:
                corr, _ = np.corrcoef(factor_values, returns_aligned_sub)[0, 1]
                factor_ics[col] = corr if not np.isnan(corr) else 0.0
        
        if method == 'maximize_ic':
            # 最大化加权IC
            def objective(weights):
                combined_factor = (factors_aligned * weights).sum(axis=1)
                corr, _ = np.corrcoef(combined_factor, returns_aligned)[0, 1]
                return -corr  # 最小化负IC（即最大化IC）
            
            # 约束：权重和为1，权重非负
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(len(factors_aligned.columns))]
            
            # 初始权重（等权重）
            x0 = np.ones(len(factors_aligned.columns)) / len(factors_aligned.columns)
            
            # 优化
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = dict(zip(factors_aligned.columns, result.x))
                # 计算优化后的IC
                combined_factor = (factors_aligned * result.x).sum(axis=1)
                optimized_ic, _ = np.corrcoef(combined_factor, returns_aligned)[0, 1]
                
                return {
                    'weights': weights,
                    'optimized_ic': optimized_ic,
                    'individual_ics': factor_ics
                }
            else:
                return {'error': '优化失败'}
        
        else:
            return {'error': f'不支持的优化方法: {method}'}
    
    def grid_search_combination(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        num_factors: int = 2,
        step: float = 0.1
    ) -> Dict:
        """
        网格搜索最优因子组合
        
        Args:
            factors: 因子DataFrame
            forward_returns: 未来收益率序列
            num_factors: 组合中的因子数
            step: 权重步长
            
        Returns:
            最优组合结果
        """
        best_ic = -np.inf
        best_combination = None
        best_weights = None
        
        # 尝试所有因子组合
        for factor_combo in combinations(factors.columns, num_factors):
            combo_factors = factors[list(factor_combo)]
            
            # 网格搜索权重
            for w1 in np.arange(0, 1 + step, step):
                w2 = 1 - w1
                weights = [w1, w2]
                
                # 计算组合IC
                combined = (combo_factors * weights).sum(axis=1)
                common_index = combined.index.intersection(forward_returns.index)
                if len(common_index) > 10:
                    corr, _ = np.corrcoef(
                        combined.loc[common_index],
                        forward_returns.loc[common_index]
                    )[0, 1]
                    
                    if not np.isnan(corr) and corr > best_ic:
                        best_ic = corr
                        best_combination = factor_combo
                        best_weights = dict(zip(factor_combo, weights))
        
        return {
            'best_combination': best_combination,
            'best_weights': best_weights,
            'best_ic': best_ic
        }


if __name__ == "__main__":
    print("=== 因子组合优化器测试 ===")
    
    optimizer = FactorCombinationOptimizer()
    
    # 模拟数据
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    factors = pd.DataFrame({
        'kline_factor_1': np.random.randn(200),
        'kline_factor_2': np.random.randn(200),
        'kline_factor_3': np.random.randn(200)
    }, index=dates)
    forward_returns = pd.Series(np.random.randn(200) * 0.01, index=dates)
    
    result = optimizer.optimize_weights(factors, forward_returns)
    print(f"优化结果: {result}")
