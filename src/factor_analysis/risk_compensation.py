"""
风险补偿分析模块
Risk Compensation Analysis Module

分析因子收益与风险的关系，评估风险调整后的收益

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats


class RiskCompensationAnalyzer:
    """
    风险补偿分析器
    
    功能：
    1. 计算因子收益 vs 波动率
    2. 计算风险调整收益（Sharpe, Sortino等）
    3. 分析风险补偿是否合理
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化风险补偿分析器
        
        Args:
            risk_free_rate: 无风险利率（年化，默认3%）
        """
        self.risk_free_rate = risk_free_rate
    
    def analyze_risk_compensation(
        self,
        factor_returns: pd.Series,
        factor_values: pd.Series = None,
        quantiles: int = 5
    ) -> Dict:
        """
        分析风险补偿
        
        Args:
            factor_returns: 因子收益序列
            factor_values: 因子值序列（可选，用于分组分析）
            quantiles: 分组数量
            
        Returns:
            风险补偿分析结果
        """
        # 1. 整体风险补偿
        overall_metrics = self._calculate_overall_metrics(factor_returns)
        
        # 2. 分组风险补偿（如果提供factor_values）
        quantile_metrics = None
        if factor_values is not None and len(factor_values) == len(factor_returns):
            quantile_metrics = self._calculate_quantile_metrics(
                factor_values, factor_returns, quantiles
            )
        
        # 3. 风险-收益散点图数据
        scatter_data = self._prepare_scatter_data(factor_returns, factor_values)
        
        return {
            'overall_metrics': overall_metrics,
            'quantile_metrics': quantile_metrics,
            'scatter_data': scatter_data,
            'risk_free_rate': self.risk_free_rate
        }
    
    def _calculate_overall_metrics(
        self,
        returns: pd.Series
    ) -> Dict:
        """
        计算整体风险指标
        """
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {
                'mean_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0
            }
        
        # 年化收益率
        mean_return = returns_clean.mean() * 252
        
        # 年化波动率
        volatility = returns_clean.std() * np.sqrt(252)
        
        # Sharpe比率
        excess_return = mean_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        
        # Sortino比率（只考虑下行波动）
        downside_returns = returns_clean[returns_clean < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0.0
        
        # 最大回撤
        cumulative = (1 + returns_clean).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar比率（年化收益 / 最大回撤）
        calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0.0
        
        return {
            'mean_return': float(mean_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio)
        }
    
    def _calculate_quantile_metrics(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        quantiles: int
    ) -> Dict:
        """
        计算各分组的风险指标
        """
        # 对齐索引
        aligned_values = factor_values.reindex(returns.index)
        aligned_returns = returns.reindex(factor_values.index)
        
        # 分组
        try:
            factor_quantiles = pd.qcut(
                aligned_values, q=quantiles, labels=False, duplicates='drop'
            )
        except:
            return None
        
        quantile_metrics = {}
        
        for q in range(quantiles):
            mask = factor_quantiles == q
            quantile_returns = aligned_returns[mask]
            
            if len(quantile_returns) > 0:
                metrics = self._calculate_overall_metrics(quantile_returns)
                quantile_metrics[f'Q{q+1}'] = metrics
        
        return quantile_metrics
    
    def _prepare_scatter_data(
        self,
        returns: pd.Series,
        factor_values: pd.Series = None
    ) -> pd.DataFrame:
        """
        准备散点图数据（收益 vs 波动率）
        """
        if factor_values is not None:
            # 如果有因子值，按因子值分组
            aligned_values = factor_values.reindex(returns.index)
            aligned_returns = returns.reindex(factor_values.index)
            
            # 分组计算
            try:
                quantiles = pd.qcut(aligned_values, q=10, labels=False, duplicates='drop')
                
                scatter_data = []
                for q in range(10):
                    mask = quantiles == q
                    q_returns = aligned_returns[mask]
                    
                    if len(q_returns) > 0:
                        scatter_data.append({
                            'quantile': q + 1,
                            'mean_return': float(q_returns.mean() * 252),
                            'volatility': float(q_returns.std() * np.sqrt(252)),
                            'sharpe': float((q_returns.mean() * 252 - self.risk_free_rate) / (q_returns.std() * np.sqrt(252)))
                            if q_returns.std() > 0 else 0.0
                        })
                
                return pd.DataFrame(scatter_data)
            except:
                pass
        
        # 如果没有因子值，返回整体数据
        returns_clean = returns.dropna()
        if len(returns_clean) > 0:
            return pd.DataFrame([{
                'mean_return': float(returns_clean.mean() * 252),
                'volatility': float(returns_clean.std() * np.sqrt(252)),
                'sharpe': float((returns_clean.mean() * 252 - self.risk_free_rate) / (returns_clean.std() * np.sqrt(252)))
                if returns_clean.std() > 0 else 0.0
            }])
        
        return pd.DataFrame()
    
    def calculate_risk_adjusted_returns(
        self,
        returns: pd.Series,
        risk_measure: str = 'sharpe'
    ) -> float:
        """
        计算风险调整收益
        
        Args:
            returns: 收益序列
            risk_measure: 风险度量 ('sharpe', 'sortino', 'calmar')
            
        Returns:
            风险调整收益指标
        """
        metrics = self._calculate_overall_metrics(returns)
        
        if risk_measure == 'sharpe':
            return metrics['sharpe_ratio']
        elif risk_measure == 'sortino':
            return metrics['sortino_ratio']
        elif risk_measure == 'calmar':
            return metrics['calmar_ratio']
        else:
            raise ValueError(f"不支持的风险度量: {risk_measure}")


if __name__ == "__main__":
    print("=== 风险补偿分析器测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.01, 500), index=dates)
    factor_values = pd.Series(np.random.randn(500).cumsum(), index=dates)
    
    # 创建分析器
    analyzer = RiskCompensationAnalyzer(risk_free_rate=0.03)
    result = analyzer.analyze_risk_compensation(returns, factor_values)
    
    print(f"\n风险补偿分析结果:")
    print(f"  年化收益率: {result['overall_metrics']['mean_return']:.4f}")
    print(f"  年化波动率: {result['overall_metrics']['volatility']:.4f}")
    print(f"  Sharpe比率: {result['overall_metrics']['sharpe_ratio']:.4f}")
    print(f"  Sortino比率: {result['overall_metrics']['sortino_ratio']:.4f}")
    print(f"  最大回撤: {result['overall_metrics']['max_drawdown']:.4f}")
