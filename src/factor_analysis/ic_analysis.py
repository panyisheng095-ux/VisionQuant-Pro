"""
Rolling IC/Sharpe分析模块
Rolling Information Coefficient and Sharpe Ratio Analysis

IC (Information Coefficient): 因子值与未来收益率的相关系数
Sharpe Ratio: 因子多空组合的夏普比率

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
import warnings


def calculate_rolling_ic(
    factor_values: pd.Series,
    returns: pd.Series,
    window: int = 252,
    method: str = 'pearson'
) -> pd.Series:
    """
    计算滚动窗口的IC序列
    
    Args:
        factor_values: K线学习因子值（胜率或得分）
        returns: 未来收益率序列
        window: 滚动窗口大小（交易日，默认252天=1年）
        method: 相关系数计算方法 ('pearson' 或 'spearman')
        
    Returns:
        IC序列（与factor_values对齐的索引）
    """
    if len(factor_values) != len(returns):
        raise ValueError("factor_values和returns长度必须一致")
    
    ic_series = []
    dates = []
    
    for i in range(window, len(factor_values)):
        factor_window = factor_values.iloc[i-window:i]
        return_window = returns.iloc[i-window:i]
        
        # 去除NaN
        valid_mask = ~(factor_window.isna() | return_window.isna())
        factor_clean = factor_window[valid_mask]
        return_clean = return_window[valid_mask]
        
        if len(factor_clean) < 10:  # 至少需要10个有效样本
            ic_series.append(np.nan)
        else:
            if method == 'pearson':
                ic, _ = stats.pearsonr(factor_clean, return_clean)
            elif method == 'spearman':
                ic, _ = stats.spearmanr(factor_clean, return_clean)
            else:
                raise ValueError(f"不支持的相关系数方法: {method}")
            
            ic_series.append(ic if not np.isnan(ic) else 0.0)
        
        dates.append(factor_values.index[i])
    
    return pd.Series(ic_series, index=dates)


def calculate_rolling_sharpe(
    factor_values: pd.Series,
    returns: pd.Series,
    window: int = 252,
    quantiles: int = 5
) -> pd.Series:
    """
    计算滚动窗口的Sharpe比率
    
    方法：将因子值分为quantiles组，做多Top组，做空Bottom组，计算组合Sharpe
    
    Args:
        factor_values: 因子值序列
        returns: 未来收益率序列
        window: 滚动窗口大小
        quantiles: 分组数量（默认5组）
        
    Returns:
        Sharpe比率序列
    """
    if len(factor_values) != len(returns):
        raise ValueError("factor_values和returns长度必须一致")
    
    sharpe_series = []
    dates = []
    
    for i in range(window, len(factor_values)):
        factor_window = factor_values.iloc[i-window:i]
        return_window = returns.iloc[i-window:i]
        
        # 去除NaN
        valid_mask = ~(factor_window.isna() | return_window.isna())
        factor_clean = factor_window[valid_mask]
        return_clean = return_window[valid_mask]
        
        if len(factor_clean) < 20:  # 至少需要20个有效样本
            sharpe_series.append(np.nan)
        else:
            # 按因子值分组
            try:
                factor_quantiles = pd.qcut(factor_clean, q=quantiles, labels=False, duplicates='drop')
                
                # Top组（因子值最高）
                top_mask = factor_quantiles == (quantiles - 1)
                top_returns = return_clean[top_mask]
                
                # Bottom组（因子值最低）
                bottom_mask = factor_quantiles == 0
                bottom_returns = return_clean[bottom_mask]
                
                if len(top_returns) > 0 and len(bottom_returns) > 0:
                    # 多空组合收益 = Top组平均收益 - Bottom组平均收益
                    long_short_return = top_returns.mean() - bottom_returns.mean()
                    
                    # 组合波动率（假设两组收益不相关）
                    combined_std = np.sqrt(
                        top_returns.std()**2 + bottom_returns.std()**2
                    ) if top_returns.std() > 0 and bottom_returns.std() > 0 else np.nan
                    
                    # Sharpe比率（年化）
                    if combined_std > 0:
                        sharpe = np.sqrt(252) * long_short_return / combined_std
                    else:
                        sharpe = 0.0
                else:
                    sharpe = 0.0
            except Exception:
                sharpe = np.nan
            
            sharpe_series.append(sharpe if not np.isnan(sharpe) else 0.0)
        
        dates.append(factor_values.index[i])
    
    return pd.Series(sharpe_series, index=dates)


class ICAnalyzer:
    """
    IC分析器
    
    功能：
    1. 计算Rolling IC
    2. 计算Rolling Sharpe
    3. IC统计指标（均值、标准差、IR等）
    4. IC衰减分析
    """
    
    def __init__(self, window: int = 252):
        """
        初始化IC分析器
        
        Args:
            window: 滚动窗口大小（交易日）
        """
        self.window = window
    
    def analyze(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        method: str = 'pearson'
    ) -> Dict:
        """
        完整IC分析
        
        Args:
            factor_values: 因子值序列
            returns: 未来收益率序列
            method: 相关系数方法
            
        Returns:
            分析结果字典
        """
        # 1. 计算Rolling IC
        ic_series = calculate_rolling_ic(factor_values, returns, self.window, method)
        
        # 2. 计算Rolling Sharpe
        sharpe_series = calculate_rolling_sharpe(factor_values, returns, self.window)
        
        # 3. IC统计指标
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0  # Information Ratio
        ic_positive_ratio = (ic_series > 0).sum() / len(ic_series) if len(ic_series) > 0 else 0
        
        # 4. Sharpe统计指标
        sharpe_mean = sharpe_series.mean()
        sharpe_std = sharpe_series.std()
        sharpe_positive_ratio = (sharpe_series > 0).sum() / len(sharpe_series) if len(sharpe_series) > 0 else 0
        
        # 5. IC显著性检验
        t_stat, p_value = stats.ttest_1samp(ic_series.dropna(), 0)
        is_significant = p_value < 0.05
        
        half_life = self._ic_half_life(ic_series)
        stability_score = self._ic_stability_score(ic_series)

        return {
            'ic_series': ic_series,
            'sharpe_series': sharpe_series,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_positive_ratio': ic_positive_ratio,
            'ic_t_stat': t_stat,
            'ic_p_value': p_value,
            'ic_significant': is_significant,
            'sharpe_mean': sharpe_mean,
            'sharpe_std': sharpe_std,
            'sharpe_positive_ratio': sharpe_positive_ratio,
            'half_life': half_life,
            'stability_score': stability_score,
            'summary': {
                'mean_ic': round(ic_mean, 4),
                'std_ic': round(ic_std, 4),
                'ir': round(ic_ir, 4),
                'positive_ratio': round(ic_positive_ratio, 2),
                'significant': is_significant,
                'mean_sharpe': round(sharpe_mean, 4),
                'half_life': None if half_life is None else round(float(half_life), 2),
                'stability_score': round(float(stability_score), 4)
            }
        }

    def _ic_half_life(self, ic_series: pd.Series) -> Optional[float]:
        """IC Half-Life（基于AR(1)近似）"""
        s = ic_series.dropna()
        if len(s) < 20:
            return None
        x = s.shift(1).dropna()
        y = s.loc[x.index]
        if len(x) < 10:
            return None
        # 估计AR(1)系数
        phi = np.corrcoef(x.values, y.values)[0, 1]
        if phi is None or phi <= 0 or phi >= 0.999:
            return None
        half_life = -np.log(2) / np.log(phi)
        return float(half_life)

    def _ic_stability_score(self, ic_series: pd.Series) -> float:
        """IC Stability Score（越高越稳定）"""
        s = ic_series.dropna()
        if len(s) == 0:
            return 0.0
        mean_ic = s.mean()
        std_ic = s.std()
        if std_ic <= 0:
            return 1.0
        score = 1 - (std_ic / (abs(mean_ic) + 1e-6))
        return float(np.clip(score, 0.0, 1.0))
    
    def calculate_ic_decay(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        max_horizon: int = 20
    ) -> pd.DataFrame:
        """
        计算IC衰减曲线（不同持有期的IC）
        
        Args:
            factor_values: 因子值序列
            returns: 未来收益率序列
            max_horizon: 最大持有期（交易日）
            
        Returns:
            DataFrame，列为不同持有期的IC
        """
        ic_decay = {}
        
        for horizon in range(1, max_horizon + 1):
            # 计算未来horizon天的累计收益率
            forward_returns = returns.rolling(horizon).sum().shift(-horizon)
            
            # 计算IC
            ic = calculate_rolling_ic(factor_values, forward_returns, self.window)
            ic_decay[f'IC_{horizon}d'] = ic
        
        return pd.DataFrame(ic_decay)


if __name__ == "__main__":
    print("=== IC分析器测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    factor_values = pd.Series(np.random.randn(500).cumsum(), index=dates)
    returns = pd.Series(np.random.randn(500) * 0.01, index=dates)
    
    # 创建分析器
    analyzer = ICAnalyzer(window=60)
    result = analyzer.analyze(factor_values, returns)
    
    print(f"\nIC分析结果:")
    print(f"  平均IC: {result['summary']['mean_ic']}")
    print(f"  IC标准差: {result['summary']['std_ic']}")
    print(f"  Information Ratio: {result['summary']['ir']}")
    print(f"  正IC比例: {result['summary']['positive_ratio']}")
    print(f"  是否显著: {result['summary']['significant']}")
    print(f"  平均Sharpe: {result['summary']['mean_sharpe']}")
