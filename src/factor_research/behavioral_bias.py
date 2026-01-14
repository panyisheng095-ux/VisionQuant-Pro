"""
行为偏差分析
Behavioral Bias Analysis

分析因子是否捕捉了市场非理性行为

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import pearsonr, spearmanr


class BehavioralBiasAnalyzer:
    """
    行为偏差分析器
    
    功能：
    1. 分析因子与情绪指标的相关性
    2. 分析因子在极端市场环境下的表现
    3. 检查因子是否与已知行为偏差相关
    """
    
    def __init__(self):
        """初始化行为偏差分析器"""
        pass
    
    def analyze_sentiment_correlation(
        self,
        factor_values: pd.Series,
        sentiment_indicator: pd.Series
    ) -> Dict:
        """
        分析因子与情绪指标的相关性
        
        Args:
            factor_values: 因子值序列
            sentiment_indicator: 情绪指标序列（如VIX、恐慌指数）
            
        Returns:
            相关性分析结果
        """
        # 对齐索引
        common_index = factor_values.index.intersection(sentiment_indicator.index)
        factor_aligned = factor_values.loc[common_index]
        sentiment_aligned = sentiment_indicator.loc[common_index]
        
        if len(common_index) < 10:
            return {'error': '数据不足'}
        
        # 计算相关性
        pearson_corr, pearson_p = pearsonr(factor_aligned, sentiment_aligned)
        spearman_corr, spearman_p = spearmanr(factor_aligned, sentiment_aligned)
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_pvalue': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_pvalue': spearman_p,
            'interpretation': self._interpret_correlation(pearson_corr)
        }
    
    def analyze_extreme_market_performance(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        extreme_threshold: float = 0.05
    ) -> Dict:
        """
        分析因子在极端市场环境下的表现
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            extreme_threshold: 极端市场阈值（如单日涨跌幅>5%）
            
        Returns:
            极端市场表现分析
        """
        # 识别极端市场日
        extreme_days = returns[abs(returns) > extreme_threshold]
        
        if len(extreme_days) == 0:
            return {'error': '未发现极端市场日'}
        
        # 对齐因子值
        extreme_factor_values = factor_values.reindex(extreme_days.index).dropna()
        
        if len(extreme_factor_values) == 0:
            return {'error': '极端市场日无对应因子值'}
        
        # 分析极端市场下的因子表现
        return {
            'num_extreme_days': len(extreme_days),
            'extreme_factor_mean': extreme_factor_values.mean(),
            'extreme_factor_std': extreme_factor_values.std(),
            'normal_factor_mean': factor_values.mean(),
            'normal_factor_std': factor_values.std(),
            'difference': extreme_factor_values.mean() - factor_values.mean()
        }
    
    def detect_herding_effect(
        self,
        factor_values: pd.Series,
        volume: pd.Series
    ) -> Dict:
        """
        检测羊群效应
        
        如果因子值与成交量高度相关，可能存在羊群效应
        
        Args:
            factor_values: 因子值序列
            volume: 成交量序列
            
        Returns:
            羊群效应检测结果
        """
        # 对齐索引
        common_index = factor_values.index.intersection(volume.index)
        factor_aligned = factor_values.loc[common_index]
        volume_aligned = volume.loc[common_index]
        
        if len(common_index) < 10:
            return {'error': '数据不足'}
        
        # 计算相关性
        corr, pvalue = pearsonr(factor_aligned, volume_aligned)
        
        return {
            'correlation': corr,
            'pvalue': pvalue,
            'has_herding': abs(corr) > 0.3 and pvalue < 0.05,
            'interpretation': '存在羊群效应' if abs(corr) > 0.3 and pvalue < 0.05 else '无明显羊群效应'
        }
    
    def _interpret_correlation(self, corr: float) -> str:
        """解释相关性"""
        if abs(corr) < 0.1:
            return '几乎无相关性'
        elif abs(corr) < 0.3:
            return '弱相关'
        elif abs(corr) < 0.5:
            return '中等相关'
        elif abs(corr) < 0.7:
            return '强相关'
        else:
            return '极强相关'


if __name__ == "__main__":
    print("=== 行为偏差分析器测试 ===")
    
    analyzer = BehavioralBiasAnalyzer()
    
    # 模拟数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    factor_values = pd.Series(np.random.randn(100), index=dates)
    sentiment = pd.Series(np.random.randn(100), index=dates)
    
    result = analyzer.analyze_sentiment_correlation(factor_values, sentiment)
    print(f"相关性分析: {result}")
