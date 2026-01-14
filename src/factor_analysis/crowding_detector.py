"""
拥挤交易检测模块
Crowding Trade Detection Module

检测因子是否被过度使用（拥挤交易）

方法：
1. 因子暴露度分布（集中度）
2. 因子收益分布（异常集中）
3. 因子相关性（与其他因子高度相关）

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats


class CrowdingDetector:
    """
    拥挤交易检测器
    
    功能：
    1. 计算因子暴露度集中度
    2. 检测异常集中度
    3. 计算拥挤度指标
    """
    
    def __init__(
        self,
        concentration_threshold: float = 0.5,
        herfindahl_threshold: float = 0.3
    ):
        """
        初始化拥挤检测器
        
        Args:
            concentration_threshold: 集中度阈值（超过此值视为拥挤）
            herfindahl_threshold: Herfindahl指数阈值
        """
        self.concentration_threshold = concentration_threshold
        self.herfindahl_threshold = herfindahl_threshold
    
    def detect_crowding(
        self,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.Series = None
    ) -> Dict:
        """
        检测拥挤交易
        
        Args:
            factor_exposures: 因子暴露度DataFrame (stocks × dates)
            factor_returns: 因子收益序列（可选）
            
        Returns:
            拥挤检测结果
        """
        # 1. 计算集中度指标
        concentration = self._calculate_concentration(factor_exposures)
        
        # 2. 计算Herfindahl指数
        herfindahl = self._calculate_herfindahl(factor_exposures)
        
        # 3. 计算因子暴露度分布
        exposure_dist = self._analyze_exposure_distribution(factor_exposures)
        
        # 4. 判断是否拥挤
        is_crowded = (
            concentration > self.concentration_threshold or
            herfindahl > self.herfindahl_threshold
        )
        
        result = {
            'is_crowded': is_crowded,
            'concentration': concentration,
            'herfindahl_index': herfindahl,
            'exposure_distribution': exposure_dist,
            'crowding_score': (concentration + herfindahl) / 2
        }
        
        # 5. 如果提供收益序列，分析收益分布
        if factor_returns is not None:
            return_dist = self._analyze_return_distribution(factor_returns)
            result['return_distribution'] = return_dist
        
        return result
    
    def _calculate_concentration(
        self,
        factor_exposures: pd.DataFrame
    ) -> float:
        """
        计算因子暴露度集中度
        
        方法：Top 10%股票的暴露度占比
        """
        # 取最新一期的暴露度
        latest_exposures = factor_exposures.iloc[:, -1] if factor_exposures.shape[1] > 0 else factor_exposures.iloc[:, 0]
        
        # 去除NaN
        exposures = latest_exposures.dropna()
        
        if len(exposures) == 0:
            return 0.0
        
        # 排序
        sorted_exposures = exposures.sort_values(ascending=False)
        
        # Top 10%的暴露度占比
        top_10_pct = int(len(sorted_exposures) * 0.1)
        if top_10_pct == 0:
            top_10_pct = 1
        
        top_exposures = sorted_exposures.head(top_10_pct)
        total_exposure = sorted_exposures.abs().sum()
        
        if total_exposure == 0:
            return 0.0
        
        concentration = top_exposures.abs().sum() / total_exposure
        
        return float(concentration)
    
    def _calculate_herfindahl(
        self,
        factor_exposures: pd.DataFrame
    ) -> float:
        """
        计算Herfindahl指数（衡量集中度）
        
        HHI = sum(wi^2)，其中wi是第i个股票的权重
        """
        # 取最新一期的暴露度
        latest_exposures = factor_exposures.iloc[:, -1] if factor_exposures.shape[1] > 0 else factor_exposures.iloc[:, 0]
        
        # 去除NaN
        exposures = latest_exposures.dropna()
        
        if len(exposures) == 0:
            return 0.0
        
        # 归一化权重
        total = exposures.abs().sum()
        if total == 0:
            return 0.0
        
        weights = exposures.abs() / total
        
        # Herfindahl指数
        hhi = (weights ** 2).sum()
        
        return float(hhi)
    
    def _analyze_exposure_distribution(
        self,
        factor_exposures: pd.DataFrame
    ) -> Dict:
        """
        分析因子暴露度分布
        """
        latest_exposures = factor_exposures.iloc[:, -1] if factor_exposures.shape[1] > 0 else factor_exposures.iloc[:, 0]
        exposures = latest_exposures.dropna()
        
        if len(exposures) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'positive_ratio': 0.0
            }
        
        return {
            'mean': float(exposures.mean()),
            'std': float(exposures.std()),
            'skewness': float(stats.skew(exposures)),
            'kurtosis': float(stats.kurtosis(exposures)),
            'positive_ratio': float((exposures > 0).sum() / len(exposures))
        }
    
    def _analyze_return_distribution(
        self,
        factor_returns: pd.Series
    ) -> Dict:
        """
        分析因子收益分布（异常集中可能表示拥挤）
        """
        returns = factor_returns.dropna()
        
        if len(returns) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
        
        return {
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'skewness': float(stats.skew(returns)),
            'kurtosis': float(stats.kurtosis(returns))
        }
    
    def calculate_crowding_timeseries(
        self,
        factor_exposures: pd.DataFrame,
        window: int = 60
    ) -> pd.Series:
        """
        计算拥挤度时间序列
        
        Args:
            factor_exposures: 因子暴露度DataFrame
            window: 滚动窗口大小
            
        Returns:
            拥挤度时间序列
        """
        crowding_scores = []
        dates = []
        
        for i in range(window, factor_exposures.shape[1]):
            window_exposures = factor_exposures.iloc[:, i-window:i]
            
            # 计算该窗口的集中度
            concentration = self._calculate_concentration(window_exposures)
            herfindahl = self._calculate_herfindahl(window_exposures)
            
            # 拥挤度得分
            crowding_score = (concentration + herfindahl) / 2
            crowding_scores.append(crowding_score)
            dates.append(factor_exposures.columns[i])
        
        return pd.Series(crowding_scores, index=dates)


if __name__ == "__main__":
    print("=== 拥挤交易检测器测试 ===")
    
    # 模拟因子暴露度数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    stocks = [f'Stock_{i}' for i in range(100)]
    
    # 模拟高度集中的暴露度（拥挤）
    factor_exposures = pd.DataFrame(
        np.random.randn(100, 100),
        index=stocks,
        columns=dates
    )
    
    # 让前10只股票暴露度很高（模拟拥挤）
    factor_exposures.iloc[:10, :] = factor_exposures.iloc[:10, :] * 5
    
    # 创建检测器
    detector = CrowdingDetector()
    result = detector.detect_crowding(factor_exposures)
    
    print(f"\n拥挤检测结果:")
    print(f"  是否拥挤: {result['is_crowded']}")
    print(f"  集中度: {result['concentration']:.4f}")
    print(f"  Herfindahl指数: {result['herfindahl_index']:.4f}")
    print(f"  拥挤度得分: {result['crowding_score']:.4f}")
