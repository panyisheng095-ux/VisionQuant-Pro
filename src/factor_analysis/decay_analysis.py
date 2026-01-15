"""
因子衰减分析模块
Factor Decay Analysis Module

分析因子IC随时间衰减的情况，判断因子失效时点

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
from datetime import datetime, timedelta


class DecayAnalyzer:
    """
    因子衰减分析器
    
    功能：
    1. 计算IC衰减曲线
    2. 检测衰减时点
    3. 预测因子失效时间
    """
    
    def __init__(self, decay_threshold: float = 0.02, window: int = 60):
        """
        初始化衰减分析器
        
        Args:
            decay_threshold: IC衰减阈值（低于此值视为衰减）
            window: 检测窗口大小（交易日）
        """
        self.decay_threshold = decay_threshold
        self.window = window
    
    def analyze_decay(
        self,
        ic_series: pd.Series,
        method: str = 'rolling_mean'
    ) -> Dict:
        """
        分析IC衰减
        
        Args:
            ic_series: Rolling IC序列
            method: 衰减检测方法 ('rolling_mean', 'linear_trend', 'exponential')
            
        Returns:
            衰减分析结果
        """
        if len(ic_series) < self.window:
            return {
                'is_decaying': False,
                'decay_start_date': None,
                'decay_rate': 0.0,
                'predicted_invalidation_date': None,
                'message': '数据不足，无法分析'
            }
        
        # 1. 检测衰减时点
        decay_start = self._detect_decay_start(ic_series, method)
        
        # 2. 计算衰减率
        decay_rate = self._calculate_decay_rate(ic_series, decay_start)
        
        # 3. 预测失效时间
        invalidation_date = self._predict_invalidation(ic_series, decay_rate)

        # 4. Change Point & CUSUM
        change_points = self._detect_change_points_cusum(ic_series)
        
        # 4. 判断是否正在衰减
        recent_ic = ic_series.tail(self.window).mean()
        is_decaying = recent_ic < self.decay_threshold or decay_start is not None
        
        return {
            'is_decaying': is_decaying,
            'decay_start_date': decay_start,
            'decay_rate': decay_rate,
            'predicted_invalidation_date': invalidation_date,
            'recent_ic_mean': float(recent_ic),
            'recent_ic_std': float(ic_series.tail(self.window).std()),
            'method': method,
            'change_points': change_points
        }

    def _detect_change_points_cusum(self, ic_series: pd.Series, threshold: float = 0.5) -> list:
        """
        使用CUSUM检测衰减拐点（简化实现）
        """
        s = ic_series.dropna()
        if len(s) < self.window:
            return []
        mean = s.mean()
        std = s.std() if s.std() > 1e-6 else 1.0
        cpos, cneg = 0.0, 0.0
        points = []
        for idx, val in s.items():
            k = (val - mean) / std
            cpos = max(0.0, cpos + k)
            cneg = min(0.0, cneg + k)
            if cpos > threshold or abs(cneg) > threshold:
                points.append(idx)
                cpos, cneg = 0.0, 0.0
        return points[-3:]  # 仅保留最近的几个拐点
    
    def _detect_decay_start(
        self,
        ic_series: pd.Series,
        method: str
    ) -> Optional[pd.Timestamp]:
        """
        检测衰减开始时间
        
        Returns:
            衰减开始日期，如果未检测到则返回None
        """
        if method == 'rolling_mean':
            # 方法1：滚动均值低于阈值
            rolling_mean = ic_series.rolling(self.window).mean()
            below_threshold = rolling_mean < self.decay_threshold
            
            if below_threshold.any():
                first_below = below_threshold.idxmax() if below_threshold.any() else None
                return first_below
        
        elif method == 'linear_trend':
            # 方法2：线性趋势检测
            for i in range(self.window, len(ic_series)):
                window_ic = ic_series.iloc[i-self.window:i]
                
                if len(window_ic.dropna()) < 10:
                    continue
                
                # 线性回归
                x = np.arange(len(window_ic))
                y = window_ic.values
                valid_mask = ~np.isnan(y)
                
                if valid_mask.sum() < 10:
                    continue
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x[valid_mask], y[valid_mask]
                )
                
                # 如果斜率显著为负且当前IC低于阈值
                if slope < -0.0001 and p_value < 0.05 and ic_series.iloc[i] < self.decay_threshold:
                    return ic_series.index[i]
        
        elif method == 'exponential':
            # 方法3：指数衰减检测
            for i in range(self.window, len(ic_series)):
                window_ic = ic_series.iloc[i-self.window:i].dropna()
                
                if len(window_ic) < 10:
                    continue
                
                # 指数衰减模型：IC(t) = IC0 * exp(-lambda * t)
                # 简化为：log(IC) = log(IC0) - lambda * t
                log_ic = np.log(np.abs(window_ic) + 1e-6)
                x = np.arange(len(log_ic))
                
                slope, _, _, p_value, _ = stats.linregress(x, log_ic)
                
                # 如果衰减率显著为负
                if slope < -0.01 and p_value < 0.05:
                    return ic_series.index[i]
        
        return None
    
    def _calculate_decay_rate(
        self,
        ic_series: pd.Series,
        decay_start: Optional[pd.Timestamp]
    ) -> float:
        """
        计算衰减率（IC下降速度）
        
        Returns:
            衰减率（每天IC下降的绝对值）
        """
        if decay_start is None:
            # 如果没有明确的衰减起点，使用最近窗口
            recent_ic = ic_series.tail(self.window)
        else:
            # 从衰减起点开始
            recent_ic = ic_series.loc[decay_start:]
        
        if len(recent_ic) < 10:
            return 0.0
        
        # 线性回归计算斜率
        x = np.arange(len(recent_ic))
        y = recent_ic.values
        valid_mask = ~np.isnan(y)
        
        if valid_mask.sum() < 10:
            return 0.0
        
        slope, _, _, _, _ = stats.linregress(x[valid_mask], y[valid_mask])
        
        return abs(slope) if slope < 0 else 0.0
    
    def _predict_invalidation(
        self,
        ic_series: pd.Series,
        decay_rate: float
    ) -> Optional[pd.Timestamp]:
        """
        预测因子失效时间（IC降至0或负值）
        
        Args:
            ic_series: IC序列
            decay_rate: 衰减率
            
        Returns:
            预测失效日期
        """
        if decay_rate <= 0:
            return None
        
        # 当前IC
        current_ic = ic_series.iloc[-1]
        
        if current_ic <= 0:
            # 已经失效
            return ic_series.index[-1]
        
        # 预测失效所需天数
        days_to_invalidation = int(current_ic / decay_rate) if decay_rate > 0 else None
        
        if days_to_invalidation is None or days_to_invalidation <= 0:
            return None
        
        # 计算失效日期
        last_date = ic_series.index[-1]
        
        # 转换为交易日（简化：假设每年252个交易日）
        invalidation_date = last_date + timedelta(days=int(days_to_invalidation * 365 / 252))
        
        return invalidation_date
    
    def get_decay_curve(
        self,
        ic_series: pd.Series,
        smooth_window: int = 20
    ) -> pd.Series:
        """
        获取衰减曲线（平滑后的IC序列）
        
        Args:
            ic_series: IC序列
            smooth_window: 平滑窗口大小
            
        Returns:
            平滑后的IC序列
        """
        return ic_series.rolling(smooth_window).mean()


if __name__ == "__main__":
    print("=== 因子衰减分析器测试 ===")
    
    # 模拟IC序列（逐渐衰减）
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # 前200天：IC较高（0.05-0.08）
    high_ic = np.random.normal(0.06, 0.01, 200)
    # 中间150天：IC逐渐下降（0.03-0.05）
    declining_ic = np.linspace(0.05, 0.02, 150)
    # 后150天：IC很低（0.00-0.02）
    low_ic = np.random.normal(0.01, 0.01, 150)
    
    ic_series = pd.Series(
        np.concatenate([high_ic, declining_ic, low_ic]),
        index=dates
    )
    
    # 创建分析器
    analyzer = DecayAnalyzer(decay_threshold=0.02, window=60)
    result = analyzer.analyze_decay(ic_series, method='linear_trend')
    
    print(f"\n衰减分析结果:")
    print(f"  是否正在衰减: {result['is_decaying']}")
    print(f"  衰减开始日期: {result['decay_start_date']}")
    print(f"  衰减率: {result['decay_rate']:.6f}")
    print(f"  预测失效日期: {result['predicted_invalidation_date']}")
    print(f"  近期IC均值: {result['recent_ic_mean']:.4f}")
