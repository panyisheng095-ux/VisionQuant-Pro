"""
多尺度K线图生成模块
Multi-Scale K-line Chart Generator

生成日线/周线/月线K线图，用于多尺度特征学习

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
from typing import Dict, Optional
from datetime import datetime
import os
import tempfile


class MultiScaleChartGenerator:
    """
    多尺度K线图生成器
    
    功能：
    1. 生成日线K线图
    2. 生成周线K线图
    3. 生成月线K线图
    4. 支持自定义时间窗口
    """
    
    def __init__(self, figsize: tuple = (3, 3), dpi: int = 50):
        """
        初始化多尺度生成器
        
        Args:
            figsize: 图像尺寸
            dpi: 图像分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # K线图样式
        self.market_colors = mpf.make_marketcolors(
            up='red',      # 上涨为红色（中国习惯）
            down='green',  # 下跌为绿色
            inherit=True
        )
        self.style = mpf.make_mpf_style(marketcolors=self.market_colors, gridstyle='')
    
    def generate_daily_chart(
        self,
        df: pd.DataFrame,
        days: int = 20,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成日线K线图
        
        Args:
            df: OHLCV数据（日线）
            days: 显示天数（默认20天）
            output_path: 输出路径（可选，如果不提供则使用临时文件）
            
        Returns:
            图像文件路径
        """
        if df is None or df.empty:
            raise ValueError("数据为空")
        
        # 确保索引是日期
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 取最近N天
        recent_df = df.tail(days).copy()
        
        # 确保列名正确
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        rename_map = {
            'open': 'Open', '开盘': 'Open',
            'high': 'High', '最高': 'High',
            'low': 'Low', '最低': 'Low',
            'close': 'Close', '收盘': 'Close',
            'volume': 'Volume', '成交量': 'Volume'
        }
        recent_df = recent_df.rename(columns=rename_map)
        
        # 生成输出路径
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        
        # 绘制K线图
        mpf.plot(
            recent_df,
            type='candle',
            style=self.style,
            savefig=dict(fname=output_path, dpi=self.dpi),
            figsize=self.figsize,
            axisoff=True,
            volume=False  # 不显示成交量
        )
        
        return output_path
    
    def generate_weekly_chart(
        self,
        df: pd.DataFrame,
        weeks: int = 20,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成周线K线图
        
        Args:
            df: OHLCV数据（日线）
            weeks: 显示周数（默认20周）
            output_path: 输出路径
            
        Returns:
            图像文件路径
        """
        if df is None or df.empty:
            raise ValueError("数据为空")
        
        # 确保索引是日期
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 重采样为周线
        weekly_df = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # 取最近N周
        recent_df = weekly_df.tail(weeks).copy()
        
        # 确保列名正确
        rename_map = {
            'open': 'Open', '开盘': 'Open',
            'high': 'High', '最高': 'High',
            'low': 'Low', '最低': 'Low',
            'close': 'Close', '收盘': 'Close',
            'volume': 'Volume', '成交量': 'Volume'
        }
        recent_df = recent_df.rename(columns=rename_map)
        
        # 生成输出路径
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        
        # 绘制K线图
        mpf.plot(
            recent_df,
            type='candle',
            style=self.style,
            savefig=dict(fname=output_path, dpi=self.dpi),
            figsize=self.figsize,
            axisoff=True,
            volume=False
        )
        
        return output_path
    
    def generate_monthly_chart(
        self,
        df: pd.DataFrame,
        months: int = 20,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成月线K线图
        
        Args:
            df: OHLCV数据（日线）
            months: 显示月数（默认20个月）
            output_path: 输出路径
            
        Returns:
            图像文件路径
        """
        if df is None or df.empty:
            raise ValueError("数据为空")
        
        # 确保索引是日期
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 重采样为月线
        monthly_df = df.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # 取最近N个月
        recent_df = monthly_df.tail(months).copy()
        
        # 确保列名正确
        rename_map = {
            'open': 'Open', '开盘': 'Open',
            'high': 'High', '最高': 'High',
            'low': 'Low', '最低': 'Low',
            'close': 'Close', '收盘': 'Close',
            'volume': 'Volume', '成交量': 'Volume'
        }
        recent_df = recent_df.rename(columns=rename_map)
        
        # 生成输出路径
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"monthly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
        
        # 绘制K线图
        mpf.plot(
            recent_df,
            type='candle',
            style=self.style,
            savefig=dict(fname=output_path, dpi=self.dpi),
            figsize=self.figsize,
            axisoff=True,
            volume=False
        )
        
        return output_path
    
    def generate_all_scales(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str] = None,
        prefix: str = ""
    ) -> Dict[str, str]:
        """
        生成所有尺度的K线图
        
        Args:
            df: OHLCV数据
            output_dir: 输出目录
            prefix: 文件名前缀
            
        Returns:
            字典：{'daily': path, 'weekly': path, 'monthly': path}
        """
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        os.makedirs(output_dir, exist_ok=True)
        
        charts = {}
        
        # 生成日线图
        daily_path = os.path.join(output_dir, f"{prefix}daily.png")
        charts['daily'] = self.generate_daily_chart(df, output_path=daily_path)
        
        # 生成周线图
        weekly_path = os.path.join(output_dir, f"{prefix}weekly.png")
        charts['weekly'] = self.generate_weekly_chart(df, output_path=weekly_path)
        
        # 生成月线图
        monthly_path = os.path.join(output_dir, f"{prefix}monthly.png")
        charts['monthly'] = self.generate_monthly_chart(df, output_path=monthly_path)
        
        return charts


if __name__ == "__main__":
    print("=== 多尺度K线图生成器测试 ===")
    
    # 模拟数据
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(50, 60, 200),
        'High': np.random.uniform(55, 65, 200),
        'Low': np.random.uniform(45, 55, 200),
        'Close': np.random.uniform(50, 60, 200),
        'Volume': np.random.uniform(1000000, 10000000, 200)
    }, index=dates)
    
    generator = MultiScaleChartGenerator()
    
    # 生成所有尺度
    charts = generator.generate_all_scales(df, prefix="test_")
    
    print(f"\n生成的K线图:")
    for scale, path in charts.items():
        print(f"  {scale}: {path}")
