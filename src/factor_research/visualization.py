"""
因子研究可视化
Factor Research Visualization

为因子研究模块提供可视化功能
Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class FactorResearchVisualizer:
    """
    因子研究可视化器
    
    功能：
    1. 行为偏差可视化
    2. 信息扩散可视化
    3. 因子相关性可视化
    4. 因子稳定性可视化
    5. 因子组合优化可视化
    """
    
    def __init__(self):
        """初始化可视化器"""
        pass
    
    def plot_behavioral_bias(
        self,
        factor_values: pd.Series,
        sentiment_indicator: pd.Series,
        title: str = "行为偏差分析"
    ) -> go.Figure:
        """
        绘制行为偏差分析图
        
        Args:
            factor_values: 因子值序列
            sentiment_indicator: 情绪指标序列
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('因子值与情绪指标', '相关性散点图'),
            vertical_spacing=0.15
        )
        
        # 时间序列
        fig.add_trace(
            go.Scatter(x=factor_values.index, y=factor_values.values,
                      name='因子值', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=sentiment_indicator.index, y=sentiment_indicator.values,
                      name='情绪指标', line=dict(color='red'), yaxis='y2'),
            row=1, col=1
        )
        
        # 散点图
        common_index = factor_values.index.intersection(sentiment_indicator.index)
        if len(common_index) > 0:
            fig.add_trace(
                go.Scatter(
                    x=factor_values.loc[common_index],
                    y=sentiment_indicator.loc[common_index],
                    mode='markers',
                    name='相关性',
                    marker=dict(size=5, opacity=0.6)
                ),
                row=2, col=1
            )
        
        fig.update_layout(title=title, height=600)
        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_yaxes(title_text="因子值", row=1, col=1)
        fig.update_xaxes(title_text="因子值", row=2, col=1)
        fig.update_yaxes(title_text="情绪指标", row=2, col=1)
        
        return fig
    
    def plot_information_diffusion(
        self,
        lag_results: Dict,
        title: str = "信息扩散分析"
    ) -> go.Figure:
        """
        绘制信息扩散分析图
        
        Args:
            lag_results: 滞后分析结果
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure()
        
        if 'error' not in lag_results:
            # 滞后分布直方图
            if 'lag_distribution' in lag_results:
                fig.add_trace(go.Histogram(
                    x=lag_results['lag_distribution'],
                    name='滞后分布',
                    nbinsx=20
                ))
            else:
                # 显示统计信息
                fig.add_annotation(
                    text=f"平均滞后: {lag_results.get('mean_lag', 'N/A')}天<br>"
                         f"反应率: {lag_results.get('reaction_rate', 0):.2%}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
        
        fig.update_layout(title=title, height=400)
        return fig
    
    def plot_factor_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "因子相关性热力图"
    ) -> go.Figure:
        """
        绘制因子相关性热力图
        
        Args:
            correlation_matrix: 相关性矩阵
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="相关系数")
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            xaxis_title="因子",
            yaxis_title="因子"
        )
        
        return fig
    
    def plot_factor_stability(
        self,
        rolling_ic: pd.Series,
        stability_metrics: Dict,
        title: str = "因子稳定性分析"
    ) -> go.Figure:
        """
        绘制因子稳定性分析图
        
        Args:
            rolling_ic: 滚动IC序列
            stability_metrics: 稳定性指标
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('滚动IC', '稳定性指标'),
            vertical_spacing=0.15
        )
        
        # 滚动IC
        fig.add_trace(
            go.Scatter(
                x=rolling_ic.index,
                y=rolling_ic.values,
                mode='lines',
                name='滚动IC',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            row=1, col=1
        )
        
        # 稳定性指标
        metrics_text = (
            f"平均IC: {stability_metrics.get('mean_ic', 0):.4f}<br>"
            f"IC标准差: {stability_metrics.get('std_ic', 0):.4f}<br>"
            f"IC-IR: {stability_metrics.get('ic_ir', 0):.4f}<br>"
            f"稳定性评分: {stability_metrics.get('stability_score', 0):.4f}"
        )
        
        fig.add_annotation(
            text=metrics_text,
            xref="paper", yref="paper",
            x=0.5, y=0.3,
            showarrow=False,
            font=dict(size=14),
            row=2, col=1
        )
        
        fig.update_layout(title=title, height=600)
        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_yaxes(title_text="IC值", row=1, col=1)
        
        return fig
    
    def plot_factor_combination(
        self,
        optimization_result: Dict,
        title: str = "因子组合优化"
    ) -> go.Figure:
        """
        绘制因子组合优化结果
        
        Args:
            optimization_result: 优化结果
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        if 'error' in optimization_result:
            fig = go.Figure()
            fig.add_annotation(
                text=f"优化失败: {optimization_result['error']}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        weights = optimization_result.get('weights', {})
        if not weights:
            fig = go.Figure()
            fig.add_annotation(
                text="无优化结果",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # 权重饼图
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.3
        )])
        
        optimized_ic = optimization_result.get('optimized_ic', 0)
        fig.update_layout(
            title=f"{title}<br>优化后IC: {optimized_ic:.4f}",
            height=500
        )
        
        return fig


if __name__ == "__main__":
    print("=== 因子研究可视化器测试 ===")
    
    visualizer = FactorResearchVisualizer()
    
    # 模拟数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    factor_values = pd.Series(np.random.randn(100), index=dates)
    sentiment = pd.Series(np.random.randn(100), index=dates)
    
    fig = visualizer.plot_behavioral_bias(factor_values, sentiment)
    print("可视化图表已创建")
