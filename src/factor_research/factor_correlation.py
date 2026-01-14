"""
因子相关性分析
Factor Correlation Analysis

分析K线因子与其他因子的相关性

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt


class FactorCorrelationAnalyzer:
    """
    因子相关性分析器
    
    功能：
    1. 计算因子间的相关性矩阵
    2. 识别高度相关的因子对
    3. 分析因子冗余性
    """
    
    def __init__(self):
        """初始化因子相关性分析器"""
        pass
    
    def calculate_correlation_matrix(
        self,
        factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算因子相关性矩阵
        
        Args:
            factors: 因子DataFrame，列为因子名，行为日期
            
        Returns:
            相关性矩阵
        """
        return factors.corr()
    
    def identify_high_correlation_pairs(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        识别高度相关的因子对
        
        Args:
            correlation_matrix: 相关性矩阵
            threshold: 相关性阈值
            
        Returns:
            高度相关的因子对列表
        """
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    high_corr_pairs.append({
                        'factor1': correlation_matrix.index[i],
                        'factor2': correlation_matrix.columns[j],
                        'correlation': corr
                    })
        
        return sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def analyze_redundancy(
        self,
        factors: pd.DataFrame,
        target_factor: str = 'kline_factor'
    ) -> Dict:
        """
        分析因子冗余性
        
        Args:
            factors: 因子DataFrame
            target_factor: 目标因子（通常是K线因子）
            
        Returns:
            冗余性分析结果
        """
        if target_factor not in factors.columns:
            return {'error': f'目标因子 {target_factor} 不存在'}
        
        corr_matrix = self.calculate_correlation_matrix(factors)
        target_corrs = corr_matrix[target_factor].drop(target_factor)
        
        return {
            'mean_correlation': target_corrs.abs().mean(),
            'max_correlation': target_corrs.abs().max(),
            'highly_correlated_factors': target_corrs[target_corrs.abs() > 0.7].to_dict(),
            'redundancy_score': target_corrs.abs().mean()  # 冗余度评分
        }
    
    def plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        output_path: str
    ):
        """
        绘制相关性热力图
        
        Args:
            correlation_matrix: 相关性矩阵
            output_path: 输出路径
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True
        )
        plt.title('Factor Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("=== 因子相关性分析器测试 ===")
    
    analyzer = FactorCorrelationAnalyzer()
    
    # 模拟数据
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    factors = pd.DataFrame({
        'kline_factor': np.random.randn(100),
        'fundamental': np.random.randn(100),
        'technical': np.random.randn(100)
    }, index=dates)
    
    corr_matrix = analyzer.calculate_correlation_matrix(factors)
    print(f"\n相关性矩阵:\n{corr_matrix}")
