"""
行业分层分析模块
Industry Stratification Analysis Module

分析因子在不同行业中的表现差异

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict


class IndustryStratifier:
    """
    行业分层分析器
    
    功能：
    1. 按行业分组计算IC
    2. 行业IC对比
    3. 行业稳定性分析
    """
    
    def __init__(self, min_stocks_per_industry: int = 5):
        """
        初始化行业分层器
        
        Args:
            min_stocks_per_industry: 每个行业最少股票数
        """
        self.min_stocks = min_stocks_per_industry
    
    def analyze_by_industry(
        self,
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
        industry_mapping: Dict[str, str],
        window: int = 252
    ) -> Dict:
        """
        按行业分析因子表现
        
        Args:
            factor_values: 因子值DataFrame (stocks × dates)
            returns: 收益率DataFrame (stocks × dates)
            industry_mapping: {stock_code: industry_name} 映射
            window: 滚动窗口大小
            
        Returns:
            行业分析结果
        """
        # 1. 按行业分组
        industry_groups = self._group_by_industry(industry_mapping, factor_values.index.tolist())
        
        # 2. 计算各行业的IC
        industry_ics = {}
        industry_sharpes = {}
        
        for industry, stocks in industry_groups.items():
            if len(stocks) < self.min_stocks:
                continue
            
            # 提取该行业的因子值和收益
            industry_factors = factor_values.loc[stocks]
            industry_returns = returns.loc[stocks]
            
            # 计算行业平均IC
            ic_series = self._calculate_industry_ic(
                industry_factors, industry_returns, window
            )
            
            if len(ic_series) > 0:
                industry_ics[industry] = {
                    'mean_ic': float(ic_series.mean()),
                    'std_ic': float(ic_series.std()),
                    'ir': float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0,
                    'positive_ratio': float((ic_series > 0).sum() / len(ic_series)),
                    'ic_series': ic_series
                }
                
                # 计算行业Sharpe
                industry_sharpes[industry] = self._calculate_industry_sharpe(
                    industry_factors, industry_returns, window
                )
        
        # 3. 行业IC排名
        ic_ranking = sorted(
            industry_ics.items(),
            key=lambda x: x[1]['mean_ic'],
            reverse=True
        )
        
        return {
            'industry_ics': industry_ics,
            'industry_sharpes': industry_sharpes,
            'ic_ranking': ic_ranking,
            'summary': self._generate_summary(industry_ics)
        }
    
    def _group_by_industry(
        self,
        industry_mapping: Dict[str, str],
        stock_list: List[str]
    ) -> Dict[str, List[str]]:
        """
        按行业分组股票
        """
        industry_groups = defaultdict(list)
        
        for stock in stock_list:
            industry = industry_mapping.get(stock, 'Unknown')
            industry_groups[industry].append(stock)
        
        return dict(industry_groups)
    
    def _calculate_industry_ic(
        self,
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
        window: int
    ) -> pd.Series:
        """
        计算行业IC序列
        
        方法：对行业内的股票取平均因子值，然后计算与平均收益的IC
        """
        # 行业平均因子值
        industry_factor = factor_values.mean(axis=0)
        
        # 行业平均收益
        industry_return = returns.mean(axis=0)
        
        # 计算Rolling IC
        from .ic_analysis import calculate_rolling_ic
        ic_series = calculate_rolling_ic(industry_factor, industry_return, window)
        
        return ic_series
    
    def _calculate_industry_sharpe(
        self,
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
        window: int
    ) -> float:
        """
        计算行业Sharpe比率
        """
        from .ic_analysis import calculate_rolling_sharpe
        
        # 行业平均因子值
        industry_factor = factor_values.mean(axis=0)
        
        # 行业平均收益
        industry_return = returns.mean(axis=0)
        
        # 计算Rolling Sharpe
        sharpe_series = calculate_rolling_sharpe(industry_factor, industry_return, window)
        
        return float(sharpe_series.mean())
    
    def _generate_summary(
        self,
        industry_ics: Dict
    ) -> Dict:
        """
        生成行业分析摘要
        """
        if not industry_ics:
            return {
                'total_industries': 0,
                'best_industry': None,
                'worst_industry': None,
                'ic_range': 0.0
            }
        
        mean_ics = {ind: data['mean_ic'] for ind, data in industry_ics.items()}
        
        best_industry = max(mean_ics.items(), key=lambda x: x[1])
        worst_industry = min(mean_ics.items(), key=lambda x: x[1])
        
        return {
            'total_industries': len(industry_ics),
            'best_industry': {
                'name': best_industry[0],
                'mean_ic': best_industry[1]
            },
            'worst_industry': {
                'name': worst_industry[0],
                'mean_ic': worst_industry[1]
            },
            'ic_range': best_industry[1] - worst_industry[1],
            'mean_ic_across_industries': np.mean(list(mean_ics.values()))
        }
    
    def create_industry_ic_table(
        self,
        industry_ics: Dict
    ) -> pd.DataFrame:
        """
        创建行业IC对比表
        """
        rows = []
        
        for industry, data in industry_ics.items():
            rows.append({
                'Industry': industry,
                'Mean IC': data['mean_ic'],
                'Std IC': data['std_ic'],
                'IR': data['ir'],
                'Positive Ratio': data['positive_ratio']
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Mean IC', ascending=False)
        
        return df


if __name__ == "__main__":
    print("=== 行业分层分析器测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    stocks = ['600519', '000858', '601318', '000002', '600036']
    industries = {
        '600519': '白酒',
        '000858': '白酒',
        '601318': '保险',
        '000002': '地产',
        '600036': '银行'
    }
    
    factor_values = pd.DataFrame(
        np.random.randn(len(stocks), len(dates)),
        index=stocks,
        columns=dates
    )
    
    returns = pd.DataFrame(
        np.random.randn(len(stocks), len(dates)) * 0.01,
        index=stocks,
        columns=dates
    )
    
    # 创建分析器
    stratifier = IndustryStratifier(min_stocks_per_industry=2)
    result = stratifier.analyze_by_industry(factor_values, returns, industries)
    
    print(f"\n行业分析结果:")
    print(f"  总行业数: {result['summary']['total_industries']}")
    if result['summary']['best_industry']:
        print(f"  最佳行业: {result['summary']['best_industry']['name']} (IC={result['summary']['best_industry']['mean_ic']:.4f})")
