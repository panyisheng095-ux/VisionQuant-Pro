"""
K线学习因子计算模块
K-line Learning Factor Calculator

核心功能：
1. 基于Top10匹配结果计算Triple Barrier标签分布
2. 保留传统胜率计算（收益率>0）
3. 混合权重设计（Triple Barrier 70% + 传统 30%）
4. 支持从HDF5快速查询历史标签

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import sys

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.triple_barrier import TripleBarrierLabeler, TripleBarrierPredictor, calculate_win_loss_ratio

# HDF5标签文件路径
HDF5_LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "triple_barrier_labels.h5")


class KLineFactorCalculator:
    """
    K线学习因子计算器
    
    核心功能：
    - 混合胜率计算（Triple Barrier + 传统胜率）
    - 支持从HDF5快速查询历史标签
    - 缓存机制优化
    """
    
    def __init__(
        self,
        triple_barrier_weight: float = 0.7,
        traditional_weight: float = 0.3,
        data_loader=None
    ):
        """
        初始化K线因子计算器
        
        Args:
            triple_barrier_weight: Triple Barrier胜率权重（默认70%）
            traditional_weight: 传统胜率权重（默认30%）
            data_loader: 数据加载器，用于获取历史价格数据
        """
        self.tb_weight = triple_barrier_weight
        self.traditional_weight = traditional_weight
        
        # Triple Barrier组件
        self.labeler = TripleBarrierLabeler(
            upper_barrier=0.05,  # 止盈+5%
            lower_barrier=0.03,  # 止损-3%
            max_holding_period=20  # 最大持有20天
        )
        self.predictor = TripleBarrierPredictor(self.labeler)
        
        # 数据加载器
        self.data_loader = data_loader
        
        # HDF5标签存储路径
        self.labels_hdf5_path = os.path.join(
            PROJECT_ROOT, "data", "indices", "triple_barrier_labels.h5"
        )
        
        # 缓存
        self._label_cache = {}
        
    def calculate_hybrid_win_rate(
        self,
        matches: List[Dict],
        query_symbol: str = None,
        query_date: str = None
    ) -> Dict:
        """
        计算混合胜率（Triple Barrier + 传统胜率）
        
        Args:
            matches: Top-K匹配结果列表，每个元素包含'symbol', 'date', 'score'
            query_symbol: 查询股票代码（可选，用于缓存key）
            query_date: 查询日期（可选，用于缓存key）
            
        Returns:
            胜率计算结果字典
        """
        if not matches:
            return {
                'hybrid_win_rate': 50.0,
                'tb_win_rate': 50.0,
                'traditional_win_rate': 50.0,
                'tb_weight': self.tb_weight,
                'traditional_weight': self.traditional_weight,
                'valid_matches': 0,
                'message': '无匹配结果'
            }
        
        # 1. 计算Triple Barrier胜率
        tb_result = self._calculate_triple_barrier_win_rate(matches)
        tb_win_rate = tb_result.get('win_rate', 50.0)
        
        # 2. 计算传统胜率（收益率>0）
        traditional_win_rate = self._calculate_traditional_win_rate(matches)
        
        # 3. 加权融合
        hybrid_win_rate = (
            tb_win_rate * self.tb_weight + 
            traditional_win_rate * self.traditional_weight
        )
        
        # 4. 确保在合理范围内
        hybrid_win_rate = max(0, min(100, hybrid_win_rate))
        
        return {
            'hybrid_win_rate': round(hybrid_win_rate, 2),
            'tb_win_rate': round(tb_win_rate, 2),
            'traditional_win_rate': round(traditional_win_rate, 2),
            'tb_weight': self.tb_weight,
            'traditional_weight': self.traditional_weight,
            'valid_matches': tb_result.get('valid_matches', 0),
            'tb_details': tb_result,
            'message': '计算成功'
        }

    def calculate_return_distribution(self, matches: List[Dict], horizon_days: int = 20) -> Dict:
        """
        更严格的收益分布估计（均值/分位数/CVaR）
        """
        if not self.data_loader or not matches:
            return {"valid": False}
        returns = []
        for m in matches:
            symbol = str(m.get("symbol", "")).zfill(6)
            date_str = str(m.get("date", ""))
            try:
                match_date = pd.to_datetime(date_str) if "-" in date_str else pd.to_datetime(date_str, format="%Y%m%d")
                df = self.data_loader.get_stock_data(symbol)
                if df is None or df.empty:
                    continue
                df.index = pd.to_datetime(df.index)
                if match_date in df.index:
                    loc = df.index.get_loc(match_date)
                    if loc + horizon_days < len(df):
                        entry = df.iloc[loc]["Close"]
                        future = df.iloc[loc + horizon_days]["Close"]
                        returns.append((future - entry) / entry * 100)
            except Exception:
                continue
        if not returns:
            return {"valid": False}
        s = pd.Series(returns)
        q05 = float(s.quantile(0.05))
        cvar = float(s[s <= q05].mean()) if (s <= q05).any() else q05
        return {
            "valid": True,
            "mean": float(s.mean()),
            "median": float(s.median()),
            "q05": q05,
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "cvar": cvar,
            "count": int(len(s))
        }
    
    def _calculate_triple_barrier_win_rate(self, matches: List[Dict]) -> Dict:
        """
        基于Top-K匹配结果计算Triple Barrier胜率
        
        策略：
        1. 优先从HDF5查询历史标签（如果已计算）
        2. 如果HDF5中没有，实时计算（需要data_loader）
        3. 统计标签分布，计算胜率
        """
        if not self.data_loader:
            # 如果没有data_loader，返回默认值
            return {
                'win_rate': 50.0,
                'valid_matches': 0,
                'message': '缺少数据加载器，无法计算Triple Barrier胜率'
            }
        
        # 尝试从HDF5查询
        labels_from_hdf5 = self._query_labels_from_hdf5(matches)
        
        # 统计标签分布
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        valid_count = 0
        
        for match in matches:
            symbol = str(match.get('symbol', '')).zfill(6)
            date_str = str(match.get('date', ''))
            
            # 从HDF5查询
            label = labels_from_hdf5.get((symbol, date_str))
            
            if label is None:
                # HDF5中没有，尝试实时计算
                label = self._calculate_single_label(symbol, date_str)
            
            if label is not None:
                valid_count += 1
                if label == 1:
                    bullish_count += 1
                elif label == -1:
                    bearish_count += 1
                else:
                    neutral_count += 1
        
        if valid_count == 0:
            return {
                'win_rate': 50.0,
                'valid_matches': 0,
                'message': '无有效标签数据'
            }
        
        # 计算胜率（看涨比例）
        win_rate = (bullish_count / valid_count) * 100
        
        return {
            'win_rate': win_rate,
            'valid_matches': valid_count,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'bullish_pct': round(bullish_count / valid_count * 100, 1),
            'message': '计算成功'
        }
    
    def _calculate_traditional_win_rate(self, matches: List[Dict], horizon_days: int = 5) -> float:
        """
        计算传统胜率（未来收益率>0 的比例，按相似度加权）
        
        说明：
        - 与“简单计数平均”不同，这里使用 TopK 匹配的 score 作为权重（相似度越高权重越大）
        - horizon_days 默认 5，与 Web UI 中“传统胜率：未来5日收益率>0 的比例”一致
        """
        if not self.data_loader:
            return 50.0
        
        weighted_positive = 0.0
        total_weight = 0.0
        valid_count = 0
        
        # score 可能为负/尺度不一：统一做非负化，避免某些实现返回距离导致权重异常
        scores = []
        for m in matches:
            try:
                scores.append(float(m.get("score", 1.0)))
            except Exception:
                scores.append(1.0)
        min_s = min(scores) if scores else 0.0
        eps = 1e-6

        for match in matches:
            symbol = str(match.get('symbol', '')).zfill(6)
            date_str = str(match.get('date', ''))
            try:
                raw_score = float(match.get("score", 1.0))
            except Exception:
                raw_score = 1.0
            weight = max(raw_score - min_s + eps, eps)
            
            try:
                # 解析日期
                if '-' in date_str:
                    match_date = pd.to_datetime(date_str)
                else:
                    match_date = pd.to_datetime(date_str, format='%Y%m%d')
                
                # 获取股票数据
                df = self.data_loader.get_stock_data(symbol)
                if df is None or df.empty:
                    continue
                
                df.index = pd.to_datetime(df.index)
                
                if match_date not in df.index:
                    continue
                
                loc = df.index.get_loc(match_date)
                
                # 计算未来 horizon_days 收益率（默认 5 日）
                if loc + horizon_days < len(df):
                    entry_price = df.iloc[loc]['Close']
                    future_price = df.iloc[loc + horizon_days]['Close']
                    return_pct = (future_price - entry_price) / entry_price
                    
                    if return_pct > 0:
                        weighted_positive += weight
                    total_weight += weight
                    valid_count += 1
                    
            except Exception:
                continue
        
        if valid_count == 0 or total_weight <= 0:
            return 50.0
        
        return (weighted_positive / total_weight) * 100
    
    def _query_labels_from_hdf5(self, matches: List[Dict]) -> Dict:
        """
        从HDF5查询Triple Barrier标签
        
        Returns:
            {(symbol, date): label} 字典
        """
        labels = {}
        
        if not os.path.exists(self.labels_hdf5_path):
            return labels
        
        try:
            import tables as tb
            
            with tb.open_file(self.labels_hdf5_path, mode='r') as h5file:
                if '/labels' not in h5file:
                    return labels
                
                table = h5file.root.labels
                
                # 构建查询条件
                symbols = [str(m.get('symbol', '')).zfill(6) for m in matches]
                dates = [str(m.get('date', '')).replace('-', '') for m in matches]
                
                # 查询（使用索引加速）
                for symbol, date in zip(symbols, dates):
                    cache_key = (symbol, date)
                    if cache_key in self._label_cache:
                        labels[cache_key] = self._label_cache[cache_key]
                        continue
                    
                    # HDF5查询
                    condition = f'(symbol == b"{symbol}") & (date == b"{date}")'
                    result = table.read_where(condition)
                    
                    if len(result) > 0:
                        label = result[0]['label']
                        labels[cache_key] = label
                        self._label_cache[cache_key] = label
        except ImportError:
            # 如果没有tables库，跳过HDF5查询
            pass
        except Exception as e:
            # 查询失败，继续使用实时计算
            pass
        
        return labels
    
    def _calculate_single_label(self, symbol: str, date_str: str) -> Optional[int]:
        """
        实时计算单个匹配的Triple Barrier标签
        
        Args:
            symbol: 股票代码
            date_str: 日期字符串
            
        Returns:
            标签值（1/0/-1）或None
        """
        if not self.data_loader:
            return None
        
        try:
            # 解析日期
            if '-' in date_str:
                match_date = pd.to_datetime(date_str)
            else:
                match_date = pd.to_datetime(date_str, format='%Y%m%d')
            
            # 获取股票数据
            df = self.data_loader.get_stock_data(symbol)
            if df is None or df.empty:
                return None
            
            df.index = pd.to_datetime(df.index)
            
            if match_date not in df.index:
                return None
            
            loc = df.index.get_loc(match_date)
            
            # 确保有足够的历史和未来数据
            if loc < 20 or loc + self.labeler.max_hold >= len(df):
                return None
            
            # 提取价格序列（从匹配日期开始，包含未来max_hold天）
            prices = df.iloc[loc:loc+self.labeler.max_hold+1]['Close']
            
            # 计算标签
            labels = self.labeler.generate_labels(prices)
            
            if not labels.empty and not pd.isna(labels.iloc[0]):
                return int(labels.iloc[0])
            
        except Exception:
            pass
        
        return None
    
    def get_win_loss_ratio(self, matches: List[Dict]) -> Tuple[float, float]:
        """
        计算胜率和盈亏比（用于凯利公式）
        
        Returns:
            (win_rate, win_loss_ratio)
        """
        return calculate_win_loss_ratio(matches)


def calculate_hybrid_win_rate(
    matches: List[Dict],
    data_loader=None,
    triple_barrier_weight: float = 0.7
) -> Dict:
    """
    便捷函数：计算混合胜率
    
    Args:
        matches: Top-K匹配结果
        data_loader: 数据加载器
        triple_barrier_weight: Triple Barrier权重
        
    Returns:
        胜率计算结果
    """
    calculator = KLineFactorCalculator(
        triple_barrier_weight=triple_barrier_weight,
        traditional_weight=1 - triple_barrier_weight,
        data_loader=data_loader
    )
    return calculator.calculate_hybrid_win_rate(matches)


if __name__ == "__main__":
    print("=== K线因子计算器测试 ===")
    
    # 模拟匹配结果
    matches = [
        {'symbol': '600519', 'date': '20231015', 'score': 0.95},
        {'symbol': '000858', 'date': '20230820', 'score': 0.92},
        {'symbol': '601318', 'date': '20231105', 'score': 0.89},
    ]
    
    # 测试（无data_loader）
    calculator = KLineFactorCalculator()
    result = calculator.calculate_hybrid_win_rate(matches)
    
    print(f"\n混合胜率计算结果:")
    print(f"  混合胜率: {result['hybrid_win_rate']}%")
    print(f"  Triple Barrier胜率: {result['tb_win_rate']}%")
    print(f"  传统胜率: {result['traditional_win_rate']}%")
    print(f"  有效匹配数: {result['valid_matches']}")
    print(f"  消息: {result['message']}")
