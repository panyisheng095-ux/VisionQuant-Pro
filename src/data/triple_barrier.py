"""
Triple Barrier Method - 业界标准的金融标签生成方法
Based on: López de Prado, M. (2018). Advances in Financial Machine Learning.

核心思想:
- 止盈线 (Upper Barrier): 达到目标收益率，标记为正类
- 止损线 (Lower Barrier): 达到止损线，标记为负类  
- 时间限制 (Vertical Barrier): 超时未触及任何边界，标记为中性

这比简单的"涨/跌"二分类更贴近实际交易决策。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TripleBarrierConfig:
    """Triple Barrier 配置"""
    upper_barrier: float = 0.05    # 止盈线 +5%
    lower_barrier: float = 0.03   # 止损线 -3%
    max_holding_period: int = 20   # 最大持有期（交易日）
    min_samples: int = 5           # 最小样本数


class TripleBarrierLabeler:
    """
    Triple Barrier 标签生成器
    
    用法:
    ```python
    labeler = TripleBarrierLabeler(upper=0.05, lower=0.03, max_hold=20)
    labels = labeler.generate_labels(price_series)
    ```
    """
    
    def __init__(
        self,
        upper_barrier: float = 0.05,
        lower_barrier: float = 0.03,
        max_holding_period: int = 20
    ):
        """
        初始化Triple Barrier标签器
        
        Args:
            upper_barrier: 止盈线，如0.05表示+5%
            lower_barrier: 止损线，如0.03表示-3%（注意：输入正数，内部会取负）
            max_holding_period: 最大持有期（交易日）
        """
        self.upper = upper_barrier
        self.lower = lower_barrier  # 存为正数，使用时取负
        self.max_hold = max_holding_period
        
    def generate_labels(
        self,
        prices: pd.Series,
        return_details: bool = False
    ) -> pd.Series:
        """
        为价格序列生成Triple Barrier标签
        
        Args:
            prices: 价格序列（通常是收盘价），索引应为日期
            return_details: 是否返回详细信息
            
        Returns:
            标签序列:
            - 1: 先触及止盈线（看涨）
            - 0: 超时未触及任何边界（震荡）
            - -1: 先触及止损线（看跌）
        """
        if len(prices) < self.max_hold + 1:
            raise ValueError(f"价格序列长度({len(prices)})不足，需要至少{self.max_hold + 1}个数据点")
        
        labels = pd.Series(index=prices.index, dtype=float)
        labels[:] = np.nan
        
        details = [] if return_details else None
        
        # 遍历每个时间点（除了最后max_hold天）
        for i in range(len(prices) - self.max_hold):
            entry_price = prices.iloc[i]
            entry_date = prices.index[i]
            
            # 计算未来max_hold天的收益率序列
            future_prices = prices.iloc[i+1:i+1+self.max_hold]
            future_returns = (future_prices - entry_price) / entry_price
            
            # 检查是否触及止盈/止损
            label, hit_day, hit_type = self._check_barriers(future_returns)
            labels.iloc[i] = label
            
            if return_details:
                details.append({
                    'date': entry_date,
                    'entry_price': entry_price,
                    'label': label,
                    'hit_day': hit_day,
                    'hit_type': hit_type,
                    'max_return': future_returns.max(),
                    'min_return': future_returns.min(),
                    'final_return': future_returns.iloc[-1] if len(future_returns) > 0 else 0
                })
        
        if return_details:
            return labels, pd.DataFrame(details)
        return labels
    
    def _check_barriers(self, returns: pd.Series) -> Tuple[int, int, str]:
        """
        检查收益率序列是否触及上下边界
        
        Returns:
            (label, hit_day, hit_type)
            - label: 1(止盈), -1(止损), 0(超时)
            - hit_day: 触及边界的天数（从1开始）
            - hit_type: 'upper', 'lower', 'timeout'
        """
        upper_touch = returns >= self.upper
        lower_touch = returns <= -self.lower
        
        # 找第一次触及上边界的位置
        upper_idx = np.where(upper_touch)[0]
        first_upper = upper_idx[0] if len(upper_idx) > 0 else np.inf
        
        # 找第一次触及下边界的位置
        lower_idx = np.where(lower_touch)[0]
        first_lower = lower_idx[0] if len(lower_idx) > 0 else np.inf
        
        # 判断哪个先触及
        if first_upper == np.inf and first_lower == np.inf:
            # 超时
            return 0, self.max_hold, 'timeout'
        elif first_upper <= first_lower:
            # 先触及止盈
            return 1, int(first_upper) + 1, 'upper'
        else:
            # 先触及止损
            return -1, int(first_lower) + 1, 'lower'
    
    def get_statistics(self, labels: pd.Series) -> Dict:
        """
        获取标签分布统计
        
        Args:
            labels: 标签序列
            
        Returns:
            统计信息字典
        """
        valid_labels = labels.dropna()
        total = len(valid_labels)
        
        if total == 0:
            return {'total': 0, 'bullish': 0, 'neutral': 0, 'bearish': 0}
        
        bullish = (valid_labels == 1).sum()
        neutral = (valid_labels == 0).sum()
        bearish = (valid_labels == -1).sum()
        
        return {
            'total': total,
            'bullish': int(bullish),
            'bullish_pct': round(bullish / total * 100, 2),
            'neutral': int(neutral),
            'neutral_pct': round(neutral / total * 100, 2),
            'bearish': int(bearish),
            'bearish_pct': round(bearish / total * 100, 2),
            'bull_bear_ratio': round(bullish / bearish, 2) if bearish > 0 else float('inf')
        }


class TripleBarrierPredictor:
    """
    基于历史相似形态预测Triple Barrier结果
    
    用法:
    ```python
    predictor = TripleBarrierPredictor(labeler)
    prediction = predictor.predict_from_matches(matches, loader)
    ```
    """
    
    def __init__(self, labeler: TripleBarrierLabeler = None):
        """
        初始化预测器
        
        Args:
            labeler: Triple Barrier标签器，如果为None则使用默认配置
        """
        self.labeler = labeler or TripleBarrierLabeler()
        
    def predict_from_matches(
        self,
        matches: List[Dict],
        data_loader,
        min_matches: int = 5
    ) -> Dict:
        """
        基于Top-K相似形态预测Triple Barrier结果
        
        Args:
            matches: 相似形态列表，每个元素包含'symbol', 'date', 'score'
            data_loader: 数据加载器，用于获取历史价格
            min_matches: 最小有效匹配数
            
        Returns:
            预测结果字典
        """
        results = []
        
        for match in matches:
            try:
                symbol = match['symbol']
                date_str = str(match['date']).replace('-', '')
                score = match.get('score', 0.5)
                
                # 加载该股票的历史数据
                df = data_loader.get_stock_data(symbol)
                if df is None or df.empty:
                    continue
                    
                df.index = pd.to_datetime(df.index)
                target_date = pd.to_datetime(date_str)
                
                if target_date not in df.index:
                    continue
                    
                loc = df.index.get_loc(target_date)
                
                # 确保有足够的未来数据
                if loc + self.labeler.max_hold >= len(df):
                    continue
                
                # 获取入场价格和未来价格
                entry_price = df.iloc[loc]['Close']
                future_prices = df.iloc[loc:loc+self.labeler.max_hold+1]['Close']
                
                # 计算收益率序列
                returns = (future_prices.iloc[1:] - entry_price) / entry_price
                
                # 检查触及情况
                label, hit_day, hit_type = self.labeler._check_barriers(returns)
                
                # 记录结果
                results.append({
                    'symbol': symbol,
                    'date': date_str,
                    'score': score,
                    'label': label,
                    'hit_day': hit_day,
                    'hit_type': hit_type,
                    'max_return': returns.max() * 100,  # 转为百分比
                    'min_return': returns.min() * 100,
                    'final_return': returns.iloc[-1] * 100 if len(returns) > 0 else 0
                })
                
            except Exception as e:
                continue
        
        if len(results) < min_matches:
            return {
                'valid': False,
                'message': f'有效匹配数不足 ({len(results)}/{min_matches})',
                'matches_count': len(results)
            }
        
        # 统计预测结果
        df_results = pd.DataFrame(results)
        
        bullish_count = (df_results['label'] == 1).sum()
        neutral_count = (df_results['label'] == 0).sum()
        bearish_count = (df_results['label'] == -1).sum()
        total = len(df_results)
        
        # 加权平均（按相似度加权）
        weights = df_results['score'].values
        weights = weights / weights.sum()  # 归一化
        
        weighted_label = (df_results['label'] * weights).sum()
        avg_max_return = df_results['max_return'].mean()
        avg_min_return = df_results['min_return'].mean()
        avg_final_return = df_results['final_return'].mean()
        
        # 计算预期收益（加权）
        expected_return = (df_results['final_return'] * weights).sum()
        
        # 计算平均触及天数
        avg_hit_day = df_results['hit_day'].mean()
        
        # 确定最终预测
        if bullish_count > bearish_count and bullish_count >= total * 0.4:
            prediction = 'BULLISH'
            confidence = bullish_count / total
        elif bearish_count > bullish_count and bearish_count >= total * 0.4:
            prediction = 'BEARISH'
            confidence = bearish_count / total
        else:
            prediction = 'NEUTRAL'
            confidence = neutral_count / total
        
        return {
            'valid': True,
            'prediction': prediction,
            'confidence': round(confidence * 100, 1),
            'weighted_label': round(weighted_label, 3),
            'bullish_count': int(bullish_count),
            'bullish_pct': round(bullish_count / total * 100, 1),
            'neutral_count': int(neutral_count),
            'neutral_pct': round(neutral_count / total * 100, 1),
            'bearish_count': int(bearish_count),
            'bearish_pct': round(bearish_count / total * 100, 1),
            'avg_max_return': round(avg_max_return, 2),
            'avg_min_return': round(avg_min_return, 2),
            'avg_final_return': round(avg_final_return, 2),
            'expected_return': round(expected_return, 2),
            'avg_hit_day': round(avg_hit_day, 1),
            'matches_count': total,
            'details': results
        }


def calculate_win_loss_ratio(matches_results: List[Dict]) -> Tuple[float, float]:
    """
    计算胜率和盈亏比（用于凯利公式）
    
    Args:
        matches_results: 匹配结果列表
        
    Returns:
        (win_rate, win_loss_ratio)
    """
    if not matches_results:
        return 0.5, 1.0
    
    wins = []
    losses = []
    
    for r in matches_results:
        final_ret = r.get('final_return', 0)
        if final_ret > 0:
            wins.append(final_ret)
        elif final_ret < 0:
            losses.append(abs(final_ret))
    
    total = len(wins) + len(losses)
    if total == 0:
        return 0.5, 1.0
    
    win_rate = len(wins) / total
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 1
    
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
    
    return win_rate, win_loss_ratio


if __name__ == "__main__":
    # 测试代码
    print("=== Triple Barrier Labeler 测试 ===")
    
    # 创建模拟价格数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = pd.Series(
        100 * (1 + np.random.randn(100).cumsum() * 0.02),
        index=dates
    )
    
    # 初始化标签器
    labeler = TripleBarrierLabeler(
        upper_barrier=0.05,  # 止盈+5%
        lower_barrier=0.03,  # 止损-3%
        max_holding_period=20
    )
    
    # 生成标签
    labels, details = labeler.generate_labels(prices, return_details=True)
    
    # 打印统计
    stats = labeler.get_statistics(labels)
    print(f"\n标签统计:")
    print(f"  总样本: {stats['total']}")
    print(f"  看涨(1): {stats['bullish']} ({stats['bullish_pct']}%)")
    print(f"  震荡(0): {stats['neutral']} ({stats['neutral_pct']}%)")
    print(f"  看跌(-1): {stats['bearish']} ({stats['bearish_pct']}%)")
    print(f"  多空比: {stats['bull_bear_ratio']}")
    
    print("\n前10个标签详情:")
    print(details.head(10).to_string())
