import pandas as pd
import numpy as np
from typing import Dict, Optional
import os
import sys

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.strategies.regime_manager import RegimeManager


class FactorMiner:
    """
    因子挖掘器（支持动态权重）
    
    功能：
    1. 计算技术指标
    2. 多因子评分（V+F+Q）
    3. 支持动态权重调整
    """
    
    def __init__(self, use_dynamic_weights: bool = True, regime_manager: RegimeManager = None):
        """
        初始化因子挖掘器
        
        Args:
            use_dynamic_weights: 是否使用动态权重
            regime_manager: Regime管理器（如果为None，则自动创建）
        """
        self.use_dynamic_weights = use_dynamic_weights
        
        if use_dynamic_weights:
            self.regime_manager = regime_manager or RegimeManager()
        else:
            self.regime_manager = None

    def _add_technical_indicators(self, df):
        """
        计算 Q 因子（量化技术面）：MA60, RSI, MACD
        这是 Web 端回测和实时分析的核心数据源
        """
        if df.empty:
            return df

        data = df.copy()
        close = data['Close']

        # 1. 计算 MA60 (趋势生命线)
        data['MA60'] = close.rolling(window=60).mean()
        # 趋势信号：股价在均线上方为 1.0 (多头)，下方为 -1.0 (空头)
        data['MA_Signal'] = np.where(close > data['MA60'], 1.0, -1.0)

        # 2. 计算 RSI (相对强弱指标) - 14天窗口
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # 处理 loss 为 0 的情况，防止报错
        rs = gain / loss.replace(0, np.nan)
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI'] = data['RSI'].fillna(50)  # 初始值填中性

        # 3. 计算 MACD (动量信号)
        # 快线12日, 慢线26日, 信号线9日
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        dif = exp12 - exp26
        dea = dif.ewm(span=9, adjust=False).mean()
        # MACD 柱状图
        data['MACD_Hist'] = (dif - dea) * 2

        # 填充开头几天的 NaN 值，保证数据整洁
        return data.fillna(method='bfill').fillna(0)

    def get_scorecard(
        self,
        visual_win_rate: float,
        factor_row: pd.Series,
        fund_data: Dict,
        factor_ics: Optional[Dict[str, float]] = None,
        returns: Optional[pd.Series] = None
    ):
        """
        [核心] 多因子评分卡系统 (V + F + Q)
        支持动态权重调整
        
        Args:
            visual_win_rate: 视觉形态胜率
            factor_row: 技术指标行
            fund_data: 财务数据
            factor_ics: 因子IC值（可选，用于动态权重）
            returns: 收益率序列（可选，用于regime识别）
            
        Returns:
            (score, action, details)
        """
        # 1. 计算各因子原始得分
        v_points = self._calculate_visual_score(visual_win_rate)
        f_points = self._calculate_fundamental_score(fund_data)
        q_points = self._calculate_technical_score(factor_row)
        
        # 2. 获取动态权重（如果启用）
        if self.use_dynamic_weights and self.regime_manager:
            try:
                # 获取regime权重
                dynamic_weights = self.regime_manager.calculate_dynamic_weights(
                    factor_values=None,  # 这里简化，实际应该传入因子值序列
                    returns=returns
                )
                weights = dynamic_weights.get('weights', {})
                
                # 应用动态权重
                v_weight = weights.get('kline_factor', 0.5)
                f_weight = weights.get('fundamental', 0.3)
                q_weight = weights.get('technical', 0.2)
                
                # 归一化到10分制
                total_weight = v_weight + f_weight + q_weight
                if total_weight > 0:
                    v_max = 3 * (v_weight / total_weight) * 10
                    f_max = 4 * (f_weight / total_weight) * 10
                    q_max = 3 * (q_weight / total_weight) * 10
                else:
                    v_max, f_max, q_max = 3, 4, 3
                
                # 按权重调整得分
                v_score = v_points * (v_max / 3) if v_max > 0 else v_points
                f_score = f_points * (f_max / 4) if f_max > 0 else f_points
                q_score = q_points * (q_max / 3) if q_max > 0 else q_points
                
                score = v_score + f_score + q_score
                
                details = {
                    '视觉分(V)': round(v_points, 2),
                    '财务分(F)': round(f_points, 2),
                    '量化分(Q)': round(q_points, 2),
                    '视觉权重': round(v_weight, 3),
                    '财务权重': round(f_weight, 3),
                    '量化权重': round(q_weight, 3),
                    'regime': dynamic_weights.get('regime', 'unknown'),
                    '权重说明': dynamic_weights.get('explain', {}),
                    '权重更新时间': dynamic_weights.get('timestamp')
                }
            except Exception as e:
                print(f"⚠️ 动态权重计算失败: {e}，使用固定权重")
                # 回退到固定权重
                score = v_points + f_points + q_points
                details = {
                    '视觉分(V)': v_points,
                    '财务分(F)': f_points,
                    '量化分(Q)': q_points
                }
        else:
            # 固定权重（传统方式）
            score = v_points + f_points + q_points
            details = {
                '视觉分(V)': v_points,
                '财务分(F)': f_points,
                '量化分(Q)': q_points
            }
        
        # 3. 最终决策
        if score >= 7:
            action = "BUY"
        elif score >= 5:
            action = "WAIT"
        else:
            action = "SELL"
        
        return round(score, 2), action, details
    
    def _calculate_visual_score(self, visual_win_rate: float) -> float:
        """计算视觉形态得分（0-3分）"""
        if visual_win_rate >= 65:
            return 3.0
        elif visual_win_rate >= 55:
            return 2.0
        elif visual_win_rate >= 45:
            return 1.0
        else:
            return 0.0
    
    def _calculate_fundamental_score(self, fund_data: Dict) -> float:
        """计算财务基本面得分（0-4分）"""
        f_points = 0.0
        
        # ROE 盈利能力
        roe = fund_data.get('roe', 0)
        if roe > 15:
            f_points += 2.0
        elif roe > 8:
            f_points += 1.0
        
        # PE 估值安全性
        pe = fund_data.get('pe_ttm', 0)
        if 0 < pe < 20:
            f_points += 2.0
        elif 20 <= pe < 40:
            f_points += 1.0
        
        return f_points
    
    def _calculate_technical_score(self, factor_row: pd.Series) -> float:
        """计算量化技术面得分（0-3分）"""
        q_points = 0.0
        
        # 1. 均线趋势
        if factor_row.get('MA_Signal', 0) > 0:
            q_points += 1.0
        
        # 2. RSI健康度
        rsi = factor_row.get('RSI', 50)
        if 30 <= rsi <= 70:
            q_points += 1.0
        
        # 3. MACD动能
        if factor_row.get('MACD_Hist', 0) > 0:
            q_points += 1.0
        
        return q_points


# === 单元测试 ===
if __name__ == "__main__":
    miner = FactorMiner()

    # 模拟数据
    mock_df = pd.DataFrame({
        'Close': [10, 11, 10.5, 12, 13, 12.5, 14] * 10  # 构造一点波动
    })

    # 测试指标计算
    df_with_factors = miner._add_technical_indicators(mock_df)
    print(">>> 技术指标计算结果预览:")
    print(df_with_factors[['Close', 'MA60', 'RSI', 'MACD_Hist']].tail())

    # 测试评分卡
    mock_fund = {'roe': 18.5, 'pe_ttm': 12.0}
    s, a, d = miner.get_scorecard(68.0, df_with_factors.iloc[-1], mock_fund)

    print("\n>>> 评分卡结果:")
    print(f"总分: {s}, 建议: {a}, 明细: {d}")