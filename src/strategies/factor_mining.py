import pandas as pd
import numpy as np


class FactorMiner:
    def __init__(self):
        pass

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

    def get_scorecard(self, visual_win_rate, factor_row, fund_data):
        """
        [核心] 多因子评分卡系统 (V + F + Q)
        用于产生初步决策，随后交给 Agent 进行‘一票否决’审阅。
        """
        score = 0
        details = {}

        # --- 因子 1: 视觉形态得分 (V) - 总分 3 分 ---
        v_points = 0
        if visual_win_rate >= 65:
            v_points = 3
        elif visual_win_rate >= 55:
            v_points = 2
        elif visual_win_rate >= 45:
            v_points = 1
        score += v_points
        details['视觉分(V)'] = v_points

        # --- 因子 2: 财务基本面得分 (F) - 总分 4 分 ---
        f_points = 0
        # ROE 盈利能力
        roe = fund_data.get('roe', 0)
        if roe > 15:
            f_points += 2
        elif roe > 8:
            f_points += 1

        # PE 估值安全性
        pe = fund_data.get('pe_ttm', 0)
        if 0 < pe < 20:
            f_points += 2
        elif 20 <= pe < 40:
            f_points += 1

        score += f_points
        details['财务分(F)'] = f_points

        # --- 因子 3: 量化技术面得分 (Q) - 总分 3 分 ---
        q_points = 0
        # 1. 均线趋势 (权重1)
        if factor_row.get('MA_Signal', 0) > 0: q_points += 1
        # 2. RSI健康度 (权重1, 处于30-70非极端区域)
        rsi = factor_row.get('RSI', 50)
        if 30 <= rsi <= 70: q_points += 1
        # 3. MACD动能 (权重1, 柱子为正)
        if factor_row.get('MACD_Hist', 0) > 0: q_points += 1

        score += q_points
        details['量化分(Q)'] = q_points

        # --- 最终决策结论 ---
        # 满分 10 分
        if score >= 7:
            action = "BUY"
        elif score >= 5:
            action = "WAIT"
        else:
            action = "SELL"

        return score, action, details


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