"""
重新计算历史胜率
Recalculate Historical Win Rates

为40万张K线图重新计算混合胜率（Triple Barrier + 传统胜率），
并更新缓存文件。

Author: VisionQuant Team
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.strategies.kline_factor import KLineFactorCalculator
from src.data.data_loader import DataLoader
from src.models.vision_engine import VisionEngine


def recalculate_win_rates():
    """
    重新计算所有K线图的历史胜率
    """
    print("🚀 开始重新计算历史胜率...")
    
    # 1. 加载元数据
    attention_meta = os.path.join(PROJECT_ROOT, "data", "indices", "meta_data_attention.csv")
    meta_csv = attention_meta if os.path.exists(attention_meta) else os.path.join(PROJECT_ROOT, "data", "indices", "meta_data.csv")
    if not os.path.exists(meta_csv):
        print(f"❌ 元数据文件不存在: {meta_csv}")
        return
    
    meta_df = pd.read_csv(meta_csv, dtype=str)
    print(f"📊 共 {len(meta_df)} 张K线图需要处理")
    
    # 2. 初始化组件
    loader = DataLoader()
    vision_engine = VisionEngine()
    vision_engine.reload_index()
    
    kline_calc = KLineFactorCalculator(
        triple_barrier_weight=0.7,
        traditional_weight=0.3,
        data_loader=loader
    )
    
    # 3. 按股票分组处理
    grouped = meta_df.groupby('symbol')
    results = []
    
    for symbol, group in tqdm(grouped, desc="处理股票"):
        try:
            # 获取该股票的数据
            df = loader.get_stock_data(symbol)
            if df.empty:
                continue
            
            # 为每个日期计算胜率
            for _, row in group.iterrows():
                date_str = str(row['date']).replace('-', '')
                
                try:
                    # 生成K线图
                    from datetime import datetime
                    import mplfinance as mpf
                    import tempfile
                    
                    match_date = pd.to_datetime(date_str, format='%Y%m%d')
                    if match_date not in df.index:
                        continue
                    
                    loc = df.index.get_loc(match_date)
                    if loc < 20:
                        continue
                    
                    # 提取最近20天数据
                    recent_df = df.iloc[loc-19:loc+1]
                    
                    # 生成临时K线图
                    temp_img = os.path.join(PROJECT_ROOT, "data", f"temp_{symbol}_{date_str}.png")
                    mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
                    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
                    mpf.plot(recent_df, type='candle', style=s,
                            savefig=dict(fname=temp_img, dpi=50), figsize=(3, 3), axisoff=True)
                    
                    # 搜索相似形态
                    matches = vision_engine.search_similar_patterns(temp_img, top_k=10)
                    
                    # 计算混合胜率
                    win_rate_result = kline_calc.calculate_hybrid_win_rate(matches)
                    
                    results.append({
                        'symbol': symbol,
                        'date': date_str,
                        'hybrid_win_rate': win_rate_result['hybrid_win_rate'],
                        'tb_win_rate': win_rate_result['tb_win_rate'],
                        'traditional_win_rate': win_rate_result['traditional_win_rate'],
                        'valid_matches': win_rate_result['valid_matches']
                    })
                    
                    # 清理临时文件
                    if os.path.exists(temp_img):
                        os.remove(temp_img)
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"❌ 处理 {symbol} 失败: {e}")
            continue
    
    # 4. 保存结果
    if results:
        result_df = pd.DataFrame(results)
        output_file = os.path.join(PROJECT_ROOT, "data", "indices", "win_rates_recalculated.csv")
        result_df.to_csv(output_file, index=False)
        print(f"\n✅ 胜率重算完成！结果保存至: {output_file}")
        print(f"📊 共计算 {len(result_df)} 条记录")
    else:
        print("⚠️ 无有效结果")


if __name__ == "__main__":
    recalculate_win_rates()
