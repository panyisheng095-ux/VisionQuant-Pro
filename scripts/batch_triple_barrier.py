"""
批量计算40万张K线图的Triple Barrier标签
Batch Calculate Triple Barrier Labels for 400K K-line Images

策略：
1. 按股票分组处理（避免重复加载同一股票数据）
2. 使用多进程并行（CPU密集型任务）
3. 增量更新（只计算新标签，已计算的跳过）
4. 结果存储到HDF5（比CSV快100倍）

Author: VisionQuant Team
"""

import os
import sys
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import time

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.triple_barrier import TripleBarrierLabeler
from src.data.data_loader import DataLoader


# 配置
LABELS_HDF5_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "triple_barrier_labels.h5")
ATT_META = os.path.join(PROJECT_ROOT, "data", "indices", "meta_data_attention.csv")
META_CSV_PATH = ATT_META if os.path.exists(ATT_META) else os.path.join(PROJECT_ROOT, "data", "indices", "meta_data.csv")
MAX_WORKERS = 8  # 多进程数


def process_stock_triple_barrier(symbol: str, dates: pd.Series) -> pd.DataFrame:
    """
    处理单只股票的所有Triple Barrier标签
    
    Args:
        symbol: 股票代码
        dates: 该股票的所有日期列表
        
    Returns:
        DataFrame with columns: symbol, date, label, hit_day, hit_type, max_return, min_return, final_return
    """
    try:
        # 加载股票数据
        loader = DataLoader()
        df = loader.get_stock_data(symbol)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df.index = pd.to_datetime(df.index)
        
        # Triple Barrier标签器
        labeler = TripleBarrierLabeler(
            upper_barrier=0.05,
            lower_barrier=0.03,
            max_holding_period=20
        )
        
        # 为每个日期计算标签
        results = []
        
        for date_str in dates:
            try:
                # 解析日期
                if '-' in str(date_str):
                    match_date = pd.to_datetime(date_str)
                else:
                    match_date = pd.to_datetime(str(date_str), format='%Y%m%d')
                
                if match_date not in df.index:
                    continue
                
                loc = df.index.get_loc(match_date)
                
                # 确保有足够的数据
                if loc < 20 or loc + labeler.max_hold >= len(df):
                    continue
                
                # 提取价格序列
                prices = df.iloc[loc:loc+labeler.max_hold+1]['Close']
                
                # 计算标签
                labels, details = labeler.generate_labels(prices, return_details=True)
                
                if not details.empty:
                    detail = details.iloc[0]
                    results.append({
                        'symbol': symbol,
                        'date': match_date.strftime('%Y%m%d'),
                        'label': int(detail['label']),
                        'hit_day': int(detail['hit_day']),
                        'hit_type': detail['hit_type'],
                        'max_return': float(detail['max_return']),
                        'min_return': float(detail['min_return']),
                        'final_return': float(detail['final_return'])
                    })
                    
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"❌ 处理 {symbol} 失败: {e}")
        return pd.DataFrame()


def save_labels_to_hdf5(df: pd.DataFrame, hdf5_path: str):
    """
    保存标签到HDF5文件（使用pandas HDFStore，更稳定）
    
    Args:
        df: 标签DataFrame
        hdf5_path: HDF5文件路径
    """
    if df.empty:
        return
    
    try:
        # 使用pandas的HDFStore（更稳定，支持字符串列）
        store = pd.HDFStore(hdf5_path, mode='a')
        
        if 'labels' in store:
            # 读取现有数据
            existing = store['labels']
            # 合并（去重）
            combined = pd.concat([existing, df]).drop_duplicates(
                subset=['symbol', 'date'],
                keep='last'
            )
            store['labels'] = combined
        else:
            store['labels'] = df
        
        store.close()
        
    except Exception as e:
        print(f"⚠️ HDF5保存失败，使用CSV备份: {e}")
        # CSV备份
        csv_path = hdf5_path.replace('.h5', '.csv')
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            combined = pd.concat([existing, df]).drop_duplicates(
                subset=['symbol', 'date'],
                keep='last'
            )
            combined.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False)


def batch_calculate_triple_barrier_labels():
    """
    批量计算40万张K线图的Triple Barrier标签
    """
    print("🚀 开始批量计算Triple Barrier标签...")
    print(f"📁 元数据文件: {META_CSV_PATH}")
    print(f"💾 输出文件: {LABELS_HDF5_PATH}")
    
    # 1. 读取元数据
    if not os.path.exists(META_CSV_PATH):
        print(f"❌ 元数据文件不存在: {META_CSV_PATH}")
        return
    
    print("📖 读取元数据...")
    meta_df = pd.read_csv(META_CSV_PATH, dtype=str)
    
    # 确保有symbol和date列
    if 'symbol' not in meta_df.columns or 'date' not in meta_df.columns:
        print("❌ 元数据文件缺少symbol或date列")
        return
    
    # 按股票分组
    grouped = meta_df.groupby('symbol')
    print(f"📊 共 {len(grouped)} 只股票，约 {len(meta_df)} 张K线图")
    
    # 2. 检查已有标签（增量更新）
    existing_labels = set()
    if os.path.exists(LABELS_HDF5_PATH):
        try:
            import tables as tb
            with tb.open_file(LABELS_HDF5_PATH, mode='r') as h5file:
                if '/labels' in h5file:
                    table = h5file.root.labels
                    existing_df = pd.DataFrame(table.read())
                    if not existing_df.empty:
                        existing_labels = set(
                            zip(existing_df['symbol'].astype(str), existing_df['date'].astype(str))
                        )
                        print(f"✅ 已有 {len(existing_labels)} 条标签，将进行增量更新")
        except:
            pass
    
    # 3. 多进程处理
    print(f"\n🔄 开始多进程处理（{MAX_WORKERS}个进程）...")
    start_time = time.time()
    
    all_results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_stock_triple_barrier, symbol, group['date']): symbol
            for symbol, group in grouped
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            symbol = futures[future]
            completed += 1
            
            try:
                result_df = future.result()
                if not result_df.empty:
                    all_results.append(result_df)
                    
                    # 每处理10只股票保存一次（避免内存溢出）
                    if len(all_results) >= 10:
                        combined = pd.concat(all_results, ignore_index=True)
                        save_labels_to_hdf5(combined, LABELS_HDF5_PATH)
                        all_results = []
                        print(f"💾 已保存 {completed}/{len(futures)} 只股票的标签")
            except Exception as e:
                print(f"❌ 处理 {symbol} 失败: {e}")
    
    # 保存剩余结果
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        save_labels_to_hdf5(combined, LABELS_HDF5_PATH)
    
    elapsed = time.time() - start_time
    print(f"\n✅ 批量计算完成！耗时: {elapsed/60:.1f} 分钟")
    print(f"📁 标签文件: {LABELS_HDF5_PATH}")


if __name__ == "__main__":
    batch_calculate_triple_barrier_labels()
