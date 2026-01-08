import os
import sys
import torch
import numpy as np
import pandas as pd
import faiss
import glob
import gc
import csv
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# === 1. 基础稳健配置 ===
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# === 2. 路径配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "images")
# 这里的模型其实用不到了(因为特征已经提好了)，但为了兼容保留
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "cae_best.pth")

INDICES_DIR = os.path.join(PROJECT_ROOT, "data", "indices")
TEMP_DIR = os.path.join(INDICES_DIR, "temp_chunks")
os.makedirs(TEMP_DIR, exist_ok=True)

# 关键文件路径
VECTORS_HUGE_MMAP = os.path.join(INDICES_DIR, "vectors_mmap.npy")  # 80GB 源文件
VECTORS_REDUCED_MMAP = os.path.join(INDICES_DIR, "vectors_reduced.npy")  # 压缩后文件
META_CSV_FILE = os.path.join(INDICES_DIR, "meta_data.csv")
INDEX_FILE = os.path.join(INDICES_DIR, "cae_faiss.bin")
PREDICTION_CACHE_FILE = os.path.join(INDICES_DIR, "prediction_cache.csv")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# 只需要引用，不需要实例化模型
from src.models.autoencoder import QuantCAE


# 占位 Dataset
class StockImageDataset(Dataset):
    def __init__(self, img_dir): pass

    def __len__(self): return 0

    def __getitem__(self, idx): return 0


class IndustrialPredictorReduced:
    def __init__(self):
        # 降维操作纯数学计算，CPU 很稳
        self.device = torch.device("cpu")
        print(f"🏭 [降维引擎] 启动... 目标维度: 1024")
        self.returns_map = {}

    def run_pipeline(self):
        # Step 1: 准备收益率
        self._step1_prepare_returns()

        # Step 2: 检查源数据
        if not os.path.exists(VECTORS_HUGE_MMAP) or not os.path.exists(META_CSV_FILE):
            print("❌ 严重错误：找不到 vectors_mmap.npy 或 meta_data.csv！")
            print("请先运行之前的 [工业引擎] 代码完成 Step 3。")
            return

        # 获取数据总量
        df_meta = pd.read_csv(META_CSV_FILE, dtype=str)
        total_rows = len(df_meta)
        del df_meta
        gc.collect()

        print(f"📊 检测到源数据: {total_rows} 条记录")

        # Step 3.5: 执行降维 (核心！)
        self._step3_5_reduce_dimensions(total_rows)

        # Step 4: 构建索引
        self._step4_build_index(total_rows)

        # Step 5: 预测
        self._step5_batch_predict(total_rows)

    def _step1_prepare_returns(self):
        print("\n[Step 1] 加载收益率表...")
        csv_files = glob.glob(os.path.join(DATA_RAW_DIR, "*.csv"))
        for f in tqdm(csv_files, desc="Returns"):
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                if len(df) < 5: continue
                symbol = os.path.basename(f).replace(".csv", "")
                future_close = df['Close'].shift(-5)
                ret = (future_close - df['Close']) / df['Close']
                for d, r in ret.items():
                    if not pd.isna(r):
                        self.returns_map[f"{symbol}_{d.strftime('%Y%m%d')}"] = r
            except:
                continue
        print(f"✅ 收益率加载完成")

    def _step3_5_reduce_dimensions(self, total_rows):
        """将 50176 维压缩到 1024 维"""
        # 如果已经压缩过，跳过
        if os.path.exists(VECTORS_REDUCED_MMAP):
            # 简单检查大小是否匹配 (1024 * 4 bytes * rows)
            expected_size = total_rows * 1024 * 4
            if os.path.getsize(VECTORS_REDUCED_MMAP) >= expected_size:
                print("\n[Step 3.5] 检测到已降维文件，跳过压缩。")
                return

        print(f"\n[Step 3.5] 执行高维特征压缩 (50176 -> 1024)...")
        print("💡 这是一个 IO 密集型操作，请耐心等待...")

        # 1. 映射源文件 (只读)
        huge_dim = 50176
        try:
            mmap_huge = np.memmap(VECTORS_HUGE_MMAP, dtype='float32', mode='r', shape=(total_rows, huge_dim))
        except:
            # 自动计算维度防崩
            file_size = os.path.getsize(VECTORS_HUGE_MMAP)
            huge_dim = file_size // (total_rows * 4)
            mmap_huge = np.memmap(VECTORS_HUGE_MMAP, dtype='float32', mode='r', shape=(total_rows, huge_dim))

        # 2. 创建目标文件
        target_dim = 1024
        mmap_small = np.memmap(VECTORS_REDUCED_MMAP, dtype='float32', mode='w+', shape=(total_rows, target_dim))

        # 3. 定义池化层 (这是降维的核心数学工具)
        # AdaptiveAvgPool1d 会自动把 50176 个数平均成 1024 个数
        pool = torch.nn.AdaptiveAvgPool1d(target_dim)

        # 4. 分批处理
        batch_size = 1000  # 每次只读 1000 行，内存占用极小 (~200MB)

        for i in tqdm(range(0, total_rows, batch_size), desc="Compressing"):
            end_i = min(i + batch_size, total_rows)

            # 读数据 (从硬盘加载到内存)
            batch_huge = mmap_huge[i: end_i].copy()

            # 转 Tensor
            batch_tensor = torch.from_numpy(batch_huge).unsqueeze(1)  # [B, 1, 50176]

            # 压缩
            with torch.no_grad():
                batch_small = pool(batch_tensor).squeeze(1).numpy()

            # 写回硬盘
            mmap_small[i: end_i] = batch_small

            # 清理
            del batch_huge, batch_tensor, batch_small

        mmap_small.flush()
        print(f"✅ 压缩完成！体积缩小 50 倍。")

    def _step4_build_index(self, total_rows):
        print("\n[Step 4] 构建 FAISS 索引 (1024维)...")

        # 如果索引已存在，跳过
        if os.path.exists(INDEX_FILE):
            print("✅ 索引文件已存在，跳过。")
            return

        dim = 1024
        # 读取压缩后的数据
        mmap_arr = np.memmap(VECTORS_REDUCED_MMAP, dtype='float32', mode='r', shape=(total_rows, dim))

        index = faiss.IndexFlatIP(dim)

        # 分批添加 (防止一次加载 1.6GB 导致瞬间卡顿，虽然 1.6GB 其实还好)
        batch_size = 50000
        for i in tqdm(range(0, total_rows, batch_size), desc="Indexing"):
            batch = mmap_arr[i: i + batch_size].copy()
            faiss.normalize_L2(batch)
            index.add(batch)
            del batch
            gc.collect()

        faiss.write_index(index, INDEX_FILE)
        print("✅ 索引构建完成。")

    def _step5_batch_predict(self, total_rows):
        print("\n[Step 5] 流式推演 (基于压缩特征)...")

        if os.path.exists(PREDICTION_CACHE_FILE):
            os.remove(PREDICTION_CACHE_FILE)

        # 加载索引
        index = faiss.read_index(INDEX_FILE)
        # 加载压缩数据
        mmap_arr = np.memmap(VECTORS_REDUCED_MMAP, dtype='float32', mode='r', shape=(total_rows, 1024))

        # 加载元数据
        df_meta = pd.read_csv(META_CSV_FILE, dtype=str)
        meta_symbols = df_meta['symbol'].values
        meta_dates = df_meta['date'].values

        batch_size = 100  # 搜索批次

        with open(PREDICTION_CACHE_FILE, 'w') as f_out:
            f_out.write("symbol,date,pred_win_rate,pred_return,confidence\n")

            for start_idx in tqdm(range(0, total_rows, batch_size), desc="Predicting"):
                end_idx = min(start_idx + batch_size, total_rows)

                # 从压缩后的 mmap 读取 Query
                batch_vecs = mmap_arr[start_idx: end_idx].copy()
                faiss.normalize_L2(batch_vecs)

                # 极速搜索
                D, I = index.search(batch_vecs, 20)

                lines = []
                for k in range(len(batch_vecs)):
                    current_idx = start_idx + k
                    curr_symbol = meta_symbols[current_idx]
                    curr_date = meta_dates[current_idx]

                    valid_ret = []
                    weights = []

                    for rank, neighbor_idx in enumerate(I[k]):
                        if neighbor_idx == current_idx: continue

                        nb_date = meta_dates[neighbor_idx]
                        if nb_date >= curr_date: continue  # 时间锁

                        nb_symbol = meta_symbols[neighbor_idx]
                        key = f"{nb_symbol}_{nb_date}"

                        if key in self.returns_map:
                            valid_ret.append(self.returns_map[key])
                            weights.append(np.exp(D[k][rank] * 5))

                        if len(valid_ret) >= 10: break

                    if len(valid_ret) >= 3:
                        wr = sum(1 for r in valid_ret if r > 0) / len(valid_ret)
                        w = np.array(weights)
                        er = np.sum(np.array(valid_ret) * (w / w.sum()))
                        lines.append(f"{curr_symbol},{curr_date},{wr * 100:.2f},{er * 100:.2f},{len(valid_ret)}\n")

                f_out.writelines(lines)
                f_out.flush()

                if start_idx % 1000 == 0: gc.collect()

        print(f"🎉 全部完成！结果保存在: {PREDICTION_CACHE_FILE}")


if __name__ == "__main__":
    engine = IndustrialPredictorReduced()
    engine.run_pipeline()