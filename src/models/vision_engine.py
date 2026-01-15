import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import faiss
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# === 1. 基础配置 ===
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# === 2. 路径配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# 优先使用 AttentionCAE，如果不存在则回退到 QuantCAE
ATTENTION_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "attention_cae_best.pth")
CAE_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "cae_best.pth")
# 索引文件路径（优先用新索引）
ATTENTION_INDEX_FILE = os.path.join(PROJECT_ROOT, "data", "indices", "cae_faiss_attention.bin")
ATTENTION_META_CSV = os.path.join(PROJECT_ROOT, "data", "indices", "meta_data_attention.csv")
INDEX_FILE = os.path.join(PROJECT_ROOT, "data", "indices", "cae_faiss.bin")
META_CSV = os.path.join(PROJECT_ROOT, "data", "indices", "meta_data.csv")
META_PKL = os.path.join(PROJECT_ROOT, "data", "indices", "meta.pkl")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.models.attention_cae import AttentionCAE


class VisionEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        
        # 1. 优先加载 AttentionCAE，如果不存在则回退到 QuantCAE
        use_attention = os.path.exists(ATTENTION_MODEL_PATH)
        
        if use_attention:
            print(f"👁️ [VisionEngine] 启动中... 加载模型: AttentionCAE")
            self.model = AttentionCAE(latent_dim=1024, num_attention_heads=8).to(self.device)
            try:
                state_dict = torch.load(ATTENTION_MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"✅ AttentionCAE 加载成功")
            except Exception as e:
                print(f"❌ AttentionCAE 权重加载失败: {e}，回退到 QuantCAE")
                use_attention = False
        
        if not use_attention:
            print(f"👁️ [VisionEngine] 启动中... 加载模型: QuantCAE (回退模式)")
            from src.models.autoencoder import QuantCAE
            self.model = QuantCAE().to(self.device)
            if os.path.exists(CAE_MODEL_PATH):
                try:
                    state_dict = torch.load(CAE_MODEL_PATH, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    print(f"✅ QuantCAE 加载成功")
                except Exception as e:
                    print(f"❌ QuantCAE 权重加载失败: {e}")
        
        # QuantCAE 需要 pool 降维，AttentionCAE 已经输出 1024 维
        self.use_attention = use_attention
        if not use_attention:
            self.pool = nn.AdaptiveAvgPool1d(1024)
        else:
            self.pool = None  # AttentionCAE 不需要 pool

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.index = None
        self.meta_data = []

    def reload_index(self):
        # 优先加载 AttentionCAE 索引
        index_file = ATTENTION_INDEX_FILE if os.path.exists(ATTENTION_INDEX_FILE) else INDEX_FILE
        meta_file = ATTENTION_META_CSV if os.path.exists(ATTENTION_META_CSV) else META_CSV
        
        if not os.path.exists(index_file):
            print(f"❌ 索引文件不存在: {index_file}")
            return False

        print(f"📥 [VisionEngine] 加载索引: {os.path.basename(index_file)}")
        try:
            self.index = faiss.read_index(index_file)
        except Exception as e:
            print(f"❌ FAISS 加载失败: {e}")
            return False

        if os.path.exists(meta_file):
            df = pd.read_csv(meta_file, dtype=str)
            self.meta_data = df.to_dict('records')
        elif os.path.exists(META_PKL):
            with open(META_PKL, 'rb') as f:
                self.meta_data = pickle.load(f)
        else:
            print(f"❌ 元数据文件不存在: {meta_file}")
            return False

        print(f"✅ 知识库就绪: {len(self.meta_data)} 条记录")
        return True

    def _image_to_vector(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.use_attention:
                    # AttentionCAE.encode() 已经返回 1024 维的 L2 归一化向量
                    feature = self.model.encode(input_tensor)
                    return feature.cpu().numpy().flatten()
                else:
                    # QuantCAE.encode() 返回 50176 维，需要 pool 降维
                    full_feature = self.model.encode(input_tensor)
                    reduced_feature = self.pool(full_feature.unsqueeze(1)).squeeze(1)
                    return reduced_feature.cpu().numpy().flatten()
        except:
            return None

    def search_similar_patterns(self, target_img_path, top_k=10, query_prices=None):
        """
        混合搜索：视觉特征 + 价格序列相关性
        
        Args:
            target_img_path: 查询K线图路径
            top_k: 返回Top-K结果
            query_prices: 查询的价格序列（20天收盘价），用于计算相关性
        """
        if self.index is None:
            if not self.reload_index(): return []

        vec = self._image_to_vector(target_img_path)
        if vec is None: return []

        vec = vec.astype('float32').reshape(1, -1)
        faiss.normalize_L2(vec)

        # === 优化1: 扩大搜索范围，获取更多候选 ===
        search_k = max(top_k * 10, 200)  # 从200个候选中筛选
        D, I = self.index.search(vec, search_k)

        candidates = []
        seen_dates = {}
        ISOLATION_DAYS = 20

        # === 优化2: 视觉候选 +（可选）价格相关性 ===
        # 注意：对“非热门股/冷门日期”，在循环里频繁拉取历史数据很容易失败。
        # 我们将相关性视为“可选增强”：算得出来就提升排序，算不出来就回退到纯视觉TopK，
        # 这样才能保证对比图几乎不可能空。
        loader = None
        price_df_cache = {}
        if query_prices is not None and len(query_prices) == 20:
            try:
                from src.data.data_loader import DataLoader
                loader = DataLoader()
            except Exception:
                loader = None

        for vector_score, idx in zip(D[0], I[0]):
            if idx >= len(self.meta_data): continue

            info = self.meta_data[idx]
            sym = str(info['symbol']).zfill(6)
            date_str = str(info['date'])

            try:
                current_dt = datetime.strptime(date_str, "%Y%m%d")
            except:
                try:
                    current_dt = datetime.strptime(date_str, "%Y-%m-%d")
                except:
                    continue

            # 时间隔离检查
            is_conflict = False
            if sym in seen_dates:
                for existing_dt in seen_dates[sym]:
                    if abs((current_dt - existing_dt).days) < ISOLATION_DAYS:
                        is_conflict = True
                        break
            if is_conflict:
                continue

            # === 优化3: 计算价格序列相关性（可选）===
            correlation = None
            if loader is not None:
                try:
                    if sym not in price_df_cache:
                        dfp = loader.get_stock_data(sym)
                        if dfp is None or dfp.empty:
                            price_df_cache[sym] = None
                        else:
                            dfp.index = pd.to_datetime(dfp.index)
                            price_df_cache[sym] = dfp
                    else:
                        dfp = price_df_cache[sym]

                    if dfp is not None and (current_dt in dfp.index):
                        loc = dfp.index.get_loc(current_dt)
                        if loc >= 19:
                            match_prices = dfp.iloc[loc - 19: loc + 1]['Close'].values
                            query_norm = (query_prices - query_prices.mean()) / (query_prices.std() + 1e-8)
                            match_norm = (match_prices - match_prices.mean()) / (match_prices.std() + 1e-8)
                            corr = np.corrcoef(query_norm, match_norm)[0, 1]
                            if not np.isnan(corr):
                                correlation = float(corr)
                except Exception:
                    correlation = None

            # === 优化4: 评分策略（相似度校准 + 相关性增强）===
            # Faiss 距离越小越相似 -> 转为 similarity 分数（0~1）
            try:
                dist = float(vector_score)
                sim_score = 1.0 / (1.0 + max(dist, 0.0))
            except Exception:
                sim_score = 0.0

            if correlation is None:
                final_score = sim_score
            else:
                # 相关性归一化到 0~1
                corr_norm = (float(correlation) + 1.0) / 2.0
                corr_norm = min(max(corr_norm, 0.0), 1.0)
                final_score = 0.7 * sim_score + 0.3 * corr_norm

            candidates.append({
                "symbol": sym,
                "date": date_str,
                "score": float(final_score),
                "vector_score": float(vector_score),
                "correlation": (None if correlation is None else float(correlation))
            })

            seen_dates.setdefault(sym, []).append(current_dt)

        # === 优化6: 排序并返回（保证Top-K） ===
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 返回Top-K
        return candidates[:top_k]


if __name__ == "__main__":
    if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
    v = VisionEngine()
    v.reload_index()
    print("Vision Engine Ready")