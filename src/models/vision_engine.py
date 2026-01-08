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
INDEX_FILE = os.path.join(PROJECT_ROOT, "data", "indices", "cae_faiss.bin")
META_CSV = os.path.join(PROJECT_ROOT, "data", "indices", "meta_data.csv")
META_PKL = os.path.join(PROJECT_ROOT, "data", "indices", "meta.pkl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "cae_best.pth")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.models.autoencoder import QuantCAE


class VisionEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        print(f"👁️ [VisionEngine] 启动中... 加载模型: QuantCAE")

        # 1. 加载模型
        self.model = QuantCAE().to(self.device)
        if os.path.exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as e:
                print(f"❌ 权重加载失败: {e}")

        self.pool = nn.AdaptiveAvgPool1d(1024)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.index = None
        self.meta_data = []

    def reload_index(self):
        if not os.path.exists(INDEX_FILE):
            return False

        print(f"📥 [VisionEngine] 加载索引...")
        try:
            self.index = faiss.read_index(INDEX_FILE)
        except Exception as e:
            print(f"❌ FAISS 加载失败: {e}")
            return False

        if os.path.exists(META_CSV):
            df = pd.read_csv(META_CSV, dtype=str)
            self.meta_data = df.to_dict('records')
        elif os.path.exists(META_PKL):
            with open(META_PKL, 'rb') as f:
                self.meta_data = pickle.load(f)
        else:
            return False

        print(f"✅ 知识库就绪: {len(self.meta_data)} 条记录")
        return True

    def _image_to_vector(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
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

        # === 优化2: 混合评分 = 视觉相似度 + 价格相关性 ===
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

            # === 优化3: 计算价格序列相关性（如果提供）===
            correlation = 0.0
            if query_prices is not None and len(query_prices) == 20:
                try:
                    # 加载匹配模式的价格序列
                    from src.data.data_loader import DataLoader
                    loader = DataLoader()
                    match_df = loader.get_stock_data(sym)
                    if not match_df.empty:
                        match_df.index = pd.to_datetime(match_df.index)
                        if current_dt in match_df.index:
                            loc = match_df.index.get_loc(current_dt)
                            if loc >= 19:  # 确保有足够的历史数据
                                match_prices = match_df.iloc[loc-19:loc+1]['Close'].values
                                # 归一化（避免绝对价格差异影响相关性）
                                query_norm = (query_prices - query_prices.mean()) / (query_prices.std() + 1e-8)
                                match_norm = (match_prices - match_prices.mean()) / (match_prices.std() + 1e-8)
                                # 计算皮尔逊相关系数
                                correlation = np.corrcoef(query_norm, match_norm)[0, 1]
                                if np.isnan(correlation):
                                    correlation = 0.0
                except Exception as e:
                    correlation = 0.0

            # === 优化4: 混合评分（视觉60% + 相关性40%）===
            # 如果相关性<0.3，说明形态相反，大幅降分
            if correlation < 0.3:
                final_score = vector_score * 0.3  # 形态相反，大幅降分
            else:
                # 正相关时，加权融合
                final_score = 0.6 * vector_score + 0.4 * correlation

            # === 优化5: 提高相似度阈值，只保留高质量匹配 ===
            if final_score < 0.85:  # 提高阈值，过滤低质量结果
                continue

            candidates.append({
                "symbol": sym,
                "date": date_str,
                "score": float(final_score),
                "vector_score": float(vector_score),
                "correlation": float(correlation)
            })

            seen_dates.setdefault(sym, []).append(current_dt)

        # === 优化6: 按最终分数重新排序 ===
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 返回Top-K
        return candidates[:top_k]


if __name__ == "__main__":
    if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
    v = VisionEngine()
    v.reload_index()
    print("Vision Engine Ready")