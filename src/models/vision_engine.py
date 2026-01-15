import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import faiss
import pickle
import glob
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
        self.model = None
        self.pool = None
        self.model_mode = None  # "attention" | "cae"

        # 1. 优先加载 AttentionCAE，如果不存在则回退到 QuantCAE
        if os.path.exists(ATTENTION_MODEL_PATH):
            if not self._load_attention_model():
                self._load_cae_model()
        else:
            self._load_cae_model()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.index = None
        self.meta_data = []
        self._pixel_cache = {}
        self._edge_cache = {}
        self._data_loader = None

    def _load_attention_model(self):
        try:
            print(f"👁️ [VisionEngine] 启动中... 加载模型: AttentionCAE")
            self.model = AttentionCAE(latent_dim=1024, num_attention_heads=8).to(self.device)
            state_dict = torch.load(ATTENTION_MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.use_attention = True
            self.pool = None
            self.model_mode = "attention"
            print(f"✅ AttentionCAE 加载成功")
            return True
        except Exception as e:
            print(f"❌ AttentionCAE 权重加载失败: {e}")
            return False

    def _load_cae_model(self):
        try:
            print(f"👁️ [VisionEngine] 启动中... 加载模型: QuantCAE (回退模式)")
            from src.models.autoencoder import QuantCAE
            self.model = QuantCAE().to(self.device)
            if os.path.exists(CAE_MODEL_PATH):
                state_dict = torch.load(CAE_MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"✅ QuantCAE 加载成功")
            self.use_attention = False
            self.pool = nn.AdaptiveAvgPool1d(1024)
            self.model_mode = "cae"
            return True
        except Exception as e:
            print(f"❌ QuantCAE 权重加载失败: {e}")
            return False

    def reload_index(self):
        # 优先加载 AttentionCAE 索引
        index_file = ATTENTION_INDEX_FILE if os.path.exists(ATTENTION_INDEX_FILE) else INDEX_FILE
        meta_file = ATTENTION_META_CSV if os.path.exists(ATTENTION_META_CSV) else META_CSV
        
        if not os.path.exists(index_file):
            print(f"❌ 索引文件不存在: {index_file}")
            return False

        # 索引与模型对齐
        index_mode = "attention" if index_file == ATTENTION_INDEX_FILE else "cae"
        if self.model_mode != index_mode:
            if index_mode == "attention":
                self._load_attention_model()
            else:
                self._load_cae_model()

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

    def _vector_score_to_similarity(self, score):
        """将FAISS返回分数统一映射到0~1"""
        try:
            if self.index is not None and self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                sim = (float(score) + 1.0) / 2.0
            else:
                sim = 1.0 / (1.0 + max(float(score), 0.0))
            return float(np.clip(sim, 0.0, 1.0))
        except Exception:
            return 0.0

    def _resolve_image_path(self, info_path, symbol, date_str):
        """从元数据或目录中定位历史K线图片"""
        if info_path and os.path.exists(info_path):
            return info_path
        img_base = os.path.join(PROJECT_ROOT, "data", "images")
        date_n = str(date_str).replace("-", "")
        candidates = [
            os.path.join(img_base, f"{symbol}_{date_n}.png"),
            os.path.join(img_base, symbol, f"{symbol}_{date_n}.png"),
            os.path.join(img_base, symbol, f"{date_n}.png"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        pattern = os.path.join(img_base, "**", f"*{symbol}*{date_n}*.png")
        matches = glob.glob(pattern, recursive=True)
        return matches[0] if matches else None

    def _load_pixel_vector(self, img_path, size=(64, 64)):
        """轻量像素向量（用于视觉重排）"""
        if not img_path:
            return None
        if img_path in self._pixel_cache:
            return self._pixel_cache[img_path]
        try:
            img = Image.open(img_path).convert("L").resize(size)
            arr = np.asarray(img, dtype=np.float32)
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            vec = arr.flatten()
            self._pixel_cache[img_path] = vec
            if len(self._pixel_cache) > 500:
                self._pixel_cache.pop(next(iter(self._pixel_cache)))
            return vec
        except Exception:
            return None

    def _cosine_sim(self, a, b):
        if a is None or b is None:
            return None
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _pearson_corr(self, a, b):
        if a is None or b is None:
            return None
        if len(a) != len(b):
            return None
        try:
            return float(np.corrcoef(a, b)[0, 1])
        except Exception:
            return None

    def _parse_date(self, date_str):
        try:
            return datetime.strptime(str(date_str), "%Y%m%d")
        except Exception:
            try:
                return datetime.strptime(str(date_str), "%Y-%m-%d")
            except Exception:
                return None

    def _load_edge_vector(self, img_path, size=(64, 64)):
        """简单边缘特征（像素差分）"""
        if not img_path:
            return None
        if img_path in self._edge_cache:
            return self._edge_cache[img_path]
        try:
            img = Image.open(img_path).convert("L").resize(size)
            arr = np.asarray(img, dtype=np.float32)
            gx = np.diff(arr, axis=1, prepend=arr[:, :1])
            gy = np.diff(arr, axis=0, prepend=arr[:1, :])
            edge = np.sqrt(gx ** 2 + gy ** 2)
            edge = (edge - edge.mean()) / (edge.std() + 1e-6)
            vec = edge.flatten()
            self._edge_cache[img_path] = vec
            if len(self._edge_cache) > 500:
                self._edge_cache.pop(next(iter(self._edge_cache)))
            return vec
        except Exception:
            return None

    def search_similar_patterns(self, target_img_path, top_k=10, query_prices=None,
                                rerank_with_pixels=True, rerank_top_k=80):
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
        search_k = max(top_k * 20, 400)  # 从更大候选中筛选
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
                if self._data_loader is None:
                    from src.data.data_loader import DataLoader
                    self._data_loader = DataLoader()
                loader = self._data_loader
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
            ret_corr = None
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
                            # 形态回报相关（差分）
                            q_ret = np.diff(query_prices) / (query_prices[:-1] + 1e-8)
                            m_ret = np.diff(match_prices) / (match_prices[:-1] + 1e-8)
                            q_ret = (q_ret - q_ret.mean()) / (q_ret.std() + 1e-8)
                            m_ret = (m_ret - m_ret.mean()) / (m_ret.std() + 1e-8)
                            corr2 = np.corrcoef(q_ret, m_ret)[0, 1]
                            if not np.isnan(corr2):
                                ret_corr = float(corr2)
                except Exception:
                    correlation = None

            # === 优化4: 评分策略（相似度校准 + 相关性增强）===
            sim_score = self._vector_score_to_similarity(vector_score)

            corr_norm = None
            if correlation is None:
                final_score = sim_score
            else:
                # 相关性归一化到 0~1
                corr_norm = (float(correlation) + 1.0) / 2.0
                corr_norm = min(max(corr_norm, 0.0), 1.0)
                # 叠加回报相关
                if ret_corr is not None:
                    ret_norm = (float(ret_corr) + 1.0) / 2.0
                    corr_norm = 0.6 * corr_norm + 0.4 * ret_norm
                final_score = 0.7 * sim_score + 0.3 * corr_norm

            candidates.append({
                "symbol": sym,
                "date": date_str,
                "score": float(final_score),
                "vector_score": float(vector_score),
                "correlation": (None if correlation is None else float(correlation)),
                "ret_corr": (None if ret_corr is None else float(ret_corr)),
                "sim_score": float(sim_score),
                "corr_norm": (None if corr_norm is None else float(corr_norm)),
                "path": info.get("path")
            })

            seen_dates.setdefault(sym, []).append(current_dt)

        # === 视觉重排：像素级相似度兜底（提升“肉眼相似”效果） ===
        if rerank_with_pixels and candidates:
            q_vec = self._load_pixel_vector(target_img_path)
            if q_vec is not None:
                q_edge = self._load_edge_vector(target_img_path)
                for c in candidates[:min(len(candidates), rerank_top_k)]:
                    img_path = self._resolve_image_path(c.get("path"), c["symbol"], c["date"])
                    v = self._load_pixel_vector(img_path)
                    e = self._load_edge_vector(img_path)
                    pix_cos = self._cosine_sim(q_vec, v)
                    pix_corr = self._pearson_corr(q_vec, v)
                    edge_cos = self._cosine_sim(q_edge, e) if q_edge is not None else None
                    pix_cos = 0.0 if pix_cos is None else pix_cos
                    pix_corr = 0.0 if pix_corr is None else pix_corr
                    edge_cos = 0.0 if edge_cos is None else edge_cos
                    pix_norm = (pix_cos + 1.0) / 2.0
                    pix_corr_norm = (pix_corr + 1.0) / 2.0
                    edge_norm = (edge_cos + 1.0) / 2.0
                    visual_sim = 0.5 * pix_norm + 0.3 * pix_corr_norm + 0.2 * edge_norm
                    corr = c.get("corr_norm")
                    corr_score = 0.5 if corr is None else corr
                    c["pixel_sim"] = visual_sim
                    c["edge_sim"] = edge_norm
                    c["score"] = 0.45 * c.get("sim_score", 0) + 0.35 * visual_sim + 0.20 * corr_score

        # === 优化6: 排序并返回（保证Top-K） ===
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 返回Top-K
        return candidates[:top_k]

    def generate_attention_heatmap(self, img_path, save_path=None, head_idx: int = 0, mode: str = "single"):
        """
        生成注意力热力图（如果模型支持注意力权重）
        """
        try:
            from src.utils.attention_visualizer import AttentionVisualizer
            if not hasattr(self.model, "get_attention_weights"):
                return None
            # 读取并预处理
            img = Image.open(img_path).convert('RGB')
            input_tensor = self.preprocess(img)
            visualizer = AttentionVisualizer(self.model, device=str(self.device))
            if mode == "all":
                fig = visualizer.visualize_multi_head_attention(
                    input_tensor, query_pos=(7, 7), save_path=save_path
                )
            else:
                fig = visualizer.visualize_single_attention(
                    input_tensor, head_idx=head_idx, query_pos=(7, 7), save_path=save_path
                )
            return save_path
        except Exception:
            return None

    def search_multi_scale_patterns(self, img_paths: dict, top_k=10, weights=None, query_prices=None,
                                    rerank_with_pixels=True, rerank_top_k=80):
        """
        多尺度检索：日/周/月图像分别检索，再加权融合
        """
        if self.index is None:
            if not self.reload_index():
                return []
        if not img_paths:
            return []
        if weights is None:
            weights = {"daily": 0.6, "weekly": 0.3, "monthly": 0.1}

        merged = {}
        for scale, path in img_paths.items():
            vec = self._image_to_vector(path)
            if vec is None:
                continue
            vec = vec.astype('float32').reshape(1, -1)
            faiss.normalize_L2(vec)
            search_k = max(top_k * 10, 200)
            D, I = self.index.search(vec, search_k)
            for vector_score, idx in zip(D[0], I[0]):
                if idx >= len(self.meta_data):
                    continue
                info = self.meta_data[idx]
                sym = str(info['symbol']).zfill(6)
                date_str = str(info['date'])
                key = (sym, date_str)
                # 距离转相似度
                sim = self._vector_score_to_similarity(vector_score)
                w = weights.get(scale, 0.0)
                merged[key] = merged.get(key, 0.0) + sim * w

        # 相关性增强（仅对日线使用）
        candidates = []
        for (sym, date_str), score in merged.items():
            candidates.append({"symbol": sym, "date": date_str, "score": float(score), "path": None})

        # 像素重排（使用日线Query）
        if rerank_with_pixels and candidates and img_paths.get("daily"):
            q_vec = self._load_pixel_vector(img_paths.get("daily"))
            if q_vec is not None:
                q_edge = self._load_edge_vector(img_paths.get("daily"))
                for c in candidates[:min(len(candidates), rerank_top_k)]:
                    img_path = self._resolve_image_path(None, c["symbol"], c["date"])
                    v = self._load_pixel_vector(img_path)
                    e = self._load_edge_vector(img_path)
                    pix_cos = self._cosine_sim(q_vec, v)
                    pix_corr = self._pearson_corr(q_vec, v)
                    edge_cos = self._cosine_sim(q_edge, e) if q_edge is not None else None
                    pix_cos = 0.0 if pix_cos is None else pix_cos
                    pix_corr = 0.0 if pix_corr is None else pix_corr
                    edge_cos = 0.0 if edge_cos is None else edge_cos
                    pix_norm = (pix_cos + 1.0) / 2.0
                    pix_corr_norm = (pix_corr + 1.0) / 2.0
                    edge_norm = (edge_cos + 1.0) / 2.0
                    visual_sim = 0.5 * pix_norm + 0.3 * pix_corr_norm + 0.2 * edge_norm
                    c["pixel_sim"] = visual_sim
                    c["edge_sim"] = edge_norm
                    c["score"] = 0.7 * c["score"] + 0.3 * visual_sim
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # 时间隔离（避免同一股票相邻日期）
        ISOLATION_DAYS = 20
        isolated = []
        seen_dates = {}
        for c in candidates:
            sym = str(c.get("symbol", "")).zfill(6)
            dt = self._parse_date(c.get("date"))
            if dt is None:
                continue
            conflict = False
            if sym in seen_dates:
                for d in seen_dates[sym]:
                    if abs((dt - d).days) < ISOLATION_DAYS:
                        conflict = True
                        break
            if conflict:
                continue
            isolated.append(c)
            seen_dates.setdefault(sym, []).append(dt)
            if len(isolated) >= top_k:
                break

        return isolated if isolated else candidates[:top_k]


if __name__ == "__main__":
    if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
    v = VisionEngine()
    v.reload_index()
    print("Vision Engine Ready")