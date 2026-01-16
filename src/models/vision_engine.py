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
from scipy.spatial.distance import cdist

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


def _dtw_distance(s1, s2, window=5):
    """
    快速DTW (Dynamic Time Warping) 距离计算
    使用Sakoe-Chiba带约束加速，复杂度从O(n²)降到O(n*window)
    
    Args:
        s1, s2: 两个价格序列
        window: 带约束窗口大小（默认5，即允许±5的时间偏移）
    """
    try:
        n, m = len(s1), len(s2)
        if n == 0 or m == 0:
            return float('inf')
        
        # 归一化到0-1（保留形态，消除绝对价格差异）
        s1 = np.array(s1, dtype=float)
        s2 = np.array(s2, dtype=float)
        s1 = (s1 - np.min(s1)) / (np.max(s1) - np.min(s1) + 1e-8)
        s2 = (s2 - np.min(s2)) / (np.max(s2) - np.min(s2) + 1e-8)
        
        # Sakoe-Chiba带约束DTW
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            # 只计算窗口内的点
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)
            for j in range(j_start, j_end):
                cost = abs(s1[i - 1] - s2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # 插入
                    dtw_matrix[i, j - 1],      # 删除
                    dtw_matrix[i - 1, j - 1]   # 匹配
                )
        
        # 归一化到0-1范围（距离越小越相似）
        return dtw_matrix[n, m] / max(n, m)
    except Exception:
        return float('inf')


def _extract_kline_features(prices):
    """
    提取K线形态的关键特征向量（用于形态匹配）
    返回8维特征：
    1. 整体趋势方向 (涨/跌)
    2. 涨跌幅
    3. 波动率
    4. 最高点位置（相对）
    5. 最低点位置（相对）
    6. 头部形态（前1/3涨跌）
    7. 中部形态（中1/3涨跌）
    8. 尾部形态（后1/3涨跌）
    """
    try:
        if len(prices) < 6:
            return None
        prices = np.array(prices, dtype=float)
        n = len(prices)
        
        # 1. 整体趋势方向 (1=涨, -1=跌)
        trend = 1 if prices[-1] > prices[0] else -1
        
        # 2. 涨跌幅 (归一化)
        change = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
        change_norm = np.tanh(change * 10)  # 映射到-1~1
        
        # 3. 波动率
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        volatility = np.std(returns) * np.sqrt(252)
        vol_norm = min(volatility / 0.5, 1.0)  # 归一化到0~1
        
        # 4-5. 最高/最低点位置 (相对位置)
        high_pos = np.argmax(prices) / (n - 1)
        low_pos = np.argmin(prices) / (n - 1)
        
        # 6-8. 分段趋势
        seg_len = n // 3
        head = (prices[seg_len] - prices[0]) / (prices[0] + 1e-8)
        mid = (prices[2 * seg_len] - prices[seg_len]) / (prices[seg_len] + 1e-8)
        tail = (prices[-1] - prices[2 * seg_len]) / (prices[2 * seg_len] + 1e-8)
        
        return np.array([
            float(trend), change_norm, vol_norm,
            high_pos, low_pos,
            np.tanh(head * 10), np.tanh(mid * 10), np.tanh(tail * 10)
        ])
    except Exception:
        return None


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
        【核心改进】DTW主导 + 视觉辅助的混合检索
        
        核心思路：
        1. FAISS只做粗筛（获取2000+候选）
        2. DTW + 形态特征做精排（真正决定"像不像"）
        3. 最终按DTW相似度排序
        
        Args:
            target_img_path: 查询K线图路径
            top_k: 返回Top-K结果
            query_prices: 查询的价格序列（20天收盘价），用于计算DTW
        """
        if self.index is None:
            if not self.reload_index(): return []

        vec = self._image_to_vector(target_img_path)
        if vec is None: return []

        vec = vec.astype('float32').reshape(1, -1)
        faiss.normalize_L2(vec)

        # === 关键改进1: 大幅扩大粗筛范围（从400→2000）===
        # 因为CNN特征不可靠，需要从更大范围中用DTW精选
        search_k = max(top_k * 200, 2000)
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

            # === 优化3: 计算价格序列相关性 + DTW + 形态特征（可选）===
            correlation = None
            ret_corr = None
            dtw_sim = None
            feature_sim = None
            match_trend = None
            
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
                            
                            # A. 价格相关性
                            query_norm = (query_prices - query_prices.mean()) / (query_prices.std() + 1e-8)
                            match_norm = (match_prices - match_prices.mean()) / (match_prices.std() + 1e-8)
                            corr = np.corrcoef(query_norm, match_norm)[0, 1]
                            if not np.isnan(corr):
                                correlation = float(corr)
                            
                            # B. 回报序列相关
                            q_ret = np.diff(query_prices) / (query_prices[:-1] + 1e-8)
                            m_ret = np.diff(match_prices) / (match_prices[:-1] + 1e-8)
                            q_ret_norm = (q_ret - q_ret.mean()) / (q_ret.std() + 1e-8)
                            m_ret_norm = (m_ret - m_ret.mean()) / (m_ret.std() + 1e-8)
                            corr2 = np.corrcoef(q_ret_norm, m_ret_norm)[0, 1]
                            if not np.isnan(corr2):
                                ret_corr = float(corr2)
                            
                            # C. DTW形态相似度 (距离越小越相似)
                            dtw_dist = _dtw_distance(query_prices, match_prices)
                            dtw_sim = max(0, 1.0 - dtw_dist)  # 转换为相似度
                            
                            # D. 形态特征相似度
                            q_feat = _extract_kline_features(query_prices)
                            m_feat = _extract_kline_features(match_prices)
                            if q_feat is not None and m_feat is not None:
                                # 趋势方向
                                match_trend = m_feat[0]
                                # 特征向量余弦相似度
                                feat_cos = np.dot(q_feat, m_feat) / (np.linalg.norm(q_feat) * np.linalg.norm(m_feat) + 1e-8)
                                feature_sim = (feat_cos + 1.0) / 2.0
                                
                except Exception:
                    correlation = None

            # === 【核心改进】DTW主导的评分策略 ===
            sim_score = self._vector_score_to_similarity(vector_score)

            # 相关性归一化
            corr_norm = None
            if correlation is not None:
                corr_norm = (float(correlation) + 1.0) / 2.0
                corr_norm = min(max(corr_norm, 0.0), 1.0)
                if ret_corr is not None:
                    ret_norm = (float(ret_corr) + 1.0) / 2.0
                    corr_norm = 0.5 * corr_norm + 0.5 * ret_norm
            
            # 【关键】DTW主导评分：DTW 50% + 相关性 30% + 形态 15% + 视觉 5%
            # 如果没有价格数据，则回退到纯视觉
            if dtw_sim is not None:
                dtw_score = dtw_sim
                feat_score = feature_sim if feature_sim is not None else 0.5
                corr_score = corr_norm if corr_norm is not None else 0.5
                combined_score = 0.50 * dtw_score + 0.30 * corr_score + 0.15 * feat_score + 0.05 * sim_score
            else:
                combined_score = sim_score * 0.3  # 无DTW则大幅降权

            candidates.append({
                "symbol": sym,
                "date": date_str,
                "score": float(combined_score),
                "vector_score": float(vector_score),
                "correlation": (None if correlation is None else float(correlation)),
                "ret_corr": (None if ret_corr is None else float(ret_corr)),
                "dtw_sim": (None if dtw_sim is None else float(dtw_sim)),
                "feature_sim": (None if feature_sim is None else float(feature_sim)),
                "match_trend": match_trend,
                "sim_score": float(sim_score),
                "corr_norm": (None if corr_norm is None else float(corr_norm)),
                "path": info.get("path")
            })

            seen_dates.setdefault(sym, []).append(current_dt)
            
            # 【性能优化】已收集足够高质量候选时提前终止
            high_quality = [c for c in candidates if c.get("dtw_sim") is not None and c["dtw_sim"] > 0.8]
            if len(high_quality) >= top_k * 3:
                break

        # === 【核心改进】DTW主导的最终排序 ===
        if query_prices is not None and len(query_prices) >= 2:
            # Step 1: 趋势方向硬约束（必须同涨同跌）
            query_trend = 1 if query_prices[-1] > query_prices[0] else -1
            
            # 只保留趋势方向一致的候选
            trend_matched = [
                c for c in candidates 
                if c.get("match_trend") is None or c.get("match_trend") == query_trend
            ]
            
            # Step 2: 优先选择有DTW分数的候选
            dtw_candidates = [c for c in trend_matched if c.get("dtw_sim") is not None]
            no_dtw_candidates = [c for c in trend_matched if c.get("dtw_sim") is None]
            
            # Step 3: DTW候选按DTW相似度排序（这是关键！）
            dtw_candidates.sort(key=lambda x: x.get("dtw_sim", 0), reverse=True)
            
            # Step 4: 无DTW候选按相关性排序
            no_dtw_candidates.sort(key=lambda x: x.get("correlation", 0) or 0, reverse=True)
            
            # Step 5: 合并（DTW优先）
            candidates = dtw_candidates + no_dtw_candidates
            
            # Step 6: 最终评分（用于显示，但排序已经按DTW完成）
            for c in candidates:
                dtw = c.get("dtw_sim", 0)
                corr = c.get("correlation", 0) or 0
                corr_norm = (corr + 1.0) / 2.0
                feat = c.get("feature_sim", 0.5)
                # 最终分数（仅供显示）
                c["score"] = 0.60 * dtw + 0.25 * corr_norm + 0.15 * feat if dtw > 0 else corr_norm * 0.5
        
        else:
            # 无价格序列时，回退到纯视觉排序
            candidates.sort(key=lambda x: x.get("sim_score", 0), reverse=True)

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