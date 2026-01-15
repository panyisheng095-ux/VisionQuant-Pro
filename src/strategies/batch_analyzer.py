import os
import glob
import sys
import pandas as pd
import numpy as np
import mplfinance as mpf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

# 注意：这里不再在顶部引用 KLineFactorCalculator，防止循环引用导致报错
# from src.strategies.kline_factor import KLineFactorCalculator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class BatchAnalyzer:
    """批量股票分析引擎（快速海选模式）"""

    def __init__(self, engines):
        # ============================================================
        # 【关键修复】先导入 KLineFactorCalculator，避免在 _load_prediction_cache 中报错
        # ============================================================
        from src.strategies.kline_factor import KLineFactorCalculator
        
        self.engines = engines
        self.max_workers = 8  # 控制并发，避免API限流
        self._data_cache = {}  # 简易缓存，加速批量
        
        # K线因子计算器（混合胜率）
        self.kline_factor_calc = KLineFactorCalculator(
            triple_barrier_weight=0.7,
            traditional_weight=0.3,
            data_loader=engines.get("loader")
        )
        
        self.prediction_cache = self._load_prediction_cache()

    def _load_prediction_cache(self):
        """加载预测缓存"""
        cache_file = os.path.join(PROJECT_ROOT, "data", "indices", "prediction_cache.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, dtype=str)
                df['date'] = df['date'].astype(str).str.replace('-', '')
                df['symbol'] = df['symbol'].astype(str).str.zfill(6)
                return df.set_index(['symbol', 'date'])['pred_win_rate'].to_dict()
            except:
                return {}
        return {}

    def analyze_batch(self, symbols, progress_callback=None):
        """
        批量分析股票（快速模式）

        Args:
            symbols: 股票代码列表
            progress_callback: 进度回调函数 (current, total, symbol)

        Returns:
            Dict: {symbol: analysis_result}
        """
        results = {}
        total = len(symbols)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_single_fast, sym): sym
                for sym in symbols
            }

            completed = 0
            for future in as_completed(futures):
                symbol = futures[future]
                completed += 1

                if progress_callback:
                    progress_callback(completed, total, symbol)

                try:
                    results[symbol] = future.result()
                except Exception as e:
                    results[symbol] = {
                        "symbol": symbol,
                        "error": str(e),
                        "score": 0,
                        "action": "ERROR"
                    }

        return results

    def _analyze_single_fast(self, symbol):
        """单股票快速分析（跳过可视化）"""
        try:
            # 1. 数据获取
            df = self._get_stock_cached(symbol)
            if df.empty:
                return {"symbol": symbol, "error": "数据获取失败", "score": 0, "action": "ERROR"}

            # 2. 财务数据
            fund_data = self.engines["fund"].get_stock_fundamentals(symbol)
            stock_name = fund_data.get('name', symbol)
            try:
                industry_name, _ = self.engines["fund"].get_industry_peers(symbol)
            except Exception:
                industry_name = "未知"

            # 3. 快速路径：预测缓存命中则跳过视觉检索（大幅加速）
            today_key = (symbol, datetime.now().strftime("%Y%m%d"))
            if today_key in self.prediction_cache:
                win_rate = float(self.prediction_cache[today_key])
                matches = []
            else:
                # 视觉搜索（简化：只取Top-10，不生成对比图）
                date_str = df.index[-1].strftime("%Y%m%d")
                q_p = self._find_existing_kline_image(symbol, date_str)
                if not q_p:
                    q_p = os.path.join(PROJECT_ROOT, "data", f"temp_batch_{symbol}.png")
                    mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
                    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
                    mpf.plot(df.tail(20), type='candle', style=s,
                             savefig=dict(fname=q_p, dpi=50), figsize=(3, 3), axisoff=True)

                matches = self.engines["vision"].search_similar_patterns(q_p, top_k=10)

            # 4. 混合胜率计算（Triple Barrier + 传统胜率）
                win_rate_result = self.kline_factor_calc.calculate_hybrid_win_rate(
                    matches, query_symbol=symbol, query_date=datetime.now().strftime("%Y%m%d")
                )
                win_rate = win_rate_result['hybrid_win_rate']
            dist = self.kline_factor_calc.calculate_return_distribution(matches, horizon_days=20) if matches else {"valid": False}

            # 5. 技术指标
            df_f = self.engines["factor"]._add_technical_indicators(df)

            # 6. 评分
            total_score, initial_action, s_details = self.engines["factor"].get_scorecard(
                win_rate, df_f.iloc[-1], fund_data
            )

            # 7. 简化AI分析（只获取核心决策）
            try:
                report = self.engines["agent"].analyze(
                    symbol, total_score, initial_action,
                    {"win_rate": win_rate},
                    df_f.iloc[-1].to_dict(), fund_data, ""
                )
                action = report.action
                confidence = report.confidence
                reasoning = report.reasoning[:150]  # 截断
            except:
                action = initial_action
                confidence = int(total_score * 10)
                reasoning = "快速分析模式"

            # 8. 预期收益（分布估计优先）
            if dist and dist.get("valid"):
                expected_return = dist.get("mean", 0.0)
            else:
                expected_return = self._estimate_return(win_rate, df_f.iloc[-1])

            return {
                "symbol": symbol,
                "name": stock_name,
                "score": round(total_score, 1),
                "action": action,
                "confidence": confidence,
                "win_rate": round(win_rate, 1),
                "expected_return": round(expected_return, 2),
                "cvar": round(dist.get("cvar", 0.0), 2) if dist and dist.get("valid") else 0.0,
                "roe": round(fund_data.get('roe', 0), 2),
                "pe_ttm": round(fund_data.get('pe_ttm', 0), 2),
                "market_cap": round(fund_data.get('total_mv', 0), 2),
                "industry": industry_name,
                "reasoning": reasoning,
                "details": s_details,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {
                "symbol": symbol,
                "error": str(e),
                "score": 0,
                "action": "ERROR"
            }

    def _get_win_rate_fast(self, symbol, matches, df):
        """快速计算胜率（优先使用缓存）"""
        # 尝试从缓存获取
        today_str = datetime.now().strftime("%Y%m%d")
        cache_key = (symbol, today_str)

        if cache_key in self.prediction_cache:
            return float(self.prediction_cache[cache_key])

        # 如果没有缓存，简化计算（只检查Top-3匹配）
        if not matches:
            return 50.0

        # 简化：基于相似度加权
        total_weight = sum(m.get('score', 0) for m in matches)
        if total_weight > 0:
            # 假设相似度高的形态胜率更高（简化假设）
            avg_similarity = np.mean([m.get('score', 0) for m in matches])
            # 映射到50-70%范围
            win_rate = 50 + (avg_similarity * 20)
            return min(70, max(50, win_rate))

        return 50.0

    def _get_stock_cached(self, symbol):
        """简单缓存，减少重复IO"""
        if symbol in self._data_cache:
            return self._data_cache[symbol]
        df = self.engines["loader"].get_stock_data(symbol)
        self._data_cache[symbol] = df
        # 控制缓存大小
        if len(self._data_cache) > 50:
            self._data_cache.pop(next(iter(self._data_cache)))
        return df

    def _find_existing_kline_image(self, symbol: str, date_str: str):
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
        if matches:
            return matches[0]
        # 回退：取该股票最新的一张图
        pattern2 = os.path.join(img_base, "**", f"{symbol}*.png")
        all_imgs = glob.glob(pattern2, recursive=True)
        if not all_imgs:
            return None
        def _extract_date(p):
            base = os.path.basename(p).replace(".png", "")
            parts = base.split("_")
            if len(parts) >= 2:
                return parts[1]
            return "00000000"
        all_imgs.sort(key=_extract_date, reverse=True)
        return all_imgs[0]

    def _estimate_return(self, win_rate, factor_row):
        """估算预期收益"""
        # 简化模型：基于胜率和趋势
        base_return = (win_rate - 50) * 0.1  # 胜率每增加1%，预期收益+0.1%

        # 趋势加成
        if factor_row.get('MA_Signal', 0) > 0:
            base_return += 1.0

        # MACD加成
        if factor_row.get('MACD_Hist', 0) > 0:
            base_return += 0.5

        return base_return