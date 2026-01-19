from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import os
import time
import sys
import base64
import glob
import mplfinance as mpf

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_loader import DataLoader
from src.data.multi_scale_generator import MultiScaleChartGenerator
from src.data.news_harvester import NewsHarvester
from src.strategies.kline_factor import KLineFactorCalculator
from src.factor_analysis.ic_analysis import ICAnalyzer
from src.strategies.portfolio_optimizer import PortfolioOptimizer
from src.strategies.factor_mining import FactorMiner
from src.strategies.fundamental import FundamentalMiner
from src.strategies.batch_analyzer import BatchAnalyzer
from src.agent.quant_agent import QuantAgent
from src.models.vision_engine import VisionEngine
from src.utils.top10_analyzer import Top10Analyzer

app = FastAPI(
    title="VisionQuant-Pro API",
    description="K线视觉学习因子投研服务（v3.0）",
    version="3.0.0",
)

# 简单审计日志
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api_access.log")

# Web App (Vue CDN) 静态目录
WEBAPP_DIR = os.path.join(PROJECT_ROOT, "web", "webapp")
if os.path.isdir(WEBAPP_DIR):
    app.mount("/static", StaticFiles(directory=WEBAPP_DIR), name="static")

# 全局单例（减少加载开销）
_vision_engine = None
_data_loader = None
_fund_miner = None
_factor_miner = None
_news_harvester = None
_batch_analyzer = None
_portfolio_optimizer = None
_quant_agent = None

def get_vision_engine():
    global _vision_engine
    if _vision_engine is None:
        _vision_engine = VisionEngine()
    return _vision_engine

def get_data_loader():
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader


def get_fund_miner():
    global _fund_miner
    if _fund_miner is None:
        _fund_miner = FundamentalMiner()
    return _fund_miner


def get_factor_miner():
    global _factor_miner
    if _factor_miner is None:
        _factor_miner = FactorMiner()
    return _factor_miner


def get_news_harvester():
    global _news_harvester
    if _news_harvester is None:
        _news_harvester = NewsHarvester()
    return _news_harvester


def get_quant_agent():
    global _quant_agent
    if _quant_agent is None:
        _quant_agent = QuantAgent()
    return _quant_agent


def get_portfolio_optimizer():
    global _portfolio_optimizer
    if _portfolio_optimizer is None:
        _portfolio_optimizer = PortfolioOptimizer()
    return _portfolio_optimizer


def get_batch_analyzer():
    global _batch_analyzer
    if _batch_analyzer is None:
        engines = {
            "loader": get_data_loader(),
            "vision": get_vision_engine(),
            "fund": get_fund_miner(),
            "factor": get_factor_miner(),
            "agent": get_quant_agent(),
        }
        _batch_analyzer = BatchAnalyzer(engines)
    return _batch_analyzer


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _encode_image_b64(img_path: str):
    try:
        if not img_path or (not os.path.exists(img_path)):
            return None
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    except Exception:
        return None


def _find_existing_kline_image(symbol: str, date_str: str):
    symbol = str(symbol).zfill(6)
    date_n = str(date_str).replace("-", "")
    img_bases = [
        os.path.join(PROJECT_ROOT, "data", "images_v2"),
        os.path.join(PROJECT_ROOT, "data", "images"),
    ]
    for img_base in img_bases:
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
    all_imgs = []
    for img_base in img_bases:
        pattern2 = os.path.join(img_base, "**", f"{symbol}*.png")
        all_imgs.extend(glob.glob(pattern2, recursive=True))
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


def _render_window_kline(window_df: pd.DataFrame, out_path: str):
    try:
        if window_df is None or window_df.empty:
            return None
        mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
        mpf.plot(
            window_df,
            type='candle',
            style=s,
            savefig=dict(fname=out_path, dpi=50),
            figsize=(3, 3),
            axisoff=True
        )
        return out_path
    except Exception:
        return None


def _render_match_image(symbol: str, date_str: str, loader: DataLoader, out_path: str):
    try:
        df = loader.get_stock_data(symbol)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        dt = pd.to_datetime(str(date_str), errors="coerce")
        if dt is pd.NaT:
            return None
        if dt not in df.index:
            candidates = df.index[df.index <= dt]
            if len(candidates) == 0:
                return None
            dt = candidates.max()
        loc = df.index.get_loc(dt)
        start = max(0, loc - 19)
        window = df.iloc[start:loc + 1].copy()
        if len(window) < 20:
            return None
        return _render_window_kline(window, out_path)
    except Exception:
        return None


def _parse_target_date(df_index, date_str: Optional[str]):
    try:
        if date_str:
            dt = pd.to_datetime(str(date_str), errors="coerce")
            if dt is not pd.NaT:
                if dt in df_index:
                    return dt
                # 回退到不晚于目标日期的最近交易日
                candidates = df_index[df_index <= dt]
                if len(candidates) > 0:
                    return candidates.max()
        # 默认最后一天
        return df_index[-1]
    except Exception:
        return df_index[-1]


def _augment_matches(matches, query_img_path, query_prices, loader, vision_engine, tmp_dir):
    if not matches:
        return matches
    q_pix = vision_engine._load_pixel_vector(query_img_path)
    q_edge = vision_engine._load_edge_vector(query_img_path)
    for m in matches:
        sym = str(m.get("symbol", "")).zfill(6)
        date_str = m.get("date")
        if m.get("pixel_sim") is None or m.get("edge_sim") is None:
            path = vision_engine._resolve_image_path(m.get("path"), sym, date_str)
            if not path:
                tmp_path = os.path.join(tmp_dir, f"tmp_match_{sym}_{date_str}.png")
                path = _render_match_image(sym, date_str, loader, tmp_path)
            if path:
                v = vision_engine._load_pixel_vector(path)
                e = vision_engine._load_edge_vector(path)
                pix_cos = vision_engine._cosine_sim(q_pix, v)
                pix_corr = vision_engine._pearson_corr(q_pix, v)
                edge_cos = vision_engine._cosine_sim(q_edge, e) if q_edge is not None else None
                pix_cos = 0.0 if pix_cos is None else pix_cos
                pix_corr = 0.0 if pix_corr is None else pix_corr
                edge_cos = 0.0 if edge_cos is None else edge_cos
                pix_norm = (pix_cos + 1.0) / 2.0
                pix_corr_norm = (pix_corr + 1.0) / 2.0
                edge_norm = (edge_cos + 1.0) / 2.0
                visual_sim = 0.5 * pix_norm + 0.3 * pix_corr_norm + 0.2 * edge_norm
                m["pixel_sim"] = float(visual_sim)
                m["edge_sim"] = float(edge_norm)
    return matches


def _compute_future_trajectories(matches, loader, horizon: int = 5):
    trajectories = []
    labels = []
    for m in matches:
        try:
            sym = str(m.get("symbol", "")).zfill(6)
            date_str = m.get("date")
            df = loader.get_stock_data(sym)
            if df is None or df.empty:
                continue
            df.index = pd.to_datetime(df.index)
            dt = pd.to_datetime(date_str, errors="coerce")
            if dt is pd.NaT or dt not in df.index:
                continue
            loc = df.index.get_loc(dt)
            if loc + horizon >= len(df):
                continue
            subset = df.iloc[loc: loc + horizon + 1]["Close"].values
            norm_path = (subset / subset[0] - 1.0) * 100
            trajectories.append(norm_path.tolist())
            labels.append(f"{sym}({date_str})")
        except Exception:
            continue
    mean_path = None
    win_rate = None
    avg_ret = None
    if trajectories:
        arr = np.vstack(trajectories)
        mean_path = np.mean(arr, axis=0).tolist()
        win_rate = float(np.mean(arr[:, -1] > 0) * 100)
        avg_ret = float(np.mean(arr[:, -1]))
    return {
        "trajectories": trajectories,
        "labels": labels,
        "mean_path": mean_path,
        "win_rate": win_rate,
        "avg_ret": avg_ret
    }

@app.middleware("http")
async def audit_log(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    cost = (time.time() - start) * 1000
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.now().isoformat()} {request.method} {request.url.path} {response.status_code} {cost:.1f}ms\n")
    except Exception:
        pass
    return response


class SingleStockRequest(BaseModel):
    symbol: str
    start_date: str = "20100101"
    end_date: str = datetime.now().strftime("%Y%m%d")


class FactorRequest(BaseModel):
    symbol: str
    start_date: str = "20100101"
    end_date: str = datetime.now().strftime("%Y%m%d")
    robust: bool = True  # v3.0 新增: 是否使用稳健统计


class PortfolioRequest(BaseModel):
    symbols: list[str]
    risk_aversion: float = 1.0
    cvar_limit: float = 0.05


class VisualSearchRequest(BaseModel):
    symbol: str
    date: Optional[str] = None
    top_k: int = 10
    multi_scale: bool = True
    include_images: bool = True


class SingleOverviewRequest(BaseModel):
    symbol: str
    visual_win_rate: Optional[float] = None
    start_date: str = "20100101"
    end_date: str = datetime.now().strftime("%Y%m%d")


class BatchAnalyzeRequest(BaseModel):
    symbols: list[str]
    risk_aversion: float = 1.0
    cvar_limit: float = 0.05


@app.get("/health")
def health():
    return {"status": "ok", "service": "visionquant-pro", "version": "3.0"}


@app.get("/")
def serve_webapp():
    index_path = os.path.join(WEBAPP_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "Web app not built yet."}


@app.post("/analyze_single_stock")
def analyze_single_stock(req: SingleStockRequest):
    loader = get_data_loader()
    df = loader.get_stock_data(req.symbol, start_date=req.start_date, end_date=req.end_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data empty")

    df.index = pd.to_datetime(df.index)
    last_close = float(df["Close"].iloc[-1])
    ret_5d = float((df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100) if len(df) > 6 else 0.0

    return {
        "symbol": req.symbol,
        "last_close": last_close,
        "ret_5d_pct": round(ret_5d, 2),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "points": len(df),
    }


@app.post("/factor/ic")
def factor_ic(req: FactorRequest):
    loader = get_data_loader()
    df = loader.get_stock_data(req.symbol, start_date=req.start_date, end_date=req.end_date)
    if df is None or df.empty or len(df) < 80:
        raise HTTPException(status_code=400, detail="Data insufficient")

    df.index = pd.to_datetime(df.index)
    returns = df["Close"].pct_change().dropna()
    
    # 模拟因子值（这里简单用动量代替，实际应用应调用 VisionEngine 批量计算）
    factor_values = df["Close"].pct_change(20).shift(1).dropna()
    
    common_idx = factor_values.index.intersection(returns.index)
    if len(common_idx) < 20:
        raise HTTPException(status_code=400, detail="Not enough overlapping data")
        
    factor_values = factor_values.loc[common_idx]
    returns = returns.loc[common_idx]

    analyzer = ICAnalyzer(window=min(60, max(20, len(factor_values) // 2)))
    # v3.0: 传递 robust 参数
    if req.robust:
        # 这里实际上 calculate_rolling_ic 内部并没有直接暴露 use_robust 参数给 analyze 方法
        # 我们需要在 analyze 方法中支持 kwargs 或者更新 analyze 签名
        # 暂时直接调用 analyze，稳健性在 analyze 内部实现
        pass
        
    ic_result = analyzer.analyze(factor_values, returns)
    
    return {
        "symbol": req.symbol,
        "ic_mean": ic_result.get("ic_mean"),
        "ic_ir": ic_result.get("ic_ir"),
        "half_life": ic_result.get("half_life"),
        "stability": ic_result.get("stability_score"),
        "points": len(factor_values),
    }


@app.post("/portfolio/optimize")
def optimize_portfolio(req: PortfolioRequest):
    """v3.0: 组合优化接口"""
    loader = get_data_loader()
    optimizer = PortfolioOptimizer()
    
    # 构造虚构的 analysis_results (仅用于演示 API 通通)
    # 实际场景应先调用 batch_analyze
    mock_results = {}
    for sym in req.symbols:
        mock_results[sym] = {
            "score": np.random.uniform(4, 9),
            "win_rate": np.random.uniform(45, 65),
            "expected_return": np.random.uniform(5, 20),
            "action": "BUY" if np.random.random() > 0.5 else "WAIT"
        }
        
    try:
        result = optimizer.optimize_multi_tier_portfolio(
            mock_results, loader,
            risk_aversion=req.risk_aversion,
            cvar_limit=req.cvar_limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/single/overview")
def single_overview(req: SingleOverviewRequest):
    """Web迁移：单只股票概览（多因子看板 + 行业 + 新闻）"""
    loader = get_data_loader()
    fund = get_fund_miner()
    factor = get_factor_miner()
    news = get_news_harvester()

    df = loader.get_stock_data(req.symbol, start_date=req.start_date, end_date=req.end_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data empty")
    df.index = pd.to_datetime(df.index)

    # 质量报告
    quality_report = {}
    try:
        quality_report = loader.quality_checker.check_data_quality(df, req.symbol)
    except Exception:
        quality_report = {}

    # 基本面与行业
    fund_data = fund.get_stock_fundamentals(req.symbol)
    try:
        ind_name, peers_df = fund.get_industry_peers(req.symbol)
        peers = peers_df.to_dict(orient="records") if peers_df is not None else []
    except Exception:
        ind_name, peers = "未知", []

    # 技术指标
    df_f = factor._add_technical_indicators(df)
    last_row = df_f.iloc[-1]
    technical = {
        "MA60": float(last_row.get("MA60", 0)) if "MA60" in last_row else None,
        "MA_Signal": float(last_row.get("MA_Signal", 0)) if "MA_Signal" in last_row else None,
        "RSI": float(last_row.get("RSI", 0)) if "RSI" in last_row else None,
        "MACD_Hist": float(last_row.get("MACD_Hist", 0)) if "MACD_Hist" in last_row else None,
    }

    # 多因子评分
    visual_win_rate = req.visual_win_rate if req.visual_win_rate is not None else 50.0
    returns = df["Close"].pct_change().dropna()
    score, action, details = factor.get_scorecard(
        visual_win_rate, last_row, fund_data, returns=returns
    )

    # 新闻
    try:
        news_text = news.get_latest_news(req.symbol)
    except Exception:
        news_text = "暂无新闻数据"

    return {
        "symbol": str(req.symbol).zfill(6),
        "name": fund_data.get("name") or "",
        "fundamentals": fund_data,
        "industry": {"name": ind_name, "peers": peers},
        "technical": technical,
        "scorecard": {"score": score, "action": action, "details": details},
        "quality_report": quality_report,
        "news": news_text,
    }


@app.post("/api/batch/analyze")
def batch_analyze(req: BatchAnalyzeRequest):
    """Web迁移：批量组合分析（核心+备选 + 权重 + 风控）"""
    symbols = [str(s).strip().zfill(6) for s in req.symbols if str(s).strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="Symbols empty")
    if len(symbols) > 30:
        symbols = symbols[:30]

    batch = get_batch_analyzer()
    loader = get_data_loader()
    optimizer = get_portfolio_optimizer()

    results = batch.analyze_batch(symbols)
    if not results:
        raise HTTPException(status_code=500, detail="Batch analyze failed")

    multi_tier = optimizer.optimize_multi_tier_portfolio(
        results, loader, min_weight=0.05, max_weight=0.25, max_positions=10,
        risk_aversion=req.risk_aversion, cvar_limit=req.cvar_limit
    )
    core_weights = multi_tier.get("core", {})
    enhanced_weights = multi_tier.get("enhanced", {})
    combined = {}
    combined.update(core_weights)
    combined.update(enhanced_weights)

    metrics = {}
    try:
        metrics = optimizer.calculate_portfolio_metrics(combined, results, loader)
    except Exception:
        metrics = {}

    return {
        "results": results,
        "tier_info": multi_tier.get("tier_info", {}),
        "weights": {
            "core": core_weights,
            "enhanced": enhanced_weights,
            "combined": combined
        },
        "metrics": metrics
    }


@app.post("/api/visual/search")
def visual_search(req: VisualSearchRequest):
    """Web迁移：视觉检索核心API（DTW主导）"""
    loader = get_data_loader()
    vision = get_vision_engine()

    df = loader.get_stock_data(req.symbol)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data empty")
    df.index = pd.to_datetime(df.index)

    target_dt = _parse_target_date(df.index, req.date)
    date_str = target_dt.strftime("%Y%m%d")

    loc = df.index.get_loc(target_dt)
    start = max(0, loc - 19)
    window_df = df.iloc[start:loc + 1]
    if len(window_df) < 20:
        raise HTTPException(status_code=400, detail="Data insufficient (need >=20 bars)")

    tmp_dir = _ensure_dir(os.path.join(PROJECT_ROOT, "data", "temp_api"))
    tag = f"{req.symbol}_{date_str}_{int(time.time())}"

    # Query图像：优先使用历史图
    q_img = _find_existing_kline_image(req.symbol, date_str)
    if not q_img:
        q_img = _render_window_kline(window_df, os.path.join(tmp_dir, f"query_{tag}.png"))

    query_prices = window_df["Close"].values

    # 多尺度检索（可选）
    try:
        if req.multi_scale:
            df_cut = df.loc[:target_dt]
            gen = MultiScaleChartGenerator(figsize=(3, 3), dpi=50)
            q_week = os.path.join(tmp_dir, f"query_week_{tag}.png")
            q_month = os.path.join(tmp_dir, f"query_month_{tag}.png")
            gen.generate_weekly_chart(df_cut, weeks=20, output_path=q_week)
            gen.generate_monthly_chart(df_cut, months=20, output_path=q_month)
            img_paths = {"daily": q_img, "weekly": q_week, "monthly": q_month}
            matches = vision.search_multi_scale_patterns(img_paths, top_k=req.top_k, query_prices=query_prices)
        else:
            matches = vision.search_similar_patterns(q_img, top_k=req.top_k, query_prices=query_prices)
    except Exception:
        matches = vision.search_similar_patterns(q_img, top_k=req.top_k, query_prices=query_prices)

    matches = _augment_matches(matches, q_img, query_prices, loader, vision, tmp_dir)

    # 混合胜率
    hybrid_win_rate = None
    hybrid_detail = None
    try:
        kline_factor_calc = KLineFactorCalculator(data_loader=loader)
        hybrid_detail = kline_factor_calc.calculate_hybrid_win_rate(
            matches,
            query_symbol=req.symbol,
            query_date=date_str
        )
        if isinstance(hybrid_detail, dict):
            hybrid_win_rate = hybrid_detail.get("hybrid_win_rate")
    except Exception:
        hybrid_win_rate = None
        hybrid_detail = None

    # 传统胜率（未来5日收益>0）
    price_df_cache = {}
    wins = 0
    total = 0
    future_returns = []
    for m in matches:
        sym = str(m.get("symbol", "")).zfill(6)
        dt = pd.to_datetime(m.get("date"), errors="coerce")
        if dt is pd.NaT:
            continue
        if sym not in price_df_cache:
            dfi = loader.get_stock_data(sym)
            if dfi is None or dfi.empty:
                price_df_cache[sym] = None
            else:
                dfi.index = pd.to_datetime(dfi.index)
                price_df_cache[sym] = dfi
        dfi = price_df_cache[sym]
        if dfi is None or (dt not in dfi.index):
            continue
        loc_i = dfi.index.get_loc(dt)
        if loc_i + 5 >= len(dfi):
            continue
        ret = (dfi["Close"].iloc[loc_i + 5] / dfi["Close"].iloc[loc_i] - 1.0) * 100
        future_returns.append(ret)
        total += 1
        if ret > 0:
            wins += 1
    traditional_win_rate = round(wins / total * 100, 2) if total > 0 else None
    avg_future_ret = round(float(np.mean(future_returns)), 2) if future_returns else None

    # 未来轨迹与均值路径
    traj_info = _compute_future_trajectories(matches, loader, horizon=5)

    # Top10多期收益/分布估计
    try:
        analyzer = Top10Analyzer(loader)
        mh_stats = analyzer.analyze_multi_horizon(matches, horizons=[5, 10, 20])
        dist_stats = analyzer.return_distribution(matches, future_days=20)
    except Exception:
        mh_stats, dist_stats = {}, {}

    # 输出结果
    result_matches = []
    for m in matches:
        sym = str(m.get("symbol", "")).zfill(6)
        date_m = m.get("date")
        img_path = vision._resolve_image_path(m.get("path"), sym, date_m)
        if not img_path:
            img_path = _render_match_image(sym, date_m, loader, os.path.join(tmp_dir, f"match_{sym}_{date_m}.png"))
        result_matches.append({
            "symbol": sym,
            "date": date_m,
            "score": m.get("score"),
            "sim_score": m.get("sim_score"),
            "correlation": m.get("correlation"),
            "ret_corr": m.get("ret_corr"),
            "dtw_sim": m.get("dtw_sim"),
            "feature_sim": m.get("feature_sim"),
            "pixel_sim": m.get("pixel_sim"),
            "edge_sim": m.get("edge_sim"),
            "image_b64": _encode_image_b64(img_path) if req.include_images else None,
        })

    return {
        "query": {
            "symbol": req.symbol,
            "date": date_str,
            "image_b64": _encode_image_b64(q_img) if req.include_images else None,
        },
        "metrics": {
            "traditional_win_rate": traditional_win_rate,
            "hybrid_win_rate": hybrid_win_rate,
            "avg_future_ret_5d_pct": avg_future_ret,
        },
        "trajectories": traj_info,
        "multi_horizon": mh_stats,
        "distribution": dist_stats,
        "hybrid_detail": hybrid_detail,
        "matches": result_matches,
    }
