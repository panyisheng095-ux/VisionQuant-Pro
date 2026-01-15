from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import pandas as pd

from src.data.data_loader import DataLoader
from src.strategies.kline_factor import KLineFactorCalculator
from src.factor_analysis.ic_analysis import ICAnalyzer


app = FastAPI(
    title="VisionQuant-Pro API",
    description="K线视觉学习因子投研服务（轻量版）",
    version="2.1.0",
)


class SingleStockRequest(BaseModel):
    symbol: str
    start_date: str = "20200101"
    end_date: str = datetime.now().strftime("%Y%m%d")


class FactorRequest(BaseModel):
    symbol: str
    start_date: str = "20200101"
    end_date: str = datetime.now().strftime("%Y%m%d")


@app.get("/health")
def health():
    return {"status": "ok", "service": "visionquant-pro"}


@app.post("/analyze_single_stock")
def analyze_single_stock(req: SingleStockRequest):
    loader = DataLoader()
    df = loader.get_stock_data(req.symbol, start_date=req.start_date, end_date=req.end_date)
    if df is None or df.empty:
        return {"symbol": req.symbol, "error": "data_empty"}

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
    loader = DataLoader()
    df = loader.get_stock_data(req.symbol, start_date=req.start_date, end_date=req.end_date)
    if df is None or df.empty or len(df) < 80:
        return {"symbol": req.symbol, "error": "data_insufficient"}

    df.index = pd.to_datetime(df.index)
    returns = df["Close"].pct_change().dropna()
    # 使用“混合胜率”因子作为代理（API轻量版）
    # 这里用滚动5日收益率作为简化代理，避免API依赖图像模型
    factor_values = df["Close"].pct_change(5).shift(-5).dropna()
    factor_values = factor_values.reindex(returns.index).dropna()
    returns = returns.reindex(factor_values.index)

    analyzer = ICAnalyzer(window=min(60, max(20, len(factor_values) // 2)))
    ic_result = analyzer.analyze(factor_values, returns)
    ic_series = ic_result.get("ic_series")

    return {
        "symbol": req.symbol,
        "ic_mean": ic_result.get("ic_mean"),
        "ic_ir": ic_result.get("ic_ir"),
        "points": int(len(ic_series)) if ic_series is not None else 0,
    }


@app.post("/backtest/buy_hold")
def backtest_buy_hold(req: SingleStockRequest):
    loader = DataLoader()
    df = loader.get_stock_data(req.symbol, start_date=req.start_date, end_date=req.end_date)
    if df is None or df.empty:
        return {"symbol": req.symbol, "error": "data_empty"}

    start = float(df["Close"].iloc[0])
    end = float(df["Close"].iloc[-1])
    ret = (end / start - 1) * 100
    return {"symbol": req.symbol, "buy_hold_return_pct": round(ret, 2)}
