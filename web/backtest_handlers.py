"""回测处理模块 - 工业级优化"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from src.strategies.transaction_cost import AdvancedTransactionCost
from src.utils.walk_forward import WalkForwardValidator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(symbol, bt_start, bt_end, bt_cap, bt_ma, bt_stop, bt_vision, 
                 bt_validation, wf_train_months, wf_test_months, eng, PROJECT_ROOT,
                 enable_stress_test: bool = False):
    """
    回测核心逻辑
    
    Args:
        symbol: 股票代码
        bt_start: 开始日期
        bt_end: 结束日期
        bt_cap: 初始资金
        bt_ma: MA周期
        bt_stop: 止损百分比
        bt_vision: AI胜率阈值
        bt_validation: 验证模式
        wf_train_months: Walk-Forward训练期（月）
        wf_test_months: Walk-Forward测试期（月）
        eng: 引擎字典
        PROJECT_ROOT: 项目根目录
        enable_stress_test: 是否启用Stress Testing
    """
    use_wf = bt_validation == "Walk-Forward验证（严格）"
    import streamlit as st
    
    try:
        logger.info(f"开始回测: {symbol}, 模式: {bt_validation}")

        # ---- 统一日期类型（修复 Timestamp vs date 比较报错）----
        # Streamlit 的 st.date_input 返回 datetime.date；而 df.index 是 Timestamp。
        # 这里强制转成 pandas Timestamp，并把 end_date 扩展到当天结束，避免边界缺失。
        bt_start_ts = pd.Timestamp(bt_start).normalize()
        bt_end_ts = pd.Timestamp(bt_end)
        if not isinstance(bt_end, datetime):
            bt_end_ts = bt_end_ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        
        with st.spinner("回测中..." if not use_wf else f"Walk-Forward验证中（{wf_train_months}月/{wf_test_months}月）..."):
            df = eng["loader"].get_stock_data(symbol, start_date=bt_start.strftime("%Y%m%d"))
            if df.empty:
                st.error("数据获取失败")
                logger.error(f"数据获取失败: {symbol}")
                return
            
            df.index = pd.to_datetime(df.index)
            df = df.loc[(df.index >= bt_start_ts) & (df.index <= bt_end_ts)].copy()
            
            if df.empty:
                st.error(f"日期范围 {bt_start} 至 {bt_end} 内无数据")
                logger.error(f"日期范围内无数据: {symbol}")
                return
            
            if use_wf:
                _run_walk_forward(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, 
                                wf_train_months, wf_test_months, eng, PROJECT_ROOT)
            else:
                _run_simple_backtest(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, eng, PROJECT_ROOT)
            
            # Stress Testing（如果启用）
            if enable_stress_test:
                _run_stress_test(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, eng, PROJECT_ROOT)
                
    except Exception as e:
        logger.exception(f"回测异常: {symbol}")
        st.error(f"回测失败: {str(e)}")
        import traceback
        with st.expander("查看详细错误"):
            st.code(traceback.format_exc())

def _run_walk_forward(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, 
                      wf_train_months, wf_test_months, eng, PROJECT_ROOT):
    """Walk-Forward验证"""
    import streamlit as st
    from src.strategies.transaction_cost import AdvancedTransactionCost
    
    train_days = wf_train_months * 21
    test_days = wf_test_months * 21
    validator = WalkForwardValidator(train_period=train_days, test_period=test_days, step_size=test_days)
    cost_calc = AdvancedTransactionCost()
    vision_map = _load_vision_map(symbol, PROJECT_ROOT)
    
    all_results = []
    for fold_id, split in enumerate(validator.split(df), 1):
        train_data = df.iloc[split.train_indices]
        test_data = df.iloc[split.test_indices]
        
        test_data = _calc_indicators(test_data, bt_ma)
        if test_data.empty:
            continue
        
        ret, bench_ret, trades = _backtest_loop(test_data, symbol, bt_cap, bt_ma, bt_stop, 
                                                bt_vision, vision_map, cost_calc)
        
        all_results.append({
            'fold': fold_id,
            'train_start': _safe_date_str(split.train_start),
            'train_end': _safe_date_str(split.train_end),
            'test_start': _safe_date_str(split.test_start),
            'test_end': _safe_date_str(split.test_end),
            'return': ret,
            'benchmark': bench_ret,
            'alpha': ret - bench_ret,
            'trades': trades
        })
    
    if all_results:
        _display_wf_results(all_results, wf_train_months, wf_test_months)

def _run_simple_backtest(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, eng, PROJECT_ROOT):
    """简单回测"""
    import streamlit as st
    from src.strategies.transaction_cost import AdvancedTransactionCost
    
    if len(df) < 50:
        st.error("数据不足")
        return
    
    df = _calc_indicators(df, bt_ma)
    if df.empty:
        st.error("数据计算失败")
        return
    
    cost_calc = AdvancedTransactionCost()
    vision_map = _load_vision_map(symbol, PROJECT_ROOT)
    
    ret, bench_ret, trades, equity, cost_summary = _backtest_loop(
        df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, vision_map, cost_calc,
        return_equity=True, return_costs=True
    )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=equity, name="VQ策略", line=dict(color='#ff4b4b', width=2)))
    bench = (df['Close'] / df['Close'].iloc[0]) * bt_cap
    fig.add_trace(go.Scatter(x=df.index, y=bench, name="基准", line=dict(color='gray', dash='dash')))
    fig.update_layout(title="策略收益曲线", height=400)
    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
    
    alpha = ret - bench_ret
    # 工业级 Sharpe：基于日收益率序列计算年化 Sharpe（允许为负，不应强行显示 N/A）
    try:
        eq = pd.Series(equity, index=df.index)
        daily_ret = eq.pct_change().dropna()
        if len(daily_ret) >= 2 and float(daily_ret.std()) > 0:
            sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std())
        else:
            sharpe = np.nan
    except Exception:
        sharpe = np.nan
    
    # 最大回撤（Q2B：15%阈值）
    try:
        roll_max = eq.cummax()
        drawdown = (eq / roll_max - 1.0).min()
        max_dd = float(drawdown) if pd.notna(drawdown) else 0.0
    except Exception:
        max_dd = 0.0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("策略收益率", f"{ret:.2f}%", delta=f"{alpha:.2f}% vs 基准")
    col2.metric("Alpha", f"{alpha:.2f}%", delta="超额收益" if alpha > 0 else "跑输基准")
    col3.metric("交易次数", f"{trades}次")
    col4.metric("夏普比率", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")
    col5.metric("最大回撤", f"{max_dd*100:.2f}%")
    if max_dd <= -0.15:
        st.warning("⚠️ 最大回撤超过 15%，风险偏高（按你的约束阈值提示）")

    # 多基线对比 + 统计检验（Q14D + Q18）
    baseline_df, baseline_returns = _compute_baseline_returns(df)
    if not baseline_df.empty:
        st.subheader("📊 基线策略对比（多基线）")
        st.dataframe(baseline_df, use_container_width=True, hide_index=True)
        # 统计显著性（与各基线的差异t检验）
        try:
            import scipy.stats as stats
            test_rows = []
            for name, b_ret in baseline_returns.items():
                aligned = pd.concat([daily_ret, b_ret], axis=1).dropna()
                if len(aligned) >= 20:
                    t_stat, p_val = stats.ttest_rel(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    test_rows.append({"基线": name, "t值": round(t_stat, 3), "p值": round(p_val, 4)})
            if test_rows:
                st.caption("统计检验（配对t检验，p值越小代表差异显著）")
                st.dataframe(pd.DataFrame(test_rows), hide_index=True, use_container_width=True)
        except Exception:
            pass

    # Transaction Cost 明细（Q4：ABCD）
    if cost_summary:
        with st.expander("💸 交易成本明细", expanded=False):
            st.json(cost_summary)

def _calc_indicators(df, bt_ma):
    """计算技术指标"""
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(bt_ma).mean()
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (exp12 - exp26) * 2
    if 'Volume' not in df.columns:
        df['Volume'] = df['Close'] * 1000000
    return df.dropna()

def _load_vision_map(symbol, PROJECT_ROOT):
    """加载AI胜率数据"""
    pred_path = os.path.join(PROJECT_ROOT, "data", "indices", "prediction_cache.csv")
    if not os.path.exists(pred_path):
        return {}
    try:
        pdf = pd.read_csv(pred_path)
        pdf['date'] = pdf['date'].astype(str).str.replace('-', '')
        pdf['symbol'] = pdf['symbol'].astype(str).str.zfill(6)
        return pdf.set_index(['symbol', 'date'])['pred_win_rate'].to_dict()
    except:
        return {}

def _backtest_loop(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, vision_map, cost_calc,
                   return_equity=False, return_costs=False):
    """回测循环核心逻辑"""
    cash, shares, equity = bt_cap, 0, []
    entry_price = 0.0
    max_turnover = 0.20
    prev_close = None
    last_buy_idx = None  # T+1约束：买入当天不能卖出
    trades_count = 0
    cost_summary = {
        "total_cost": 0.0,
        "commission": 0.0,
        "slippage": 0.0,
        "market_impact": 0.0,
        "opportunity_cost": 0.0,
        "trade_count": 0
    }
    
    for _, row in df.iterrows():
        # 先取价格，再用作缺省值（修复 UnboundLocalError: p）
        p = float(row["Close"])
        ma20 = float(row.get("MA20", p))
        ma60 = float(row.get("MA60", p))
        macd = float(row.get("MACD", 0))
        date_str = row.name.strftime("%Y%m%d")
        ai_win = vision_map.get((symbol, date_str), 50.0)
        volume = float(row.get('Volume', df['Close'].mean() * 1000000))

        # A股涨跌停与停牌处理（Q9D）
        daily_ret = None
        if prev_close is not None and prev_close > 0:
            daily_ret = (p - prev_close) / prev_close
        is_limit_up = daily_ret is not None and daily_ret >= 0.095
        is_limit_down = daily_ret is not None and daily_ret <= -0.095
        is_suspended = (not np.isfinite(volume)) or volume <= 0
        
        target_pos = _calc_target_position(p, ma60, ma20, macd, ai_win, bt_vision)
        total_assets = cash + shares * p
        target_shares = int(total_assets * target_pos / p) if p > 0 else 0
        diff = target_shares - shares
        
        if total_assets > 0 and abs(diff * p) / total_assets > max_turnover:
            max_trade = int(total_assets * max_turnover / p)
            diff = max_trade if diff > 0 else -max_trade
        
        if abs(diff * p) > total_assets * 0.1:
            if is_suspended:
                equity.append(cash + shares * p)
                prev_close = p
                continue
            if diff > 0 and is_limit_up:
                equity.append(cash + shares * p)
                prev_close = p
                continue
            if diff < 0 and is_limit_down:
                equity.append(cash + shares * p)
                prev_close = p
                continue
            # T+1：买入当天不允许卖出
            if diff < 0 and last_buy_idx is not None and row.name <= last_buy_idx:
                equity.append(cash + shares * p)
                prev_close = p
                continue

            trade_value = abs(diff * p)
            volatility = df['Close'].pct_change().std() if len(df) > 1 else 0.02
            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.02
            
            try:
                cost_result = cost_calc.calculate_cost(trade_value, p, max(volume, 1), volatility, diff > 0)
                total_cost = cost_result.get('total_cost', trade_value * 0.001)
                # 动态交易成本（波动/流动性敏感）
                liquidity_ratio = abs(diff) / max(volume, 1)
                cost_mult = 1.0 + min(1.5, max(0.0, volatility - 0.2)) + min(1.0, liquidity_ratio * 10)
                total_cost *= cost_mult
            except:
                total_cost = trade_value * 0.001
                cost_result = {}
            
            if diff > 0 and cash >= diff * p + total_cost:
                cash -= diff * p + total_cost
                shares += diff
                if entry_price == 0:
                    entry_price = p
                last_buy_idx = row.name
                trades_count += 1
            elif diff < 0:
                pnl = (p - entry_price) / entry_price if entry_price > 0 and shares > 0 else 0
                if pnl < -bt_stop / 100:
                    diff = -shares
                cash += abs(diff) * p - total_cost
                shares += diff
                if shares == 0:
                    entry_price = 0
                trades_count += 1

            if cost_result:
                cost_summary["total_cost"] += float(cost_result.get("total_cost", 0))
                cost_summary["commission"] += float(cost_result.get("commission", 0))
                cost_summary["slippage"] += float(cost_result.get("slippage", 0))
                cost_summary["market_impact"] += float(cost_result.get("market_impact", 0))
                cost_summary["opportunity_cost"] += float(cost_result.get("opportunity_cost", 0))
                cost_summary["trade_count"] += 1
        
        equity.append(cash + shares * p)
        prev_close = p
    
    ret = (equity[-1] - bt_cap) / bt_cap * 100 if equity else 0
    bench_ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100 if len(df) > 0 else 0
    trades = trades_count if trades_count > 0 else (sum(1 for e in equity if e != equity[0]) if len(equity) > 1 else 0)

    if return_equity and return_costs:
        return ret, bench_ret, trades, equity, cost_summary
    if return_equity:
        return ret, bench_ret, trades, equity
    return (ret, bench_ret, trades)

def _compute_baseline_returns(df):
    """多基线收益率对比（Q14D）"""
    try:
        close = df["Close"].astype(float)
        returns = close.pct_change().fillna(0.0)

        # Buy & Hold
        buy_hold = (1.0 + returns).cumprod().iloc[-1] - 1.0

        # MA交叉
        ma_signal = (df["MA20"] > df["MA60"]).astype(float)
        ma_ret = (1.0 + returns * ma_signal.shift(1).fillna(0.0)).cumprod().iloc[-1] - 1.0

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        rsi_signal = (rsi < 30).astype(float).fillna(0.0)
        rsi_ret = (1.0 + returns * rsi_signal.shift(1).fillna(0.0)).cumprod().iloc[-1] - 1.0

        # MACD
        macd_signal = (df["MACD"] > 0).astype(float)
        macd_ret = (1.0 + returns * macd_signal.shift(1).fillna(0.0)).cumprod().iloc[-1] - 1.0

        # Momentum(20)
        mom = close.pct_change(20)
        mom_signal = (mom > 0).astype(float).fillna(0.0)
        mom_ret = (1.0 + returns * mom_signal.shift(1).fillna(0.0)).cumprod().iloc[-1] - 1.0

        data = [
            {"基线": "Buy&Hold", "收益率": f"{buy_hold*100:.2f}%"},
            {"基线": "MA 20/60", "收益率": f"{ma_ret*100:.2f}%"},
            {"基线": "RSI(14)", "收益率": f"{rsi_ret*100:.2f}%"},
            {"基线": "MACD>0", "收益率": f"{macd_ret*100:.2f}%"},
            {"基线": "Momentum(20)", "收益率": f"{mom_ret*100:.2f}%"},
        ]
        baseline_returns = {
            "Buy&Hold": returns,
            "MA 20/60": returns * ma_signal.shift(1).fillna(0.0),
            "RSI(14)": returns * rsi_signal.shift(1).fillna(0.0),
            "MACD>0": returns * macd_signal.shift(1).fillna(0.0),
            "Momentum(20)": returns * mom_signal.shift(1).fillna(0.0),
        }
        return pd.DataFrame(data), baseline_returns
    except Exception:
        return pd.DataFrame(), {}

def _calc_target_position(p, ma60, ma20, macd, ai_win, bt_vision):
    """计算目标仓位"""
    if p > ma60:
        return 1.0 if (macd > 0 or p > ma20) else (0.81 if ai_win >= bt_vision else 0.03)
    else:
        return 0.50 if ai_win >= bt_vision + 2 else 0.03

def _safe_date_str(date_obj):
    """安全日期格式化"""
    try:
        return date_obj.strftime('%Y-%m-%d') if hasattr(date_obj, 'strftime') else str(date_obj)
    except:
        return str(date_obj)

def _display_wf_results(all_results, wf_train_months, wf_test_months):
    """显示Walk-Forward结果"""
    import streamlit as st
    
    results_df = pd.DataFrame(all_results)
    st.markdown("### Walk-Forward验证结果")
    st.dataframe(results_df, use_container_width=True, height=300)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df['fold'], y=results_df['return'], mode='lines+markers',
                            name='策略收益', line=dict(color='#ff4b4b', width=2)))
    fig.add_trace(go.Scatter(x=results_df['fold'], y=results_df['benchmark'], mode='lines+markers',
                            name='基准收益', line=dict(color='gray', dash='dash')))
    fig.update_layout(title=f"Walk-Forward验证结果（{len(all_results)}个fold）",
                     xaxis_title="Fold", yaxis_title="收益率 (%)", height=400)
    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
    
    avg_return = results_df['return'].mean()
    avg_alpha = results_df['alpha'].mean()
    std_return = results_df['return'].std()
    win_rate = (results_df['return'] > 0).sum() / len(results_df) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("平均收益率", f"{avg_return:.2f}%", delta=f"±{std_return:.2f}%")
    col2.metric("平均Alpha", f"{avg_alpha:.2f}%", delta="超额收益" if avg_alpha > 0 else "跑输基准")
    col3.metric("胜率", f"{win_rate:.1f}%", delta="优秀" if win_rate > 60 else "一般")
    col4.metric("Fold数量", f"{len(all_results)}个")

def run_stratified_backtest_batch(symbols, eng, bt_ma=60, bt_stop=8, bt_vision=57):
    """
    分层回测：行业/市值/风格 + 显著性检验
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    from src.backtest.stock_stratifier import StockStratifier
    from src.strategies.transaction_cost import AdvancedTransactionCost

    rows = []
    loader = eng["loader"]
    for sym in symbols:
        try:
            data = eng["fund"].get_stock_fundamentals(sym)
            ind, _ = eng["fund"].get_industry_peers(sym)
            df = loader.get_stock_data(sym)
            if df is None or df.empty or len(df) < 80:
                continue
            df.index = pd.to_datetime(df.index)
            df = _calc_indicators(df, bt_ma)
            if df.empty:
                continue
            # 风格：动量 or 均值回归
            mom60 = (df["Close"].iloc[-1] / df["Close"].iloc[-60] - 1) if len(df) > 60 else 0.0
            style = "momentum" if mom60 > 0 else "mean_reversion"
            rows.append({
                "symbol": sym,
                "market_cap": data.get("total_mv", 0),
                "industry": ind or "未知",
                "style": style
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    strat_df = pd.DataFrame(rows)
    stratifier = StockStratifier()
    strat_df = stratifier.stratify_combined(strat_df, market_cap_col="market_cap", industry_col="industry")
    strat_df["stratum"] = strat_df["stratum"].astype(str) + "_" + strat_df["style"].astype(str)

    results = []
    cost_calc = AdvancedTransactionCost()
    vision_map = {}
    for stratum in strat_df["stratum"].unique():
        sub = strat_df[strat_df["stratum"] == stratum]
        rets = []
        alphas = []
        for _, row in sub.iterrows():
            sym = row["symbol"]
            try:
                df = loader.get_stock_data(sym)
                if df is None or df.empty or len(df) < 80:
                    continue
                df.index = pd.to_datetime(df.index)
                df = _calc_indicators(df, bt_ma)
                if df.empty:
                    continue
                # 用视觉阈值生成交易逻辑
                ret, bench_ret, _ = _backtest_loop(
                    df, sym, 100000, bt_ma, bt_stop, bt_vision, vision_map, cost_calc
                )
                rets.append(ret)
                alphas.append(ret - bench_ret)
            except Exception:
                continue
        if len(rets) == 0:
            continue
        # 统计显著性：alpha 是否显著 > 0
        t_stat, p_val = stats.ttest_1samp(alphas, 0) if len(alphas) >= 3 else (0.0, 1.0)
        results.append({
            "分层": stratum,
            "样本数": len(rets),
            "平均收益": round(float(np.mean(rets)), 2),
            "平均Alpha": round(float(np.mean(alphas)), 2),
            "p值": round(float(p_val), 4)
        })
    return pd.DataFrame(results)

def _run_stress_test(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, eng, PROJECT_ROOT):
    """Stress Testing - 极端市场条件测试"""
    import streamlit as st
    
    try:
        from src.backtest.stress_testing import StressTester
        
        st.divider()
        st.subheader("🔥 Stress Testing - 极端市场测试")
        
        with st.spinner("运行Stress测试中..."):
            tester = StressTester()
            
            # 生成交易信号（简化版，使用回测逻辑）
            df_indicators = _calc_indicators(df, bt_ma)
            if df_indicators.empty:
                st.warning("数据不足，无法进行Stress测试")
                return
            
            # 构建信号序列（简化：基于MA和AI胜率）
            vision_map = _load_vision_map(symbol, PROJECT_ROOT)
            signals = pd.Series(0.0, index=df_indicators.index)
            win_rates = pd.Series(50.0, index=df_indicators.index)
            
            for idx, row in df_indicators.iterrows():
                # 先取价格，再用作缺省值（修复 Stress Testing: p 未初始化）
                p = float(row["Close"])
                ma60 = float(row.get("MA60", p))
                ma20 = float(row.get("MA20", p))
                macd = float(row.get("MACD", 0))
                date_str = idx.strftime("%Y%m%d")
                ai_win = vision_map.get((symbol, date_str), 50.0)
                target_pos = _calc_target_position(p, ma60, ma20, macd, ai_win, bt_vision)
                signals.loc[idx] = 1.0 if target_pos > 0.5 else -1.0 if target_pos < 0.3 else 0.0
                win_rates.loc[idx] = ai_win
            
            # 运行Stress测试（选择关键场景）
            key_scenarios = ['financial_crisis_2008', 'covid_crash_2020', 'market_crash_2015']
            stress_results = {}
            
            for scenario_name in key_scenarios:
                try:
                    scenario = tester.scenarios.get(scenario_name)
                    if not scenario or not scenario.start_date:
                        continue
                    
                    # 检查数据是否包含该场景期间
                    scenario_start = pd.to_datetime(scenario.start_date)
                    scenario_end = pd.to_datetime(scenario.end_date)
                    
                    if scenario_start > df.index[-1] or scenario_end < df.index[0]:
                        continue
                    
                    stress_df = tester._extract_stress_period(df, scenario.start_date, scenario.end_date)
                    if stress_df.empty or len(stress_df) < 20:
                        continue
                    
                    # 简化回测（使用现有逻辑）
                    stress_indicators = _calc_indicators(stress_df, bt_ma)
                    if not stress_indicators.empty:
                        cost_calc = AdvancedTransactionCost()
                        ret, bench_ret, trades = _backtest_loop(
                            stress_indicators, symbol, bt_cap, bt_ma, bt_stop, 
                            bt_vision, vision_map, cost_calc
                        )
                        
                        stress_results[scenario.name] = {
                            'return': ret,
                            'benchmark': bench_ret,
                            'alpha': ret - bench_ret,
                            'trades': trades,
                            'period': f"{scenario.start_date} ~ {scenario.end_date}"
                        }
                except Exception as e:
                    logger.warning(f"Stress场景 {scenario_name} 测试失败: {e}")
                    continue
            
            # 显示结果
            if stress_results:
                st.markdown("#### Stress测试结果")
                stress_df = pd.DataFrame([
                    {
                        '场景': name,
                        '期间': result['period'],
                        '策略收益': f"{result['return']:.2f}%",
                        '基准收益': f"{result['benchmark']:.2f}%",
                        'Alpha': f"{result['alpha']:.2f}%",
                        '交易次数': result['trades']
                    }
                    for name, result in stress_results.items()
                ])
                st.dataframe(stress_df, use_container_width=True, hide_index=True)
                
                # 可视化
                fig = go.Figure()
                scenarios = list(stress_results.keys())
                returns = [stress_results[s]['return'] for s in scenarios]
                benchmarks = [stress_results[s]['benchmark'] for s in scenarios]
                
                fig.add_trace(go.Bar(x=scenarios, y=returns, name='策略收益', marker_color='#ff4b4b'))
                fig.add_trace(go.Bar(x=scenarios, y=benchmarks, name='基准收益', marker_color='gray'))
                fig.update_layout(title="Stress测试收益对比", height=300, barmode='group')
                st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
            else:
                # ---- 工业级兜底：样本内自动压力窗口（避免 2022-2026 数据无法测历史危机）----
                st.info("当前数据不包含预定义历史场景。已自动改用“样本内压力窗口”(最差回撤/最差滚动收益/最高波动)进行测试。")

                window = 60  # ~3个月交易日
                if len(df_indicators) < window + 10:
                    st.warning("数据量不足，无法进行样本内压力测试（需要更长区间）")
                    return

                close = df_indicators["Close"].astype(float)
                rets = close.pct_change().dropna()

                # 1) 最差滚动累计收益窗口（用几何累计收益）
                roll_cum = (1.0 + rets).rolling(window).apply(lambda x: float(np.prod(x) - 1.0), raw=True)
                worst_ends = roll_cum.nsmallest(3).dropna().index.tolist()

                # 2) 最高波动窗口（波动爆发）
                roll_vol = rets.rolling(window).std() * np.sqrt(252)
                vol_end = roll_vol.nlargest(1).dropna().index.tolist()

                # 3) “熔断”代理：单日最大下跌，取其窗口
                min_day = rets.nsmallest(1).dropna().index.tolist()

                # 4) 低流动性窗口（成交量滚动均值最低）
                vol_series = df_indicators.get("Volume", pd.Series(index=df_indicators.index, data=np.nan))
                roll_liq = vol_series.rolling(window).mean()
                low_liq_end = roll_liq.nsmallest(1).dropna().index.tolist()

                end_dates = []
                for d in worst_ends + vol_end + min_day + low_liq_end:
                    if d not in end_dates:
                        end_dates.append(d)

                auto_results = {}
                for j, end_dt in enumerate(end_dates, 1):
                    start_dt = end_dt - pd.Timedelta(days=window * 2)  # 用日历日放宽，后续按索引截取
                    # 对齐到实际交易日区间
                    segment = df_indicators.loc[(df_indicators.index >= start_dt) & (df_indicators.index <= end_dt)].copy()
                    if len(segment) < 20:
                        continue

                    cost_calc = AdvancedTransactionCost()
                    r, b, t = _backtest_loop(segment, symbol, bt_cap, bt_ma, bt_stop, bt_vision, vision_map, cost_calc)
                    if end_dt in vol_end:
                        label = "波动爆发窗口"
                    elif end_dt in min_day:
                        label = "熔断代理窗口"
                    elif end_dt in low_liq_end:
                        label = "低流动性窗口"
                    else:
                        label = f"最差滚动收益窗口#{j}"
                    auto_results[label] = {
                        "return": r,
                        "benchmark": b,
                        "alpha": r - b,
                        "trades": t,
                        "period": f"{segment.index[0].date()} ~ {segment.index[-1].date()}",
                    }

                if not auto_results:
                    st.warning("样本内压力窗口计算失败或数据不足")
                    return

                auto_df = pd.DataFrame([
                    {
                        "场景": name,
                        "期间": v["period"],
                        "策略收益": f'{v["return"]:.2f}%',
                        "基准收益": f'{v["benchmark"]:.2f}%',
                        "Alpha": f'{v["alpha"]:.2f}%',
                        "交易次数": v["trades"],
                    }
                    for name, v in auto_results.items()
                ])
                st.dataframe(auto_df, use_container_width=True, hide_index=True)

                fig = go.Figure()
                scenarios = list(auto_results.keys())
                returns = [auto_results[s]["return"] for s in scenarios]
                benchmarks = [auto_results[s]["benchmark"] for s in scenarios]
                fig.add_trace(go.Bar(x=scenarios, y=returns, name="策略收益", marker_color="#ff4b4b"))
                fig.add_trace(go.Bar(x=scenarios, y=benchmarks, name="基准收益", marker_color="gray"))
                fig.update_layout(title="样本内 Stress Testing 收益对比", height=320, barmode="group")
                st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
                
    except ImportError:
        st.warning("Stress Testing模块未找到，跳过Stress测试")
        logger.warning("Stress Testing模块导入失败")
    except Exception as e:
        logger.exception("Stress测试异常")
        st.warning(f"Stress测试失败: {str(e)}")

def run_stratified_backtest_batch(symbols, eng, bt_ma=60, bt_stop=8, bt_vision=57):
    """
    分层回测：行业/市值/风格 + 显著性检验
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    from src.backtest.stock_stratifier import StockStratifier
    from src.strategies.transaction_cost import AdvancedTransactionCost

    rows = []
    loader = eng["loader"]
    for sym in symbols:
        try:
            data = eng["fund"].get_stock_fundamentals(sym)
            ind, _ = eng["fund"].get_industry_peers(sym)
            df = loader.get_stock_data(sym)
            if df is None or df.empty or len(df) < 80:
                continue
            df.index = pd.to_datetime(df.index)
            df = _calc_indicators(df, bt_ma)
            if df.empty:
                continue
            # 风格：动量 or 均值回归
            mom60 = (df["Close"].iloc[-1] / df["Close"].iloc[-60] - 1) if len(df) > 60 else 0.0
            style = "momentum" if mom60 > 0 else "mean_reversion"
            rows.append({
                "symbol": sym,
                "market_cap": data.get("total_mv", 0),
                "industry": ind or "未知",
                "style": style
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    strat_df = pd.DataFrame(rows)
    stratifier = StockStratifier()
    strat_df = stratifier.stratify_combined(strat_df, market_cap_col="market_cap", industry_col="industry")
    strat_df["stratum"] = strat_df["stratum"].astype(str) + "_" + strat_df["style"].astype(str)

    results = []
    # 分层回测也使用简化成本，确保一致性
    cost_calc = _get_simplified_cost_calc()
    vision_map = {}
    for stratum in strat_df["stratum"].unique():
        sub = strat_df[strat_df["stratum"] == stratum]
        rets = []
        alphas = []
        for _, row in sub.iterrows():
            sym = row["symbol"]
            try:
                df = loader.get_stock_data(sym)
                if df is None or df.empty or len(df) < 80:
                    continue
                df.index = pd.to_datetime(df.index)
                df = _calc_indicators(df, bt_ma)
                if df.empty:
                    continue
                # 用视觉阈值生成交易逻辑
                ret, bench_ret, _ = _backtest_loop(
                    df, sym, 100000, bt_ma, bt_stop, bt_vision, vision_map, cost_calc, strict_t1=False
                )
                rets.append(ret)
                alphas.append(ret - bench_ret)
            except Exception:
                continue
        if len(rets) == 0:
            continue
        # 统计显著性：alpha 是否显著 > 0
        t_stat, p_val = stats.ttest_1samp(alphas, 0) if len(alphas) >= 3 else (0.0, 1.0)
        results.append({
            "分层": stratum,
            "样本数": len(rets),
            "平均收益": round(float(np.mean(rets)), 2),
            "平均Alpha": round(float(np.mean(alphas)), 2),
            "p值": round(float(p_val), 4)
        })
    return pd.DataFrame(results)
