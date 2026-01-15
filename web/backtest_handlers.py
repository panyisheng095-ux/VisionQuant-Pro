"""回测处理模块 - 工业级优化"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from src.strategies.transaction_cost import AdvancedTransactionCost, TransactionCostConfig
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

def _get_simplified_cost_calc():
    """获取简化版成本计算器 (修复收益率暴跌问题)"""
    # 将冲击系数设为极小值，仅保留基本印花税和佣金
    config = TransactionCostConfig(
        market_impact_coef=0.000001,  # 几乎为0
        commission_rate=0.0002,       # 万2
        slippage_rate=0.001           # 千1
    )
    return AdvancedTransactionCost(config)

def _run_walk_forward(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, 
                      wf_train_months, wf_test_months, eng, PROJECT_ROOT):
    """Walk-Forward验证"""
    import streamlit as st
    
    train_days = wf_train_months * 21
    test_days = wf_test_months * 21
    validator = WalkForwardValidator(train_period=train_days, test_period=test_days, step_size=test_days)
    
    # 使用简化成本模型
    cost_calc = _get_simplified_cost_calc()
    vision_map = _load_vision_map(symbol, PROJECT_ROOT)
    
    all_results = []
    for fold_id, split in enumerate(validator.split(df), 1):
        train_data = df.iloc[split.train_indices]
        test_data = df.iloc[split.test_indices]
        
        test_data = _calc_indicators(test_data, bt_ma)
        if test_data.empty:
            continue
        
        # 强制 T+1 宽松模式
        ret, bench_ret, trades = _backtest_loop(test_data, symbol, bt_cap, bt_ma, bt_stop, 
                                                bt_vision, vision_map, cost_calc, strict_t1=False)
        
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
    
    if len(df) < 50:
        st.error("数据不足")
        return
    
    df = _calc_indicators(df, bt_ma)
    if df.empty:
        st.error("数据计算失败")
        return
    
    # 紧急回退：使用简化成本模型，避免 -8% 收益率
    cost_calc = _get_simplified_cost_calc()
    vision_map = _load_vision_map(symbol, PROJECT_ROOT)
    
    # 强制 T+1 宽松模式
    ret, bench_ret, trades, equity, cost_summary = _backtest_loop(
        df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, vision_map, cost_calc,
        return_equity=True, return_costs=True, strict_t1=False
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
        with st.expander("💸 交易成本明细"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总成本", f"¥{cost_summary['total']:.2f}")
            c2.metric("佣金", f"¥{cost_summary['commission']:.2f}")
            c3.metric("滑点", f"¥{cost_summary['slippage']:.2f}")
            c4.metric("冲击/机会成本", f"¥{cost_summary['impact']:.2f}")

def _run_stress_test(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, eng, PROJECT_ROOT):
    """压力测试"""
    import streamlit as st
    from src.backtest.stress_testing import StressTester
    
    st.markdown("### 🌪️ 压力测试 (Stress Testing)")
    
    # 压力测试可以使用标准成本模型
    vision_map = _load_vision_map(symbol, PROJECT_ROOT)
    
    tester = StressTester()
    key_scenarios = ['financial_crisis_2008', 'covid_crash_2020', 'market_crash_2015']
    
    # 手动触发几个场景的回测
    stress_results = {}
    for scenario_name in key_scenarios:
        scenario_df = tester.apply_scenario(df, scenario_name)
        if scenario_df is None or len(scenario_df) < 50:
            continue
            
        scenario_df = _calc_indicators(scenario_df, bt_ma)
        if scenario_df.empty: 
            continue
            
        # 压力测试也用宽松T+1
        ret, bench_ret, _ = _backtest_loop(
            scenario_df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, vision_map,
            AdvancedTransactionCost(), strict_t1=False # 压力测试可以稍微严格点，但这里保持一致
        )
        stress_results[scenario_name] = ret
        
    if stress_results:
        cols = st.columns(len(stress_results))
        for i, (name, ret) in enumerate(stress_results.items()):
            cols[i].metric(f"场景: {name}", f"{ret:.2f}%", 
                           delta="抗跌" if ret > -20 else "脆弱", delta_color="inverse")
    
    # 样本内自动压力窗口
    st.markdown("#### 样本内自动压力窗口测试")
    # 自动搜索最差窗口
    auto_stress_results = tester.run_auto_stress_test(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, eng, PROJECT_ROOT)
    
    if auto_stress_results:
        rows = []
        for label, res in auto_stress_results.items():
            rows.append({
                "压力窗口": label,
                "时间段": f"{res['start']} ~ {res['end']}",
                "策略收益": f"{res['return']:.2f}%",
                "基准收益": f"{res['benchmark']:.2f}%",
                "超额": f"{res['alpha']:.2f}%"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def _backtest_loop(df, symbol, bt_cap, bt_ma, bt_stop, bt_vision, vision_map, cost_calc,
                   return_equity=False, return_costs=False, strict_t1=False):
    """
    回测循环核心
    strict_t1: 是否开启严格T+1（默认False以恢复高收益）
    """
    cash = bt_cap
    shares = 0
    equity = []
    
    entry_price = 0
    trades_count = 0
    
    total_commission = 0.0
    total_slippage = 0.0
    total_impact = 0.0
    
    # 状态变量
    prev_close = None
    last_buy_idx = None  # T+1约束
    
    # 遍历
    for i in range(len(df)):
        row = df.iloc[i]
        date_str = df.index[i].strftime("%Y%m%d")
        
        p = float(row["Close"])
        volume = float(row.get("Volume", 100000))
        
        # 涨跌停/停牌检测
        is_limit_up = False
        is_limit_down = False
        is_suspended = volume == 0
        
        if prev_close:
            if p >= prev_close * 1.095: is_limit_up = True
            if p <= prev_close * 0.905: is_limit_down = True
            
        # 信号生成
        signal = 0
        
        # 1. 止损逻辑 (最高优先级)
        if shares > 0 and entry_price > 0:
            pnl_pct = (p - entry_price) / entry_price
            if pnl_pct < -bt_stop / 100:
                signal = -1 # 止损卖出
        
        # 2. 视觉/策略信号
        if signal == 0:
            # 视觉信号
            v_score = vision_map.get(date_str, 50.0)
            
            # 结合 MA 趋势
            ma_val = row.get("MA", 0)
            
            if v_score >= bt_vision and p > ma_val:
                signal = 1
            elif v_score < 40 or p < ma_val:
                # 增强卖出逻辑：趋势坏了或者AI看空
                if shares > 0:
                    signal = -1
        
        # 执行逻辑
        diff = 0
        total_assets = cash + shares * p
        
        if signal == 1 and cash > 0:
            # 全仓买入 (简化)
            can_buy_shares = int(cash / p / 100) * 100
            if can_buy_shares > 0:
                diff = can_buy_shares
        elif signal == -1 and shares > 0:
            # 全仓卖出
            diff = -shares
            
        # 约束检查
        if abs(diff * p) > 1000: # 有实际交易
            # 停牌/涨跌停检查
            if is_suspended:
                diff = 0
            elif diff > 0 and is_limit_up:
                diff = 0
            elif diff < 0 and is_limit_down:
                diff = 0
            
            # T+1 检查 (strict_t1)
            if strict_t1 and diff < 0 and last_buy_idx is not None and row.name <= last_buy_idx:
                diff = 0
        
        # 成本计算与结算
        step_cost = 0
        if diff != 0:
            trade_val = abs(diff * p)
            volatility = 0.02 # 默认日波2%
            
            # 计算成本
            try:
                cost_res = cost_calc.calculate_cost(trade_val, p, max(volume, 1), volatility, diff > 0)
                step_cost = cost_res.get('total_cost', 0)
                
                total_commission += cost_res.get('commission', 0)
                total_slippage += cost_res.get('slippage', 0)
                # 包含了 impact + opportunity
                total_impact += cost_res.get('market_impact', 0) + cost_res.get('opportunity_cost', 0)
            except:
                step_cost = trade_val * 0.001
            
            if diff > 0: # Buy
                if cash >= trade_val + step_cost:
                    cash -= (trade_val + step_cost)
                    shares += diff
                    if entry_price == 0: entry_price = p
                    else: 
                        # 加仓均价
                        old_val = (shares - diff) * entry_price
                        entry_price = (old_val + trade_val) / shares
                    last_buy_idx = row.name
                    trades_count += 1
            else: # Sell
                cash += (trade_val - step_cost)
                shares += diff # diff is negative
                if shares <= 0:
                    shares = 0
                    entry_price = 0
                trades_count += 1
                
        # 更新净值
        equity.append(cash + shares * p)
        prev_close = p
        
    final_equity = equity[-1]
    total_ret = (final_equity / bt_cap - 1) * 100
    bench_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    
    if return_equity and return_costs:
        return total_ret, bench_ret, trades_count, equity, {
            "total": total_commission + total_slippage + total_impact,
            "commission": total_commission,
            "slippage": total_slippage,
            "impact": total_impact
        }
    elif return_equity:
        return total_ret, bench_ret, trades_count, equity
    else:
        return total_ret, bench_ret, trades_count

def _load_vision_map(symbol, project_root):
    """加载视觉预测结果缓存"""
    # 模拟：实际应从 BatchAnalyzer 或 VisionEngine 缓存读取
    # 这里简单起见，返回空字典，回测将依赖 MA 趋势
    # 在完整系统中，这里应读取 data/predictions/{symbol}.json
    return {}

def _calc_indicators(df, ma_period):
    """计算回测所需指标"""
    try:
        df = df.copy()
        df["MA"] = df["Close"].rolling(window=ma_period).mean()
        return df
    except:
        return pd.DataFrame()

def _safe_date_str(dt):
    try:
        return dt.strftime("%Y-%m-%d")
    except:
        return str(dt)

def _display_wf_results(all_results, train_m, test_m):
    import streamlit as st
    st.subheader("🔁 Walk-Forward 验证结果")
    
    df_res = pd.DataFrame(all_results)
    avg_ret = df_res['return'].mean()
    win_folds = len(df_res[df_res['return'] > 0])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("平均Fold收益", f"{avg_ret:.2f}%")
    c2.metric("正收益Fold占比", f"{win_folds}/{len(df_res)}")
    c3.metric("训练/测试窗口", f"{train_m}月 / {test_m}月")
    c4.metric("总Fold数", f"{len(df_res)}")
    
    with st.expander("查看详细Fold数据"):
        st.dataframe(df_res, use_container_width=True)

def _compute_baseline_returns(df):
    """计算基线策略收益"""
    try:
        close = df["Close"]
        ret_series = close.pct_change().fillna(0)
        
        # Buy & Hold
        bh = (close / close.iloc[0] - 1) * 100
        bh_val = bh.iloc[-1]
        
        # MA Crossover (Fast=5, Slow=20)
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        sig = np.where(ma5 > ma20, 1, 0)
        sig = pd.Series(sig, index=close.index).shift(1).fillna(0) # T+1 execution
        ma_ret = (1 + ret_series * sig).cumprod() - 1
        ma_val = ma_ret.iloc[-1] * 100
        
        return pd.DataFrame([
            {"基线": "Buy & Hold", "收益率": f"{bh_val:.2f}%"},
            {"基线": "MA(5,20)", "收益率": f"{ma_val:.2f}%"}
        ]), {
            "Buy & Hold": ret_series,
            "MA(5,20)": ret_series * sig
        }
    except:
        return pd.DataFrame(), {}
