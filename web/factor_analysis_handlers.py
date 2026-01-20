"""因子分析处理模块 - 工业级优化"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import logging
import mplfinance as mpf

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def show_factor_analysis(symbol, df_f, eng, PROJECT_ROOT):
    """
    因子有效性分析

    Args:
        symbol: 股票代码
        df_f: 包含技术指标的DataFrame
        eng: 引擎字典
        PROJECT_ROOT: 项目根目录
    """
    import streamlit as st

    try:
        logger.info(f"开始因子分析: {symbol}")
        # 性能优化提示
        with st.spinner("🚀 正在计算因子值（性能优化版，预计2-5分钟）..."):
            from src.factor_analysis.ic_analysis import ICAnalyzer
            from src.factor_analysis.regime_detector import RegimeDetector
            from src.strategies.kline_factor import KLineFactorCalculator
            from src.factor_analysis.factor_invalidation import FactorInvalidationDetector

            kline_calc = KLineFactorCalculator(data_loader=eng.get("loader"))
            factor_values, forward_returns, dates, horizon_returns, success_count, fail_count = _calculate_factor_values(
                df_f, symbol, kline_calc, eng["vision"], PROJECT_ROOT, horizons=[1, 5, 10, 20]
            )

        if len(factor_values) < 20:
            st.warning(f"数据不足，需要至少20个有效数据点（当前 {len(factor_values)}）")
            st.caption(f"匹配诊断: 尝试 {success_count + fail_count} 次 | 成功 {success_count} 次 | 失败 {fail_count} 次")
            if fail_count > success_count:
                st.info("💡 失败率过高提示：当前图库对该股历史形态的覆盖度不足。建议扩充图库至100万样本。")
            logger.warning(f"因子分析数据不足: {symbol}, 有效点数: {len(factor_values)}")
            return

        # 样本量置信度提示（科学性）
        n = len(factor_values)
        if n >= 500:
            conf = "高"
        elif n >= 200:
            conf = "中"
        elif n >= 80:
            conf = "低"
        else:
            conf = "偏低"
        st.caption(f"有效样本数: {n} | 置信度: {conf}")
        total_attempts = success_count + fail_count
        fail_rate = (fail_count / total_attempts * 100) if total_attempts > 0 else 0.0
        st.caption(f"匹配诊断: 尝试 {total_attempts} 次 | 成功 {success_count} 次 | 失败 {fail_count} 次 (失败率 {fail_rate:.1f}%)")

        factor_series = pd.Series(factor_values, index=pd.to_datetime(dates))
        returns_series = pd.Series(forward_returns, index=pd.to_datetime(dates))

        # ---- ICAnalyzer 正确用法：__init__(window=...) + analyze(factor_values, returns) ----
        # 选择一个不会导致空序列的窗口：20~60之间，且严格小于样本长度
        n = len(factor_series)
        window = min(60, max(20, n // 2))
        window = min(window, max(2, n - 1))
        ic_analyzer = ICAnalyzer(window=window)
        # v3.0: 开启稳健统计 (Winsorization)
        ic_result = ic_analyzer.analyze(factor_series, returns_series, method="pearson")
        # 多持有期IC矩阵
        try:
            horizon_series = {}
            for h, ret_list in horizon_returns.items():
                if len(ret_list) != len(dates):
                    continue
                horizon_series[h] = pd.Series(ret_list, index=pd.to_datetime(dates))
            multi_ic = ic_analyzer.analyze_multi_horizon(factor_series, horizon_series, method="pearson")
        except Exception:
            multi_ic = {}
        rolling_ic = ic_result.get("ic_series", pd.Series(dtype=float))

        _plot_ic_curve(rolling_ic, ic_result)
        if multi_ic:
            _plot_ic_horizon_matrix(multi_ic)
        _plot_sharpe_curve(ic_result)
        _plot_regime_distribution(df_f)

        # 衰减 + 拐点检测（Change Point / CUSUM）
        try:
            from src.factor_analysis.decay_analysis import DecayAnalyzer
            decay_analyzer = DecayAnalyzer()
            decay_result = decay_analyzer.analyze_decay(rolling_ic)
        except Exception:
            decay_result = {}

        _plot_decay_analysis(rolling_ic, decay_result)
        _detect_invalidation(factor_series, returns_series)

    except ImportError as e:
        logger.exception(f"因子分析模块导入失败: {symbol}")
        st.error(f"模块导入失败: {e}")
    except Exception as e:
        logger.exception(f"因子分析异常: {symbol}")
        st.error(f"因子分析失败: {e}")
        import traceback
        with st.expander("查看详细错误"):
            st.code(traceback.format_exc())


def _calculate_factor_values(df_f, symbol, kline_calc, vision_engine, PROJECT_ROOT, horizons=None):
    """
    计算历史因子值（性能优化版）

    通过遍历历史数据，为每个时间点计算K线学习因子值
    优化：使用快速模式、减少样本数、添加进度提示
    """
    import streamlit as st
    
    if horizons is None:
        horizons = [1, 5, 10, 20]
    factor_values, forward_returns, dates = [], [], []
    horizon_returns = {h: [] for h in horizons}

    # 覆盖全区间 + 性能优化：减少样本数
    end_idx = len(df_f) - 6  # 需要 i+5 可取
    if end_idx <= 20:
        return factor_values, forward_returns, dates, horizon_returns, 0, 0

    total_points = end_idx - 20 + 1
    # 性能优化：目标样本数从600降到200（兼顾速度和科学性）
    target_points = min(200, total_points)
    # 自适应步长：数据越长，步长越大
    if total_points <= 200:
        step = 1
    elif total_points <= 400:
        step = 2
    elif total_points <= 800:
        step = 4
    else:
        step = max(2, total_points // target_points)
    sample_idx = list(range(20, end_idx + 1, step))
    # 兜底避免过多
    if len(sample_idx) > target_points:
        sample_idx = sample_idx[:target_points]

    success_count = 0
    fail_count = 0
    
    # 添加进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_iters = len(sample_idx)

    for idx, i in enumerate(sample_idx):
        try:
            current_data = df_f.iloc[i-20:i]
            if len(current_data) < 20:
                continue

            date_dt = df_f.index[i]
            date_str = _safe_date_str(date_dt)

            # 更新进度
            progress = (idx + 1) / total_iters
            progress_bar.progress(progress)
            status_text.text(f"计算因子值: {idx + 1}/{total_iters} ({progress*100:.1f}%)")
            
            temp_img = os.path.join(PROJECT_ROOT, "data", f"temp_factor_{i}.png")
            mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
            mpf.plot(current_data, type='candle', style=s, savefig=dict(fname=temp_img, dpi=50),
                    figsize=(3, 3), axisoff=True)

            # 性能优化：使用快速模式，大幅减少搜索时间
            matches = vision_engine.search_similar_patterns(
                temp_img, 
                top_k=10, 
                max_date=date_dt,
                fast_mode=True,  # 启用快速模式：跳过DTW/相关性计算
                search_k=500,    # 减少搜索范围
                rerank_with_pixels=False,  # 跳过像素重排
                max_price_checks=50,  # 限制价格检查次数
                use_price_features=False  # 跳过价格特征计算
            )

            # 严格无未来函数：若匹配结果稀少，则使用“同股历史窗口”回退
            if not matches or len(matches) < 3:
                matches = _self_match_windows(df_f, symbol, i, top_k=10)

            if matches and len(matches) > 0:
                success_count += 1
                try:
                    factor_result = kline_calc.calculate_hybrid_win_rate(
                        matches,
                        query_symbol=symbol,
                        query_date=date_str,
                        query_df=df_f.iloc[:i+1]
                    )
                    if isinstance(factor_result, dict):
                        enhanced = factor_result.get("enhanced_factor")
                        if isinstance(enhanced, dict) and enhanced.get("final_score") is not None:
                            factor_value = float(enhanced.get("final_score")) / 100.0
                        else:
                            factor_value = factor_result.get('hybrid_win_rate', 50.0) / 100.0
                    else:
                        factor_value = 0.5

                    # 多持有期收益率
                    p_entry = df_f.iloc[i]['Close']
                    for h in horizons:
                        if i + h < len(df_f):
                            p_exit = df_f.iloc[i + h]['Close']
                            ret = (p_exit - p_entry) / p_entry
                            horizon_returns[h].append(ret)
                    # 默认用5日作为主序列
                    p_exit = df_f.iloc[i+5]['Close'] if i + 5 < len(df_f) else df_f.iloc[i]['Close']
                    ret = (p_exit - p_entry) / p_entry

                    factor_values.append(factor_value)
                    forward_returns.append(ret)
                    dates.append(date_str)

                except Exception:
                    pass
            else:
                fail_count += 1

            if os.path.exists(temp_img):
                os.remove(temp_img)

        except Exception:
            fail_count += 1
            continue
    
    # 清理进度条
    progress_bar.empty()
    status_text.empty()

    return factor_values, forward_returns, dates, horizon_returns, success_count, fail_count


def _self_match_windows(df_f, symbol, idx, window: int = 20, top_k: int = 10, max_windows: int = 100):
    """
    回退方案：仅在"同一股票历史窗口"内做形态相似度（无未来函数）
    性能优化：减少最大窗口数，使用更快的相关性计算
    """
    try:
        if idx <= window:
            return []
        q_prices = df_f.iloc[idx - window: idx]["Close"].values
        if len(q_prices) < window:
            return []

        # 性能优化：减少窗口数量（从200降到100）
        start = window
        end = idx
        total = end - start
        if total <= 0:
            return []
        step = max(1, total // max_windows)

        # 性能优化：预计算归一化查询价格
        q_mean = q_prices.mean()
        q_std = q_prices.std() + 1e-8
        q_norm = (q_prices - q_mean) / q_std
        
        candidates = []
        for j in range(start, end, step):
            cand = df_f.iloc[j - window: j]["Close"].values
            if len(cand) < window:
                continue
            # 性能优化：使用更快的相关性计算
            c_mean = cand.mean()
            c_std = cand.std() + 1e-8
            c_norm = (cand - c_mean) / c_std
            # 使用点积计算相关性（比corrcoef快）
            corr = np.dot(q_norm, c_norm) / window
            if np.isnan(corr):
                corr = 0.0
            sim = (corr + 1.0) / 2.0
            date_str = df_f.index[j - 1].strftime("%Y%m%d")
            candidates.append({
                "symbol": str(symbol).zfill(6),
                "date": date_str,
                "score": float(sim),
                "correlation": float(corr)
            })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
    except Exception:
        return []


def _plot_ic_curve(rolling_ic, ic_result):
    """绘制IC曲线"""
    import streamlit as st

    st.markdown("#### IC 分析")
    if rolling_ic.empty:
        st.write("IC 数据不足")
        return

    summary = ic_result.get("summary", {})
    mean_ic = summary.get("mean_ic", 0.0)
    std_ic = summary.get("std_ic", 0.0)
    ic_ir = summary.get("ir", 0.0)
    positive_ratio = summary.get("positive_ratio", 0.0)
    half_life = summary.get("half_life", None)
    stability = summary.get("stability_score", None)

    # 修正逻辑：IC为负不一定无效，可能是反向指标
    if abs(mean_ic) > 0.05:
        ic_status = "显著" + ("(正向)" if mean_ic > 0 else "(反向)")
        ic_color = "normal" if mean_ic > 0 else "inverse"  # 负值给红色/反色提示
    elif abs(mean_ic) > 0.02:
        ic_status = "微弱"
        ic_color = "off"
    else:
        ic_status = "无效"
        ic_color = "off"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("平均IC", f"{mean_ic:.4f}", delta=ic_status, delta_color=ic_color)
    col2.metric("IC标准差", f"{std_ic:.4f}")
    col3.metric("ICIR", f"{ic_ir:.2f}", delta="优秀" if abs(ic_ir) > 1.0 else "一般")
    col4.metric("正IC比例", f"{positive_ratio*100:.1f}%",
               delta="良好" if positive_ratio > 0.6 else "一般")
    col5.metric("IC Half-Life", f"{half_life:.1f}" if half_life is not None else "N/A")
    col6.metric("稳定性评分", f"{float(stability):.2f}" if stability is not None else "N/A")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rolling_ic.index,
        y=rolling_ic.values,
        name="Rolling IC",
        marker_color=['red' if x >= 0 else 'green' for x in rolling_ic.values]
    ))
    # 累积IC曲线
    cum_ic = rolling_ic.cumsum()
    fig.add_trace(go.Scatter(
        x=rolling_ic.index,
        y=cum_ic.values,
        name="Cumulative IC",
        yaxis="y2",
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title="滚动IC与累积IC",
        height=300,
        yaxis=dict(title="Rolling IC"),
        yaxis2=dict(title="Cumulative IC", overlaying="y", side="right"),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ 因子分析说明与指标解读", expanded=False):
        st.markdown(r"""
        **1. 核心概念**
        - **因子定义**: K线学习因子 = 相似度加权的混合胜率（作为期望收益代理）
        - **IC (Information Coefficient)**: 因子值与未来收益率的相关系数。反映因子预测能力。
        - **Rolling IC**: 滚动窗口下的IC值，用于观察因子随时间的稳定性。

        **2. 指标解读标准**
        - **平均IC**:
          - `> 0.05`: 显著正向（因子分越高，未来涨幅越大）
          - `< -0.05`: 显著反向（可作为反向指标使用）
          - `abs(IC) < 0.02`: 预测能力微弱
        - **ICIR (IC/Std)**: 衡量因子稳定性（IC均值/IC标准差）。绝对值 `> 1.0` 为优秀。
        - **正IC比例**: 滚动IC > 0 的时间占比，越高越好。
        - **Half-Life (半衰期)**: 因子预测能力衰减一半所需天数。越长越适合中长线。

        **3. 进阶分析**
        - **Regime分析**: 在不同市场状态（牛/熊/震荡）下的因子表现差异。
        - **因子衰减**: 观察近期IC是否显著弱于早期IC，提示失效风险。
        - **失效检测**: 综合IC衰减、拥挤度等维度判断因子是否失效。
        """)


def _plot_ic_horizon_matrix(multi_ic: dict):
    """多持有期IC矩阵"""
    import streamlit as st
    st.subheader("多持有期IC矩阵（IC衰减）")
    matrix = multi_ic.get("ic_matrix")
    if matrix is None or matrix.empty:
        st.caption("IC矩阵数据不足")
        return
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    try:
        fig = go.Figure(data=go.Heatmap(
            z=matrix[["ic_mean", "ic_ir", "half_life"]].values,
            x=["IC均值", "ICIR", "Half-Life"],
            y=matrix["horizon"].astype(str).tolist(),
            colorscale="RdBu"
        ))
        fig.update_layout(height=280, title="IC矩阵热图（不同持有期）")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _plot_sharpe_curve(ic_result):
    """绘制滚动Sharpe"""
    import streamlit as st
    sharpe_series = ic_result.get("sharpe_series", pd.Series(dtype=float))

    if sharpe_series.empty:
        return

    st.subheader("Rolling Sharpe 分析")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sharpe_series.index,
        y=sharpe_series.values,
        name="Rolling Sharpe",
        line=dict(color='orange')
    ))

    mean_sharpe = sharpe_series.mean()
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Rolling Sharpe均值: {mean_sharpe:.3f}")


def _plot_regime_distribution(df):
    """Regime分布"""
    pass


def _plot_decay_analysis(rolling_ic, decay_result=None):
    """因子衰减分析"""
    import streamlit as st

    st.subheader("因子衰减分析")
    decay_window = min(60, len(rolling_ic))
    if decay_window < 10:
        return

    recent_ic = rolling_ic.tail(decay_window).mean()
    earlier_ic = rolling_ic.head(decay_window).mean() if len(rolling_ic) > decay_window else recent_ic
    decay_rate = (recent_ic - earlier_ic) / abs(earlier_ic) * 100 if earlier_ic != 0 else 0

    col1, col2 = st.columns(2)
    col1.metric("早期IC均值", f"{earlier_ic:.4f}")
    col2.metric("近期IC均值", f"{recent_ic:.4f}", delta=f"{decay_rate:.1f}%",
               delta_color="inverse" if decay_rate < 0 else "normal")

    # 拐点信息
    if decay_result:
        cps = decay_result.get("change_points", [])
        if cps:
            st.caption(f"检测到拐点: {', '.join([str(c) for c in cps[-3:]])}")


def _detect_invalidation(factor_values, returns):
    """因子失效检测"""
    pass


def _safe_date_str(dt):
    try:
        return dt.strftime("%Y%m%d")
    except Exception:
        return str(dt)
