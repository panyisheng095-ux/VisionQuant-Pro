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
        from src.factor_analysis.ic_analysis import ICAnalyzer
        from src.factor_analysis.regime_detector import RegimeDetector
        from src.strategies.kline_factor import KLineFactorCalculator
        from src.factor_analysis.factor_invalidation import FactorInvalidationDetector
        
        kline_calc = KLineFactorCalculator(data_loader=eng.get("loader"))
        factor_values, forward_returns, dates = _calculate_factor_values(
            df_f, symbol, kline_calc, eng["vision"], PROJECT_ROOT
        )
        
        if len(factor_values) < 20:
            st.warning("数据不足，需要至少20个有效数据点")
            logger.warning(f"因子分析数据不足: {symbol}, 有效点数: {len(factor_values)}")
            return
        
        factor_series = pd.Series(factor_values, index=pd.to_datetime(dates))
        returns_series = pd.Series(forward_returns, index=pd.to_datetime(dates))

        # ---- ICAnalyzer 正确用法：__init__(window=...) + analyze(factor_values, returns) ----
        # 选择一个不会导致空序列的窗口：20~60之间，且严格小于样本长度
        n = len(factor_series)
        window = min(60, max(20, n // 2))
        window = min(window, max(2, n - 1))
        ic_analyzer = ICAnalyzer(window=window)
        ic_result = ic_analyzer.analyze(factor_series, returns_series, method="pearson")
        rolling_ic = ic_result.get("ic_series", pd.Series(dtype=float))

        _plot_ic_curve(rolling_ic, ic_result)
        _plot_regime_distribution(df_f)
        _plot_decay_analysis(rolling_ic)
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

def _calculate_factor_values(df_f, symbol, kline_calc, vision_engine, PROJECT_ROOT):
    """
    计算历史因子值
    
    通过遍历历史数据，为每个时间点计算K线学习因子值
    """
    factor_values, forward_returns, dates = [], [], []
    
    # 限制计算量，但要覆盖全区间（原实现只算最前200个点，导致“只看到2020”）
    end_idx = len(df_f) - 6  # 需要 i+5 可取
    if end_idx <= 20:
        return factor_values, forward_returns, dates

    max_points = min(200, end_idx - 20 + 1)
    sample_idx = np.linspace(20, end_idx, num=max_points, dtype=int)
    sample_idx = sorted(set(int(x) for x in sample_idx))

    for i in sample_idx:
        try:
            current_data = df_f.iloc[i-20:i]
            if len(current_data) < 20:
                continue
            
            temp_img = os.path.join(PROJECT_ROOT, "data", f"temp_factor_{i}.png")
            mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
            mpf.plot(current_data, type='candle', style=s, savefig=dict(fname=temp_img, dpi=50), 
                    figsize=(3, 3), axisoff=True)
            
            matches = vision_engine.search_similar_patterns(temp_img, top_k=10)
            
            if matches and len(matches) > 0:
                try:
                    date_str = _safe_date_str(df_f.index[i])
                    factor_result = kline_calc.calculate_hybrid_win_rate(
                        matches, 
                        query_symbol=symbol, 
                        query_date=date_str
                    )
                    if isinstance(factor_result, dict):
                        factor_value = factor_result.get('hybrid_win_rate', 50.0) / 100.0
                    else:
                        factor_value = 0.5
                    
                    # 计算未来5日收益率
                    if i + 5 < len(df_f):
                        future_return = (df_f.iloc[i+5]['Close'] - df_f.iloc[i]['Close']) / df_f.iloc[i]['Close']
                        factor_values.append(factor_value)
                        forward_returns.append(future_return)
                        dates.append(df_f.index[i])
                except Exception as e:
                    logger.debug(f"计算因子值失败 {i}: {e}")
                    continue
            
            if os.path.exists(temp_img):
                os.remove(temp_img)
        except:
            continue
    
    return factor_values, forward_returns, dates

def _plot_ic_curve(rolling_ic, ic_result):
    """绘制IC曲线"""
    import streamlit as st

    if rolling_ic is None or len(rolling_ic) == 0:
        st.warning("Rolling IC 为空：样本量不足或窗口过大。请扩大时间范围后再试。")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_ic.index, y=rolling_ic.values, mode='lines',
                            name='Rolling IC', line=dict(color='blue', width=2)))
    fig.add_hline(y=0.05, line_dash="dash", line_color="green", annotation_text="IC阈值(0.05)")
    fig.add_hline(y=-0.05, line_dash="dash", line_color="red")
    fig.update_layout(title="IC曲线分析", height=300)
    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
    
    summary = (ic_result or {}).get("summary", {})
    mean_ic = float(summary.get("mean_ic", rolling_ic.mean()))
    std_ic = float(summary.get("std_ic", rolling_ic.std()))
    ic_ir = float(summary.get("ir", mean_ic / std_ic if std_ic > 0 else 0.0))
    positive_ratio = float(summary.get("positive_ratio", (rolling_ic > 0).mean() if len(rolling_ic) else 0.0))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("平均IC", f"{mean_ic:.4f}", delta="有效" if mean_ic > 0.05 else "无效")
    col2.metric("IC标准差", f"{std_ic:.4f}")
    col3.metric("ICIR", f"{ic_ir:.2f}", delta="优秀" if abs(ic_ir) > 1.0 else "一般")
    col4.metric("正IC比例", f"{positive_ratio*100:.1f}%",
               delta="良好" if positive_ratio > 0.6 else "一般")

def _plot_regime_distribution(df_f):
    """绘制Regime分布"""
    import streamlit as st
    from src.factor_analysis.regime_detector import RegimeDetector
    
    st.subheader("市场Regime识别")
    prices = df_f["Close"].copy()
    returns = prices.pct_change().dropna()
    regime_detector = RegimeDetector()
    regimes = regime_detector.detect_regime(returns, prices=prices.reindex(returns.index))
    regime_counts = regimes.value_counts()
    
    colors_map = {
        "bull_market": "green",
        "bear_market": "red",
        "oscillating": "goldenrod",
        "unknown": "gray",
    }
    fig = go.Figure(data=[go.Bar(x=regime_counts.index, y=regime_counts.values,
                                marker_color=[colors_map.get(r, 'gray') for r in regime_counts.index])])
    fig.update_layout(title="市场Regime分布", height=300)
    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

def _plot_decay_analysis(rolling_ic):
    """因子衰减分析"""
    import streamlit as st
    
    st.subheader("因子衰减分析")
    decay_window = min(60, len(rolling_ic))
    recent_ic = rolling_ic.tail(decay_window).mean()
    earlier_ic = rolling_ic.head(decay_window).mean() if len(rolling_ic) > decay_window else recent_ic
    decay_rate = (recent_ic - earlier_ic) / abs(earlier_ic) * 100 if earlier_ic != 0 else 0
    
    col1, col2 = st.columns(2)
    col1.metric("早期IC均值", f"{earlier_ic:.4f}")
    col2.metric("近期IC均值", f"{recent_ic:.4f}", delta=f"{decay_rate:.1f}%",
               delta_color="inverse" if decay_rate < 0 else "normal")

def _detect_invalidation(factor_values, forward_returns):
    """因子失效检测"""
    import streamlit as st
    
    try:
        from src.factor_analysis.factor_invalidation import FactorInvalidationDetector
        
        st.subheader("因子失效检测")
        detector = FactorInvalidationDetector()
        result = detector.detect_invalidation(factor_values, forward_returns)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("失效得分", f"{result['invalidation_score']:.2f}",
                   delta="失效" if result['is_invalidated'] else "有效",
                   delta_color="inverse" if result['is_invalidated'] else "normal")
        # 不同版本返回字段不一致：以当前实现 ic_result/decay_result 为准
        ic_failed = bool((result.get("ic_result") or {}).get("is_failed", False))
        decay_failed = bool((result.get("decay_result") or {}).get("is_decaying", False))
        col2.metric("IC状态", "失效" if ic_failed else "正常")
        col3.metric("衰减状态", "失效" if decay_failed else "正常")
        
        if result['is_invalidated']:
            st.warning("⚠️ 因子可能已失效，建议降低权重或暂停使用")
    except Exception as e:
        st.info(f"因子失效检测暂不可用: {e}")

def _safe_date_str(date_obj):
    """安全日期格式化"""
    try:
        if hasattr(date_obj, 'strftime'):
            return date_obj.strftime('%Y%m%d')
        elif isinstance(date_obj, pd.Timestamp):
            return date_obj.strftime('%Y%m%d')
        else:
            return str(date_obj).replace('-', '').replace(' ', '')[:8]
    except:
        return str(date_obj)
