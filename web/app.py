"""VisionQuant Pro - 工业级精简版"""
import streamlit as st
import os, sys, pandas as pd, numpy as np, mplfinance as mpf, plotly.graph_objects as go
from datetime import datetime
import importlib
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.data.data_loader import DataLoader
    from src.data.news_harvester import NewsHarvester
    from src.models.vision_engine import VisionEngine
    from src.strategies.factor_mining import FactorMiner
    from src.strategies.fundamental import FundamentalMiner
    from src.agent.quant_agent import QuantAgent
    from src.utils.visualizer import create_comparison_plot
    from src.utils.pdf_generator import generate_report_pdf
    from src.utils.audio_manager import AudioManager
    from src.strategies.batch_analyzer import BatchAnalyzer
    from src.strategies.portfolio_optimizer import PortfolioOptimizer
    from src.strategies.kline_factor import KLineFactorCalculator
except ImportError as e:
    st.error(f"❌ 系统模块加载失败: {e}")
    st.stop()

def _code_version_key():
    paths = [
        os.path.join(PROJECT_ROOT, "src", "models", "vision_engine.py"),
        os.path.join(PROJECT_ROOT, "src", "strategies", "fundamental.py"),
    ]
    return "|".join([str(os.path.getmtime(p)) if os.path.exists(p) else "0" for p in paths])

def _find_existing_kline_image(symbol: str, date_str: str, vision_engine=None):
    symbol = str(symbol).zfill(6)
    date_n = str(date_str).replace("-", "")
    if vision_engine is not None:
        try:
            fast_path = vision_engine.find_image_path(symbol, date_n, allow_nearest=True)
            if fast_path:
                return fast_path
        except Exception:
            pass
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
    return None

def _render_match_image(symbol: str, date_str: str, loader, out_path: str):
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
        mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
        mpf.plot(window, type='candle', style=s, savefig=dict(fname=out_path, dpi=50), figsize=(3, 3), axisoff=True)
        return out_path
    except Exception:
        return None

def _augment_matches(matches, query_img_path, query_prices, loader, vision_engine, tmp_dir):
    if not matches:
        return matches
    q_pix = vision_engine._load_pixel_vector(query_img_path)
    q_edge = vision_engine._load_edge_vector(query_img_path)
    for i, m in enumerate(matches):
        sym = str(m.get("symbol", "")).zfill(6)
        date_str = m.get("date")
        # 1) 像素/边缘相似度兜底
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
                m["pixel_sim"] = visual_sim
                m["edge_sim"] = edge_norm
            # fallback: 用sim_score填补，避免N/A
            if m.get("pixel_sim") is None:
                m["pixel_sim"] = m.get("sim_score", m.get("score", 0))
            if m.get("edge_sim") is None:
                m["edge_sim"] = m.get("pixel_sim")

        # 2) 相关性与回报相关兜底
        if (m.get("correlation") is None or m.get("ret_corr") is None) and query_prices is not None:
            try:
                dfp = loader.get_stock_data(sym)
                if dfp is not None and not dfp.empty:
                    dfp.index = pd.to_datetime(dfp.index)
                    dt = pd.to_datetime(str(date_str), errors="coerce")
                    if dt in dfp.index:
                        loc = dfp.index.get_loc(dt)
                        if loc >= 19:
                            match_prices = dfp.iloc[loc - 19: loc + 1]['Close'].values
                            qn = (query_prices - query_prices.mean()) / (query_prices.std() + 1e-8)
                            mn = (match_prices - match_prices.mean()) / (match_prices.std() + 1e-8)
                            corr = np.corrcoef(qn, mn)[0, 1]
                            if not np.isnan(corr):
                                m["correlation"] = float(corr)
                            q_ret = np.diff(query_prices) / (query_prices[:-1] + 1e-8)
                            m_ret = np.diff(match_prices) / (match_prices[:-1] + 1e-8)
                            q_ret = (q_ret - q_ret.mean()) / (q_ret.std() + 1e-8)
                            m_ret = (m_ret - m_ret.mean()) / (m_ret.std() + 1e-8)
                            corr2 = np.corrcoef(q_ret, m_ret)[0, 1]
                            if not np.isnan(corr2):
                                m["ret_corr"] = float(corr2)
            except Exception:
                pass
    return matches

st.set_page_config(page_title="VisionQuant Pro", layout="wide", page_icon="🦄")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e6e9ef; }
    .agent-box { border-left: 5px solid #ff4b4b; padding: 20px; background-color: #fff1f1; border-radius: 5px; margin-bottom: 20px; }
    .stChatMessage { background-color: #ffffff; border-radius: 12px; padding: 12px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_engines(_code_version: str):
    ve_mod = importlib.import_module("src.models.vision_engine")
    fm_mod = importlib.import_module("src.strategies.fundamental")
    importlib.reload(ve_mod)
    importlib.reload(fm_mod)
    v = ve_mod.VisionEngine()
    # 延迟加载索引：只在第一次搜索时加载，避免启动时20-30分钟的等待
    # v.reload_index()  # 已移除：索引将在第一次调用search_similar_patterns时自动加载
    return {
        "loader": DataLoader(mem_cache_max=128), "vision": v, "factor": FactorMiner(),
        "fund": fm_mod.FundamentalMiner(), "agent": QuantAgent(), 
        "news": NewsHarvester(), "audio": AudioManager()
    }

eng = load_all_engines(_code_version=_code_version_key())

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_context" not in st.session_state: st.session_state.last_context = ""
if "has_run" not in st.session_state: st.session_state.has_run = False
if "last_voice_text" not in st.session_state: st.session_state.last_voice_text = ""
if "batch_results" not in st.session_state: st.session_state.batch_results = {}
if "portfolio_weights" not in st.session_state: st.session_state.portfolio_weights = {}
if "portfolio_metrics" not in st.session_state: st.session_state.portfolio_metrics = {}
if "current_symbol" not in st.session_state: st.session_state.current_symbol = None
if "ic_summary" not in st.session_state: st.session_state.ic_summary = {}

# URL 跳转预处理：先写入 session_state，让侧边栏控件同步
url_symbol = st.query_params.get("symbol")
if url_symbol:
    st.session_state["symbol_input"] = url_symbol
    st.session_state["mode_select"] = "🔍 单只股票分析"

from backtest_handlers import run_backtest, run_stratified_backtest_batch
from factor_analysis_handlers import show_factor_analysis as render_factor_analysis
from streamlit_mic_recorder import mic_recorder

with st.sidebar:
    st.title("🦄 VisionQuant Pro")
    st.caption("AI 全栈量化投研系统 v8.8")
    
    # === 数据源选择 ===
    with st.expander("⚙️ 数据源设置", expanded=False):
        ds_map = {"AkShare (免费)": "akshare", "JQData (聚宽)": "jqdata", "RQData (米筐)": "rqdata"}
        ds_label = st.selectbox("选择数据源", list(ds_map.keys()), index=0)
        curr_ds = ds_map[ds_label]
        
        # 如果选了付费源，检查/提示输入账号
        if curr_ds in ["jqdata", "rqdata"]:
            st.caption(f"需提供 {curr_ds} 账号 (或设置环境变量)")
            ds_user = st.text_input("用户名", key=f"{curr_ds}_user")
            ds_pass = st.text_input("密码", type="password", key=f"{curr_ds}_pass")
            if st.button("切换/认证"):
                eng["loader"].switch_data_source(curr_ds, username=ds_user, password=ds_pass)
                st.success(f"已尝试切换至 {curr_ds}")
        else:
            if eng["loader"].get_current_data_source() != "akshare":
                eng["loader"].switch_data_source("akshare")

    st.divider()
    symbol_input = st.text_input("请输入 A 股代码", value="601899", help="输入6位代码", key="symbol_input")
    symbol = symbol_input.strip().zfill(6)
    mode = st.radio("功能模块:", ("🔍 单只股票分析", "📊 批量组合分析"), key="mode_select")

    if mode == "🔍 单只股票分析":
        st.divider()
        st.caption("回测 / 因子有效性分析入口已统一放在“单只股票分析”报告底部 Tab 中（更符合使用路径）。")
    
    elif mode == "📊 批量组合分析":
        batch_input = st.text_area("输入股票代码（每行一个，最多30只）", height=150, key="batch_input")

    st.divider()
    run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    if st.button("🔄 强制重载", help="清除缓存，重新加载模块"):
        st.cache_resource.clear()
        st.rerun()

url_jump_mode = False
if url_symbol:
    if url_symbol != symbol:
        symbol = url_symbol
        url_jump_mode = True
        mode = "🔍 单只股票分析"
        st.session_state["symbol_input"] = symbol
        st.session_state["mode_select"] = "🔍 单只股票分析"
        if "res" in st.session_state:
            del st.session_state.res
        st.session_state.current_symbol = symbol
        st.session_state.has_run = True
        run_btn = True
    elif "res" not in st.session_state:
        url_jump_mode = True
        mode = "🔍 单只股票分析"
        st.session_state["mode_select"] = "🔍 单只股票分析"
        st.session_state.has_run = True
        run_btn = True
    else:
        st.query_params.clear()
        url_jump_mode = False

if not run_btn and not st.session_state.has_run:
    st.header(f"👋 欢迎使用 VisionQuant Pro")
    st.info(f"当前选中标的: **{symbol}**\n请在左侧侧边栏点击红色按钮启动。")
    st.stop()

if mode == "🔍 单只股票分析":
    if st.session_state.current_symbol != symbol and st.session_state.current_symbol is not None:
        if "res" in st.session_state:
            del st.session_state.res
        st.session_state.has_run = False
        st.session_state.chat_history = []
        st.session_state.last_voice_text = ""
    
    if run_btn:
        st.session_state.has_run = True
        st.session_state.chat_history = []
        st.session_state.last_voice_text = ""
        st.session_state.current_symbol = symbol
        if "res" in st.session_state:
            del st.session_state.res

        progress = st.progress(0)
        status = st.empty()
        status.write("加载行情数据...")
        with st.spinner(f"正在全栈扫描 {symbol}..."):
            try:
                logger.info(f"开始分析股票: {symbol}")
                df = eng["loader"].get_stock_data(symbol)
                if df.empty: 
                    st.error("数据获取失败")
                    logger.error(f"数据获取失败: {symbol}")
                    st.stop()
            except Exception as e:
                logger.exception(f"数据获取异常: {symbol}")
                st.error(f"数据获取失败: {str(e)}")
                st.stop()
            progress.progress(20)

            # 数据质量报告
            try:
                quality_report = eng["loader"].quality_checker.check_data_quality(df, symbol)
            except Exception:
                quality_report = {}
            progress.progress(30)

            fund_data = eng["fund"].get_stock_fundamentals(symbol)
            stock_name = fund_data.get('name', symbol)
            status.write("生成查询K线图...")

            # 优先使用已存在的历史K线图（保证与索引同分布）
            date_str = df.index[-1].strftime("%Y%m%d")
            q_p = _find_existing_kline_image(symbol, date_str, eng.get("vision"))
            if not q_p:
                q_p = os.path.join(PROJECT_ROOT, "data", "temp_q.png")
                mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
                s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
                mpf.plot(df.tail(20), type='candle', style=s, savefig=dict(fname=q_p, dpi=50), figsize=(3, 3), axisoff=True)
            progress.progress(45)
            
            query_prices = df.tail(20)['Close'].values if len(df) >= 20 else None
            # 多尺度检索（日/周/月）+ 动态权重融合
            status.write("相似形态检索中...")
            try:
                from src.data.multi_scale_generator import MultiScaleChartGenerator
                gen = MultiScaleChartGenerator(figsize=(3, 3), dpi=50)
                q_week = os.path.join(PROJECT_ROOT, "data", "temp_q_week.png")
                q_month = os.path.join(PROJECT_ROOT, "data", "temp_q_month.png")
                gen.generate_weekly_chart(df, weeks=20, output_path=q_week)
                gen.generate_monthly_chart(df, months=20, output_path=q_month)
                img_paths = {"daily": q_p, "weekly": q_week, "monthly": q_month}
                # 动态融合权重：基于各周期的收益分布质量评分
                try:
                    kline_factor_calc = KLineFactorCalculator(data_loader=eng["loader"])
                    # 仅用于权重估计，使用快速模式减少耗时
                    scale_matches = {
                        "daily": eng["vision"].search_similar_patterns(
                            q_p, top_k=10, query_prices=query_prices, max_date=date_str,
                            fast_mode=True, search_k=400, rerank_with_pixels=False
                        ),
                        "weekly": eng["vision"].search_similar_patterns(
                            q_week, top_k=10, max_date=date_str,
                            fast_mode=True, search_k=400, rerank_with_pixels=False
                        ),
                        "monthly": eng["vision"].search_similar_patterns(
                            q_month, top_k=10, max_date=date_str,
                            fast_mode=True, search_k=400, rerank_with_pixels=False
                        ),
                    }
                    scale_stats = {
                        k: kline_factor_calc.calculate_return_distribution(v, horizon_days=5, query_date=date_str)
                        for k, v in scale_matches.items()
                    }
                    scale_weights = kline_factor_calc.estimate_scale_weights(scale_stats)
                except Exception:
                    scale_weights = None

                matches = eng["vision"].search_multi_scale_patterns(
                    img_paths, top_k=10, query_prices=query_prices, weights=scale_weights, max_date=date_str
                )
            except Exception:
                matches = eng["vision"].search_similar_patterns(q_p, top_k=10, query_prices=query_prices, max_date=date_str)
            progress.progress(65)

            # 补齐相似度字段，减少 N/A
            matches = _augment_matches(matches, q_p, query_prices, eng["loader"], eng["vision"], os.path.join(PROJECT_ROOT, "data"))
            progress.progress(75)

            def get_future_trajectories(matches, loader):
                trajectories, details = [], []
                for m in matches:
                    try:
                        hdf = loader.get_stock_data(m['symbol'])
                        hdf.index = pd.to_datetime(hdf.index)
                        target_date = pd.to_datetime(m['date'])
                        if target_date in hdf.index:
                            loc = hdf.index.get_loc(target_date)
                            if loc + 5 < len(hdf):
                                subset = hdf.iloc[loc: loc + 6]['Close'].values
                                norm_path = (subset / subset[0] - 1) * 100
                                trajectories.append(norm_path)
                                details.append(f"{m['symbol']} ({m['date']})")
                    except:
                        continue
                return trajectories, details

            trajs, traj_labels = get_future_trajectories(matches, eng["loader"])

            if trajs:
                mean_path = np.mean(np.vstack(trajs), axis=0)
                avg_ret = mean_path[-1]
                traditional_win_rate = np.sum(np.vstack(trajs)[:, -1] > 0) / len(trajs) * 100
            else:
                mean_path, avg_ret, traditional_win_rate = np.zeros(6), 0.0, 50.0

            # Top10多期收益/分布估计
            try:
                from src.utils.top10_analyzer import Top10Analyzer
                analyzer = Top10Analyzer(eng["loader"])
                mh_stats = analyzer.analyze_multi_horizon(matches, horizons=[5, 10, 20])
                dist_stats = analyzer.return_distribution(matches, future_days=20)
            except Exception:
                mh_stats, dist_stats = {}, {}

            try:
                kline_factor_calc = KLineFactorCalculator(data_loader=eng["loader"])
                query_date_str = datetime.now().strftime('%Y%m%d')
                hybrid_win_rate_result = kline_factor_calc.calculate_hybrid_win_rate(
                    matches, 
                    query_symbol=symbol,
                    query_date=query_date_str,
                    query_df=df
                )
                if isinstance(hybrid_win_rate_result, dict):
                    hybrid_win_rate = hybrid_win_rate_result.get('hybrid_win_rate', traditional_win_rate)
                else:
                    hybrid_win_rate = traditional_win_rate
                    hybrid_win_rate_result = None
                logger.info(f"混合胜率计算成功: {symbol}, 胜率={hybrid_win_rate:.1f}%")
            except Exception as e:
                logger.warning(f"混合胜率计算失败，使用传统胜率: {symbol}, 错误={str(e)}")
                hybrid_win_rate = traditional_win_rate
                hybrid_win_rate_result = None
            progress.progress(85)
            
            win_rate = hybrid_win_rate if hybrid_win_rate is not None else traditional_win_rate
            enhanced_factor = None
            enhanced_score = None
            if isinstance(hybrid_win_rate_result, dict):
                enhanced_factor = hybrid_win_rate_result.get("enhanced_factor")
                if isinstance(enhanced_factor, dict):
                    enhanced_score = enhanced_factor.get("final_score")
            # 多因子评分使用增强因子分数（若有），否则回退混合胜率
            win_rate_for_score = enhanced_score if enhanced_score is not None else win_rate

            df_f = eng["factor"]._add_technical_indicators(df)
            news_text = eng["news"].get_latest_news(symbol)
            ind_name, peers_df = eng["fund"].get_industry_peers(symbol)
            progress.progress(95)

            returns = df['Close'].pct_change().dropna()
            total_score, initial_action, s_details = eng["factor"].get_scorecard(
                win_rate_for_score, df_f.iloc[-1], fund_data, returns=returns if not returns.empty else None
            )

            ic_summary = (st.session_state.get("ic_summary") or {}).get(symbol)
            report = eng["agent"].analyze(
                symbol, total_score, initial_action,
                {"win_rate": win_rate, "score": 0.9},
                df_f.iloc[-1].to_dict(), fund_data, news_text,
                ic_summary=ic_summary
            )

            c_p = os.path.join(PROJECT_ROOT, "data", "comparison.png")
            create_comparison_plot(q_p, matches, c_p)

            res_dict = {
                "name": stock_name, "c_p": c_p, "trajs": trajs, "mean": mean_path,
                "win": win_rate, "ret": avg_ret, "labels": traj_labels,
                "score": total_score, "act": initial_action, "det": s_details,
                "fund": fund_data, "df_f": df_f, "ind": ind_name, "peers": peers_df,
                "news": news_text, "rep": report,
                "mh_stats": mh_stats, "dist_stats": dist_stats,
                "matches": matches, "q_p": q_p,
                "quality_report": quality_report
            }
            if enhanced_factor:
                res_dict["enhanced_factor"] = enhanced_factor
                res_dict["enhanced_score"] = enhanced_score
            
            if hybrid_win_rate_result and hybrid_win_rate is not None:
                res_dict["hybrid_win_rate"] = hybrid_win_rate
                res_dict["traditional_win_rate"] = traditional_win_rate
                res_dict["tb_win_rate"] = hybrid_win_rate_result.get('tb_win_rate', 0)
                res_dict["win_rate_type"] = "混合胜率"
            else:
                res_dict["win_rate_type"] = "传统胜率"
            
            st.session_state.res = res_dict
            progress.progress(100)
            status.empty()
            ic_summary_txt = ""
            if ic_summary:
                ic_summary_txt = f"IC均值: {ic_summary.get('mean_ic')} | IR: {ic_summary.get('ir')} | 显著性: {ic_summary.get('significant')}"

            st.session_state.last_context = f"""
            股票名称: {stock_name} ({symbol})
            当前时间: {datetime.now().strftime('%Y-%m-%d')}
            --- 量化数据 ---
            AI评分: {total_score}/10
            趋势信号: {initial_action}
            形态胜率: {win_rate:.1f}%
            IC摘要: {ic_summary_txt}
            --- 财务数据 ---
            ROE: {fund_data.get('roe')}%
            PE(TTM): {fund_data.get('pe_ttm')}
            --- 舆情摘要 ---
            {news_text[:500]}
            --- 初始观点 ---
            {report.reasoning}
            """

            if url_jump_mode:
                st.session_state.clear_url_after_render = True

    if "res" in st.session_state:
        if st.session_state.get("clear_url_after_render", False):
            st.query_params.clear()
            st.session_state.clear_url_after_render = False
        
        d = st.session_state.res
        display_name = (d.get("name") or "").strip()
        if (not display_name) or (display_name == symbol):
            st.markdown(f"# 📊 深度投研报告: {symbol}")
        else:
            st.markdown(f"# 📊 深度投研报告: {display_name} ({symbol})")

        st.subheader("1. 视觉模式识别")
        with st.expander("ℹ️ 数据来源说明", expanded=False):
            st.markdown("""
            **Top10相似K线匹配**:
            - 使用AttentionCAE模型提取K线形态特征
            - 通过FAISS向量数据库搜索历史相似模式
            - 匹配结果包含：股票代码、日期、相似度分数
            - 计算这些历史模式的未来表现作为预测依据
            """)
        if d.get("quality_report"):
            with st.expander("🧪 数据质量报告", expanded=False):
                qr = d["quality_report"]
                st.write(f"质量评分: {qr.get('score', 'N/A')}")
                st.write(f"样本量: {qr.get('data_points', 'N/A')}")
                st.write(f"时间范围: {qr.get('date_range', {}).get('start')} ~ {qr.get('date_range', {}).get('end')}")
                if qr.get("missing_stats"):
                    st.write(f"缺失率: {qr['missing_stats'].get('missing_ratio', 0)*100:.2f}%")
                    by_col = qr["missing_stats"].get("by_column", {})
                    if by_col:
                        fig_miss = go.Figure()
                        fig_miss.add_trace(go.Bar(x=list(by_col.keys()), y=list(by_col.values())))
                        fig_miss.update_layout(height=250, title="缺失值分布")
                        st.plotly_chart(fig_miss, use_container_width=True)
                if qr.get("adjust_integrity"):
                    adj = qr["adjust_integrity"]
                    if adj.get("available"):
                        st.write(f"复权完整性: {adj.get('column')} 缺失率 {adj.get('missing_ratio', 0)*100:.2f}%")
                    else:
                        st.write("复权完整性: 未提供复权列")
                if qr.get("warnings"):
                    st.write("警告: " + "; ".join(qr.get("warnings", [])[:5]))
        st.image(d['c_p'], use_container_width=True)

        # 相似度分解（视觉相似度/相关性）
        if d.get("matches"):
            rows = []
            for m in d["matches"]:
                vector_score = m.get("vector_score")
                corr = m.get("correlation")
                sim_score = m.get("sim_score")
                if sim_score is None:
                    if vector_score is not None:
                        sim_score = 1.0 / (1.0 + max(float(vector_score), 0.0))
                    else:
                        sim_score = m.get("score", 0)
                corr_norm = None if corr is None else (float(corr) + 1.0) / 2.0
                pix_sim = m.get("pixel_sim")
                edge_sim = m.get("edge_sim")
                ret_corr = m.get("ret_corr")
                rows.append({
                    "股票": f"{m.get('symbol')}",
                    "日期": f"{m.get('date')}",
                    "相似度": round(float(sim_score), 4),
                    "像素相似": round(float(pix_sim), 4) if pix_sim is not None else 0.0,
                    "边缘相似": round(float(edge_sim), 4) if edge_sim is not None else 0.0,
                    "相关性": round(float(corr_norm), 4) if corr_norm is not None else 0.0,
                    "回报相关": round(float((ret_corr+1)/2), 4) if ret_corr is not None else 0.0,
                    "最终分": round(float(m.get("score", 0)), 4)
                })
            with st.expander("🔍 相似度分解（可解释）", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 注意力热力图（如果模型支持）
        try:
            if hasattr(eng["vision"].model, "get_attention_weights"):
                with st.expander("🔥 注意力热力图（解释性）", expanded=False):
                    mode = st.selectbox("显示方式", ["多头(全部)", "单头"], index=0, key="attn_mode")
                    heat_path = os.path.join(PROJECT_ROOT, "data", "temp_attention.png")
                    if mode == "多头(全部)":
                        eng["vision"].generate_attention_heatmap(d.get("q_p"), save_path=heat_path, mode="all")
                        st.image(heat_path, use_container_width=True)
                    else:
                        num_heads = getattr(eng["vision"].model, "num_attention_heads", 8)
                        head_idx = st.slider("选择注意力头", 0, max(0, num_heads - 1), 0, key="attn_head")
                        eng["vision"].generate_attention_heatmap(d.get("q_p"), save_path=heat_path, head_idx=head_idx, mode="single")
                        st.image(heat_path, use_container_width=True)
                    if os.path.exists(heat_path):
                        os.remove(heat_path)
        except Exception:
            pass
        if d['trajs']:
            fig = go.Figure()
            for i, p in enumerate(d['trajs']):
                fig.add_trace(go.Scatter(y=p, mode='lines', line=dict(color='rgba(200,200,200,0.5)', width=1),
                                         name=d['labels'][i]))
            fig.add_trace(go.Scatter(y=d['mean'], mode='lines+markers', line=dict(color='#d62728', width=3), name='平均预期'))
            fig.update_layout(title=f"未来5日走势推演 (胜率: {d['win']:.0f}%)", xaxis_title="天数", yaxis_title="收益%", height=400)
            st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("历史胜率", f"{d['win']:.1f}%")
            c2.metric("预期收益", f"{d['ret']:.2f}%")
            if d.get("enhanced_score") is not None:
                c3.metric("增强因子分", f"{d['enhanced_score']:.2f}")
                c4.metric("信号强度", d.get("enhanced_factor", {}).get("signal_level", "N/A"))
            else:
                c3.metric("增强因子分", "N/A")
                c4.metric("信号强度", "N/A")
            
            # 胜率计算公式说明
            if d.get('win_rate_type') == '混合胜率' and 'hybrid_win_rate' in d:
                with c3:
                    with st.expander("📐 胜率计算公式", expanded=False):
                        st.markdown("""
                        **混合胜率 = Triple Barrier胜率 × 70% + 传统胜率 × 30%**
                        
                        - **Triple Barrier胜率**: 基于止盈(+5%)、止损(-3%)、最大持有20天的标签统计
                        - **传统胜率（相似度加权）**: 未来5日收益率>0 的比例，按 Top10 匹配的相似度 `score` 加权
                        - **数据来源**: Top10相似K线模式的历史表现
                        """)
                        if 'tb_win_rate' in d:
                            st.caption(f"TB胜率: {d.get('tb_win_rate', 0):.1f}% | 传统胜率: {d.get('traditional_win_rate', 0):.1f}%")

            # 多期收益曲线（5/10/20）
            mh = d.get("mh_stats", {})
            if mh.get("valid") and mh.get("horizon_stats"):
                hs = mh["horizon_stats"]
                mh_fig = go.Figure()
                for h, stats in hs.items():
                    mh_fig.add_trace(go.Scatter(
                        x=[h], y=[stats.get("avg_return", 0)],
                        mode="markers+text", text=[f"{h}日"],
                        name=f"{h}日"
                    ))
                mh_fig.update_layout(title="多期收益预期（5/10/20日）", xaxis_title="持有期(天)", yaxis_title="均值收益(%)", height=280)
                st.plotly_chart(mh_fig, use_container_width=True)

            # 收益分布估计
            dist = d.get("dist_stats", {})
            if dist.get("valid"):
                with st.expander("📊 收益分布估计（更严格）", expanded=False):
                    st.write(f"样本数: {dist.get('count')}")
                    st.write(f"均值: {dist.get('mean'):.2f}% | 中位数: {dist.get('median'):.2f}%")
                    st.write(f"分位数: Q05={dist.get('q05'):.2f}%, Q25={dist.get('q25'):.2f}%, Q75={dist.get('q75'):.2f}%")
                    st.write(f"CVaR(5%): {dist.get('cvar'):.2f}%")

            # 复合因子解释（分布 + 情境 + 量价）
            if d.get("enhanced_factor"):
                ef = d["enhanced_factor"]
                with st.expander("🧭 情境感知与量价复合因子（新增）", expanded=False):
                    st.write(f"最佳持有期: {ef.get('best_horizon', 'N/A')} 天")
                    st.write(f"信号强度: {ef.get('signal_level', 'N/A')} | 增强因子分: {ef.get('final_score', 'N/A')}")
                    context = ef.get("context", {})
                    st.caption(f"Regime: {context.get('regime')} | 波动率: {context.get('volatility')} | 流动性评分: {context.get('liquidity_score')}")
                    money = ef.get("money_features", {})
                    if money:
                        st.write("量价/资金特征:")
                        st.json(money)
                    dist_map = ef.get("dist_map", {})
                    if dist_map:
                        rows = []
                        for h, stats in dist_map.items():
                            if not stats or not stats.get("valid"):
                                continue
                            rows.append({
                                "持有期": h,
                                "均值": round(stats.get("mean", 0), 2),
                                "胜率": round(stats.get("win_rate", 0), 2),
                                "CVaR": round(stats.get("cvar", 0), 2),
                                "偏度": stats.get("skew"),
                                "峰度": stats.get("kurt"),
                                "赔率": stats.get("odds")
                            })
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        c_left, c_right = st.columns([1.5, 1])
        with c_left:
            st.subheader("2. 量化多因子看板")
            with st.expander("ℹ️ 因子说明", expanded=False):
                st.markdown("""
                **多因子评分系统 (V+F+Q)**:
                - **V (视觉因子)**: K线学习因子胜率，权重60%
                - **F (基本面因子)**: ROE、PE、PB等，权重20%
                - **Q (技术因子)**: MA、RSI、MACD等，权重20%
                - **动态权重**: 根据市场regime自动调整
                """)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AI 总评分", f"{d['score']}/10", delta=d['act'])
            fund_ok = (d.get("fund", {}) or {}).get("_ok", {})
            m2.metric("ROE", f"{d['fund'].get('roe')}%" if fund_ok.get("finance") else "N/A")
            m3.metric("PE", f"{d['fund'].get('pe_ttm')}" if fund_ok.get("spot") else "N/A")
            m4.metric("趋势", "看涨" if d['df_f'].iloc[-1]['MA_Signal'] > 0 else "看跌")

            with st.expander("📊 杜邦分析 & 因子明细"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**杜邦拆解**")
                    if fund_ok.get("finance"):
                        st.write(f"净利率: {d['fund'].get('net_profit_margin')}%")
                        st.write(f"周转率: {d['fund'].get('asset_turnover')}")
                        st.write(f"权益乘数: {d['fund'].get('leverage')}x")
                with col_b:
                    st.write("**技术因子**")
                    st.json(d['det'])

            # 解释性评分（V/F/Q贡献）
            det = d.get("det", {})
            if det.get("视觉权重") is not None or det.get("regime"):
                with st.expander("🧭 Regime 动态权重", expanded=False):
                    st.write(f"当前Regime: {det.get('regime', 'N/A')}")
                    weights_df = pd.DataFrame([
                        {"因子": "视觉(V)", "权重": det.get("视觉权重", "N/A")},
                        {"因子": "基本面(F)", "权重": det.get("财务权重", "N/A")},
                        {"因子": "技术(Q)", "权重": det.get("量化权重", "N/A")},
                    ])
                    st.dataframe(weights_df, use_container_width=True, hide_index=True)
            try:
                v = float(det.get("视觉分(V)", 0))
                f = float(det.get("财务分(F)", 0))
                q = float(det.get("量化分(Q)", 0))
                total = v + f + q if (v + f + q) > 0 else 1.0
                contrib = pd.DataFrame([
                    {"因子": "视觉(V)", "贡献": f"{v/total*100:.1f}%"},
                    {"因子": "基本面(F)", "贡献": f"{f/total*100:.1f}%"},
                    {"因子": "技术(Q)", "贡献": f"{q/total*100:.1f}%"},
                ])
                with st.expander("🧠 可解释性评分贡献", expanded=False):
                    st.dataframe(contrib, use_container_width=True, hide_index=True)
            except Exception:
                pass


        with c_right:
            st.subheader(f"3. 行业对标 ({d['ind']})")
            st.dataframe(d['peers'], hide_index=True)

        st.divider()
        st.subheader("4. 新闻舆情")
        with st.expander("ℹ️ 数据来源", expanded=False):
            st.markdown("""
            **新闻数据来源**: 
            - 通过akshare接口获取最新新闻
            - 包含公司公告、行业动态、市场资讯
            """)
        st.info(d['news'])

        # ---- 单只股票页：先因子/回测，最后 AI ----
        st.divider()
        tab_bt, tab_fa = st.tabs(["🧪 回测", "📈 因子有效性分析"])

        with tab_bt:
            st.subheader("🧪 策略模拟回测")
            with st.expander("ℹ️ 回测说明", expanded=False):
                st.markdown("""
                **回测策略逻辑**:
                - **仓位计算**: 基于MA60、MA20、MACD和AI胜率阈值
                - **Transaction Cost**: 佣金(0.1%) + 滑点(0.1%) + 市场冲击 + 机会成本
                - **Turnover约束**: 单日最大换手率20%
                - **涨跌停/停牌约束**: 涨停不追、跌停不砍、停牌不交易（A股执行约束）
                - **止损机制**: 达到止损阈值(-8%)时强制平仓
                - **Walk-Forward**: 滚动窗口验证，防止未来函数泄漏
                """)

            cbt1, cbt2, cbt3, cbt4 = st.columns(4)
            with cbt1:
                bt_start_val = st.date_input("开始日期", value=st.session_state.get("bt_start", datetime(2010, 1, 1)), key="bt_start")
            with cbt2:
                bt_end_val = st.date_input("结束日期", value=st.session_state.get("bt_end", datetime.now()), key="bt_end")
            with cbt3:
                bt_cap_val = st.number_input("初始资金", value=st.session_state.get("bt_cap", 100000), key="bt_cap")
            with cbt4:
                bt_ma_val = st.slider("MA周期", 20, 120, st.session_state.get("bt_ma", 60), key="bt_ma")

            cbt5, cbt6, cbt7 = st.columns(3)
            with cbt5:
                bt_stop_val = st.slider("止损%", 1, 20, st.session_state.get("bt_stop", 8), key="bt_stop")
            with cbt6:
                bt_vision_val = st.slider("AI胜率阈值", 40, 80, st.session_state.get("bt_vision", 57), key="bt_vision")
            with cbt7:
                bt_validation_val = st.selectbox("验证模式", ["简单回测", "Walk-Forward验证（严格）"], index=0 if st.session_state.get("bt_validation", "简单回测") == "简单回测" else 1, key="bt_validation")

            if bt_validation_val == "Walk-Forward验证（严格）":
                wf1, wf2 = st.columns(2)
                with wf1:
                    wf_train_months_val = st.slider("训练期(月)", 6, 36, st.session_state.get("wf_train_months", 24), key="wf_train_months")
                with wf2:
                    wf_test_months_val = st.slider("测试期(月)", 1, 12, st.session_state.get("wf_test_months", 6), key="wf_test_months")
            else:
                wf_train_months_val, wf_test_months_val = 24, 6

            enable_stress = st.checkbox(
                "启用Stress Testing",
                value=st.session_state.get("enable_stress", False),
                key="enable_stress",
                help="在极端市场条件下测试策略鲁棒性（2008金融危机、2020疫情崩盘、2015股灾）",
            )
            strict_no_future = st.checkbox(
                "严格无未来函数（更慢）",
                value=st.session_state.get("strict_no_future", True),
                key="strict_no_future",
                help="仅使用当前日期及之前的相似形态，避免未来数据泄漏"
            )
            if strict_no_future:
                cbt8, cbt9 = st.columns(2)
                with cbt8:
                    ai_stride_val = st.slider(
                        "AI评估步长(天)",
                        1, 20,
                        st.session_state.get("ai_stride", 5),
                        key="ai_stride",
                        help="步长越大越快，但精度会下降；设为1表示逐日评估"
                    )
                with cbt9:
                    ai_fast_mode_val = st.checkbox(
                        "快速AI评估（向量近似）",
                        value=st.session_state.get("ai_fast_mode", True),
                        key="ai_fast_mode",
                        help="跳过DTW/相关性计算，显著加速但精度略降"
                    )
            else:
                ai_stride_val, ai_fast_mode_val = 1, False

            if st.button("开始回测", key="backtest_btn"):
                run_backtest(
                    symbol, bt_start_val, bt_end_val, bt_cap_val, bt_ma_val,
                    bt_stop_val, bt_vision_val, bt_validation_val,
                    wf_train_months_val, wf_test_months_val, eng, PROJECT_ROOT,
                    enable_stress_test=enable_stress, strict_no_future=strict_no_future,
                    ai_stride=ai_stride_val, ai_fast_mode=ai_fast_mode_val
                )

        with tab_fa:
            st.subheader("📈 因子有效性分析")
            render_factor_analysis(symbol, d["df_f"], eng, PROJECT_ROOT)

        st.divider()
        st.subheader("5. AI 基金经理终审")
        color = "green" if d['rep'].action == "BUY" else "red" if d['rep'].action == "SELL" else "orange"
        st.markdown(f"""
        <div class="agent-box">
            <h2 style="color:{color}; margin:0">{d['rep'].action}</h2>
            <p>信心: {d['rep'].confidence}% | 风险: {d['rep'].risk_level}</p>
            <hr><p>{d['rep'].reasoning}</p>
        </div>
        """, unsafe_allow_html=True)

        pdf_p = os.path.join(PROJECT_ROOT, "data", f"Report_{symbol}.pdf")
        if st.button("📄 导出 PDF"):
            generate_report_pdf(f"{d['name']}({symbol})", d['rep'], d['c_p'], pdf_p)
            with open(pdf_p, "rb") as f:
                st.download_button("下载 PDF", f, file_name=f"VQ_{symbol}.pdf")

        st.divider()
        st.subheader("💬 智能对话")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): 
                st.markdown(msg["content"])

        c_mic, c_input = st.columns([1, 8])
        user_voice_text = None
        with c_mic:
            st.write(" ")
            audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format='wav')
        if audio:
            transcribed = eng["audio"].transcribe(audio['bytes'])
            if transcribed and transcribed != st.session_state.last_voice_text:
                user_voice_text = transcribed
                st.session_state.last_voice_text = transcribed
        with c_input:
            text_input = st.chat_input("输入问题...")
        final_input = user_voice_text if user_voice_text else text_input

        if final_input:
            st.session_state.chat_history.append({"role": "user", "content": final_input})
            st.rerun()

        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            user_q = st.session_state.chat_history[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    resp = eng["agent"].chat(user_q, st.session_state.last_context)
                    st.markdown(resp)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})

elif mode == "📊 批量组合分析":
    if run_btn:
        symbols = [s.strip().zfill(6) for s in batch_input.split('\n') if s.strip()][:30]
        if len(symbols) == 0:
            st.error("❌ 请输入至少一只股票代码")
            st.stop()
        
        st.session_state.has_run = True
        if "batch_results" in st.session_state:
            del st.session_state.batch_results
        if "multi_tier_result" in st.session_state:
            del st.session_state.multi_tier_result
        if "portfolio_metrics" in st.session_state:
            del st.session_state.portfolio_metrics
        
        batch_analyzer = BatchAnalyzer(eng)
        portfolio_optimizer = PortfolioOptimizer()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, symbol):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"正在分析 {symbol} ({current}/{total})...")
        
        batch_results = batch_analyzer.analyze_batch(symbols, progress_callback=update_progress)
        st.session_state.batch_results = batch_results
        progress_bar.progress(1.0)
        status_text.text("✅ 分析完成")
        progress_bar.empty()
        status_text.empty()

        if not batch_results:
            st.error("批量分析失败或无有效数据")
            st.stop()

        # 统一组合优化（即使没有 BUY 也能输出“增强组合”）
        multi_tier_result = portfolio_optimizer.optimize_multi_tier_portfolio(
            batch_results, eng["loader"], min_weight=0.05, max_weight=0.25, max_positions=10
        )
        st.session_state.multi_tier_result = multi_tier_result

        buy_stocks = {k: v for k, v in batch_results.items() if v.get('action') == 'BUY' and v.get('score', 0) >= 7}
        wait_stocks = {k: v for k, v in batch_results.items() if v.get('action') == 'WAIT'}
        sell_stocks = {k: v for k, v in batch_results.items() if v.get('action') == 'SELL'}

        def _goto_symbol(sym: str):
            if "res" in st.session_state:
                del st.session_state.res
            st.session_state.current_symbol = None
            st.session_state.has_run = False
            st.session_state["symbol_input"] = sym
            st.session_state["mode_select"] = "🔍 单只股票分析"
            st.query_params.update({"symbol": sym, "mode": "detail"})
            st.rerun()

        tier_info = multi_tier_result.get("tier_info", {})
        if tier_info:
            st.info(
                f"组合策略: {tier_info.get('strategy', '-')}"
                f" | 优化器: {tier_info.get('optimizer', 'Black-Litterman')}"
                f" | 说明: {tier_info.get('description', '-')}"
            )

        st.subheader("✅ 组合结果（核心 + 备选）")
        core_weights = multi_tier_result.get('core', {})
        enhanced_weights = multi_tier_result.get('enhanced', {})
        combined_weights = {}
        combined_weights.update(core_weights)
        combined_weights.update(enhanced_weights)

        def _render_weights_table(title, weights):
            st.markdown(f"### {title}")
            if not weights:
                st.info("暂无可用组合")
                return
            rows = []
            for sym, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                data = batch_results.get(sym, {})
                rows.append({
                    "股票代码": sym,
                    "股票名称": data.get("name", sym),
                    "权重": f"{w*100:.1f}%",
                    "评分": f"{data.get('score', 0):.1f}/10",
                    "胜率": f"{data.get('win_rate', 0):.1f}%",
                    "预期收益": f"{data.get('expected_return', 0):.2f}%"
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            for sym, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                data = batch_results.get(sym, {})
                c1, c2, c3, c4 = st.columns([3, 1, 1, 4])
                with c1:
                    if st.button(f"📊 {data.get('name', sym)} ({sym})", key=f"link_{title}_{sym}", use_container_width=True):
                        _goto_symbol(sym)
                with c2:
                    st.write(f"**{data.get('score', 0):.1f}/10**")
                with c3:
                    st.write(f"{w*100:.1f}%")
                with c4:
                    st.write(f"{data.get('action', 'WAIT')} - {data.get('reasoning', '')[:60]}")

        _render_weights_table("核心推荐组合", core_weights)
        _render_weights_table("备选增强组合", enhanced_weights)

        st.subheader("📌 仓位设计与风控设置")
        c1, c2, c3 = st.columns(3)
        c1.metric("最小仓位", "5%")
        c2.metric("最大仓位", "25%")
        c3.metric("最大持仓数", "10")
        st.caption("止盈/止损参考：标签止盈 +5%、标签止损 -3%；回测止损默认 -8%（可在单股回测中调整）")

        if combined_weights:
            st.subheader("📊 组合权重图表")
            # 拥挤交易指标
            hhi = sum([w**2 for w in combined_weights.values()])
            top3 = sum(sorted(combined_weights.values(), reverse=True)[:3])
            st.caption(f"拥挤度(HHI): {hhi:.4f} | 前三集中度: {top3*100:.1f}%")
            
            # 组合指标与风险预算
            try:
                metrics = portfolio_optimizer.calculate_portfolio_metrics(combined_weights, batch_results, eng["loader"])
                if metrics:
                    st.subheader("🧾 组合风险指标")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("期望收益", f"{metrics.get('expected_return', 0):.2f}%")
                    m2.metric("风险(波动)", f"{metrics.get('risk', 0):.2f}%")
                    m3.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    m4.metric("CVaR(5%)", f"{metrics.get('cvar', 0):.2f}%")
                    if metrics.get("risk_budget"):
                        with st.expander("风险预算分解", expanded=False):
                            rb = pd.DataFrame(
                                [{"symbol": k, "risk_contrib": v} for k, v in metrics["risk_budget"].items()]
                            )
                            st.dataframe(rb, use_container_width=True, hide_index=True)
            except Exception:
                pass

            # 再平衡建议（基于上次权重 + 换手上限）
            prev_weights = st.session_state.get("portfolio_weights", {})
            try:
                rebalance_weights, rebalance_info = portfolio_optimizer.propose_rebalance(
                    prev_weights, combined_weights, max_turnover=0.20
                )
                st.session_state.portfolio_weights = combined_weights
                with st.expander("🔁 再平衡建议（换手≤20%）", expanded=False):
                    st.write(f"预计换手: {rebalance_info.get('turnover', 0)*100:.1f}%")
                    r_df = pd.DataFrame([
                        {"symbol": s, "current": round(prev_weights.get(s, 0)*100, 1), "target": round(rebalance_weights.get(s, 0)*100, 1)}
                        for s in set(prev_weights) | set(rebalance_weights)
                    ])
                    st.dataframe(r_df, use_container_width=True, hide_index=True)
            except Exception:
                pass
            labels = [f"{batch_results[s].get('name', s)}({s})" for s in combined_weights.keys()]
            values = [combined_weights[s] for s in combined_weights.keys()]
            pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
            pie.update_layout(height=320, title="组合权重分布")
            st.plotly_chart(pie, use_container_width=True)

            bar = go.Figure()
            bar.add_trace(go.Bar(x=labels, y=[batch_results[s].get('score', 0) for s in combined_weights.keys()],
                                 name="评分", marker_color="#ff4b4b"))
            bar.update_layout(height=300, title="评分对比")
            st.plotly_chart(bar, use_container_width=True)

            scatter = go.Figure()
            for s in combined_weights.keys():
                scatter.add_trace(go.Scatter(
                    x=[batch_results[s].get('win_rate', 0)],
                    y=[batch_results[s].get('expected_return', 0)],
                    mode='markers+text',
                    text=[s],
                    marker=dict(size=max(8, combined_weights[s]*200), color="#1f77b4"),
                    name=s
                ))
            scatter.update_layout(height=320, title="胜率 vs 预期收益", xaxis_title="胜率(%)", yaxis_title="预期收益(%)")
            st.plotly_chart(scatter, use_container_width=True)

            st.subheader("🕯️ 组合Top3 K线展示")
            top_syms = [s for s, _ in sorted(combined_weights.items(), key=lambda x: x[1], reverse=True)[:3]]
            if top_syms:
                cols = st.columns(len(top_syms))
                for i, sym in enumerate(top_syms):
                    try:
                        dfk = eng["loader"].get_stock_data(sym)
                        if dfk is None or dfk.empty:
                            continue
                        tmp_img = os.path.join(PROJECT_ROOT, "data", f"temp_batch_k_{sym}.png")
                        mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
                        sstyle = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
                        mpf.plot(dfk.tail(60), type='candle', style=sstyle,
                                 savefig=dict(fname=tmp_img, dpi=80), figsize=(4, 3), axisoff=True)
                        with cols[i]:
                            st.image(tmp_img, caption=f"{sym}", use_container_width=True)
                        if os.path.exists(tmp_img):
                            os.remove(tmp_img)
                    except Exception:
                        continue

            # 分层回测（行业/市值/风格 + 显著性）
            with st.expander("🧪 分层回测（行业/市值/风格）", expanded=False):
                if st.button("运行分层回测", key="strat_bt_btn"):
                    strat_df = run_stratified_backtest_batch(list(batch_results.keys()), eng)
                    if strat_df is not None and not strat_df.empty:
                        st.dataframe(strat_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("分层样本不足或数据不可用")

            # 权重动态变化（简化：基于20日动量的月度再平衡）
            try:
                st.subheader("📈 组合权重动态变化")
                top_syms = list(combined_weights.keys())[:6]
                weight_df = pd.DataFrame()
                for sym in top_syms:
                    dfw = eng["loader"].get_stock_data(sym)
                    if dfw is None or dfw.empty:
                        continue
                    dfw.index = pd.to_datetime(dfw.index)
                    dfw = dfw.tail(180)
                    mom = dfw["Close"].pct_change(20)
                    dfw = dfw.assign(mom=mom)
                    dfw = dfw.resample("M").last().dropna()
                    weight_df[sym] = dfw["mom"]
                if not weight_df.empty:
                    # 归一化为权重
                    weight_df = weight_df.apply(lambda x: x - x.min() + 1e-6)
                    weight_df = weight_df.div(weight_df.sum(axis=1), axis=0)
                    fig_w = go.Figure()
                    for sym in weight_df.columns:
                        fig_w.add_trace(go.Scatter(x=weight_df.index, y=weight_df[sym], mode="lines", name=sym))
                    fig_w.update_layout(height=320, title="月度权重演化（动量驱动）")
                    st.plotly_chart(fig_w, use_container_width=True)
            except Exception:
                pass

            # 滚动收益热图
            try:
                st.subheader("🧊 滚动收益热图（20日）")
                heat_syms = list(combined_weights.keys())[:8]
                heat_data = []
                heat_index = None
                for sym in heat_syms:
                    dfh = eng["loader"].get_stock_data(sym)
                    if dfh is None or dfh.empty:
                        continue
                    dfh.index = pd.to_datetime(dfh.index)
                    dfh = dfh.tail(200)
                    roll = dfh["Close"].pct_change(20) * 100
                    if heat_index is None:
                        heat_index = roll.index
                    heat_data.append(roll.reindex(heat_index).fillna(0).values)
                if heat_data:
                    heat = go.Figure(data=go.Heatmap(
                        z=np.array(heat_data),
                        x=[d.strftime("%Y-%m-%d") for d in heat_index],
                        y=heat_syms,
                        colorscale="RdYlGn"
                    ))
                    heat.update_layout(height=320)
                    st.plotly_chart(heat, use_container_width=True)
            except Exception:
                pass
        
        if wait_stocks or sell_stocks:
            st.divider()
            st.subheader("⚠️ 观望/卖出列表")
            all_other = {**wait_stocks, **sell_stocks}
            if all_other:
                for symbol, data in sorted(all_other.items(), key=lambda x: x[1].get('score', 0)):
                    col1, col2, col3 = st.columns([3, 1, 4])
                    with col1:
                        if st.button(f"📊 {data.get('name', symbol)} ({symbol})", 
                                   key=f"link_other_{symbol}", use_container_width=True):
                            _goto_symbol(symbol)
                    with col2:
                        st.write(f"**{data.get('score', 0):.1f}/10**")
                    with col3:
                        st.write(f"{data.get('action', 'WAIT')} - {data.get('reasoning', '')[:50]}")
                    st.divider()
    
    else:
        st.info("👈 请在左侧输入股票代码并点击启动")
