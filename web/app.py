import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import mplfinance as mpf
import plotly.graph_objects as go
from datetime import datetime
import pickle
from streamlit_mic_recorder import mic_recorder
import importlib

# ================= 路径与环境配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
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
except ImportError as e:
    st.error(f"❌ 系统模块加载失败: {e}. 请确保 src 目录下文件完整。")
    st.stop()

# ================= 代码版本（用于缓存失效 + 热更新） =================
def _code_version_key() -> str:
    """
    Streamlit 会缓存 resource；但 Python import 默认不会热更新。
    这里用源码 mtime 作为 cache key，并在 load_all_engines 内部 importlib.reload，
    以确保你改了 src 代码后无需手动重启也能生效。
    """
    paths = [
        os.path.join(PROJECT_ROOT, "src", "models", "vision_engine.py"),
        os.path.join(PROJECT_ROOT, "src", "strategies", "fundamental.py"),
    ]
    parts = []
    for p in paths:
        try:
            parts.append(str(os.path.getmtime(p)))
        except Exception:
            parts.append("0")
    return "|".join(parts)

# ================= 页面配置 =================
st.set_page_config(page_title="VisionQuant Pro", layout="wide", page_icon="🦄")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e6e9ef; }
    /* 核心决策框样式 */
    .agent-box { border-left: 5px solid #ff4b4b; padding: 20px; background-color: #fff1f1; border-radius: 5px; margin-bottom: 20px; }
    /* 聊天气泡 */
    .stChatMessage { background-color: #ffffff; border-radius: 12px; padding: 12px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    </style>
""", unsafe_allow_html=True)


# ================= 引擎初始化 =================
@st.cache_resource
def load_all_engines(_code_version: str):
    # 强制热重载关键模块（Vision/Fundamental），避免“改了代码网页还是旧效果”
    ve_mod = importlib.import_module("src.models.vision_engine")
    fm_mod = importlib.import_module("src.strategies.fundamental")
    importlib.reload(ve_mod)
    importlib.reload(fm_mod)

    VisionEngineReloaded = ve_mod.VisionEngine
    FundamentalMinerReloaded = fm_mod.FundamentalMiner

    v = VisionEngineReloaded()
    v.reload_index()
    return {
        "loader": DataLoader(), "vision": v, "factor": FactorMiner(),
        "fund": FundamentalMinerReloaded(), "agent": QuantAgent(), "news": NewsHarvester(),
        "audio": AudioManager()
    }


eng = load_all_engines(_code_version=_code_version_key())

# === Session State 初始化 ===
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "last_context" not in st.session_state: st.session_state.last_context = ""
if "has_run" not in st.session_state: st.session_state.has_run = False
# 新增：防复读锁
if "last_voice_text" not in st.session_state: st.session_state.last_voice_text = ""
# 批量分析结果缓存
if "batch_results" not in st.session_state: st.session_state.batch_results = {}
if "portfolio_weights" not in st.session_state: st.session_state.portfolio_weights = {}
if "portfolio_metrics" not in st.session_state: st.session_state.portfolio_metrics = {}
# 当前分析的股票代码（用于检测切换）
if "current_symbol" not in st.session_state: st.session_state.current_symbol = None

# =========================================================
#  侧边栏 (Sidebar)
# =========================================================
with st.sidebar:
    st.title("🦄 VisionQuant Pro")
    st.caption("AI 全栈量化投研系统 v8.8")
    st.divider()

    symbol_input = st.text_input("请输入 A 股代码", value="601899", help="输入6位代码")
    symbol = symbol_input.strip().zfill(6)

    mode = st.radio("功能模块:", ("🔍 单只股票分析", "📊 批量组合分析"))

    if mode == "🔍 单只股票分析":
        st.divider()
        st.subheader("分析选项")
        show_backtest = st.checkbox("🧪 显示策略回测", value=False, help="在分析结果中显示回测功能")
        show_factor_analysis = st.checkbox("📈 显示因子有效性分析", value=False, help="在分析结果中显示因子分析功能")
        
        if show_backtest:
            st.markdown("**回测参数**")
            bt_start = st.date_input("开始日期", datetime(2022, 1, 1), key="bt_start")
            bt_end = st.date_input("结束日期", datetime.now(), key="bt_end")
            bt_cap = st.number_input("初始本金", 100000, key="bt_cap")
            bt_ma = st.slider("趋势线周期 (MA)", 20, 120, 60, key="bt_ma")
            bt_stop = st.slider("止损阈值 (%)", 3, 15, 8, key="bt_stop")
            bt_vision = st.slider("AI 介入阈值 (Win%)", 50, 70, 57, key="bt_vision")
            
            # Walk-Forward验证选项
            bt_validation = st.radio(
                "验证方式",
                ("简单回测", "Walk-Forward验证（严格）"),
                help="Walk-Forward验证模拟真实交易，使用滚动窗口防止未来函数泄漏",
                key="bt_validation"
            )
            if bt_validation == "Walk-Forward验证（严格）":
                wf_train_months = st.slider("训练期（月）", 6, 36, 24, help="每次训练使用的历史数据长度", key="wf_train_months")
                wf_test_months = st.slider("测试期（月）", 3, 12, 6, help="每次测试的时间长度", key="wf_test_months")
    
    if mode == "📊 批量组合分析":
        st.divider()
        st.subheader("批量分析参数")
        batch_input = st.text_area(
            "请输入股票代码（每行一个，最多30只）",
            value="601899\n600519\n000001",
            height=150,
            help="每行一个6位股票代码"
        )
        max_positions = st.slider("最大持仓数量", 5, 15, 10)
        min_weight = st.slider("最小仓位 (%)", 3, 10, 5) / 100
        max_weight = st.slider("最大仓位 (%)", 15, 30, 20) / 100

    st.divider()
    # ================== 强制重载（解决缓存导致的 N/A / 旧逻辑不生效） ==================
    if st.button("🔄 强制重载引擎（清缓存）", use_container_width=True, help="当你更新代码/数据后，点击此按钮让 Fundamental/Vision 等引擎重新初始化"):
        try:
            load_all_engines.clear()
        except Exception:
            # 兼容不同streamlit版本
            st.cache_resource.clear()

        # 清空常见结果缓存，避免旧数据混入
        for k in ["res", "batch_results", "multi_tier_result", "portfolio_metrics", "portfolio_weights"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.has_run = False
        st.rerun()

    run_btn = st.button("🚀 立即开始分析", type="primary", use_container_width=True)

    if st.button("🧹 清空对话历史"):
        st.session_state.chat_history = []
        st.session_state.last_voice_text = ""
        st.rerun()
    
    # 添加返回按钮（当有URL参数时显示）
    if "symbol" in st.query_params and "mode" in st.query_params:
        if st.button("🔙 返回主界面", use_container_width=True):
            st.query_params.clear()
            if "res" in st.session_state:
                del st.session_state.res
            st.session_state.current_symbol = None
            st.session_state.has_run = False
        st.rerun()

# =========================================================
#  主界面逻辑
# =========================================================

# 检查URL参数（详情页跳转）
query_params = st.query_params
url_jump_mode = False
if "symbol" in query_params and "mode" in query_params and query_params["mode"] == "detail":
    url_symbol = query_params["symbol"].strip().zfill(6)
    # 如果URL中的股票代码与侧边栏不同，使用URL中的
    if url_symbol != symbol:
        symbol = url_symbol
        url_jump_mode = True
        mode = "🔍 实盘深度研判"
        # 清空旧结果
        if "res" in st.session_state:
            del st.session_state.res
        st.session_state.current_symbol = symbol
        st.session_state.has_run = True
        run_btn = True
    elif "res" not in st.session_state:
        # 如果没有结果，触发分析
        url_jump_mode = True
        mode = "🔍 实盘深度研判"
        st.session_state.has_run = True
        run_btn = True
    else:
        # 已有结果，清除URL参数，恢复正常模式
        st.query_params.clear()
        url_jump_mode = False

# 显示欢迎页面（仅在未运行且未点击按钮时）
if not run_btn and not st.session_state.has_run:
    st.header(f"👋 欢迎使用 VisionQuant Pro")
    st.info(f"当前选中标的: **{symbol}**\n请在左侧侧边栏点击红色按钮启动。")
    st.stop()

# --- 模式 A: 单只股票分析（整合因子分析和回测） ---
if mode == "🔍 单只股票分析":
    # 检测股票切换：如果symbol变化，清空旧结果和状态
    if st.session_state.current_symbol != symbol and st.session_state.current_symbol is not None:
        if "res" in st.session_state:
            del st.session_state.res
        st.session_state.has_run = False  # 重置运行状态，允许重新分析
        st.session_state.chat_history = []  # 清空聊天历史
        st.session_state.last_voice_text = ""  # 重置语音锁
    
    if run_btn:
        # 每次点击按钮都重新分析（即使股票代码相同，也允许重新分析）
        st.session_state.has_run = True
        st.session_state.chat_history = []  # 每次新分析清空旧聊天
        st.session_state.last_voice_text = ""  # 重置语音锁
        st.session_state.current_symbol = symbol  # 更新当前股票
        
        # 清空旧结果，强制重新生成
        if "res" in st.session_state:
            del st.session_state.res

        with st.spinner(f"正在全栈扫描 {symbol} (视觉+财务+舆情)..."):
            # 1. 数据
            df = eng["loader"].get_stock_data(symbol)
            if df.empty: st.error("数据获取失败"); st.stop()

            fund_data = eng["fund"].get_stock_fundamentals(symbol)
            stock_name = fund_data.get('name', symbol)

            # 2. 视觉匹配（优化：传入价格序列用于相关性计算）
            q_p = os.path.join(PROJECT_ROOT, "data", "temp_q.png")
            mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
            mpf.plot(df.tail(20), type='candle', style=s, savefig=dict(fname=q_p, dpi=50), figsize=(3, 3), axisoff=True)
            
            # 提取查询价格序列（最近20天收盘价，用于相关性计算）
            query_prices = df.tail(20)['Close'].values if len(df) >= 20 else None
            matches = eng["vision"].search_similar_patterns(q_p, top_k=10, query_prices=query_prices)


            # 轨迹计算
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

            # 3. 使用新的K线因子计算器（混合胜率）
            hybrid_win_rate_result = None
            hybrid_win_rate = None
            try:
                kline_factor_calc = KLineFactorCalculator()
                hybrid_win_rate_result = kline_factor_calc.calculate_hybrid_win_rate(matches, df)
                if hybrid_win_rate_result and 'hybrid_win_rate' in hybrid_win_rate_result:
                    hybrid_win_rate = hybrid_win_rate_result.get('hybrid_win_rate', traditional_win_rate)
                else:
                    hybrid_win_rate = traditional_win_rate
            except Exception as e:
                # 如果Triple Barrier标签不可用，使用传统胜率
                hybrid_win_rate = traditional_win_rate
                hybrid_win_rate_result = None
            
            # 使用混合胜率（如果可用）
            win_rate = hybrid_win_rate if hybrid_win_rate is not None else traditional_win_rate

            # 3. 因子与新闻
            df_f = eng["factor"]._add_technical_indicators(df)
            news_text = eng["news"].get_latest_news(symbol)
            ind_name, peers_df = eng["fund"].get_industry_peers(symbol)

            # 4. 打分（使用动态权重）
            # 获取当前市场regime和动态权重
            returns = df['Close'].pct_change().dropna()
            try:
                regime_weights = eng["regime_manager"].calculate_dynamic_weights(returns=returns)
                dynamic_weights = regime_weights.get('weights', {})
                current_regime = regime_weights.get('regime', 'unknown')
            except:
                dynamic_weights = None
                current_regime = 'unknown'
            
            # 使用动态权重评分（如果可用）
            if dynamic_weights:
                total_score, initial_action, s_details = eng["factor"].get_scorecard(
                    win_rate, df_f.iloc[-1], fund_data,
                    returns=returns
                )
            else:
                total_score, initial_action, s_details = eng["factor"].get_scorecard(win_rate, df_f.iloc[-1], fund_data)

            # 5. Agent
            report = eng["agent"].analyze(symbol, total_score, initial_action,
                                          {"win_rate": win_rate, "score": 0.9},
                                          df_f.iloc[-1].to_dict(), fund_data, news_text)

            # 6. 对比图
            c_p = os.path.join(PROJECT_ROOT, "data", "comparison.png")
            create_comparison_plot(q_p, matches, c_p)

            # === 保存结果到 Session ===
            res_dict = {
                "name": stock_name, "c_p": c_p, "trajs": trajs, "mean": mean_path,
                "win": win_rate, "ret": avg_ret, "labels": traj_labels,
                "score": total_score, "act": initial_action, "det": s_details,
                "fund": fund_data, "df_f": df_f, "ind": ind_name, "peers": peers_df,
                "news": news_text, "rep": report
            }
            
            # 保存混合胜率信息（如果计算了）
            if hybrid_win_rate_result and hybrid_win_rate is not None:
                res_dict["hybrid_win_rate"] = hybrid_win_rate
                res_dict["traditional_win_rate"] = traditional_win_rate
                res_dict["tb_win_rate"] = hybrid_win_rate_result.get('tb_win_rate', 0)
                res_dict["win_rate_type"] = "混合胜率"
            else:
                res_dict["win_rate_type"] = "传统胜率"
            
            st.session_state.res = res_dict

            # 构建上下文给 Chat 用
            st.session_state.last_context = f"""
            股票名称: {stock_name} ({symbol})
            当前时间: {datetime.now().strftime('%Y-%m-%d')}
            --- 量化数据 ---
            AI评分: {total_score}/10
            趋势信号: {initial_action}
            形态胜率: {win_rate:.1f}%
            --- 财务数据 ---
            ROE: {fund_data.get('roe')}%
            PE(TTM): {fund_data.get('pe_ttm')}
            --- 舆情摘要 ---
            {news_text[:500]}
            --- 初始观点 ---
            {report.reasoning}
            """

            # 注意：这里我们不再把初始报告塞进 chat_history，避免重复显示
            
            # 如果是从URL跳转来的，清除URL参数，恢复正常交互
            if url_jump_mode:
                # 延迟清除，确保结果已保存
                st.session_state.clear_url_after_render = True

    # === 渲染界面 ===
    # 显示结果（如果有的话）
    if "res" in st.session_state:
        # 如果标记了需要清除URL，现在清除
        if st.session_state.get("clear_url_after_render", False):
            st.query_params.clear()
            st.session_state.clear_url_after_render = False
        
        d = st.session_state.res

        # 标题：避免出现 “300286 (300286)” 这种重复
        display_name = (d.get("name") or "").strip()
        if (not display_name) or (display_name == symbol):
            st.markdown(f"# 📊 深度投研报告: {symbol}")
        else:
            st.markdown(f"# 📊 深度投研报告: {display_name} ({symbol})")

        # 1. 视觉
        st.subheader("1. 视觉模式识别")
        st.image(d['c_p'], use_container_width=True)
        if d['trajs']:
            fig = go.Figure()
            for i, p in enumerate(d['trajs']):
                fig.add_trace(go.Scatter(y=p, mode='lines', line=dict(color='rgba(200,200,200,0.5)', width=1),
                                         name=d['labels'][i]))
            fig.add_trace(
                go.Scatter(y=d['mean'], mode='lines+markers', line=dict(color='#d62728', width=3), name='平均预期'))
            fig.update_layout(title=f"未来5日走势推演 (胜率: {d['win']:.0f}%)", xaxis_title="天数", yaxis_title="收益%",
                              height=400)
            st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("历史胜率", f"{d['win']:.1f}%")
            c2.metric("预期收益", f"{d['ret']:.2f}%", delta_color="normal")

        # 2. 量化
        st.divider()
        c_left, c_right = st.columns([1.5, 1])
        with c_left:
            st.subheader("2. 量化多因子看板")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AI 总评分", f"{d['score']}/10", delta=d['act'])
            # 基本面抓取失败时，不要把默认0展示成真实值
            fund_ok = (d.get("fund", {}) or {}).get("_ok", {})
            spot_ok = bool(fund_ok.get("spot"))
            finance_ok = bool(fund_ok.get("finance"))

            roe_val = d["fund"].get("roe")
            pe_val = d["fund"].get("pe_ttm")

            m2.metric("ROE", f"{roe_val}%" if finance_ok else "N/A")
            m3.metric("PE", f"{pe_val}" if spot_ok else "N/A")
            m4.metric("趋势", "看涨" if d['df_f'].iloc[-1]['MA_Signal'] > 0 else "看跌")

            with st.expander("📊 杜邦分析 & 因子明细"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**杜邦拆解**")
                    if finance_ok:
                        st.write(f"净利率: {d['fund'].get('net_profit_margin')}%")
                        st.write(f"周转率: {d['fund'].get('asset_turnover')}")
                        st.write(f"权益乘数: {d['fund'].get('leverage')}x")
                    else:
                        st.info("⚠️ 财务报表指标抓取失败，已隐藏杜邦拆解（避免用默认值0误导）。")
                with col_b:
                    st.write("**技术因子**")
                    st.json(d['det'])

            # 基本面抓取失败的提示（收敛在量化看板区，不影响其他模块）
            if (not spot_ok) or (not finance_ok):
                errs = (d.get("fund", {}) or {}).get("_err", [])
                st.warning("⚠️ 基本面数据获取不完整：可能是 akshare 拉取失败/网络波动/接口字段变更。已用 N/A 展示缺失项。")
                if errs:
                    with st.expander("查看基本面抓取错误详情"):
                        st.write("\n".join([f"- {e}" for e in errs]))

        with c_right:
            st.subheader(f"3. 行业对标 ({d['ind']})")
            st.dataframe(d['peers'], hide_index=True)

        # 3. 新闻
        st.divider()
        st.subheader("4. 新闻舆情")
        st.info(d['news'])

        # 4. Agent 决策书 (这里只显示一次，静态的)
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

        # PDF 导出
        pdf_p = os.path.join(PROJECT_ROOT, "data", f"Report_{symbol}.pdf")
        if st.button("📄 导出 PDF"):
            generate_report_pdf(f"{d['name']}({symbol})", d['rep'], d['c_p'], pdf_p)
            with open(pdf_p, "rb") as f:
                st.download_button("下载 PDF", f, file_name=f"VQ_{symbol}.pdf")

        # === 6. 交互问答 (Interactive Chat) ===
        st.divider()
        st.subheader("💬 智能对话")

        # 显示历史
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        c_mic, c_input = st.columns([1, 8])
        user_voice_text = None

        with c_mic:
            st.write(" ")
            # 录音组件
            audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format='wav')

        if audio:
            # === 防复读核心：对比 audio 字节流的哈希或简单判断是否刚处理过 ===
            # 这里简化逻辑：直接调用识别，但利用 session_state 锁住不让它重复上屏

            # 调用识别
            transcribed = eng["audio"].transcribe(audio['bytes'])
            if transcribed:
                # 只有当识别出的文字和上一次不一样，或者确实是新录音时才处理
                # 最简单的防复读：检查是否与 session_state.last_voice_text 相同
                if transcribed != st.session_state.last_voice_text:
                    user_voice_text = transcribed
                    st.session_state.last_voice_text = transcribed  # 更新锁
            else:
                # 识别失败不弹窗干扰
                pass

        with c_input:
            text_input = st.chat_input("输入问题...")

        # 统一提交逻辑
        final_input = user_voice_text if user_voice_text else text_input

        if final_input:
            # 用户上屏
            st.session_state.chat_history.append({"role": "user", "content": final_input})

            # 强制刷新以立刻显示用户问题
            st.rerun()

        # 处理 AI 回复 (在 rerun 后执行，此时 chat_history 最后一类是 user)
        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
            user_q = st.session_state.chat_history[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    resp = eng["agent"].chat(user_q, st.session_state.last_context)
                    st.markdown(resp)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
        
        # === 整合功能：回测和因子分析 ===
        # 检查是否显示回测
        if show_backtest and "res" in st.session_state:
            st.divider()
            st.subheader("🧪 策略模拟回测")
            
            # 回测参数（从侧边栏重新获取，确保使用最新值）
            bt_start_val = st.session_state.get("bt_start", datetime(2022, 1, 1))
            bt_end_val = st.session_state.get("bt_end", datetime.now())
            bt_cap_val = st.session_state.get("bt_cap", 100000)
            bt_ma_val = st.session_state.get("bt_ma", 60)
            bt_stop_val = st.session_state.get("bt_stop", 8)
            bt_vision_val = st.session_state.get("bt_vision", 57)
            bt_validation_val = st.session_state.get("bt_validation", "简单回测")
            wf_train_months_val = st.session_state.get("wf_train_months", 24)
            wf_test_months_val = st.session_state.get("wf_test_months", 6)
            
            if st.button("开始回测", key="backtest_btn"):
                _run_backtest_integrated(symbol, bt_start_val, bt_end_val, bt_cap_val, bt_ma_val, 
                                        bt_stop_val, bt_vision_val, bt_validation_val, 
                                        wf_train_months_val, wf_test_months_val)
        
        # 检查是否显示因子分析
        if show_factor_analysis and "res" in st.session_state:
            st.divider()
            st.subheader("📈 因子有效性分析")
            _show_factor_analysis_integrated(symbol, d["df_f"])

# --- 模式 B: 批量组合分析 ---
elif mode == "📊 批量组合分析":
    if run_btn:
        # 解析股票代码
        symbols = [s.strip().zfill(6) for s in batch_input.split('\n') if s.strip()][:30]
        
        if len(symbols) == 0:
            st.error("❌ 请输入至少一只股票代码")
            st.stop()
        
        st.session_state.has_run = True
        # 清空旧结果
        if "batch_results" in st.session_state:
            del st.session_state.batch_results
        if "multi_tier_result" in st.session_state:
            del st.session_state.multi_tier_result
        if "portfolio_metrics" in st.session_state:
            del st.session_state.portfolio_metrics
        
        # 初始化批量分析器
        batch_analyzer = BatchAnalyzer(eng)
        portfolio_optimizer = PortfolioOptimizer()
        
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, symbol):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"正在分析 {symbol} ({current}/{total})...")
        
        # 批量分析
        with st.spinner("正在批量分析..."):
            results = batch_analyzer.analyze_batch(symbols, update_progress)
        
        # 组合优化 - 使用新的多层优化
        with st.spinner("正在优化组合配置..."):
            multi_tier_result = portfolio_optimizer.optimize_multi_tier_portfolio(
                results, eng["loader"], 
                min_weight=min_weight, max_weight=max_weight,
                max_positions=max_positions
            )
            
            # 合并所有权重用于计算总体指标
            all_weights = {}
            all_weights.update(multi_tier_result['core'])
            all_weights.update(multi_tier_result['enhanced'])
            
            portfolio_metrics = portfolio_optimizer.calculate_portfolio_metrics(
                all_weights, results, eng["loader"]
            ) if all_weights else {}
        
        # 保存结果
        st.session_state.batch_results = results
        st.session_state.multi_tier_result = multi_tier_result
        st.session_state.portfolio_metrics = portfolio_metrics
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ 批量分析完成！共分析 {len(symbols)} 只股票")
    
    # 显示结果
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        multi_tier = st.session_state.multi_tier_result
        portfolio_metrics = st.session_state.portfolio_metrics
        
        # 提取多层结果
        core_weights = multi_tier.get('core', {})
        enhanced_weights = multi_tier.get('enhanced', {})
        tier_info = multi_tier.get('tier_info', {})
        
        all_weights = {}
        all_weights.update(core_weights)
        all_weights.update(enhanced_weights)
        
        st.markdown("# 📊 批量组合分析报告")
        
        # 筛选股票
        buy_stocks = {k: v for k, v in results.items() 
                      if v.get('action') == 'BUY' and v.get('score', 0) >= 7}
        enhanced_stocks = {k: v for k, v in results.items() 
                          if v.get('score', 0) >= 6 and v.get('action') != 'SELL' and k not in buy_stocks}
        wait_stocks = {k: v for k, v in results.items() 
                      if v.get('action') == 'WAIT' and v.get('score', 0) < 6}
        sell_stocks = {k: v for k, v in results.items() 
                      if v.get('action') == 'SELL' or v.get('score', 0) < 5}
        
        # 策略说明
        strategy_emojis = {
            'core_only': '🎯',
            'mixed': '⚖️',
            'enhanced_only': '⚠️'
        }
        strategy = tier_info.get('strategy', 'unknown')
        st.info(f"{strategy_emojis.get(strategy, '📊')} **配置策略：** {tier_info.get('description', '')}")
        
        # 组合指标 - 始终显示
        st.subheader("📊 组合配置指标")
        if all_weights and portfolio_metrics:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("组合预期收益", f"{portfolio_metrics.get('expected_return', 0):.2f}%")
            col2.metric("组合风险", f"{portfolio_metrics.get('risk', 0):.2f}%")
            col3.metric("夏普比率", f"{portfolio_metrics.get('sharpe_ratio', 0):.2f}")
            col4.metric("总持仓数", f"{len(all_weights)}只")
            
            # 分层统计
            col1, col2, col3 = st.columns(3)
            col1.metric("核心推荐", f"{tier_info.get('core_count', 0)}只", 
                       help="评分≥7且action=BUY")
            col2.metric("备选增强", f"{tier_info.get('enhanced_count', 0)}只", 
                       help="评分≥6且action≠SELL")
            col3.metric("观察监控", f"{len(wait_stocks) + len(sell_stocks)}只",
                       help="评分<6或action=SELL")
        else:
            st.warning("⚠️ 暂无符合条件的股票，无法生成组合配置。建议调整参数或等待更好的市场机会。")
        
        st.divider()
        
        # 仓位分配 - 分层展示
        if all_weights and len(all_weights) > 0:
            st.subheader("💰 组合仓位分配")
            
            # 双饼图：核心 vs 增强
            if core_weights and enhanced_weights:
                col_pie1, col_pie2 = st.columns(2)
                
                with col_pie1:
                    core_total = sum(core_weights.values())
                    fig_core = go.Figure(data=[go.Pie(
                        labels=[f"{results[s].get('name', s)[:4]}" for s in core_weights.keys()],
                        values=[w/core_total*100 for w in core_weights.values()],
                        hole=0.4,
                        marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
                    )])
                    fig_core.update_layout(
                        title=f"核心推荐 (70%仓位)", 
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_core, config={"displayModeBar": False}, use_container_width=True)
                
                with col_pie2:
                    enhanced_total = sum(enhanced_weights.values())
                    fig_enhanced = go.Figure(data=[go.Pie(
                        labels=[f"{results[s].get('name', s)[:4]}" for s in enhanced_weights.keys()],
                        values=[w/enhanced_total*100 for w in enhanced_weights.values()],
                        hole=0.4,
                        marker=dict(colors=['#C7CEEA', '#B5EAD7', '#FFB6B9', '#FFDAB9', '#E0BBE4'])
                    )])
                    fig_enhanced.update_layout(
                        title=f"备选增强 (30%仓位)", 
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_enhanced, config={"displayModeBar": False}, use_container_width=True)
            
            else:
                # 单饼图
                col_chart, col_list = st.columns([2, 1])
                
                with col_chart:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=[f"{results[s].get('name', s)} ({s})" for s in all_weights.keys()],
                        values=[w*100 for w in all_weights.values()],
                        hole=0.4,
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    fig_pie.update_layout(title="仓位分配", height=400)
                    st.plotly_chart(fig_pie, config={"displayModeBar": False}, use_container_width=True)
                
                with col_list:
                    st.write("**详细仓位**")
                    for symbol, weight in sorted(all_weights.items(), 
                                               key=lambda x: x[1], reverse=True):
                        data = results.get(symbol, {})
                        tier_tag = "🎯核心" if symbol in core_weights else "⚡增强"
                        st.write(f"{tier_tag} **{data.get('name', symbol)}** ({symbol})")
                        st.write(f"仓位: {weight*100:.1f}% | 评分: {data.get('score', 0):.1f}/10")
                        st.write("---")
            
            st.divider()
        
        # 核心推荐组合详情
        if buy_stocks:
            st.subheader(f"🎯 核心推荐组合 ({len(buy_stocks)}只)")
            
            # 按评分排序
            sorted_buy = sorted(buy_stocks.items(), 
                              key=lambda x: x[1].get('score', 0), reverse=True)
            
            # 创建表格展示（使用Streamlit原生表格）
            for idx, (symbol, data) in enumerate(sorted_buy):
                weight = core_weights.get(symbol, 0)
                col1, col2, col3, col4, col5, col6, col7 = st.columns([3, 1, 1, 1, 1, 1, 1])
                
                with col1:
                    # 使用按钮作为超链接
                    if st.button(f"📊 {data.get('name', symbol)} ({symbol})", 
                               key=f"link_{symbol}", use_container_width=True):
                        # 清空当前结果，准备显示新股票
                        if "res" in st.session_state:
                            del st.session_state.res
                        st.session_state.current_symbol = None
                        st.session_state.has_run = False
                        st.query_params.update({"symbol": symbol, "mode": "detail"})
                        st.rerun()
                
                with col2:
                    st.metric("评分", f"{data.get('score', 0):.1f}", label_visibility="collapsed")
                with col3:
                    st.metric("胜率", f"{data.get('win_rate', 0):.1f}%", label_visibility="collapsed")
                with col4:
                    st.metric("预期", f"{data.get('expected_return', 0):.2f}%", label_visibility="collapsed")
                with col5:
                    st.metric("仓位", f"{weight*100:.1f}%" if weight > 0 else "-", label_visibility="collapsed")
                with col6:
                    st.metric("ROE", f"{data.get('roe', 0):.1f}%", label_visibility="collapsed")
                with col7:
                    st.write(f"**🎯 {data.get('action', 'BUY')}**")
                
                if idx < len(sorted_buy) - 1:
                    st.divider()
        
        # 备选增强组合（评分≥6且非SELL）
        if enhanced_stocks:
            st.subheader(f"⚡ 备选增强组合 ({len(enhanced_stocks)}只)")
            st.caption("评分≥6，可作为辅助配置，建议谨慎操作")
            
            sorted_enhanced = sorted(enhanced_stocks.items(), 
                                   key=lambda x: x[1].get('score', 0), reverse=True)
            
            for idx, (symbol, data) in enumerate(sorted_enhanced[:10]):  # 最多展示10只
                weight = enhanced_weights.get(symbol, 0)
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 2])
                
                with col1:
                    if st.button(f"📊 {data.get('name', symbol)} ({symbol})", 
                               key=f"link_enh_{symbol}", use_container_width=True):
                        if "res" in st.session_state:
                            del st.session_state.res
                        st.session_state.current_symbol = None
                        st.session_state.has_run = False
                        st.query_params.update({"symbol": symbol, "mode": "detail"})
                        st.rerun()
                
                with col2:
                    st.metric("评分", f"{data.get('score', 0):.1f}", label_visibility="collapsed")
                with col3:
                    st.metric("胜率", f"{data.get('win_rate', 0):.1f}%", label_visibility="collapsed")
                with col4:
                    st.metric("仓位", f"{weight*100:.1f}%" if weight > 0 else "-", label_visibility="collapsed")
                with col5:
                    st.write(f"**⚡ {data.get('action', 'WAIT')}**")
                
                if idx < len(sorted_enhanced[:10]) - 1:
                    st.divider()
        
        # K线图网格预览（整合核心+增强）
        if buy_stocks or enhanced_stocks:
            st.subheader("📊 股票K线图预览")
            display_stocks = list(buy_stocks.items())[:6] + list(enhanced_stocks.items())[:3]
            
            cols = st.columns(3)
            for idx, (symbol, data) in enumerate(display_stocks[:9]):
                with cols[idx % 3]:
                    try:
                        df = eng["loader"].get_stock_data(symbol)
                        if not df.empty:
                            tier_tag = "🎯核心" if symbol in buy_stocks else "⚡增强"
                            fig_mini = go.Figure()
                            fig_mini.add_trace(go.Candlestick(
                                x=df.tail(20).index,
                                open=df.tail(20)['Open'],
                                high=df.tail(20)['High'],
                                low=df.tail(20)['Low'],
                                close=df.tail(20)['Close']
                            ))
                            fig_mini.update_layout(
                                title=f"{tier_tag} {data.get('name', symbol)} ({symbol})",
                                height=200,
                                xaxis_rangeslider_visible=False,
                                margin=dict(l=0, r=0, t=30, b=0)
                            )
                            st.plotly_chart(fig_mini, config={"displayModeBar": False}, use_container_width=True)
                    except:
                        st.write(f"{data.get('name', symbol)} - 数据加载失败")
        
        # 观望/卖出列表
        if wait_stocks or sell_stocks:
            st.divider()
            st.subheader("⚠️ 观望/卖出列表")
            
            all_other = {**wait_stocks, **sell_stocks}
            if all_other:
                for symbol, data in sorted(all_other.items(), 
                                         key=lambda x: x[1].get('score', 0)):
                    col1, col2, col3 = st.columns([3, 1, 4])
                    with col1:
                        if st.button(f"📊 {data.get('name', symbol)} ({symbol})", 
                                   key=f"link_other_{symbol}", use_container_width=True):
                            # 清空当前结果，准备显示新股票
                            if "res" in st.session_state:
                                del st.session_state.res
                            st.session_state.current_symbol = None
                            st.session_state.has_run = False
                            st.query_params.update({"symbol": symbol, "mode": "detail"})
                            st.rerun()
                    with col2:
                        st.write(f"**{data.get('score', 0):.1f}/10**")
                    with col3:
                        st.write(f"{data.get('action', 'WAIT')} - {data.get('reasoning', '')[:50]}")
                    st.divider()
    
    else:
        st.info("👈 请在左侧输入股票代码并点击启动")

# === 辅助函数：整合的回测和因子分析 ===

def _show_factor_analysis_integrated(symbol, df_f):
    """整合的因子分析函数 - 优化版"""
    try:
        from src.factor_analysis.ic_analysis import ICAnalyzer
        from src.factor_analysis.regime_detector import RegimeDetector
        from src.strategies.kline_factor import KLineFactorCalculator
        
        # 使用K线学习因子作为因子值（更准确）
        # 需要传入data_loader以支持Triple Barrier计算
        try:
            kline_factor_calc = KLineFactorCalculator(data_loader=eng["loader"])
        except:
            kline_factor_calc = KLineFactorCalculator()
        
        # 计算历史因子值（使用Top10匹配的胜率）
        factor_values_list = []
        forward_returns_list = []
        dates_list = []
        
        # 遍历历史数据，计算每个时间点的因子值
        for i in range(20, len(df_f) - 5):
            try:
                # 获取当前时间点的K线图
                current_data = df_f.iloc[i-20:i]
                if len(current_data) < 20:
                    continue
                
                # 生成临时K线图用于匹配
                temp_img = os.path.join(PROJECT_ROOT, "data", f"temp_factor_{i}.png")
                mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
                s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
                mpf.plot(current_data, type='candle', style=s, savefig=dict(fname=temp_img, dpi=50), 
                        figsize=(3, 3), axisoff=True)
                
                # 搜索相似模式
                matches = eng["vision"].search_similar_patterns(temp_img, top_k=10)
                
                if matches and len(matches) > 0:
                    # 计算混合胜率作为因子值
                    try:
                        # 获取日期字符串
                        try:
                            if hasattr(df_f.index[i], 'strftime'):
                                date_str = df_f.index[i].strftime('%Y%m%d')
                            elif isinstance(df_f.index[i], pd.Timestamp):
                                date_str = df_f.index[i].strftime('%Y%m%d')
                            else:
                                date_str = str(df_f.index[i]).replace('-', '').replace(' ', '')[:8]
                        except:
                            date_str = str(i)  # 如果日期转换失败，使用索引
                        
                        factor_result = kline_factor_calc.calculate_hybrid_win_rate(
                            matches, 
                            query_symbol=symbol, 
                            query_date=date_str
                        )
                        
                        # 确保返回的是字典
                        if not isinstance(factor_result, dict):
                            factor_result = {'hybrid_win_rate': 50.0}
                        
                        # 确保返回的是字典且包含hybrid_win_rate
                        if isinstance(factor_result, dict) and 'hybrid_win_rate' in factor_result:
                            factor_value = factor_result.get('hybrid_win_rate', 50.0) / 100.0  # 归一化到0-1
                        else:
                            factor_value = 0.5  # 默认值
                    except Exception as e:
                        factor_value = 0.5  # 默认值
                        continue
                    
                    # 未来5日收益率
                    future_return = (df_f.iloc[i+5]['Close'] - df_f.iloc[i]['Close']) / df_f.iloc[i]['Close']
                    
                    factor_values_list.append(factor_value)
                    forward_returns_list.append(future_return)
                    dates_list.append(df_f.index[i])
                
                # 清理临时文件
                if os.path.exists(temp_img):
                    os.remove(temp_img)
                    
            except Exception as e:
                continue
        
        if len(factor_values_list) > 20:
            # 转换为Series
            factor_values = pd.Series(factor_values_list, index=dates_list)
            forward_returns = pd.Series(forward_returns_list, index=dates_list)
            
            # IC分析
            ic_analyzer = ICAnalyzer(factor_values, forward_returns)
            rolling_ic = ic_analyzer.calculate_rolling_ic(window=min(20, len(factor_values)//2))
            
            # 绘制IC曲线
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_ic.index,
                y=rolling_ic.values,
                mode='lines',
                name='Rolling IC',
                line=dict(color='blue', width=2)
            ))
            fig.add_hline(y=0.05, line_dash="dash", line_color="green", 
                         annotation_text="IC阈值(0.05)")
            fig.add_hline(y=-0.05, line_dash="dash", line_color="red")
            fig.update_layout(title="IC曲线分析", height=300)
            st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
            
            # 显示IC统计
            ic_stats = ic_analyzer.get_ic_statistics(rolling_ic)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("平均IC", f"{ic_stats['mean_ic']:.4f}", 
                         delta="有效" if ic_stats['mean_ic'] > 0.05 else "无效")
            with col2:
                st.metric("IC标准差", f"{ic_stats['std_ic']:.4f}")
            with col3:
                st.metric("ICIR", f"{ic_stats['ic_ir']:.2f}", 
                         delta="优秀" if abs(ic_stats['ic_ir']) > 1.0 else "一般")
            with col4:
                st.metric("正IC比例", f"{ic_stats['positive_ic_ratio']*100:.1f}%",
                         delta="良好" if ic_stats['positive_ic_ratio'] > 0.6 else "一般")
            
            # Regime识别图
            st.subheader("市场Regime识别")
            regime_detector = RegimeDetector(df_f['Close'])
            regimes = regime_detector.detect_regime()
            regime_counts = regimes.value_counts()
            
            colors_map = {'Bull': 'green', 'Bear': 'red', 'Oscillating': 'yellow'}
            fig_regime = go.Figure(data=[go.Bar(
                x=regime_counts.index,
                y=regime_counts.values,
                marker_color=[colors_map.get(r, 'gray') for r in regime_counts.index]
            )])
            fig_regime.update_layout(title="市场Regime分布", height=300)
            st.plotly_chart(fig_regime, config={"displayModeBar": False}, use_container_width=True)
            
            # 因子衰减分析
            st.subheader("因子衰减分析")
            decay_window = min(60, len(rolling_ic))
            recent_ic = rolling_ic.tail(decay_window).mean()
            earlier_ic = rolling_ic.head(decay_window).mean() if len(rolling_ic) > decay_window else recent_ic
            decay_rate = (recent_ic - earlier_ic) / abs(earlier_ic) * 100 if earlier_ic != 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("早期IC均值", f"{earlier_ic:.4f}")
            with col2:
                st.metric("近期IC均值", f"{recent_ic:.4f}", 
                         delta=f"{decay_rate:.1f}%", 
                         delta_color="inverse" if decay_rate < 0 else "normal")
            
            # 因子失效多维度检测
            try:
                from src.factor_analysis.factor_invalidation import FactorInvalidationDetector
                st.subheader("因子失效检测")
                invalidation_detector = FactorInvalidationDetector()
                invalidation_result = invalidation_detector.detect_invalidation(
                    factor_values=factor_values,
                    returns=forward_returns
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("失效得分", f"{invalidation_result['invalidation_score']:.2f}",
                             delta="失效" if invalidation_result['is_invalidated'] else "有效",
                             delta_color="inverse" if invalidation_result['is_invalidated'] else "normal")
                with col2:
                    st.metric("IC状态", "失效" if invalidation_result['ic_invalidation']['is_invalidated'] else "正常")
                with col3:
                    st.metric("衰减状态", "失效" if invalidation_result['decay_invalidation']['is_invalidated'] else "正常")
                
                if invalidation_result['is_invalidated']:
                    st.warning("⚠️ 因子可能已失效，建议降低权重或暂停使用")
            except Exception as e:
                st.info(f"因子失效检测功能暂不可用: {e}")
            
        else:
            st.warning("数据不足，无法进行因子分析。需要至少20个有效数据点。")
    except ImportError as e:
        st.error(f"模块导入失败: {e}")
        st.info("提示：请确保因子分析模块已正确安装")
    except Exception as e:
        st.error(f"因子分析失败: {e}")
        import traceback
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())

def _run_backtest_integrated(symbol, bt_start, bt_end, bt_cap, bt_ma, bt_stop, bt_vision, 
                             bt_validation, wf_train_months, wf_test_months):
    """整合的回测函数"""
    # #region agent log
    import json
    log_path = "/Users/bytedance/PycharmProjects/.cursor/debug.log"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A",
                "location": "web/app.py:898",
                "message": "Function entry",
                "data": {
                    "symbol": symbol,
                    "bt_validation": bt_validation,
                    "wf_train_months": wf_train_months,
                    "wf_test_months": wf_test_months
                },
                "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
            }) + "\n")
    except: pass
    # #endregion
    
    use_walk_forward = bt_validation == "Walk-Forward验证（严格）"
    
    with st.spinner("回测中..." if not use_walk_forward else f"Walk-Forward验证中（训练期{wf_train_months}月，测试期{wf_test_months}月）..."):
        df_bt = eng["loader"].get_stock_data(symbol, start_date=bt_start.strftime("%Y%m%d"))
        
        # #region agent log
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "B",
                    "location": "web/app.py:904",
                    "message": "Data loaded",
                    "data": {
                        "df_empty": df_bt.empty,
                        "df_shape": list(df_bt.shape) if not df_bt.empty else None
                    },
                    "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
                }) + "\n")
        except: pass
        # #endregion
        
        if not df_bt.empty:
            df_bt.index = pd.to_datetime(df_bt.index)
            mask = (df_bt.index >= pd.to_datetime(bt_start)) & (df_bt.index <= pd.to_datetime(bt_end))
            df_bt = df_bt.loc[mask].copy()
            
            # #region agent log
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C",
                        "location": "web/app.py:909",
                        "message": "After filtering",
                        "data": {
                            "df_shape": list(df_bt.shape),
                            "use_walk_forward": use_walk_forward
                        },
                        "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
                    }) + "\n")
            except: pass
            # #endregion
            
            if use_walk_forward:
                # 真正实现Walk-Forward验证
                from src.utils.walk_forward import WalkForwardValidator
                
                # 转换月份为交易日（约21个交易日/月）
                train_days = wf_train_months * 21
                test_days = wf_test_months * 21
                step_days = wf_test_months * 21  # 每次滚动一个测试期
                
                # #region agent log
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "D",
                            "location": "web/app.py:920",
                            "message": "Before WalkForwardValidator",
                            "data": {
                                "train_days": train_days,
                                "test_days": test_days,
                                "step_days": step_days
                            },
                            "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
                        }) + "\n")
                except: pass
                # #endregion
                
                validator = WalkForwardValidator(
                    train_period=train_days,
                    test_period=test_days,
                    step_size=step_days
                )
                
                all_results = []
                fold_count = 0
                
                # #region agent log
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "E",
                            "location": "web/app.py:928",
                            "message": "Before split iteration",
                            "data": {},
                            "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
                        }) + "\n")
                except: pass
                # #endregion
                
                for split in validator.split(df_bt):
                    # #region agent log
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "F",
                                "location": "web/app.py:929",
                                "message": "Split object attributes",
                                "data": {
                                    "has_train_start": hasattr(split, "train_start"),
                                    "has_train_end": hasattr(split, "train_end"),
                                    "has_test_start": hasattr(split, "test_start"),
                                    "has_test_end": hasattr(split, "test_end"),
                                    "has_train_indices": hasattr(split, "train_indices"),
                                    "has_test_indices": hasattr(split, "test_indices"),
                                    "split_type": str(type(split)),
                                    "split_dir": dir(split)[:10] if hasattr(split, "__dict__") else None
                                },
                                "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
                            }) + "\n")
                    except Exception as e:
                        try:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "F",
                                    "location": "web/app.py:929",
                                    "message": "Error checking split attributes",
                                    "data": {"error": str(e)},
                                    "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
                                }) + "\n")
                        except: pass
                    # #endregion
                    
                    fold_count += 1
                    train_data = df_bt.iloc[split.train_indices]
                    test_data = df_bt.iloc[split.test_indices]
                    
                    # 在训练集上计算技术指标（模拟训练过程）
                    train_data = train_data.copy()
                    train_data['MA20'] = train_data['Close'].rolling(window=20).mean()
                    train_data['MA60'] = train_data['Close'].rolling(window=bt_ma).mean()
                    exp12 = train_data['Close'].ewm(span=12, adjust=False).mean()
                    exp26 = train_data['Close'].ewm(span=26, adjust=False).mean()
                    train_data['MACD'] = (exp12 - exp26) * 2
                    
                    # 在测试集上应用策略
                    test_data = test_data.copy()
                    test_data['MA20'] = test_data['Close'].rolling(window=20).mean()
                    test_data['MA60'] = test_data['Close'].rolling(window=bt_ma).mean()
                    exp12 = test_data['Close'].ewm(span=12, adjust=False).mean()
                    exp26 = test_data['Close'].ewm(span=26, adjust=False).mean()
                    test_data['MACD'] = (exp12 - exp26) * 2
                    # 确保有Volume列（如果没有则使用默认值）
                    if 'Volume' not in test_data.columns:
                        test_data['Volume'] = test_data['Close'] * 1000000  # 默认成交量
                    test_data = test_data.dropna()
                    
                    # 加载AI胜率数据（使用混合胜率，如果可用）
                    pred_path = os.path.join(PROJECT_ROOT, "data", "indices", "prediction_cache.csv")
                    vision_map = {}
                    if os.path.exists(pred_path):
                        try:
                            pdf = pd.read_csv(pred_path)
                            pdf['date'] = pdf['date'].astype(str).str.replace('-', '')
                            pdf['symbol'] = pdf['symbol'].astype(str).str.zfill(6)
                            vision_map = pdf.set_index(['symbol', 'date'])['pred_win_rate'].to_dict()
                        except:
                            pass
                    
                    # 回测逻辑（与之前相同，但只在测试集上运行）
                    # 集成Transaction Cost模型和Turnover约束
                    from src.strategies.transaction_cost import AdvancedTransactionCost
                    transaction_cost_calc = AdvancedTransactionCost()
                    
                    cash, shares, equity = bt_cap, 0, []
                    trade_log = []
                    entry_price = 0.0
                    daily_turnover = []  # 记录每日换手率
                    max_daily_turnover = 0.20  # 最大单日换手率20%
                    
                    for _, row in test_data.iterrows():
                        p = row['Close']
                        ma20 = row.get('MA20', p)
                        ma60 = row.get('MA60', p)
                        macd = row.get('MACD', 0)
                        date_str = row.name.strftime("%Y%m%d")
                        ai_win = vision_map.get((symbol, date_str), 50.0)
                        # 安全获取Volume，如果不存在则使用默认值
                        try:
                            volume = float(row.get('Volume', row.get('volume', test_data['Close'].mean() * 1000000)))
                        except:
                            volume = test_data['Close'].mean() * 1000000 if len(test_data) > 0 else 1000000
                        
                        target_pos = 0.0
                        if p > ma60:
                            if macd > 0 or p > ma20:
                                target_pos = 1.0
                            elif ai_win >= bt_vision:
                                target_pos = 0.81
                        else:
                            if ai_win >= bt_vision + 2:
                                target_pos = 0.50
                            else:
                                target_pos = 0.03
                        
                        total_assets = cash + shares * p
                        target_val = total_assets * target_pos
                        target_shares = int(target_val / p) if p > 0 else 0
                        diff = target_shares - shares
                        
                        # Turnover约束：计算当日换手率
                        if total_assets > 0:
                            daily_turnover_pct = abs(diff * p) / total_assets
                            if daily_turnover_pct > max_daily_turnover:
                                # 限制换手率
                                max_trade_value = total_assets * max_daily_turnover
                                if diff > 0:
                                    diff = int(max_trade_value / p)
                                else:
                                    diff = -int(max_trade_value / p)
                        
                        if abs(diff * p) > total_assets * 0.1:
                            trade_value = abs(diff * p)
                            
                            # 计算Transaction Cost（添加错误处理）
                            try:
                                volatility = test_data['Close'].pct_change().std() if len(test_data) > 1 else 0.02
                                if pd.isna(volatility) or volatility <= 0:
                                    volatility = 0.02
                                cost_result = transaction_cost_calc.calculate_cost(
                                    trade_size=trade_value,
                                    price=p,
                                    volume=max(volume, 1),  # 确保volume > 0
                                    volatility=volatility,
                                    is_buy=(diff > 0)
                                )
                                total_cost = cost_result.get('total_cost', trade_value * 0.001)  # 默认0.1%成本
                            except Exception as e:
                                # 如果Transaction Cost计算失败，使用简单成本模型
                                total_cost = trade_value * 0.001  # 默认0.1%成本
                            
                            if diff > 0:
                                cost = diff * p + total_cost
                                if cash >= cost:
                                    cash -= cost
                                    shares += diff
                                    if entry_price == 0:
                                        entry_price = p
                                    trade_log.append({
                                        'date': date_str, 
                                        'action': 'BUY', 
                                        'price': p,
                                        'shares': diff,
                                        'cost': total_cost
                                    })
                            elif diff < 0:
                                pnl = (p - entry_price) / entry_price if entry_price > 0 and shares > 0 else 0
                                if pnl < -bt_stop / 100:
                                    diff = -shares
                                revenue = abs(diff) * p - total_cost
                                cash += revenue
                                shares += diff
                                if shares == 0:
                                    entry_price = 0
                                trade_log.append({
                                    'date': date_str, 
                                    'action': 'SELL', 
                                    'price': p,
                                    'shares': abs(diff),
                                    'cost': total_cost
                                })
                        
                        equity.append(cash + shares * p)
                        daily_turnover.append(abs(diff * p) / total_assets if total_assets > 0 else 0)
                    
                    if equity:
                        ret = (equity[-1] - bt_cap) / bt_cap * 100
                        bench_ret = (test_data['Close'].iloc[-1] - test_data['Close'].iloc[0]) / test_data['Close'].iloc[0] * 100
                        
                        # #region agent log
                        try:
                            train_start_str = split.train_start.strftime('%Y-%m-%d') if hasattr(split.train_start, 'strftime') else str(split.train_start)
                            train_end_str = split.train_end.strftime('%Y-%m-%d') if hasattr(split.train_end, 'strftime') else str(split.train_end)
                            test_start_str = split.test_start.strftime('%Y-%m-%d') if hasattr(split.test_start, 'strftime') else str(split.test_start)
                            test_end_str = split.test_end.strftime('%Y-%m-%d') if hasattr(split.test_end, 'strftime') else str(split.test_end)
                            
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps({
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                    "hypothesisId": "G",
                                    "location": "web/app.py:1015",
                                    "message": "Before appending result",
                                    "data": {
                                        "train_start_type": str(type(split.train_start)),
                                        "train_start_str": train_start_str,
                                        "ret": ret,
                                        "bench_ret": bench_ret
                                    },
                                    "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
                                }) + "\n")
                        except Exception as e:
                            try:
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps({
                                        "sessionId": "debug-session",
                                        "runId": "run1",
                                        "hypothesisId": "G",
                                        "location": "web/app.py:1015",
                                        "message": "Error formatting dates",
                                        "data": {"error": str(e)},
                                        "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
                                    }) + "\n")
                            except: pass
                        # #endregion
                        
                        # 安全地格式化日期
                        try:
                            train_start_str = split.train_start.strftime('%Y-%m-%d') if hasattr(split.train_start, 'strftime') else str(split.train_start)
                            train_end_str = split.train_end.strftime('%Y-%m-%d') if hasattr(split.train_end, 'strftime') else str(split.train_end)
                            test_start_str = split.test_start.strftime('%Y-%m-%d') if hasattr(split.test_start, 'strftime') else str(split.test_start)
                            test_end_str = split.test_end.strftime('%Y-%m-%d') if hasattr(split.test_end, 'strftime') else str(split.test_end)
                        except Exception as e:
                            # 如果日期格式化失败，使用索引位置
                            train_start_str = f"Fold{fold_count}_TrainStart"
                            train_end_str = f"Fold{fold_count}_TrainEnd"
                            test_start_str = f"Fold{fold_count}_TestStart"
                            test_end_str = f"Fold{fold_count}_TestEnd"
                        
                        all_results.append({
                            'fold': fold_count,
                            'train_start': train_start_str,
                            'train_end': train_end_str,
                            'test_start': test_start_str,
                            'test_end': test_end_str,
                            'return': ret,
                            'benchmark': bench_ret,
                            'alpha': ret - bench_ret,
                            'trades': len(trade_log)
                        })
                
                # 显示Walk-Forward结果 - 专业风格
                if all_results:
                    results_df = pd.DataFrame(all_results)
                    
                    # 显示结果表格
                    st.markdown("### Walk-Forward验证结果")
                    st.dataframe(results_df, use_container_width=True, height=300)
                    
                    # 绘制多fold结果 - 专业图表
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['fold'],
                        y=results_df['return'],
                        mode='lines+markers',
                        name='策略收益',
                        line=dict(color='#ff4b4b', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=results_df['fold'],
                        y=results_df['benchmark'],
                        mode='lines+markers',
                        name='基准收益',
                        line=dict(color='gray', dash='dash')
                    ))
                    fig.update_layout(
                        title=f"Walk-Forward验证结果（{fold_count}个fold，训练期{wf_train_months}月，测试期{wf_test_months}月）",
                        xaxis_title="Fold",
                        yaxis_title="收益率 (%)",
                        height=400,
                        plot_bgcolor='#0d1117',
                        paper_bgcolor='#0d1117',
                        font=dict(color='#c9d1d9', size=11),
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='#161b22', bordercolor='#30363d'),
                        xaxis=dict(gridcolor='#30363d', linecolor='#30363d'),
                        yaxis=dict(gridcolor='#30363d', linecolor='#30363d')
                    )
                    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
                    
                    # 统计指标
                    avg_return = results_df['return'].mean()
                    avg_alpha = results_df['alpha'].mean()
                    std_return = results_df['return'].std()
                    win_rate = (results_df['return'] > 0).sum() / len(results_df) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("平均收益率", f"{avg_return:.2f}%", 
                                 delta=f"±{std_return:.2f}%")
                    with col2:
                        st.metric("平均Alpha", f"{avg_alpha:.2f}%",
                                 delta="超额收益" if avg_alpha > 0 else "跑输基准")
                    with col3:
                        st.metric("胜率", f"{win_rate:.1f}%",
                                 delta="优秀" if win_rate > 60 else "一般")
                    with col4:
                        st.metric("Fold数量", f"{fold_count}个")
                else:
                    st.warning("Walk-Forward验证未生成结果，可能数据不足")
            else:
                # 简单回测（原有逻辑）
                if len(df_bt) > 50:
                    df_bt['MA20'] = df_bt['Close'].rolling(window=20).mean()
                    df_bt['MA60'] = df_bt['Close'].rolling(window=bt_ma).mean()
                    exp12 = df_bt['Close'].ewm(span=12, adjust=False).mean()
                    exp26 = df_bt['Close'].ewm(span=26, adjust=False).mean()
                    df_bt['MACD'] = (exp12 - exp26) * 2
                    # 确保有Volume列
                    if 'Volume' not in df_bt.columns:
                        df_bt['Volume'] = df_bt['Close'] * 1000000  # 默认成交量
                    df_bt = df_bt.dropna()
                    
                    # 加载AI胜率数据（使用混合胜率，如果可用）
                    pred_path = os.path.join(PROJECT_ROOT, "data", "indices", "prediction_cache.csv")
                    vision_map = {}
                    if os.path.exists(pred_path):
                        try:
                            pdf = pd.read_csv(pred_path)
                            pdf['date'] = pdf['date'].astype(str).str.replace('-', '')
                            pdf['symbol'] = pdf['symbol'].astype(str).str.zfill(6)
                            vision_map = pdf.set_index(['symbol', 'date'])['pred_win_rate'].to_dict()
                        except:
                            pass
                    
                    # 集成Transaction Cost模型和Turnover约束
                    from src.strategies.transaction_cost import AdvancedTransactionCost
                    transaction_cost_calc = AdvancedTransactionCost()
                    
                    cash, shares, equity = bt_cap, 0, []
                    trade_log = []
                    entry_price = 0.0
                    max_daily_turnover = 0.20  # 最大单日换手率20%
                    
                    for _, row in df_bt.iterrows():
                        p = row['Close']
                        ma20 = row.get('MA20', p)
                        ma60 = row.get('MA60', p)
                        macd = row.get('MACD', 0)
                        date_str = row.name.strftime("%Y%m%d")
                        ai_win = vision_map.get((symbol, date_str), 50.0)
                        # 安全获取Volume
                        try:
                            volume = float(row.get('Volume', row.get('volume', df_bt['Close'].mean() * 1000000)))
                        except:
                            volume = df_bt['Close'].mean() * 1000000 if len(df_bt) > 0 else 1000000
                        
                        target_pos = 0.0
                        if p > ma60:
                            if macd > 0 or p > ma20:
                                target_pos = 1.0
                            elif ai_win >= bt_vision:
                                target_pos = 0.81
                        else:
                            if ai_win >= bt_vision + 2:
                                target_pos = 0.50
                            else:
                                target_pos = 0.03
                        
                        total_assets = cash + shares * p
                        target_val = total_assets * target_pos
                        target_shares = int(target_val / p) if p > 0 else 0
                        diff = target_shares - shares
                        
                        # Turnover约束
                        if total_assets > 0:
                            daily_turnover_pct = abs(diff * p) / total_assets
                            if daily_turnover_pct > max_daily_turnover:
                                max_trade_value = total_assets * max_daily_turnover
                                if diff > 0:
                                    diff = int(max_trade_value / p)
                                else:
                                    diff = -int(max_trade_value / p)
                        
                        if abs(diff * p) > total_assets * 0.1:
                            trade_value = abs(diff * p)
                            
                            # 计算Transaction Cost（添加错误处理）
                            try:
                                volatility = df_bt['Close'].pct_change().std() if len(df_bt) > 1 else 0.02
                                if pd.isna(volatility) or volatility <= 0:
                                    volatility = 0.02
                                cost_result = transaction_cost_calc.calculate_cost(
                                    trade_size=trade_value,
                                    price=p,
                                    volume=max(volume, 1),  # 确保volume > 0
                                    volatility=volatility,
                                    is_buy=(diff > 0)
                                )
                                total_cost = cost_result.get('total_cost', trade_value * 0.001)  # 默认0.1%成本
                            except Exception as e:
                                # 如果Transaction Cost计算失败，使用简单成本模型
                                total_cost = trade_value * 0.001  # 默认0.1%成本
                            
                            if diff > 0:
                                cost = diff * p + total_cost
                                if cash >= cost:
                                    cash -= cost
                                    shares += diff
                                    if entry_price == 0:
                                        entry_price = p
                                    trade_log.append({
                                        'date': date_str, 
                                        'action': 'BUY', 
                                        'price': p,
                                        'cost': total_cost
                                    })
                            elif diff < 0:
                                pnl = (p - entry_price) / entry_price if entry_price > 0 and shares > 0 else 0
                                if pnl < -bt_stop / 100:
                                    diff = -shares
                                revenue = abs(diff) * p - total_cost
                                cash += revenue
                                shares += diff
                                if shares == 0:
                                    entry_price = 0
                                trade_log.append({
                                    'date': date_str, 
                                    'action': 'SELL', 
                                    'price': p,
                                    'cost': total_cost
                                })
                        
                        equity.append(cash + shares * p)
                    
                    # 绘制收益曲线 - 专业风格
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_bt.index, y=equity, name="VQ 策略", 
                                           line=dict(color='#ff4b4b', width=2)))
                    bench = (df_bt['Close'] / df_bt['Close'].iloc[0]) * bt_cap
                    fig.add_trace(go.Scatter(x=df_bt.index, y=bench, name="基准（买入持有）", 
                                           line=dict(color='gray', dash='dash')))
                    fig.update_layout(title="策略收益曲线", height=400)
                    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
                    
                    ret = (equity[-1] - bt_cap) / bt_cap * 100
                    bench_ret = (df_bt['Close'].iloc[-1] - df_bt['Close'].iloc[0]) / df_bt['Close'].iloc[0] * 100
                    alpha = ret - bench_ret
                    sharpe = (ret / 100) / (np.std(np.diff(equity)) / np.mean(equity)) if len(equity) > 1 and np.std(np.diff(equity)) > 0 else 0
                    
                    # 指标展示
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("策略收益率", f"{ret:.2f}%", 
                                 delta=f"{alpha:.2f}% vs 基准", 
                                 delta_color="normal" if alpha > 0 else "inverse")
                    with col2:
                        st.metric("Alpha", f"{alpha:.2f}%",
                                 delta="超额收益" if alpha > 0 else "跑输基准")
                    with col3:
                        st.metric("交易次数", f"{len(trade_log)}次")
                    with col4:
                        st.metric("夏普比率", f"{sharpe:.2f}" if sharpe > 0 else "N/A",
                                 delta="优秀" if sharpe > 1.5 else "一般")
                else:
                    st.error("数据不足")
        else:
            st.error("数据获取失败")
