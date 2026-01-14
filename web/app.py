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
    v.reload_index()
    return {
        "loader": DataLoader(), "vision": v, "factor": FactorMiner(),
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

from backtest_handlers import run_backtest
from factor_analysis_handlers import show_factor_analysis as render_factor_analysis
from streamlit_mic_recorder import mic_recorder

with st.sidebar:
    st.title("🦄 VisionQuant Pro")
    st.caption("AI 全栈量化投研系统 v8.8")
    st.divider()
    symbol_input = st.text_input("请输入 A 股代码", value="601899", help="输入6位代码")
    symbol = symbol_input.strip().zfill(6)
    mode = st.radio("功能模块:", ("🔍 单只股票分析", "📊 批量组合分析"))
    
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

url_symbol = st.query_params.get("symbol")
url_jump_mode = False
if url_symbol:
    if url_symbol != symbol:
        symbol = url_symbol
        url_jump_mode = True
        mode = "🔍 单只股票分析"
        if "res" in st.session_state:
            del st.session_state.res
        st.session_state.current_symbol = symbol
        st.session_state.has_run = True
        run_btn = True
    elif "res" not in st.session_state:
        url_jump_mode = True
        mode = "🔍 单只股票分析"
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

            fund_data = eng["fund"].get_stock_fundamentals(symbol)
            stock_name = fund_data.get('name', symbol)

            q_p = os.path.join(PROJECT_ROOT, "data", "temp_q.png")
            mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
            mpf.plot(df.tail(20), type='candle', style=s, savefig=dict(fname=q_p, dpi=50), figsize=(3, 3), axisoff=True)
            
            query_prices = df.tail(20)['Close'].values if len(df) >= 20 else None
            matches = eng["vision"].search_similar_patterns(q_p, top_k=10, query_prices=query_prices)

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

            try:
                kline_factor_calc = KLineFactorCalculator(data_loader=eng["loader"])
                query_date_str = datetime.now().strftime('%Y%m%d')
                hybrid_win_rate_result = kline_factor_calc.calculate_hybrid_win_rate(
                    matches, 
                    query_symbol=symbol,
                    query_date=query_date_str
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
            
            win_rate = hybrid_win_rate if hybrid_win_rate is not None else traditional_win_rate

            df_f = eng["factor"]._add_technical_indicators(df)
            news_text = eng["news"].get_latest_news(symbol)
            ind_name, peers_df = eng["fund"].get_industry_peers(symbol)

            returns = df['Close'].pct_change().dropna()
            try:
                regime_weights = eng.get("regime_manager", None)
                if regime_weights:
                    regime_weights = regime_weights.calculate_dynamic_weights(returns=returns)
                    dynamic_weights = regime_weights.get('weights', {})
                else:
                    dynamic_weights = None
            except:
                dynamic_weights = None
            
            if dynamic_weights:
                total_score, initial_action, s_details = eng["factor"].get_scorecard(win_rate, df_f.iloc[-1], fund_data, returns=returns)
            else:
                total_score, initial_action, s_details = eng["factor"].get_scorecard(win_rate, df_f.iloc[-1], fund_data)

            report = eng["agent"].analyze(symbol, total_score, initial_action, {"win_rate": win_rate, "score": 0.9},
                                          df_f.iloc[-1].to_dict(), fund_data, news_text)

            c_p = os.path.join(PROJECT_ROOT, "data", "comparison.png")
            create_comparison_plot(q_p, matches, c_p)

            res_dict = {
                "name": stock_name, "c_p": c_p, "trajs": trajs, "mean": mean_path,
                "win": win_rate, "ret": avg_ret, "labels": traj_labels,
                "score": total_score, "act": initial_action, "det": s_details,
                "fund": fund_data, "df_f": df_f, "ind": ind_name, "peers": peers_df,
                "news": news_text, "rep": report
            }
            
            if hybrid_win_rate_result and hybrid_win_rate is not None:
                res_dict["hybrid_win_rate"] = hybrid_win_rate
                res_dict["traditional_win_rate"] = traditional_win_rate
                res_dict["tb_win_rate"] = hybrid_win_rate_result.get('tb_win_rate', 0)
                res_dict["win_rate_type"] = "混合胜率"
            else:
                res_dict["win_rate_type"] = "传统胜率"
            
            st.session_state.res = res_dict
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
        st.image(d['c_p'], use_container_width=True)
        if d['trajs']:
            fig = go.Figure()
            for i, p in enumerate(d['trajs']):
                fig.add_trace(go.Scatter(y=p, mode='lines', line=dict(color='rgba(200,200,200,0.5)', width=1),
                                         name=d['labels'][i]))
            fig.add_trace(go.Scatter(y=d['mean'], mode='lines+markers', line=dict(color='#d62728', width=3), name='平均预期'))
            fig.update_layout(title=f"未来5日走势推演 (胜率: {d['win']:.0f}%)", xaxis_title="天数", yaxis_title="收益%", height=400)
            st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("历史胜率", f"{d['win']:.1f}%")
            c2.metric("预期收益", f"{d['ret']:.2f}%")
            
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
                - **止损机制**: 达到止损阈值(-8%)时强制平仓
                - **Walk-Forward**: 滚动窗口验证，防止未来函数泄漏
                """)

            cbt1, cbt2, cbt3, cbt4 = st.columns(4)
            with cbt1:
                bt_start_val = st.date_input("开始日期", value=st.session_state.get("bt_start", datetime(2022, 1, 1)), key="bt_start")
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

            if st.button("开始回测", key="backtest_btn"):
                run_backtest(
                    symbol, bt_start_val, bt_end_val, bt_cap_val, bt_ma_val,
                    bt_stop_val, bt_vision_val, bt_validation_val,
                    wf_train_months_val, wf_test_months_val, eng, PROJECT_ROOT,
                    enable_stress_test=enable_stress
                )

        with tab_fa:
            st.subheader("📈 因子有效性分析")
            with st.expander("ℹ️ 因子分析说明", expanded=False):
                st.markdown("""
                **因子有效性分析内容**:
                - **IC分析**: 因子值与未来收益率的相关系数（Information Coefficient）
                - **Rolling IC**: 滚动窗口IC统计，观察因子稳定性
                - **Regime分析**: 不同市场状态（牛市/熊市/震荡）下的因子表现
                - **因子衰减**: 因子有效性随时间的变化趋势
                - **因子失效检测**: 多维度检测因子是否失效（IC衰减、拥挤度等）
                """)
            with st.expander("📌 指标怎么解读（简版）", expanded=False):
                st.markdown(r"""
                - **平均IC**：越大越好；\(|IC|<0.02\) 通常很弱，\(|IC|>0.05\) 才有研究价值  
                - **ICIR**：\(\text{IC均值}/\text{IC标准差}\)，衡量稳定性；>1 较强  
                - **正IC比例**：Rolling IC > 0 的占比，越高越好  
                - **Regime分布**：当前样本处于牛/熊/震荡/未知的比例（样本不足会出现 unknown）  
                - **衰减**：近期IC相对早期IC变弱则提示“衰减风险”  
                """)
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
        
        buy_stocks = {k: v for k, v in batch_results.items() if v.get('action') == 'BUY' and v.get('score', 0) >= 7}
        wait_stocks = {k: v for k, v in batch_results.items() if v.get('action') == 'WAIT'}
        sell_stocks = {k: v for k, v in batch_results.items() if v.get('action') == 'SELL'}
        
        if buy_stocks:
            st.subheader("✅ 推荐买入列表")
            multi_tier_result = portfolio_optimizer.optimize_multi_tier_portfolio(
                buy_stocks, eng["loader"], min_weight=0.05, max_weight=0.25, max_positions=10
            )
            st.session_state.multi_tier_result = multi_tier_result
            
            st.markdown("### 核心推荐组合")
            core_weights = multi_tier_result.get('core', {})
            if core_weights:
                core_df = pd.DataFrame([
                    {"股票代码": sym, "股票名称": batch_results[sym].get('name', sym), 
                     "权重": f"{w*100:.1f}%", "评分": f"{batch_results[sym].get('score', 0):.1f}/10"}
                    for sym, w in sorted(core_weights.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(core_df, hide_index=True, use_container_width=True)
                
                total_weight = sum(core_weights.values())
                st.info(f"核心组合总权重: {total_weight*100:.1f}%")
            
            st.markdown("### 备选增强组合")
            enhanced_weights = multi_tier_result.get('enhanced', {})
            if enhanced_weights:
                enhanced_df = pd.DataFrame([
                    {"股票代码": sym, "股票名称": batch_results[sym].get('name', sym), 
                     "权重": f"{w*100:.1f}%", "评分": f"{batch_results[sym].get('score', 0):.1f}/10"}
                    for sym, w in sorted(enhanced_weights.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(enhanced_df, hide_index=True, use_container_width=True)
        
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
