import os
import sys
from dotenv import load_dotenv

try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GOOGLE_API_KEY")


class TradeDecision(BaseModel):
    action: str = Field(description="决策: BUY, SELL, or WAIT")
    confidence: int = Field(description="0-100")
    risk_level: str = Field(description="High, Medium, Low")
    reasoning: str = Field(description="分析理由")


class QuantAgent:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=TradeDecision)
        self.chain = None
        self.llm = None

        if not API_KEY:
            return

        # === 核心修正：优先使用你账号支持的高级模型 ===
        # 根据你之前的 check-api 结果，优先调用 2.5 和 2.0 系列
        candidate_models = [
            "gemini-2.5-pro",  # 首选：最强逻辑
            "gemini-2.0-flash-exp",  # 次选：速度最快
            "gemini-1.5-pro",  # 备选
            "gemini-pro"  # 兜底
        ]

        for model in candidate_models:
            try:
                # transport="rest" 是防断连的关键
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=API_KEY,
                    temperature=0.4,
                    transport="rest"
                )
                # 尝试一次空调用来验证连接
                llm.invoke("Hi")
                self.llm = llm
                print(f"✅ Agent 成功接入模型: {model}")
                break
            except Exception as e:
                # print(f"   跳过 {model}: {e}")
                continue

        if not self.llm:
            print("❌ 所有模型连接失败，请检查 API Key 额度或网络。")
            return

        template = """
        你是一位华尔街顶级对冲基金首席风险官 (CRO)。请根据以下多模态数据撰写深度投资备忘录。

        【数据快照】
        股票: {symbol} | 日期: {date}
        量化评分: {total_score}/10 | 初步建议: {initial_action}

        [技术面]
        形态胜率: {win_rate}% | 趋势: {ma_trend} | RSI: {rsi} | MACD: {macd}

        [基本面]
        PE: {pe_ttm} | PB: {pb} | ROE: {roe}%

        [舆情]
        {news_summary}

        【决策逻辑】
        1. **一票否决**：若 ROE<0 或 PE>60，或者新闻有重大利空（立案/调查），强制 **SELL/WAIT**。
        2. **趋势共振**：只有当 形态胜率>60% 且 趋势向上 时，才建议 **BUY**。
        3. **深度分析**：请详细解释数据之间的冲突（例如：为什么形态好但基本面差要回避？）。

        {format_instructions}
        """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["symbol", "date", "total_score", "initial_action", "win_rate", "ma_trend", "rsi", "macd",
                             "pe_ttm", "pb", "roe", "news_summary"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | self.llm | self.parser

    def analyze(self, symbol, total_score, initial_action, visual_data, factor_data, fund_data, news_text=""):
        if not self.chain: return self._fallback_result("API 连接失败")

        from datetime import datetime
        try:
            return self.chain.invoke({
                "symbol": symbol,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total_score": total_score,
                "initial_action": initial_action,
                "win_rate": visual_data.get('win_rate', 50),
                "ma_trend": factor_data.get('MA_Signal', 0),
                "rsi": round(factor_data.get('RSI', 50), 2),
                "macd": round(factor_data.get('MACD_Hist', 0), 4),
                "pe_ttm": fund_data.get('pe_ttm', 0),
                "pb": fund_data.get('pb', 0),
                "roe": fund_data.get('roe', 0),
                "news_summary": str(news_text)[:2000]
            })
        except Exception as e:
            return self._fallback_result(f"AI 生成中断: {str(e)[:100]}")

    def chat(self, user_question, context_str):
        if not self.llm: return "AI 未连接"
        try:
            messages = [
                SystemMessage(content=f"你是一个专业的基金经理。背景数据：\n{context_str}"),
                HumanMessage(content=user_question)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except:
            return "网络繁忙，请重试。"

    def _fallback_result(self, reason):
        return TradeDecision(action="WAIT", confidence=0, risk_level="Unknown", reasoning=reason)


if __name__ == "__main__":
    agent = QuantAgent()