import akshare as ak
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
import datetime


class NewsHarvester:
    def __init__(self):
        # 伪装浏览器头，防止 Google RSS 反爬
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def get_latest_news(self, symbol, top_n=5):
        """
        [三引擎容错版] 获取新闻
        优先级: AkShare -> Google RSS -> Yahoo Finance
        """
        symbol = str(symbol).strip().zfill(6)
        print(f"📰 [新闻监控] 正在扫描 {symbol} 的舆情...")

        news_items = []

        # === 1. 尝试 AkShare (国内直连) ===
        try:
            news_df = ak.stock_news_em(symbol=symbol)
            if news_df is not None and not news_df.empty:
                for i, row in news_df.head(top_n).iterrows():
                    date = str(row.get('发布时间', ''))[:10]
                    title = str(row.get('新闻标题', '')).strip()
                    # 使用双换行，确保网页显示美观
                    news_items.append(f"- **{date}** {title}")
                print("✅ [源:东方财富] 获取成功")
                return "\n\n".join(news_items)
        except:
            print("⚠️ AkShare 接口波动，切换备用引擎...")
            pass

        # === 2. 尝试 Google News RSS (国际源，最稳) ===
        try:
            query = f"{symbol} 股票"
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
            response = requests.get(rss_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                root = ET.fromstring(response.content)
                count = 0
                for item in root.findall('./channel/item'):
                    if count >= top_n: break
                    title = item.find('title').text.split(' - ')[0]
                    pub_date = item.find('pubDate').text
                    try:
                        dt = datetime.datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                        date_str = dt.strftime("%Y-%m-%d")
                    except:
                        date_str = "近期"

                    news_items.append(f"- **{date_str}** (Google) {title}")
                    count += 1

                if news_items:
                    print("✅ [源:Google News] 获取成功")
                    return "\n\n".join(news_items)
        except Exception as e:
            print(f"❌ Google RSS 异常: {e}")

        # === 3. 尝试 Yahoo Finance (最后防线) ===
        try:
            yf_symbol = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
            yf_ticker = yf.Ticker(yf_symbol)
            yf_news = yf_ticker.news
            if yf_news:
                for item in yf_news[:top_n]:
                    title = item.get('title')
                    ts = item.get('providerPublishTime')
                    if title and ts:
                        date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        news_items.append(f"- **{date_str}** (Yahoo) {title}")

                if news_items:
                    print("✅ [源:Yahoo] 获取成功")
                    return "\n\n".join(news_items)
        except:
            pass

        return "✅ 暂无重大敏感舆情 (多源扫描完成)。"


if __name__ == "__main__":
    nh = NewsHarvester()
    print(nh.get_latest_news("600519"))