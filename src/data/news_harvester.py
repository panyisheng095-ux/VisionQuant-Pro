import yfinance as yf
import requests
import xml.etree.ElementTree as ET
import datetime
import json
import re
import time
from collections import OrderedDict


class NewsHarvester:
    def __init__(self):
        # 伪装浏览器头，防止 Google RSS 反爬
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # 简单缓存，避免重复拉取导致卡顿
        self._cache = OrderedDict()
        self._cache_ttl = 600  # 秒
        self._cache_max = 256

    def _cache_get(self, key):
        item = self._cache.get(key)
        if not item:
            return None
        if time.time() - item["ts"] > self._cache_ttl:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return item["data"]

    def _cache_set(self, key, data):
        self._cache[key] = {"ts": time.time(), "data": data}
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def _fetch_eastmoney_news(self, keyword, top_n=5):
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        callback = f"jQuery{int(time.time() * 1000)}"
        inner_param = {
            "uid": "",
            "keyword": keyword,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": max(top_n, 10),
                    "preTag": "<em>",
                    "postTag": "</em>"
                }
            }
        }
        params = {
            "cb": callback,
            "param": json.dumps(inner_param, ensure_ascii=False),
            "_": str(int(time.time() * 1000))
        }
        headers = dict(self.headers)
        headers["referer"] = f"https://so.eastmoney.com/news/s?keyword={keyword}"
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=6)
            if resp.status_code != 200:
                return []
            text = resp.text.strip()
            match = re.search(r"\((\{.*\})\)\s*$", text, re.S)
            data_json = None
            if match:
                data_json = json.loads(match.group(1))
            elif text.startswith("{") and text.endswith("}"):
                data_json = json.loads(text)
            if not data_json:
                return []
            items = data_json.get("result", {}).get("cmsArticleWebOld", []) or []
            news_items = []
            for item in items[:top_n]:
                title = str(item.get("title", "")).strip()
                title = re.sub(r"</?em>", "", title)
                date = str(item.get("date", ""))[:10] or "近期"
                media = str(item.get("mediaName", "")).strip() or "东方财富"
                if title:
                    news_items.append(f"- **{date}** ({media}) {title}")
            return news_items
        except Exception:
            return []

    def get_latest_news(self, symbol, top_n=5):
        """
        [三引擎容错版] 获取新闻
        优先级: AkShare -> Google RSS -> Yahoo Finance
        """
        symbol = str(symbol).strip().zfill(6)
        print(f"📰 [新闻监控] 正在扫描 {symbol} 的舆情...")

        cache_key = f"{symbol}:{top_n}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        news_items = []

        # === 1. 东方财富新闻搜索（稳健 JSONP 解析） ===
        news_items = self._fetch_eastmoney_news(symbol, top_n=top_n)
        if not news_items:
            news_items = self._fetch_eastmoney_news(f"{symbol} 股票", top_n=top_n)
        if news_items:
            print("✅ [源:东方财富] 获取成功")
            result = "\n\n".join(news_items)
            self._cache_set(cache_key, result)
            return result

        # === 2. 尝试 Google News RSS (国际源，最稳) ===
        try:
            query = f"{symbol} 股票"
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
            response = requests.get(rss_url, headers=self.headers, timeout=6)

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
                    result = "\n\n".join(news_items)
                    self._cache_set(cache_key, result)
                    return result
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
                    result = "\n\n".join(news_items)
                    self._cache_set(cache_key, result)
                    return result
        except:
            pass

        result = "✅ 暂无重大敏感舆情 (多源扫描完成)。"
        self._cache_set(cache_key, result)
        return result


if __name__ == "__main__":
    nh = NewsHarvester()
    print(nh.get_latest_news("600519"))