import akshare as ak
import pandas as pd
import os
import time
from tqdm import tqdm
from datetime import datetime, timedelta

# === 路径配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


class DataLoader:
    def __init__(self):
        if not os.path.exists(DATA_RAW_DIR):
            os.makedirs(DATA_RAW_DIR)
        self.data_dir = DATA_RAW_DIR

    def get_stock_data(self, symbol, start_date="20200101", end_date=None, adjust="qfq"):
        """
        [智能更新版] 获取股票数据
        逻辑：
        1. 本地无文件 -> 下载
        2. 本地有文件 -> 检查最新日期
           - 如果数据滞后 -> 重新下载覆盖 (保持数据最新)
           - 如果数据是最新的 -> 直接读取 (极速)
        """
        if end_date is None:
            # 获取当前现实世界的日期
            end_date = datetime.now().strftime("%Y%m%d")

        symbol = str(symbol).strip().zfill(6)
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        need_download = False
        df = pd.DataFrame()

        # === 1. 检查本地缓存 ===
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not df.empty:
                    # 获取本地数据的最后一天
                    last_date_in_file = df.index[-1].date()
                    today = datetime.now().date()

                    # 如果今天是周末，我们要往前推到最近的交易日（简单处理：如果最后日期 < 昨天，就更新）
                    # 严谨逻辑：如果最后一条数据不是今天(或最近交易日)，就更新
                    # 这里为了简化：只要最后日期小于今天，就尝试更新
                    if last_date_in_file < today:
                        # print(f"🔄 数据滞后 ({last_date_in_file})，正在更新 {symbol}...")
                        need_download = True
                    else:
                        # print(f"✅ 数据已是最新 ({last_date_in_file})")
                        need_download = False
                else:
                    need_download = True
            except:
                need_download = True
        else:
            need_download = True

        # === 2. 执行下载 (如果需要) ===
        if need_download:
            print(f"⬇️ [联网更新] 正在拉取 {symbol} 最新行情...")
            try:
                # 重新下载全量数据 (覆盖模式)
                # AkShare 接口很快，直接覆盖比增量append更不容易出错
                df_new = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                            start_date=start_date, end_date=end_date, adjust=adjust)

                if df_new is None or df_new.empty:
                    # 如果下载失败但本地有旧数据，就暂时用旧的
                    if not df.empty:
                        print(f"⚠️ 网络下载失败，降级使用本地旧数据")
                        return df
                    return pd.DataFrame()

                # 格式化
                rename_map = {
                    "日期": "Date", "开盘": "Open", "收盘": "Close",
                    "最高": "High", "最低": "Low", "成交量": "Volume"
                }
                df_new = df_new.rename(columns=rename_map)
                df_new['Date'] = pd.to_datetime(df_new['Date'])
                df_new.set_index('Date', inplace=True)

                # 保存覆盖
                df_new.to_csv(file_path)
                return df_new

            except Exception as e:
                print(f"❌ 更新失败: {e}")
                if not df.empty: return df  # 返回旧数据兜底
                return pd.DataFrame()

        return df

    def get_top300_stocks(self):
        """获取全A股列表并按市值排序"""
        try:
            df = ak.stock_zh_a_spot_em()
            if '总市值' in df.columns:
                df = df.sort_values(by='总市值', ascending=False)
            df = df.head(300)
            return df[['代码', '名称']].rename(columns={'代码': 'code', '名称': 'name'})
        except Exception as e:
            print(f"❌ 获取名单失败: {e}")
            return pd.DataFrame()

    def download_batch_data(self, stock_list, start_date="20200101"):
        """批量下载"""
        print(f"⬇️ [批量维护] 正在检查并更新 {len(stock_list)} 只股票...")
        for _, row in tqdm(stock_list.iterrows(), total=len(stock_list)):
            symbol = str(row['code']).zfill(6)
            self.get_stock_data(symbol, start_date=start_date)
            # 稍微快一点，因为大部分可能不需要下载
            time.sleep(0.01)


if __name__ == "__main__":
    loader = DataLoader()
    # 测试更新逻辑
    df = loader.get_stock_data("601899")
    print(f"最新数据日期: {df.index[-1]}")