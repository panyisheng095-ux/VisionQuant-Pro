import akshare as ak
import pandas as pd
import os
import time
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Optional

# === 路径配置 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# 日志（不强行覆盖全局 logging 配置，交由入口处统一配置）
logger = logging.getLogger(__name__)

# 导入数据源适配器
from .data_source import DataSource, AkshareDataSource
from .jqdata_adapter import JQDataAdapter
from .rqdata_adapter import RQDataAdapter
from .quality_checker import DataQualityChecker


class DataLoader:
    """
    数据加载器（支持多数据源切换）
    
    支持的数据源：
    - 'akshare': 免费数据源（默认）
    - 'jqdata': 聚宽数据源（需要认证）
    - 'rqdata': 米筐数据源（需要认证）
    """
    
    def __init__(self, data_source: str = 'akshare', **kwargs):
        """
        初始化数据加载器
        
        Args:
            data_source: 数据源名称 ('akshare', 'jqdata', 'rqdata')
            **kwargs: 数据源特定参数
                - 对于jqdata: username, password
                - 对于rqdata: username, password
        """
        if not os.path.exists(DATA_RAW_DIR):
            os.makedirs(DATA_RAW_DIR)
        self.data_dir = DATA_RAW_DIR
        
        # 初始化数据源
        self.data_source_name = data_source
        self.data_source = self._init_data_source(data_source, **kwargs)
        
        # 初始化数据质量检查器
        self.quality_checker = DataQualityChecker()
        self.enable_quality_check = kwargs.get('enable_quality_check', True)
    
    def _init_data_source(self, source_name: str, **kwargs) -> DataSource:
        """
        初始化数据源
        
        Args:
            source_name: 数据源名称
            **kwargs: 数据源参数
            
        Returns:
            DataSource实例
        """
        if source_name == 'akshare':
            return AkshareDataSource()
        elif source_name == 'jqdata':
            username = kwargs.get('username') or kwargs.get('jq_username')
            password = kwargs.get('password') or kwargs.get('jq_password')
            return JQDataAdapter(username=username, password=password)
        elif source_name == 'rqdata':
            username = kwargs.get('username') or kwargs.get('rq_username')
            password = kwargs.get('password') or kwargs.get('rq_password')
            return RQDataAdapter(username=username, password=password)
        else:
            logger.warning("未知数据源: %s，使用 akshare 作为默认", source_name)
            return AkshareDataSource()
    
    def switch_data_source(self, source_name: str, **kwargs):
        """
        切换数据源
        
        Args:
            source_name: 新数据源名称
            **kwargs: 数据源参数
        """
        self.data_source_name = source_name
        self.data_source = self._init_data_source(source_name, **kwargs)
        logger.info("已切换到数据源: %s", source_name)
    
    def get_current_data_source(self) -> str:
        """获取当前数据源名称"""
        return self.data_source_name

    def get_stock_data(self, symbol, start_date="20200101", end_date=None, adjust="qfq", use_cache=True):
        """
        [智能更新版] 获取股票数据（支持多数据源）
        
        逻辑：
        1. 如果use_cache=True，先检查本地缓存
        2. 如果数据滞后或不存在，从当前数据源下载
        3. 如果当前数据源不可用，回退到akshare
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型
            use_cache: 是否使用本地缓存
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        symbol = str(symbol).strip().zfill(6)
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        need_download = False
        df = pd.DataFrame()

        # === 1. 检查本地缓存（如果启用） ===
        if use_cache and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not df.empty:
                    last_date_in_file = df.index[-1].date()
                    today = datetime.now().date()
                    
                    if last_date_in_file < today:
                        need_download = True
                    else:
                        return df  # 数据已是最新，直接返回
                else:
                    need_download = True
            except Exception as e:
                logger.warning("读取本地缓存失败 %s (%s): %s", symbol, file_path, e)
                need_download = True
        else:
            need_download = True

        # === 2. 从数据源下载（如果需要） ===
        if need_download:
            # 尝试从当前数据源获取
            if self.data_source and self.data_source.is_available():
                print(f"⬇️ [{self.data_source_name}] 正在拉取 {symbol} 最新行情...")
                df_new = self.data_source.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
                
                if df_new is not None and not df_new.empty:
                    # 数据质量检查
                    if self.enable_quality_check:
                        quality_result = self.quality_checker.check_data_quality(df_new, symbol)
                        if not quality_result['is_valid']:
                            print(f"⚠️ [{symbol}] 数据质量检查未通过 (得分: {quality_result['score']}/100)")
                            if quality_result['score'] < 50:
                                print(f"  错误: {quality_result['errors']}")
                                # 质量太差：优先使用旧数据；没有旧数据则继续走回退数据源逻辑
                                if not df.empty:
                                    return df  # 返回旧数据
                                df_new = None  # 触发回退
                    
                    # 质量不通过且没有旧数据：继续走回退，不在此处保存/返回
                    if df_new is not None and not df_new.empty:
                        # 保存到本地缓存
                        if use_cache:
                            df_new.to_csv(file_path)
                        return df_new
                else:
                    print(f"⚠️ [{self.data_source_name}] 获取数据失败，尝试回退...")
            
            # 回退到akshare（如果当前不是akshare）
            if self.data_source_name != 'akshare':
                print(f"🔄 回退到akshare数据源...")
                fallback_source = AkshareDataSource()
                if fallback_source.is_available():
                    df_new = fallback_source.get_stock_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust
                    )
                    if df_new is not None and not df_new.empty:
                        if use_cache:
                            df_new.to_csv(file_path)
                        return df_new
            
            # 如果所有数据源都失败，返回旧数据（如果有）
            if not df.empty:
                print(f"⚠️ 所有数据源获取失败，使用本地旧数据")
                return df
            
            return pd.DataFrame()

        return df

    def get_top300_stocks(self):
        """获取全A股列表并按市值排序"""
        # 优先使用当前数据源
        if self.data_source and self.data_source.is_available():
            try:
                stock_list = self.data_source.get_stock_list()
                if not stock_list.empty:
                    # 如果有市值信息，按市值排序
                    if 'market_cap' in stock_list.columns:
                        stock_list = stock_list.sort_values(by='market_cap', ascending=False)
                    return stock_list.head(300)
            except Exception as e:
                print(f"⚠️ [{self.data_source_name}] 获取股票列表失败: {e}")
        
        # 回退到akshare
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