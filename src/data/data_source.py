"""
数据源抽象层
Data Source Abstraction Layer

统一数据接口，支持多数据源（akshare、聚宽、米筐）

Author: VisionQuant Team
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime


class DataSource(ABC):
    """
    数据源抽象基类
    
    所有数据源必须实现这些接口
    """
    
    @abstractmethod
    def get_stock_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        获取股票OHLCV数据
        
        Args:
            symbol: 股票代码（6位，如'600519'）
            start_date: 开始日期（格式：'YYYYMMDD'）
            end_date: 结束日期（格式：'YYYYMMDD'）
            adjust: 复权类型（'qfq'前复权, 'hfq'后复权, 'bfq'不复权）
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index: DatetimeIndex
        """
        pass
    
    @abstractmethod
    def get_index_data(
        self,
        index_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取指数数据
        
        Args:
            index_code: 指数代码（如'000001'上证指数）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            DataFrame with columns: symbol, name, market, etc.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查数据源是否可用
        
        Returns:
            True if available, False otherwise
        """
        pass
    
    def _format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统一数据格式
        
        将不同数据源的数据格式转换为标准格式
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        # 确保有必要的列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 标准化列名（如果存在中文列名）
        column_mapping = {
            '开盘': 'Open', '开盘价': 'Open',
            '收盘': 'Close', '收盘价': 'Close',
            '最高': 'High', '最高价': 'High',
            '最低': 'Low', '最低价': 'Low',
            '成交量': 'Volume', '成交额': 'Amount',
            '日期': 'Date', '时间': 'Date', 'date': 'Date'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 确保索引是日期
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass
        
        # 确保数据类型正确
        for col in required_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 按日期排序
        df.sort_index(inplace=True)
        
        return df


class AkshareDataSource(DataSource):
    """
    AkShare数据源（免费，但可能不稳定）
    """
    
    def __init__(self):
        """初始化AkShare数据源"""
        try:
            import akshare as ak
            self.ak = ak
            self._available = True
        except ImportError:
            self._available = False
            print("⚠️ AkShare未安装，请运行: pip install akshare")
    
    def get_stock_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """获取股票数据（AkShare）"""
        if not self.is_available():
            return pd.DataFrame()
        
        try:
            symbol = str(symbol).strip().zfill(6)
            
            if start_date is None:
                start_date = "20100101"
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
            
            df = self.ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            
            return self._format_data(df)
        except Exception as e:
            print(f"❌ AkShare获取数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def get_index_data(
        self,
        index_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """获取指数数据（AkShare）"""
        if not self.is_available():
            return pd.DataFrame()
        
        try:
            if start_date is None:
                start_date = "20100101"
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
            
            df = self.ak.index_zh_a_hist(
                symbol=index_code,
                period="daily",
                start_date=start_date,
                end_date=end_date
            )
            
            return self._format_data(df)
        except Exception as e:
            print(f"❌ AkShare获取指数数据失败 {index_code}: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表（AkShare）"""
        if not self.is_available():
            return pd.DataFrame()
        
        try:
            df = self.ak.stock_info_a_code_name()
            return df
        except Exception as e:
            print(f"❌ AkShare获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """检查AkShare是否可用"""
        return self._available


if __name__ == "__main__":
    print("=== 数据源抽象层测试 ===")
    
    # 测试AkShare数据源
    akshare_source = AkshareDataSource()
    if akshare_source.is_available():
        print("✅ AkShare数据源可用")
        # 测试获取数据
        df = akshare_source.get_stock_data('600519', start_date='20240101', end_date='20241231')
        if not df.empty:
            print(f"✅ 成功获取数据，共 {len(df)} 条记录")
            print(df.head())
    else:
        print("❌ AkShare数据源不可用")
