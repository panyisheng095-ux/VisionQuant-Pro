"""
聚宽数据源适配器
JoinQuant Data Source Adapter

封装聚宽API，提供统一的数据接口

使用前需要：
1. pip install jqdatasdk
2. 在聚宽官网注册账号
3. 调用 jq.auth(username, password) 进行认证

Author: VisionQuant Team
"""

import pandas as pd
from typing import Optional
from datetime import datetime
import warnings

from .data_source import DataSource


class JQDataAdapter(DataSource):
    """
    聚宽数据源适配器
    
    聚宽是专业的量化数据平台，数据质量高、稳定性好
    """
    
    def __init__(self, username: str = None, password: str = None):
        """
        初始化聚宽适配器
        
        Args:
            username: 聚宽用户名（可选，也可通过环境变量设置）
            password: 聚宽密码（可选，也可通过环境变量设置）
        """
        self.username = username
        self.password = password
        self.jq = None
        self._authenticated = False
        
        # 尝试导入jqdatasdk
        try:
            import jqdatasdk as jq
            self.jq = jq
            self._available = True
        except ImportError:
            self._available = False
            print("⚠️ jqdatasdk未安装，请运行: pip install jqdatasdk")
            return
        
        # 尝试认证
        if username and password:
            self.authenticate(username, password)
        else:
            # 尝试从环境变量读取
            import os
            env_username = os.getenv('JQDATA_USERNAME')
            env_password = os.getenv('JQDATA_PASSWORD')
            if env_username and env_password:
                self.authenticate(env_username, env_password)
    
    def authenticate(self, username: str, password: str) -> bool:
        """
        聚宽认证
        
        Args:
            username: 聚宽用户名
            password: 聚宽密码
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self.jq.auth(username, password)
            self.username = username
            self._authenticated = True
            print("✅ 聚宽认证成功")
            return True
        except Exception as e:
            print(f"❌ 聚宽认证失败: {e}")
            self._authenticated = False
            return False
    
    def get_stock_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        获取股票数据（聚宽）
        
        聚宽股票代码格式：'600519.XSHG' (上交所) 或 '000001.XSHE' (深交所)
        """
        if not self.is_available() or not self._authenticated:
            return pd.DataFrame()
        
        try:
            # 转换股票代码格式
            jq_symbol = self._convert_symbol(symbol)
            if not jq_symbol:
                return pd.DataFrame()
            
            # 转换日期格式
            if start_date:
                start_date = pd.to_datetime(start_date, format='%Y%m%d')
            else:
                start_date = datetime(2010, 1, 1)
            
            if end_date:
                end_date = pd.to_datetime(end_date, format='%Y%m%d')
            else:
                end_date = datetime.now()
            
            # 获取数据
            df = self.jq.get_price(
                jq_symbol,
                start_date=start_date,
                end_date=end_date,
                frequency='daily',
                fields=['open', 'close', 'high', 'low', 'volume'],
                skip_paused=True,  # 跳过停牌日
                fq=adjust  # 复权类型
            )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 转换列名为标准格式
            df = df.rename(columns={
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume'
            })
            
            return self._format_data(df)
            
        except Exception as e:
            print(f"❌ 聚宽获取股票数据失败 {symbol}: {e}")
            return pd.DataFrame()
    
    def get_index_data(
        self,
        index_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取指数数据（聚宽）
        
        聚宽指数代码格式：'000001.XSHG' (上证指数)
        """
        if not self.is_available() or not self._authenticated:
            return pd.DataFrame()
        
        try:
            # 转换指数代码格式
            jq_index = self._convert_index_code(index_code)
            if not jq_index:
                return pd.DataFrame()
            
            # 转换日期格式
            if start_date:
                start_date = pd.to_datetime(start_date, format='%Y%m%d')
            else:
                start_date = datetime(2010, 1, 1)
            
            if end_date:
                end_date = pd.to_datetime(end_date, format='%Y%m%d')
            else:
                end_date = datetime.now()
            
            # 获取数据
            df = self.jq.get_price(
                jq_index,
                start_date=start_date,
                end_date=end_date,
                frequency='daily',
                fields=['open', 'close', 'high', 'low', 'volume']
            )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # 转换列名
            df = df.rename(columns={
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume'
            })
            
            return self._format_data(df)
            
        except Exception as e:
            print(f"❌ 聚宽获取指数数据失败 {index_code}: {e}")
            return pd.DataFrame()
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表（聚宽）"""
        if not self.is_available() or not self._authenticated:
            return pd.DataFrame()
        
        try:
            # 获取所有股票
            stocks = self.jq.get_all_securities(types=['stock'], date=None)
            
            # 转换为DataFrame
            df = stocks.reset_index()
            df = df.rename(columns={'index': 'symbol', 'display_name': 'name'})
            
            return df
            
        except Exception as e:
            print(f"❌ 聚宽获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def _convert_symbol(self, symbol: str) -> Optional[str]:
        """
        转换股票代码格式
        
        输入：'600519' (6位数字)
        输出：'600519.XSHG' 或 '000001.XSHE'
        """
        symbol = str(symbol).strip().zfill(6)
        
        # 判断是上交所还是深交所
        if symbol.startswith(('60', '68', '30')):  # 上交所：60, 68(科创板), 30(科创板)
            return f"{symbol}.XSHG"
        elif symbol.startswith(('00', '30')):  # 深交所：00, 30(创业板)
            return f"{symbol}.XSHE"
        else:
            print(f"⚠️ 无法识别股票代码: {symbol}")
            return None
    
    def _convert_index_code(self, index_code: str) -> Optional[str]:
        """
        转换指数代码格式
        
        输入：'000001' (上证指数)
        输出：'000001.XSHG'
        """
        index_code = str(index_code).strip().zfill(6)
        
        # 常见指数映射
        index_mapping = {
            '000001': '000001.XSHG',  # 上证指数
            '399001': '399001.XSHE',  # 深证成指
            '399006': '399006.XSHE',  # 创业板指
            '000300': '000300.XSHG',  # 沪深300
            '000905': '000905.XSHG',  # 中证500
        }
        
        if index_code in index_mapping:
            return index_mapping[index_code]
        else:
            # 默认尝试上交所
            return f"{index_code}.XSHG"
    
    def is_available(self) -> bool:
        """检查聚宽是否可用"""
        return self._available and self._authenticated
    
    def logout(self):
        """登出聚宽"""
        if self.jq and self._authenticated:
            try:
                self.jq.logout()
                self._authenticated = False
                print("✅ 聚宽已登出")
            except:
                pass


if __name__ == "__main__":
    print("=== 聚宽数据源适配器测试 ===")
    
    # 注意：需要提供真实的用户名和密码
    # adapter = JQDataAdapter(username='your_username', password='your_password')
    # 
    # if adapter.is_available():
    #     print("✅ 聚宽数据源可用")
    #     df = adapter.get_stock_data('600519', start_date='20240101', end_date='20241231')
    #     if not df.empty:
    #         print(f"✅ 成功获取数据，共 {len(df)} 条记录")
    #     adapter.logout()
    # else:
    #     print("❌ 聚宽数据源不可用（需要认证）")
    
    print("ℹ️ 请提供聚宽用户名和密码进行测试")
