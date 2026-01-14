"""
数据质量检查模块
Data Quality Checker

检查数据完整性和准确性

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class DataQualityChecker:
    """
    数据质量检查器
    
    功能：
    1. 检查数据完整性（缺失值、异常值）
    2. 检查数据准确性（价格合理性、成交量合理性）
    3. 检查数据一致性（OHLC关系、时间连续性）
    4. 生成质量报告
    """
    
    def __init__(self):
        """初始化数据质量检查器"""
        pass
    
    def check_data_quality(
        self,
        df: pd.DataFrame,
        symbol: str = None
    ) -> Dict:
        """
        全面检查数据质量
        
        Args:
            df: 股票数据DataFrame
            symbol: 股票代码（可选）
            
        Returns:
            质量检查结果字典
        """
        if df is None or df.empty:
            return {
                'is_valid': False,
                'errors': ['数据为空'],
                'warnings': [],
                'score': 0.0
            }
        
        errors = []
        warnings = []
        score = 100.0
        
        # 1. 检查必要列
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"缺少必要列: {missing_columns}")
            score -= 20.0
        
        # 2. 检查缺失值
        missing_stats = self._check_missing_values(df)
        if missing_stats['total_missing'] > 0:
            warnings.append(f"存在缺失值: {missing_stats['total_missing']}个")
            score -= missing_stats['missing_ratio'] * 10
        
        # 3. 检查OHLC关系
        ohlc_errors = self._check_ohlc_consistency(df)
        if ohlc_errors:
            errors.extend(ohlc_errors)
            score -= len(ohlc_errors) * 5
        
        # 4. 检查价格合理性
        price_warnings = self._check_price_reasonableness(df)
        if price_warnings:
            warnings.extend(price_warnings)
            score -= len(price_warnings) * 2
        
        # 5. 检查成交量合理性
        volume_warnings = self._check_volume_reasonableness(df)
        if volume_warnings:
            warnings.extend(volume_warnings)
            score -= len(volume_warnings) * 1
        
        # 6. 检查时间连续性
        time_issues = self._check_time_continuity(df)
        if time_issues:
            warnings.extend(time_issues)
            score -= len(time_issues) * 1
        
        # 7. 检查异常值
        outlier_warnings = self._check_outliers(df)
        if outlier_warnings:
            warnings.extend(outlier_warnings)
            score -= len(outlier_warnings) * 1
        
        score = max(0.0, score)
        is_valid = len(errors) == 0 and score >= 70.0
        
        return {
            'is_valid': is_valid,
            'score': round(score, 2),
            'errors': errors,
            'warnings': warnings,
            'missing_stats': missing_stats,
            'data_points': len(df),
            'date_range': {
                'start': str(df.index[0]) if len(df) > 0 else None,
                'end': str(df.index[-1]) if len(df) > 0 else None
            }
        }
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """检查缺失值"""
        missing_count = df.isnull().sum()
        total_missing = missing_count.sum()
        missing_ratio = total_missing / (len(df) * len(df.columns)) if len(df) > 0 else 0
        
        return {
            'total_missing': int(total_missing),
            'missing_ratio': float(missing_ratio),
            'by_column': missing_count.to_dict()
        }
    
    def _check_ohlc_consistency(self, df: pd.DataFrame) -> List[str]:
        """检查OHLC一致性"""
        errors = []
        
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            return errors
        
        # High >= max(Open, Close)
        invalid_high = df[df['High'] < df[['Open', 'Close']].max(axis=1)]
        if len(invalid_high) > 0:
            errors.append(f"High < max(Open, Close) 的记录: {len(invalid_high)}条")
        
        # Low <= min(Open, Close)
        invalid_low = df[df['Low'] > df[['Open', 'Close']].min(axis=1)]
        if len(invalid_low) > 0:
            errors.append(f"Low > min(Open, Close) 的记录: {len(invalid_low)}条")
        
        # High >= Low
        invalid_range = df[df['High'] < df['Low']]
        if len(invalid_range) > 0:
            errors.append(f"High < Low 的记录: {len(invalid_range)}条")
        
        return errors
    
    def _check_price_reasonableness(self, df: pd.DataFrame) -> List[str]:
        """检查价格合理性"""
        warnings = []
        
        if 'Close' not in df.columns:
            return warnings
        
        # 检查价格是否为0或负数
        zero_prices = df[df['Close'] <= 0]
        if len(zero_prices) > 0:
            warnings.append(f"收盘价为0或负数的记录: {len(zero_prices)}条")
        
        # 检查价格异常波动（单日涨跌幅超过20%）
        if len(df) > 1:
            returns = df['Close'].pct_change()
            extreme_moves = df[abs(returns) > 0.20]
            if len(extreme_moves) > 0:
                warnings.append(f"单日涨跌幅超过20%的记录: {len(extreme_moves)}条（可能是停牌复牌或数据错误）")
        
        # 检查价格是否在合理范围（假设A股价格在0.1-1000之间）
        extreme_prices = df[(df['Close'] < 0.1) | (df['Close'] > 1000)]
        if len(extreme_prices) > 0:
            warnings.append(f"价格超出合理范围(0.1-1000)的记录: {len(extreme_prices)}条")
        
        return warnings
    
    def _check_volume_reasonableness(self, df: pd.DataFrame) -> List[str]:
        """检查成交量合理性"""
        warnings = []
        
        if 'Volume' not in df.columns:
            return warnings
        
        # 检查成交量为负数
        negative_volume = df[df['Volume'] < 0]
        if len(negative_volume) > 0:
            warnings.append(f"成交量为负数的记录: {len(negative_volume)}条")
        
        # 检查成交量异常（超过平均值的10倍）
        if len(df) > 1:
            avg_volume = df['Volume'].mean()
            extreme_volume = df[df['Volume'] > avg_volume * 10]
            if len(extreme_volume) > 0:
                warnings.append(f"成交量异常大(>10倍均值)的记录: {len(extreme_volume)}条")
        
        return warnings
    
    def _check_time_continuity(self, df: pd.DataFrame) -> List[str]:
        """检查时间连续性"""
        warnings = []
        
        if len(df) < 2:
            return warnings
        
        # 检查索引是否为DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            warnings.append("索引不是DatetimeIndex")
            return warnings
        
        # 检查是否有重复日期
        duplicate_dates = df.index.duplicated()
        if duplicate_dates.any():
            warnings.append(f"存在重复日期: {duplicate_dates.sum()}个")
        
        # 检查时间间隔（正常情况下应该是交易日，但可能有节假日）
        # 这里只检查是否有异常大的间隔（>10天）
        date_diffs = df.index.to_series().diff()
        large_gaps = date_diffs[date_diffs > timedelta(days=10)]
        if len(large_gaps) > 0:
            warnings.append(f"存在异常大的时间间隔(>10天): {len(large_gaps)}个（可能是停牌）")
        
        return warnings
    
    def _check_outliers(self, df: pd.DataFrame) -> List[str]:
        """检查异常值（使用IQR方法）"""
        warnings = []
        
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in numeric_columns if col in df.columns]
        
        for col in available_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    warnings.append(f"{col}列存在异常值: {len(outliers)}个")
        
        return warnings
    
    def generate_quality_report(
        self,
        df: pd.DataFrame,
        symbol: str = None
    ) -> str:
        """
        生成数据质量报告（文本格式）
        """
        result = self.check_data_quality(df, symbol)
        
        report = f"""
=== 数据质量报告 ===
股票代码: {symbol or 'N/A'}
数据点数: {result['data_points']}
日期范围: {result['date_range']['start']} 至 {result['date_range']['end']}

质量得分: {result['score']}/100
状态: {'✅ 通过' if result['is_valid'] else '❌ 未通过'}

错误 ({len(result['errors'])}):
"""
        if result['errors']:
            for error in result['errors']:
                report += f"  ❌ {error}\n"
        else:
            report += "  无错误\n"
        
        report += f"\n警告 ({len(result['warnings'])}):\n"
        if result['warnings']:
            for warning in result['warnings']:
                report += f"  ⚠️ {warning}\n"
        else:
            report += "  无警告\n"
        
        if result['missing_stats']['total_missing'] > 0:
            report += f"\n缺失值统计:\n"
            report += f"  总缺失数: {result['missing_stats']['total_missing']}\n"
            report += f"  缺失比例: {result['missing_stats']['missing_ratio']:.2%}\n"
        
        return report


if __name__ == "__main__":
    print("=== 数据质量检查器测试 ===")
    
    # 模拟数据
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(50, 60, 100),
        'High': np.random.uniform(55, 65, 100),
        'Low': np.random.uniform(45, 55, 100),
        'Close': np.random.uniform(50, 60, 100),
        'Volume': np.random.uniform(1000000, 10000000, 100)
    }, index=dates)
    
    # 添加一些错误数据
    df.loc[dates[10], 'High'] = 30  # High < Low
    df.loc[dates[20], 'Close'] = np.nan  # 缺失值
    
    checker = DataQualityChecker()
    result = checker.check_data_quality(df, '600519')
    
    print(checker.generate_quality_report(df, '600519'))
