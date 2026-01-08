import os
import sys
import pandas as pd
import numpy as np
import mplfinance as mpf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class BatchAnalyzer:
    """批量股票分析引擎（快速海选模式）"""
    
    def __init__(self, engines):
        self.engines = engines
        self.max_workers = 8  # 控制并发，避免API限流
        self.prediction_cache = self._load_prediction_cache()
    
    def _load_prediction_cache(self):
        """加载预测缓存"""
        cache_file = os.path.join(PROJECT_ROOT, "data", "indices", "prediction_cache.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, dtype=str)
                df['date'] = df['date'].astype(str).str.replace('-', '')
                df['symbol'] = df['symbol'].astype(str).str.zfill(6)
                return df.set_index(['symbol', 'date'])['pred_win_rate'].to_dict()
            except:
                return {}
        return {}
    
    def analyze_batch(self, symbols, progress_callback=None):
        """
        批量分析股票（快速模式）
        
        Args:
            symbols: 股票代码列表
            progress_callback: 进度回调函数 (current, total, symbol)
        
        Returns:
            Dict: {symbol: analysis_result}
        """
        results = {}
        total = len(symbols)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_single_fast, sym): sym 
                for sym in symbols
            }
            
            completed = 0
            for future in as_completed(futures):
                symbol = futures[future]
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total, symbol)
                
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    results[symbol] = {
                        "symbol": symbol,
                        "error": str(e),
                        "score": 0,
                        "action": "ERROR"
                    }
        
        return results
    
    def _analyze_single_fast(self, symbol):
        """单股票快速分析（跳过可视化）"""
        try:
            # 1. 数据获取
            df = self.engines["loader"].get_stock_data(symbol)
            if df.empty:
                return {"symbol": symbol, "error": "数据获取失败", "score": 0, "action": "ERROR"}
            
            # 2. 财务数据
            fund_data = self.engines["fund"].get_stock_fundamentals(symbol)
            stock_name = fund_data.get('name', symbol)
            
            # 3. 视觉搜索（简化：只取Top-3，不生成对比图）
            q_p = os.path.join(PROJECT_ROOT, "data", f"temp_batch_{symbol}.png")
            mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='')
            mpf.plot(df.tail(20), type='candle', style=s, 
                    savefig=dict(fname=q_p, dpi=50), figsize=(3, 3), axisoff=True)
            
            matches = self.engines["vision"].search_similar_patterns(q_p, top_k=3)
            
            # 4. 快速胜率计算（使用缓存或简化计算）
            win_rate = self._get_win_rate_fast(symbol, matches, df)
            
            # 5. 技术指标
            df_f = self.engines["factor"]._add_technical_indicators(df)
            
            # 6. 评分
            total_score, initial_action, s_details = self.engines["factor"].get_scorecard(
                win_rate, df_f.iloc[-1], fund_data
            )
            
            # 7. 简化AI分析（只获取核心决策）
            try:
                report = self.engines["agent"].analyze(
                    symbol, total_score, initial_action,
                    {"win_rate": win_rate},
                    df_f.iloc[-1].to_dict(), fund_data, ""
                )
                action = report.action
                confidence = report.confidence
                reasoning = report.reasoning[:150]  # 截断
            except:
                action = initial_action
                confidence = int(total_score * 10)
                reasoning = "快速分析模式"
            
            # 8. 预期收益（简化）
            expected_return = self._estimate_return(win_rate, df_f.iloc[-1])
            
            return {
                "symbol": symbol,
                "name": stock_name,
                "score": round(total_score, 1),
                "action": action,
                "confidence": confidence,
                "win_rate": round(win_rate, 1),
                "expected_return": round(expected_return, 2),
                "roe": round(fund_data.get('roe', 0), 2),
                "pe_ttm": round(fund_data.get('pe_ttm', 0), 2),
                "reasoning": reasoning,
                "details": s_details,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {
                "symbol": symbol,
                "error": str(e),
                "score": 0,
                "action": "ERROR"
            }
    
    def _get_win_rate_fast(self, symbol, matches, df):
        """快速计算胜率（优先使用缓存）"""
        # 尝试从缓存获取
        today_str = datetime.now().strftime("%Y%m%d")
        cache_key = (symbol, today_str)
        
        if cache_key in self.prediction_cache:
            return float(self.prediction_cache[cache_key])
        
        # 如果没有缓存，简化计算（只检查Top-3匹配）
        if not matches:
            return 50.0
        
        # 简化：基于相似度加权
        total_weight = sum(m.get('score', 0) for m in matches)
        if total_weight > 0:
            # 假设相似度高的形态胜率更高（简化假设）
            avg_similarity = np.mean([m.get('score', 0) for m in matches])
            # 映射到50-70%范围
            win_rate = 50 + (avg_similarity * 20)
            return min(70, max(50, win_rate))
        
        return 50.0
    
    def _estimate_return(self, win_rate, factor_row):
        """估算预期收益"""
        # 简化模型：基于胜率和趋势
        base_return = (win_rate - 50) * 0.1  # 胜率每增加1%，预期收益+0.1%
        
        # 趋势加成
        if factor_row.get('MA_Signal', 0) > 0:
            base_return += 1.0
        
        # MACD加成
        if factor_row.get('MACD_Hist', 0) > 0:
            base_return += 0.5
        
        return base_return
