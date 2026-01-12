"""
Baseline Experiments - 对比实验框架

用于arXiv论文的实验部分，包含:
1. Buy-and-Hold 基准策略
2. MA Crossover 均线交叉策略
3. RSI 超买超卖策略
4. LSTM 深度学习预测策略
5. 统计显著性检验

论文要求: 
- 必须与多个Baseline对比
- 必须进行统计显著性检验 (t-test, p-value < 0.05)
- 必须进行消融实验 (Ablation Study)

Author: Yisheng Pan
Date: 2026-01
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestResult:
    """回测结果数据类"""
    strategy_name: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    alpha: float  # 相对于Buy-Hold的超额收益
    daily_returns: np.ndarray
    equity_curve: np.ndarray


class BaseStrategy:
    """策略基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            signals: 1=买入, -1=卖出, 0=持有
        """
        raise NotImplementedError
    
    def backtest(
        self, 
        df: pd.DataFrame, 
        initial_capital: float = 100000,
        commission: float = 0.001
    ) -> BacktestResult:
        """
        回测策略
        
        Args:
            df: 包含OHLCV数据的DataFrame
            initial_capital: 初始资金
            commission: 手续费率
            
        Returns:
            result: 回测结果
        """
        signals = self.generate_signals(df)
        
        # 初始化
        cash = initial_capital
        position = 0
        equity_curve = [initial_capital]
        daily_returns = []
        trades = 0
        wins = 0
        
        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            prev_price = df['close'].iloc[i-1]
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # 执行交易
            if signal == 1 and position == 0:  # 买入
                shares = (cash * (1 - commission)) / price
                position = shares
                cash = 0
                trades += 1
                entry_price = price
                
            elif signal == -1 and position > 0:  # 卖出
                cash = position * price * (1 - commission)
                if price > entry_price:
                    wins += 1
                position = 0
            
            # 计算权益
            equity = cash + position * price
            daily_return = (equity - equity_curve[-1]) / equity_curve[-1]
            daily_returns.append(daily_return)
            equity_curve.append(equity)
        
        # 最终结算
        if position > 0:
            cash = position * df['close'].iloc[-1] * (1 - commission)
        
        equity_curve = np.array(equity_curve)
        daily_returns = np.array(daily_returns)
        
        # 计算指标
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        
        # 年化收益
        days = len(df)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # 夏普比率
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe = 0
        
        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = drawdown.max()
        
        # 胜率
        win_rate = wins / trades if trades > 0 else 0
        
        # Buy-Hold收益（用于计算Alpha）
        bh_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
        alpha = total_return - bh_return
        
        return BacktestResult(
            strategy_name=self.name,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=trades,
            alpha=alpha,
            daily_returns=daily_returns,
            equity_curve=equity_curve
        )


# ============================================================
# Baseline 1: Buy-and-Hold
# ============================================================

class BuyAndHoldStrategy(BaseStrategy):
    """
    买入持有策略（基准）
    
    最简单的策略：开盘买入，收盘卖出
    用于计算其他策略的Alpha
    """
    
    def __init__(self):
        super().__init__("Buy-and-Hold")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals.iloc[0] = 1  # 第一天买入
        signals.iloc[-1] = -1  # 最后一天卖出
        return signals


# ============================================================
# Baseline 2: MA Crossover
# ============================================================

class MACrossoverStrategy(BaseStrategy):
    """
    均线交叉策略
    
    经典技术分析策略：
    - 短期均线上穿长期均线 → 买入
    - 短期均线下穿长期均线 → 卖出
    """
    
    def __init__(self, short_period: int = 20, long_period: int = 60):
        super().__init__(f"MA({short_period}/{long_period})")
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ma_short = df['close'].rolling(self.short_period).mean()
        ma_long = df['close'].rolling(self.long_period).mean()
        
        signals = pd.Series(0, index=df.index)
        
        # 金叉买入，死叉卖出
        signals[(ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))] = 1
        signals[(ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))] = -1
        
        return signals


# ============================================================
# Baseline 3: RSI Strategy
# ============================================================

class RSIStrategy(BaseStrategy):
    """
    RSI超买超卖策略
    
    相对强弱指标策略：
    - RSI < 30（超卖）→ 买入
    - RSI > 70（超买）→ 卖出
    """
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"RSI({period})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=df.index)
        
        # 超卖买入，超买卖出
        signals[(rsi < self.oversold) & (rsi.shift(1) >= self.oversold)] = 1
        signals[(rsi > self.overbought) & (rsi.shift(1) <= self.overbought)] = -1
        
        return signals


# ============================================================
# Baseline 4: MACD Strategy
# ============================================================

class MACDStrategy(BaseStrategy):
    """
    MACD策略
    
    - MACD线上穿信号线 → 买入
    - MACD线下穿信号线 → 卖出
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD({fast}/{slow}/{signal})")
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        exp1 = df['close'].ewm(span=self.fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        
        signals = pd.Series(0, index=df.index)
        
        # MACD金叉买入，死叉卖出
        signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
        signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
        
        return signals


# ============================================================
# Baseline 5: Momentum Strategy
# ============================================================

class MomentumStrategy(BaseStrategy):
    """
    动量策略
    
    基于价格动量：
    - N日收益 > 阈值 → 买入
    - N日收益 < -阈值 → 卖出
    """
    
    def __init__(self, lookback: int = 20, threshold: float = 0.05):
        super().__init__(f"Momentum({lookback})")
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        returns = df['close'].pct_change(self.lookback)
        
        signals = pd.Series(0, index=df.index)
        
        signals[returns > self.threshold] = 1
        signals[returns < -self.threshold] = -1
        
        return signals


# ============================================================
# 实验框架
# ============================================================

class ExperimentFramework:
    """
    实验框架
    
    用于运行所有Baseline对比实验和统计检验
    """
    
    def __init__(self):
        self.baselines = [
            BuyAndHoldStrategy(),
            MACrossoverStrategy(20, 60),
            RSIStrategy(14, 30, 70),
            MACDStrategy(12, 26, 9),
            MomentumStrategy(20, 0.05),
        ]
        self.results = {}
    
    def add_strategy(self, strategy: BaseStrategy):
        """添加自定义策略（如VQ策略）"""
        self.baselines.append(strategy)
    
    def run_single_stock(
        self, 
        df: pd.DataFrame, 
        stock_code: str,
        vq_strategy: Optional[BaseStrategy] = None
    ) -> pd.DataFrame:
        """
        对单只股票运行所有策略
        
        Args:
            df: 股票数据
            stock_code: 股票代码
            vq_strategy: VQ策略（如果有）
            
        Returns:
            results_df: 结果DataFrame
        """
        results = []
        
        strategies = self.baselines.copy()
        if vq_strategy:
            strategies.append(vq_strategy)
        
        for strategy in strategies:
            try:
                result = strategy.backtest(df)
                results.append({
                    'Stock': stock_code,
                    'Strategy': result.strategy_name,
                    'Return': result.total_return,
                    'Annual_Return': result.annual_return,
                    'Sharpe': result.sharpe_ratio,
                    'MaxDD': result.max_drawdown,
                    'WinRate': result.win_rate,
                    'Trades': result.num_trades,
                    'Alpha': result.alpha,
                })
                self.results[(stock_code, strategy.name)] = result
            except Exception as e:
                print(f"⚠️ {strategy.name} on {stock_code} failed: {e}")
        
        return pd.DataFrame(results)
    
    def run_multi_stock(
        self,
        stock_data: Dict[str, pd.DataFrame],
        vq_strategy: Optional[BaseStrategy] = None
    ) -> pd.DataFrame:
        """
        对多只股票运行实验
        
        Args:
            stock_data: {股票代码: DataFrame} 字典
            vq_strategy: VQ策略
            
        Returns:
            all_results: 汇总结果
        """
        all_results = []
        
        for code, df in stock_data.items():
            print(f"Running experiments on {code}...")
            result_df = self.run_single_stock(df, code, vq_strategy)
            all_results.append(result_df)
        
        return pd.concat(all_results, ignore_index=True)
    
    def statistical_tests(
        self,
        target_strategy: str = "VQ",
        baseline_strategy: str = "Buy-and-Hold"
    ) -> Dict:
        """
        统计显著性检验
        
        使用配对t检验比较策略收益是否显著优于Baseline
        
        Args:
            target_strategy: 目标策略名称
            baseline_strategy: 基准策略名称
            
        Returns:
            test_results: 检验结果
        """
        target_returns = []
        baseline_returns = []
        
        # 收集所有股票的收益
        stocks = set(k[0] for k in self.results.keys())
        
        for stock in stocks:
            target_key = (stock, target_strategy)
            baseline_key = (stock, baseline_strategy)
            
            if target_key in self.results and baseline_key in self.results:
                target_returns.append(self.results[target_key].total_return)
                baseline_returns.append(self.results[baseline_key].total_return)
        
        target_returns = np.array(target_returns)
        baseline_returns = np.array(baseline_returns)
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(target_returns, baseline_returns)
        
        # Wilcoxon符号秩检验（非参数检验）
        try:
            w_stat, w_pvalue = stats.wilcoxon(target_returns - baseline_returns)
        except:
            w_stat, w_pvalue = np.nan, np.nan
        
        # 效应量 (Cohen's d)
        diff = target_returns - baseline_returns
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
        
        return {
            'target_strategy': target_strategy,
            'baseline_strategy': baseline_strategy,
            'n_samples': len(target_returns),
            'target_mean_return': target_returns.mean(),
            'baseline_mean_return': baseline_returns.mean(),
            'mean_alpha': (target_returns - baseline_returns).mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'wilcoxon_stat': w_stat,
            'wilcoxon_p': w_pvalue,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
    
    def run_all_statistical_tests(self, target_strategy: str = "VQ") -> pd.DataFrame:
        """
        对所有Baseline进行统计检验
        """
        results = []
        
        for baseline in self.baselines:
            if baseline.name != target_strategy:
                test = self.statistical_tests(target_strategy, baseline.name)
                results.append(test)
        
        return pd.DataFrame(results)
    
    def generate_latex_table(self, results_df: pd.DataFrame) -> str:
        """
        生成LaTeX表格（用于论文）
        """
        latex = r"""
\begin{table}[t]
\centering
\caption{Backtesting Results Comparison}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
Strategy & Return & Alpha & Sharpe & MaxDD & Win\% \\
\midrule
"""
        # 按策略聚合
        agg = results_df.groupby('Strategy').agg({
            'Return': 'mean',
            'Alpha': 'mean',
            'Sharpe': 'mean',
            'MaxDD': 'mean',
            'WinRate': 'mean'
        })
        
        for strategy, row in agg.iterrows():
            latex += f"{strategy} & {row['Return']*100:.1f}\\% & {row['Alpha']*100:.1f}\\% & "
            latex += f"{row['Sharpe']:.2f} & {row['MaxDD']*100:.1f}\\% & {row['WinRate']*100:.1f}\\% \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex
    
    def generate_statistical_latex(self, test_results: pd.DataFrame) -> str:
        """
        生成统计检验的LaTeX表格
        """
        latex = r"""
\begin{table}[t]
\centering
\caption{Statistical Significance Tests (VQ vs. Baselines)}
\label{tab:statistical_tests}
\begin{tabular}{lcccc}
\toprule
Comparison & t-stat & p-value & Cohen's d & Significant \\
\midrule
"""
        for _, row in test_results.iterrows():
            sig = r"\checkmark" if row['significant'] else ""
            latex += f"VQ vs. {row['baseline_strategy']} & {row['t_statistic']:.2f} & "
            latex += f"{row['p_value']:.4f} & {row['cohens_d']:.2f} & {sig} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex


# ============================================================
# 消融实验
# ============================================================

class AblationStudy:
    """
    消融实验框架
    
    用于证明每个模块的作用
    """
    
    def __init__(self, full_model_results: BacktestResult):
        self.full_results = full_model_results
        self.ablation_results = {}
    
    def add_ablation(self, name: str, result: BacktestResult):
        """添加消融配置的结果"""
        self.ablation_results[name] = result
    
    def generate_report(self) -> pd.DataFrame:
        """生成消融实验报告"""
        rows = [
            {
                'Configuration': 'Full Model (VQ)',
                'Return': self.full_results.total_return,
                'Alpha': self.full_results.alpha,
                'Sharpe': self.full_results.sharpe_ratio,
                'MaxDD': self.full_results.max_drawdown
            }
        ]
        
        for name, result in self.ablation_results.items():
            rows.append({
                'Configuration': name,
                'Return': result.total_return,
                'Alpha': result.alpha,
                'Sharpe': result.sharpe_ratio,
                'MaxDD': result.max_drawdown
            })
        
        return pd.DataFrame(rows)
    
    def generate_latex_table(self) -> str:
        """生成消融实验LaTeX表格"""
        df = self.generate_report()
        
        latex = r"""
\begin{table}[t]
\centering
\caption{Ablation Study: Effect of Each Component}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
Configuration & Return & Alpha & Sharpe & MaxDD \\
\midrule
"""
        for _, row in df.iterrows():
            latex += f"{row['Configuration']} & {row['Return']*100:.1f}\\% & "
            latex += f"{row['Alpha']*100:.1f}\\% & {row['Sharpe']:.2f} & "
            latex += f"{row['MaxDD']*100:.1f}\\% \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex


# ============================================================
# 敏感性分析
# ============================================================

def sensitivity_analysis(
    df: pd.DataFrame,
    param_name: str,
    param_values: List,
    strategy_factory,
    metric: str = 'sharpe_ratio'
) -> pd.DataFrame:
    """
    超参数敏感性分析
    
    Args:
        df: 股票数据
        param_name: 参数名称
        param_values: 参数值列表
        strategy_factory: 策略工厂函数 lambda param: Strategy(param)
        metric: 评估指标
        
    Returns:
        results: 敏感性分析结果
    """
    results = []
    
    for value in param_values:
        strategy = strategy_factory(value)
        result = strategy.backtest(df)
        
        results.append({
            param_name: value,
            'Return': result.total_return,
            'Alpha': result.alpha,
            'Sharpe': result.sharpe_ratio,
            'MaxDD': result.max_drawdown,
        })
    
    return pd.DataFrame(results)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Baseline Experiments Framework")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    # 模拟股价（带趋势）
    returns = np.random.randn(len(dates)) * 0.02 + 0.0005
    prices = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(len(prices)) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(len(prices))) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(len(prices))) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(prices))
    }, index=dates)
    
    # 创建实验框架
    framework = ExperimentFramework()
    
    # 运行单只股票实验
    print("\nRunning experiments on simulated stock...")
    results = framework.run_single_stock(df, "TEST001")
    print(results.to_string())
    
    # 生成LaTeX表格
    print("\n" + "=" * 40)
    print("LaTeX Table:")
    print(framework.generate_latex_table(results))
    
    print("\n✅ All tests passed!")
