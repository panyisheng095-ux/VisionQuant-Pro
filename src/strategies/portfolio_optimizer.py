import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """马科维茨均值-方差组合优化器"""
    
    def __init__(self, risk_free_rate=0.03):
        """
        Args:
            risk_free_rate: 无风险利率（年化，默认3%）
        """
        self.risk_free_rate = risk_free_rate / 252  # 转换为日利率
    
    def optimize_multi_tier_portfolio(self, analysis_results, loader, 
                                     min_weight=0.05, max_weight=0.25,
                                     max_positions=10, risk_aversion=1.0):
        """
        三层分级组合优化（新逻辑）
        
        返回结构：
        {
            'core': {symbol: weight},  # 核心推荐组合
            'enhanced': {symbol: weight},  # 备选增强组合
            'tier_info': {
                'strategy': 'xxx',  # 策略类型
                'core_count': N,
                'enhanced_count': M
            }
        }
        """
        # 1. 分层筛选
        core_stocks = {
            k: v for k, v in analysis_results.items()
            if v.get('action') == 'BUY' and v.get('score', 0) >= 7
        }
        
        enhanced_stocks = {
            k: v for k, v in analysis_results.items()
            if v.get('score', 0) >= 6 and v.get('action') != 'SELL' 
            and k not in core_stocks
        }
        
        core_count = len(core_stocks)
        
        # 2. 根据核心股票数量选择策略
        if core_count >= 5:
            # 策略A：全部投资核心推荐
            weights = self._optimize_single_tier(
                core_stocks, loader, min_weight, max_weight, 
                max_positions, risk_aversion
            )
            return {
                'core': weights,
                'enhanced': {},
                'tier_info': {
                    'strategy': 'core_only',
                    'core_count': len(weights),
                    'enhanced_count': 0,
                    'description': '核心推荐股票充足，全仓配置核心组合'
                }
            }
        
        elif core_count >= 2:
            # 策略B：70%核心 + 30%增强
            core_weights = self._optimize_single_tier(
                core_stocks, loader, 0.10, 0.25, 
                core_count, risk_aversion
            )
            enhanced_weights = self._optimize_single_tier(
                enhanced_stocks, loader, 0.05, 0.15, 
                min(5, len(enhanced_stocks)), risk_aversion
            )
            
            # 混合权重
            mixed_core = {k: v * 0.7 for k, v in core_weights.items()}
            mixed_enhanced = {k: v * 0.3 for k, v in enhanced_weights.items()}
            
            return {
                'core': mixed_core,
                'enhanced': mixed_enhanced,
                'tier_info': {
                    'strategy': 'mixed',
                    'core_count': len(mixed_core),
                    'enhanced_count': len(mixed_enhanced),
                    'description': '核心推荐较少，70%核心+30%增强配置'
                }
            }
        
        else:
            # 策略C：只展示备选增强（带风险提示）
            enhanced_weights = self._optimize_single_tier(
                enhanced_stocks, loader, 0.05, 0.20, 
                min(max_positions, len(enhanced_stocks)), risk_aversion
            )
            
            return {
                'core': {},
                'enhanced': enhanced_weights,
                'tier_info': {
                    'strategy': 'enhanced_only',
                    'core_count': 0,
                    'enhanced_count': len(enhanced_weights),
                    'description': '暂无高确定性推荐，仅展示备选组合，建议谨慎配置'
                }
            }
    
    def _optimize_single_tier(self, stocks, loader, min_weight, max_weight, 
                             max_positions, risk_aversion):
        """优化单层组合"""
        if len(stocks) == 0:
            return {}
        
        # 按评分排序，取Top N
        sorted_stocks = sorted(
            stocks.items(),
            key=lambda x: x[1].get('score', 0),
            reverse=True
        )[:max_positions]
        
        if len(sorted_stocks) == 0:
            return {}
        
        symbols = [s[0] for s in sorted_stocks]
        
        # 计算期望收益和协方差矩阵
        expected_returns = self._calculate_expected_returns(sorted_stocks)
        cov_matrix = self._calculate_covariance_matrix(symbols, loader)
        
        if cov_matrix is None or len(cov_matrix) == 0:
            return self._simple_weight_allocation(sorted_stocks, min_weight, max_weight)
        
        # 马科维茨优化
        try:
            weights = self._markowitz_optimize(
                expected_returns, cov_matrix, 
                min_weight, max_weight, risk_aversion
            )
        except:
            return self._simple_weight_allocation(sorted_stocks, min_weight, max_weight)
        
        # 构建结果字典
        result = {}
        for i, (symbol, _) in enumerate(sorted_stocks):
            if i < len(weights):
                result[symbol] = round(weights[i], 4)
        
        return result
    
    def optimize_weights(self, analysis_results, loader, 
                        min_weight=0.05, max_weight=0.20,
                        max_positions=10, risk_aversion=1.0):
        """
        基于马科维茨模型优化组合权重（保持兼容性）
        
        Args:
            analysis_results: 批量分析结果 {symbol: {score, win_rate, ...}}
            loader: DataLoader实例，用于获取历史数据计算协方差
            min_weight: 最小仓位（5%）
            max_weight: 最大仓位（20%）
            max_positions: 最大持仓数量
            risk_aversion: 风险厌恶系数（越大越保守）
        
        Returns:
            Dict: {symbol: weight}
        """
        # 调用新的多层优化方法
        multi_tier = self.optimize_multi_tier_portfolio(
            analysis_results, loader, min_weight, max_weight,
            max_positions, risk_aversion
        )
        
        # 合并所有权重返回（兼容旧接口）
        all_weights = {}
        all_weights.update(multi_tier['core'])
        all_weights.update(multi_tier['enhanced'])
        
        return all_weights
    
    def _calculate_expected_returns(self, sorted_stocks):
        """计算期望收益向量"""
        returns = []
        for symbol, data in sorted_stocks:
            # 基于视觉胜率和预期收益
            win_rate = data.get('win_rate', 50) / 100
            expected_ret = data.get('expected_return', 0) / 100
            
            # 简化：期望收益 = 胜率 * 预期收益
            er = win_rate * expected_ret + (1 - win_rate) * (-expected_ret * 0.5)
            returns.append(er)
        
        return np.array(returns)
    
    def _calculate_covariance_matrix(self, symbols, loader, lookback_days=60):
        """计算协方差矩阵（基于历史收益率）"""
        returns_data = []
        valid_symbols = []
        
        for symbol in symbols:
            try:
                df = loader.get_stock_data(symbol)
                if len(df) < lookback_days:
                    continue
                
                # 计算日收益率
                df = df.tail(lookback_days)
                returns = df['Close'].pct_change().dropna()
                
                if len(returns) >= 30:  # 至少需要30天数据
                    returns_data.append(returns.values)
                    valid_symbols.append(symbol)
            except:
                continue
        
        if len(returns_data) < 2:
            return None
        
        # 对齐长度（取最短的）
        min_len = min(len(r) for r in returns_data)
        returns_data = [r[-min_len:] for r in returns_data]
        
        # 计算协方差矩阵
        returns_matrix = np.array(returns_data)
        cov_matrix = np.cov(returns_matrix)
        
        # 如果矩阵不是正定的，添加小的正则项
        if not self._is_positive_definite(cov_matrix):
            cov_matrix += np.eye(len(cov_matrix)) * 1e-6
        
        # 更新symbols列表
        symbols[:] = valid_symbols[:len(cov_matrix)]
        
        return cov_matrix
    
    def _is_positive_definite(self, matrix):
        """检查矩阵是否正定"""
        try:
            np.linalg.cholesky(matrix)
            return True
        except:
            return False
    
    def _markowitz_optimize(self, expected_returns, cov_matrix, 
                           min_weight, max_weight, risk_aversion):
        """
        马科维茨均值-方差优化
        
        目标函数：maximize (w^T * μ - λ * w^T * Σ * w)
        其中：w是权重向量，μ是期望收益，Σ是协方差矩阵，λ是风险厌恶系数
        """
        n = len(expected_returns)
        
        # 目标函数：负的夏普比率（因为minimize）
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # 夏普比率（简化，假设无风险利率为0）
            if portfolio_risk > 0:
                sharpe = portfolio_return / portfolio_risk
            else:
                sharpe = 0
            
            # 风险惩罚项
            risk_penalty = risk_aversion * portfolio_risk
            
            return -(sharpe - risk_penalty)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # 权重和为1
        ]
        
        # 边界条件
        bounds = [(min_weight, max_weight) for _ in range(n)]
        
        # 初始猜测（等权重）
        x0 = np.array([1.0 / n] * n)
        
        # 优化
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            # 归一化（确保和为1）
            weights = weights / np.sum(weights)
            return weights
        else:
            # 优化失败，返回等权重
            return np.array([1.0 / n] * n)
    
    def _simple_weight_allocation(self, sorted_stocks, min_weight, max_weight):
        """简化权重分配（按评分比例）"""
        scores = [s[1].get('score', 0) for s in sorted_stocks]
        total_score = sum(scores)
        
        if total_score == 0:
            # 等权重
            n = len(sorted_stocks)
            return {s[0]: round(1.0/n, 4) for s in sorted_stocks}
        
        weights = {}
        for symbol, data in sorted_stocks:
            base_weight = data.get('score', 0) / total_score
            # 限制在[min_weight, max_weight]范围内
            weight = max(min_weight, min(max_weight, base_weight))
            weights[symbol] = round(weight, 4)
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v/total, 4) for k, v in weights.items()}
        
        return weights
    
    def calculate_portfolio_metrics(self, weights, analysis_results, loader):
        """计算组合指标"""
        if not weights:
            return {}
        
        symbols = list(weights.keys())
        expected_returns = []
        
        for symbol in symbols:
            data = analysis_results.get(symbol, {})
            er = data.get('expected_return', 0) / 100
            expected_returns.append(er)
        
        # 组合期望收益
        portfolio_return = sum(weights[s] * er for s, er in zip(symbols, expected_returns))
        
        # 计算组合风险（简化）
        try:
            cov_matrix = self._calculate_covariance_matrix(symbols, loader)
            if cov_matrix is not None:
                w_vec = np.array([weights[s] for s in symbols])
                portfolio_risk = np.sqrt(np.dot(w_vec, np.dot(cov_matrix, w_vec)))
            else:
                portfolio_risk = 0.02  # 默认2%
        except:
            portfolio_risk = 0.02
        
        # 夏普比率
        if portfolio_risk > 0:
            sharpe_ratio = portfolio_return / portfolio_risk
        else:
            sharpe_ratio = 0
        
        return {
            "expected_return": round(portfolio_return * 100, 2),
            "risk": round(portfolio_risk * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }
