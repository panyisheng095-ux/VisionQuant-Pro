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
                                     max_positions=10, risk_aversion=1.0,
                                     cvar_limit: float = 0.05, cvar_alpha: float = 0.05):
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
        # 1. 分层筛选（放宽规则：保证中小样本也能输出可用组合）
        optimizer_name = "Black-Litterman"
        ranked = sorted(
            analysis_results.items(),
            key=lambda x: x[1].get('score', 0),
            reverse=True
        )
        if not ranked:
            return {
                'core': {},
                'enhanced': {},
                'tier_info': {
                    'strategy': 'empty',
                    'optimizer': optimizer_name,
                    'core_count': 0,
                    'enhanced_count': 0,
                    'description': '无有效股票数据'
                }
            }

        n_total = len(ranked)
        target_total = min(max_positions, max(3, int(np.ceil(n_total * 0.6)) + 1))
        target_total = min(target_total, n_total)
        target_core = min(max(2, int(np.ceil(target_total * 0.6))), target_total)

        buy = [sym for sym, d in ranked if d.get('action') == 'BUY']
        wait = [sym for sym, d in ranked if d.get('action') == 'WAIT']
        sell = [sym for sym, d in ranked if d.get('action') == 'SELL']

        core_syms = []
        for sym in buy:
            if len(core_syms) < target_core:
                core_syms.append(sym)
        for sym in wait:
            if len(core_syms) < target_core:
                core_syms.append(sym)
        for sym in sell:
            if len(core_syms) < target_core:
                core_syms.append(sym)

        core_set = set(core_syms)
        remaining = [sym for sym, _ in ranked if sym not in core_set]
        enhanced_syms = remaining[:max(0, target_total - len(core_syms))]

        core_stocks = {s: analysis_results[s] for s in core_syms}
        enhanced_stocks = {s: analysis_results[s] for s in enhanced_syms}

        # 2. 权重优化（核心/增强同时存在时 70/30；否则单层 100%）
        core_weights = self._optimize_single_tier(
            core_stocks, loader, min_weight, max_weight,
            max_positions, risk_aversion, cvar_limit, cvar_alpha
        ) if core_stocks else {}

        enhanced_weights = self._optimize_single_tier(
            enhanced_stocks, loader, 0.05, 0.20,
            max_positions, risk_aversion, cvar_limit, cvar_alpha
        ) if enhanced_stocks else {}

        if core_weights and enhanced_weights:
            mixed_core = {k: v * 0.7 for k, v in core_weights.items()}
            mixed_enhanced = {k: v * 0.3 for k, v in enhanced_weights.items()}
            strategy = "mixed_relaxed"
            desc = "规则放宽：核心优先 BUY，其次 WAIT；增强补齐剩余，确保组合可用"
        elif core_weights:
            mixed_core = core_weights
            mixed_enhanced = {}
            strategy = "core_only_relaxed"
            desc = "规则放宽：用高分股票构建核心组合"
        else:
            mixed_core = {}
            mixed_enhanced = enhanced_weights
            strategy = "enhanced_only_relaxed"
            desc = "规则放宽：以高分股票构建备选组合"

        return {
            'core': mixed_core,
            'enhanced': mixed_enhanced,
            'tier_info': {
                'strategy': strategy,
                'optimizer': optimizer_name,
                'core_count': len(mixed_core),
                'enhanced_count': len(mixed_enhanced),
                'description': desc
            }
        }
    
    def _optimize_single_tier(self, stocks, loader, min_weight, max_weight, 
                             max_positions, risk_aversion, cvar_limit, cvar_alpha):
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
        view_confidences = self._calculate_view_confidences(sorted_stocks)
        cov_matrix = self._calculate_covariance_matrix(symbols, loader)
        
        if cov_matrix is None or len(cov_matrix) == 0:
            return self._simple_weight_allocation(sorted_stocks, min_weight, max_weight)
        
        # Black-Litterman 优化（Q10：不选C/D）
        try:
            bl_returns = self._black_litterman_expected_returns(
                expected_returns, cov_matrix, view_confidences=view_confidences
            )
            weights = self._markowitz_optimize(
                bl_returns, cov_matrix,
                min_weight, max_weight, risk_aversion,
                cvar_limit=cvar_limit, cvar_alpha=cvar_alpha
            )
        except Exception:
            # 回退到 Markowitz
            try:
                weights = self._markowitz_optimize(
                    expected_returns, cov_matrix,
                    min_weight, max_weight, risk_aversion,
                    cvar_limit=cvar_limit, cvar_alpha=cvar_alpha
                )
            except Exception:
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
                           min_weight, max_weight, risk_aversion,
                           cvar_limit: float = 0.05, cvar_alpha: float = 0.05):
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
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 权重和为1
            {'type': 'ineq', 'fun': lambda w: cvar_limit - self._portfolio_cvar(w, expected_returns, cov_matrix, cvar_alpha)}
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

    def _black_litterman_expected_returns(self, expected_returns, cov_matrix,
                                          view_confidences=None,
                                          tau: float = 0.05, delta: float = 2.5):
        """
        计算 Black-Litterman 融合后的期望收益
        - 市场均衡收益 pi = delta * Sigma * w_mkt
        - 观点 Q = expected_returns（来自视觉因子）
        """
        n = len(expected_returns)
        if n == 0:
            return expected_returns

        # 市场权重（等权）
        w_mkt = np.array([1.0 / n] * n)

        # 均衡收益
        pi = delta * cov_matrix.dot(w_mkt)

        # 观点矩阵
        P = np.eye(n)
        Q = expected_returns.reshape(-1, 1)

        # Omega（观点不确定性）：置信度越高，不确定性越低
        base_omega = np.diag(np.diag(P.dot(tau * cov_matrix).dot(P.T)))
        if view_confidences is None:
            view_confidences = np.ones(n)
        view_confidences = np.clip(np.array(view_confidences), 0.2, 1.0)
        omega = base_omega / view_confidences.reshape(-1, 1)

        # BL 公式
        inv_tau_sigma = np.linalg.inv(tau * cov_matrix)
        inv_omega = np.linalg.inv(omega)
        middle = np.linalg.inv(inv_tau_sigma + P.T.dot(inv_omega).dot(P))
        mu_bl = middle.dot(inv_tau_sigma.dot(pi.reshape(-1, 1)) + P.T.dot(inv_omega).dot(Q))

        return mu_bl.flatten()

    def _calculate_view_confidences(self, sorted_stocks):
        """根据因子置信度生成观点权重（0.2-1.0）"""
        confs = []
        for _, data in sorted_stocks:
            try:
                raw_conf = float(data.get("confidence", 0))
            except Exception:
                raw_conf = 0.0
            score = float(data.get("score", 0)) / 10.0
            win = float(data.get("win_rate", 50)) / 100.0
            conf = raw_conf / 100.0 if raw_conf > 0 else 0.5 * score + 0.5 * win
            conf = min(max(conf, 0.2), 1.0)
            confs.append(conf)
        return confs

    def _portfolio_cvar(self, weights, expected_returns, cov_matrix, alpha: float = 0.05):
        """正态近似CVaR（损失为正）"""
        mu = float(np.dot(weights, expected_returns))
        sigma = float(np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))))
        if sigma <= 1e-8:
            return 0.0
        # 正态分布CVaR系数（固定近似，避免随机性）
        z = -1.645 if alpha <= 0.05 else -1.282
        phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z * z)
        cvar = -(mu - sigma * (phi / max(alpha, 1e-6)))
        return max(cvar, 0.0)

    def propose_rebalance(self, current_weights, target_weights, max_turnover: float = 0.2):
        """根据换手上限生成再平衡建议"""
        if not current_weights:
            return target_weights, {"turnover": 0.0}
        all_syms = set(current_weights) | set(target_weights)
        diffs = {s: target_weights.get(s, 0) - current_weights.get(s, 0) for s in all_syms}
        turnover = sum(abs(v) for v in diffs.values())
        if turnover <= max_turnover:
            return target_weights, {"turnover": turnover}
        scale = max_turnover / turnover if turnover > 0 else 1.0
        adj = {s: current_weights.get(s, 0) + diffs[s] * scale for s in all_syms}
        # 归一化
        total = sum(max(v, 0) for v in adj.values())
        if total > 0:
            adj = {k: max(v, 0) / total for k, v in adj.items()}
        return adj, {"turnover": max_turnover}
    
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

        # CVaR（正态近似）
        cov_matrix = self._calculate_covariance_matrix(symbols, loader)
        cvar = None
        if cov_matrix is not None:
            w_vec = np.array([weights[s] for s in symbols])
            cvar = self._portfolio_cvar(w_vec, np.array(expected_returns), cov_matrix)

        # 风险贡献（Budget）
        risk_budget = {}
        if cov_matrix is not None:
            w_vec = np.array([weights[s] for s in symbols])
            port_vol = np.sqrt(np.dot(w_vec, np.dot(cov_matrix, w_vec)))
            if port_vol > 0:
                mrc = np.dot(cov_matrix, w_vec) / port_vol
                for i, s in enumerate(symbols):
                    risk_budget[s] = round(float(w_vec[i] * mrc[i]), 6)
        
        return {
            "expected_return": round(portfolio_return * 100, 2),
            "risk": round(portfolio_risk * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "cvar": round(float(cvar) * 100, 2) if cvar is not None else None,
            "risk_budget": risk_budget
        }
