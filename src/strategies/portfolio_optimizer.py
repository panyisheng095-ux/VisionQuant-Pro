import numpy as np
import pandas as pd
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
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
                                     cvar_limit: float = 0.05, cvar_alpha: float = 0.05,
                                     max_drawdown_limit: float = 0.15):
        """
        三层分级组合优化（新逻辑）
        """
        # 1. 分层筛选（放宽规则：保证中小样本也能输出可用组合）
        optimizer_name = "Black-Litterman (Robust)"
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
            max_positions, risk_aversion, cvar_limit, cvar_alpha, max_drawdown_limit
        ) if core_stocks else {}

        enhanced_weights = self._optimize_single_tier(
            enhanced_stocks, loader, 0.05, 0.20,
            max_positions, risk_aversion, cvar_limit, cvar_alpha, max_drawdown_limit
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
                             max_positions, risk_aversion, cvar_limit, cvar_alpha, max_drawdown_limit=0.15):
        """优化单层组合"""
        if len(stocks) == 0:
            return {}
        
        # 按评分排序，取前N个
        sorted_stocks = sorted(
            stocks.items(),
            key=lambda x: x[1].get('score', 0),
            reverse=True
        )[:max_positions]
        
        symbols = [s[0] for s in sorted_stocks]
        n = len(symbols)
        
        # 计算期望收益和协方差矩阵
        expected_returns = self._calculate_expected_returns(sorted_stocks)
        view_confidences = self._calculate_view_confidences_advanced(sorted_stocks, loader) # 升级版置信度
        cov_matrix = self._calculate_covariance_matrix(symbols, loader)
        
        if cov_matrix is None:
            # 降级为简单加权
            return self._simple_weight_allocation(sorted_stocks, min_weight, max_weight)
        
        # Black-Litterman 优化
        try:
            bl_returns = self._black_litterman_expected_returns(
                expected_returns, cov_matrix, view_confidences=view_confidences
            )
            weights = self._markowitz_optimize(
                bl_returns, cov_matrix,
                min_weight, max_weight, risk_aversion,
                cvar_limit=cvar_limit, cvar_alpha=cvar_alpha,
                max_drawdown_limit=max_drawdown_limit
            )
        except Exception:
            # 回退到 Markowitz
            try:
                weights = self._markowitz_optimize(
                    expected_returns, cov_matrix,
                    min_weight, max_weight, risk_aversion,
                    cvar_limit=cvar_limit, cvar_alpha=cvar_alpha,
                    max_drawdown_limit=max_drawdown_limit
                )
            except Exception:
                return self._simple_weight_allocation(sorted_stocks, min_weight, max_weight)
        
        result = {}
        for i, sym in enumerate(symbols):
            if weights[i] > 0.001:  # 忽略极小权重
                result[sym] = weights[i]
                
        return result

    def _markowitz_optimize(self, expected_returns, cov_matrix, 
                           min_weight, max_weight, risk_aversion,
                           cvar_limit: float = 0.05, cvar_alpha: float = 0.05,
                           max_drawdown_limit: float = 0.15):
        """马科维茨优化求解（含高级风控约束）"""
        n = len(expected_returns)
        
        # 目标函数：最大化效用 (E[R] - lambda * Risk)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.dot(weights, np.dot(cov_matrix, weights))
            # MaxDD 惩罚项 (近似：Volatility * 2)
            mdd_penalty = max(0, np.sqrt(portfolio_risk) * 2 - max_drawdown_limit) * 10.0
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_risk - mdd_penalty)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 权重和为1
            {'type': 'ineq', 'fun': lambda w: cvar_limit - self._portfolio_cvar(w, expected_returns, cov_matrix, cvar_alpha)}
        ]
        
        # 边界条件
        bounds = tuple((min_weight, max_weight) for _ in range(n))
        
        # 初始猜测
        init_guess = np.array([1.0/n] * n)
        
        result = minimize(objective, init_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if not result.success:
            raise RuntimeError("Optimization failed")
            
        return result.x

    def _black_litterman_expected_returns(self, expected_returns, cov_matrix,
                                          view_confidences=None,
                                          tau: float = 0.05, delta: float = 2.5):
        """
        Black-Litterman 模型计算后验期望收益
        
        Args:
            expected_returns: 投资者观点（AI预测收益）
            cov_matrix: 协方差矩阵
            view_confidences: 观点置信度 (0~1)
            tau: 缩放系数
            delta: 风险厌恶系数
        """
        n = len(expected_returns)
        
        # 隐含均衡收益 (Implied Equilibrium Returns)
        # 简化：假设市场均衡收益等于无风险利率 + 风险溢价
        # Pi = delta * Sigma * w_mkt
        # 这里简化为所有股票均值为 0.05
        pi = np.full(n, 0.05 / 252) 
        
        # 观点矩阵
        P = np.eye(n)
        Q = expected_returns.reshape(-1, 1)

        # Omega（观点不确定性）：置信度越高，不确定性越低
        base_omega = np.diag(np.diag(P.dot(tau * cov_matrix).dot(P.T)))
        if view_confidences is None:
            view_confidences = np.ones(n)
        
        # 确保置信度在合理范围
        view_confidences = np.clip(np.array(view_confidences), 0.1, 0.99)
        # Omega = P * Sigma * P' * ( (1/conf) - 1 )
        uncertainty_multiplier = (1.0 / view_confidences) - 1.0
        omega = base_omega * uncertainty_multiplier.reshape(-1, 1)
        
        # BL 公式
        # E[R] = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1 * [(tau*Sigma)^-1 Pi + P' Omega^-1 Q]
        tau_cov_inv = np.linalg.inv(tau * cov_matrix)
        omega_inv = np.linalg.inv(omega)
        
        M_inverse = np.linalg.inv(tau_cov_inv + P.T.dot(omega_inv).dot(P))
        term1 = tau_cov_inv.dot(pi.reshape(-1, 1))
        term2 = P.T.dot(omega_inv).dot(Q)
        
        mu_bl = M_inverse.dot(term1 + term2)
        return mu_bl.flatten()

    def _calculate_expected_returns(self, sorted_stocks):
        """从分析结果提取期望收益"""
        returns = []
        for _, data in sorted_stocks:
            # 优先使用已有的预期收益字段
            er = data.get('expected_return', 0)
            # 转换为日收益率 (假设输入是年化或总收益)
            # 这里假设 data['expected_return'] 是百分比，如 15.5
            daily_er = (er / 100) / 20  # 假设是20日预期收益
            returns.append(daily_er)
        return np.array(returns)

    def _calculate_view_confidences(self, sorted_stocks):
        """(Deprecated) 简单置信度"""
        return self._calculate_view_confidences_advanced(sorted_stocks, None)

    def _calculate_view_confidences_advanced(self, sorted_stocks, loader):
        """
        基于分布特征计算观点置信度 (13D)
        
        Confidence = f(Score, WinRate, DistributionStd)
        分布越紧密（方差小），置信度越高。
        """
        confs = []
        for sym, data in sorted_stocks:
            base_conf = 0.5
            
            # 1. 基础分
            score = float(data.get("score", 0)) / 10.0
            win_rate = float(data.get("win_rate", 50)) / 100.0
            
            # 2. 分布特征 (如果存在)
            dist_std = 0.05 # 默认假设5%波动
            matches = data.get("matches", [])
            if matches and loader:
                # 尝试计算 Top10 收益的标准差
                try:
                    # 这里简化：假设 matches 里已有收益率或者重新获取太慢
                    # 如果 data 里有 cvar 或 volatility 字段最好
                    # 也可以用 "similarity" 的一致性来代表置信度
                    scores = [m.get('score', 0) for m in matches]
                    if scores:
                        score_std = np.std(scores)
                        # 相似度方差越小，置信度越高
                        dist_modifier = max(0, 0.1 - score_std) * 2 
                        base_conf += dist_modifier
                except:
                    pass
            
            # 3. 综合计算
            # 胜率越高 -> 置信度高
            # 评分越高 -> 置信度高
            conf = 0.4 * score + 0.4 * win_rate + 0.2 * base_conf
            confs.append(min(max(conf, 0.1), 0.95))
            
        return confs

    def _calculate_covariance_matrix(self, symbols, loader):
        """计算协方差矩阵（并行拉取，增强鲁棒性）"""
        if not symbols:
            return None

        prices = pd.DataFrame()

        def _fetch_prices(sym: str):
            try:
                df = loader.get_stock_data(sym)
                if df is None or df.empty or "Close" not in df.columns:
                    return sym, None
                series = df["Close"].astype(float)
                series.name = sym
                return sym, series
            except Exception:
                return sym, None

        max_workers = min(max(1, cpu_count() * 2), 8, len(symbols))
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_fetch_prices, s): s for s in symbols}
                for future in as_completed(futures):
                    sym, series = future.result()
                    if series is not None:
                        prices[sym] = series
        else:
            for sym in symbols:
                _, series = _fetch_prices(sym)
                if series is not None:
                    prices[sym] = series

        if prices.empty:
            return None

        # 如果缺失过多，避免维度不一致引发优化失败
        if prices.shape[1] != len(symbols):
            return None

        # 计算日收益率
        returns = prices.pct_change().dropna(how="all")
        returns = returns.dropna(axis=1, how="all")
        if returns.empty or len(returns) < 30:
            return None

        cov_matrix = returns.cov().values
        return cov_matrix

    def _simple_weight_allocation(self, sorted_stocks, min_weight, max_weight):
        """简单权重分配（按分数比例）"""
        total_score = sum([x[1].get('score', 0) for x in sorted_stocks])
        weights = {}
        
        if total_score == 0:
            count = len(sorted_stocks)
            for sym, _ in sorted_stocks:
                weights[sym] = 1.0 / count
        else:
            for sym, data in sorted_stocks:
                score = data.get('score', 0)
                raw_weight = score / total_score
                # 简单截断
                w = max(min_weight, min(raw_weight, max_weight))
                weights[sym] = w
        
        # 归一化
        total_w = sum(weights.values())
        if total_w > 0:
            for k in weights:
                weights[k] /= total_w
                
        return weights

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

    def calculate_portfolio_metrics(self, weights, analysis_results, loader):
        """计算组合指标"""
        if not weights:
            return {}
            
        symbols = list(weights.keys())
        
        # 期望收益 (日)
        expected_returns = []
        for s in symbols:
            data = analysis_results.get(s, {})
            er = data.get('expected_return', 0)
            daily_er = (er / 100) / 20 
            expected_returns.append(daily_er)
            
        cov_matrix = self._calculate_covariance_matrix(symbols, loader)
        
        if cov_matrix is None:
            return {
                "expected_return": 0,
                "risk": 0,
                "sharpe_ratio": 0
            }
            
        # 组合风险 (日波动率)
        w_vec = np.array([weights[s] for s in symbols])
        portfolio_risk = np.sqrt(np.dot(w_vec, np.dot(cov_matrix, w_vec)))
        
        # 组合期望收益
        portfolio_return = sum(weights[s] * er for s, er in zip(symbols, expected_returns))
        
        # 夏普比率
        if portfolio_risk > 0:
            sharpe_ratio = portfolio_return / portfolio_risk
        else:
            sharpe_ratio = 0

        # CVaR（正态近似）
        cvar = None
        if cov_matrix is not None:
            cvar = self._portfolio_cvar(w_vec, np.array(expected_returns), cov_matrix)

        # 风险贡献（Budget）
        risk_budget = {}
        if cov_matrix is not None:
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
