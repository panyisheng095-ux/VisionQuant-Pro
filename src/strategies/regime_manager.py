"""
Regime管理器
Regime Manager

实时识别市场regime，并提供权重配置

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import yaml
import os

from src.factor_analysis.regime_detector import RegimeDetector, MarketRegime
from src.strategies.dynamic_weighting import DynamicWeightManager


class RegimeManager:
    """
    Regime管理器
    
    功能：
    1. 实时识别市场regime
    2. 加载权重配置
    3. 计算动态权重
    4. 提供权重建议
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        data_loader=None
    ):
        """
        初始化Regime管理器
        
        Args:
            config_path: 权重配置文件路径
            data_loader: 数据加载器（用于获取市场数据）
        """
        # 默认配置路径
        if config_path is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
            config_path = os.path.join(PROJECT_ROOT, "config", "factor_weights.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Regime检测器
        self.regime_detector = RegimeDetector()
        
        # 动态权重管理器
        self.weight_manager = DynamicWeightManager()
        
        # 数据加载器
        self.data_loader = data_loader
        
        # 当前regime缓存
        self._current_regime = None
        self._current_regime_date = None

    @staticmethod
    def _to_float(val, default: float = 0.0) -> float:
        try:
            if val is None:
                return float(default)
            if isinstance(val, str):
                val = val.strip()
                if val == "":
                    return float(default)
            return float(val)
        except Exception:
            return float(default)

    @staticmethod
    def _normalize_returns(returns):
        if returns is None:
            return None
        try:
            if isinstance(returns, pd.DataFrame):
                if returns.shape[1] == 0:
                    return None
                returns = returns.iloc[:, 0]
            elif isinstance(returns, (list, tuple, np.ndarray)):
                returns = pd.Series(returns)
            series = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                return None
            return series.astype(float)
        except Exception:
            return None
    
    def _load_config(self) -> Dict:
        """加载权重配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                return config
            except Exception as e:
                print(f"⚠️ 加载配置文件失败: {e}，使用默认配置")
        else:
            print(f"⚠️ 配置文件不存在: {self.config_path}，使用默认配置")
        
        # 默认配置
        return {
            'regime_weights': {
                'bull_market': {'kline_factor': 0.60, 'fundamental': 0.20, 'technical': 0.20},
                'bear_market': {'kline_factor': 0.40, 'fundamental': 0.40, 'technical': 0.20},
                'oscillating': {'kline_factor': 0.50, 'fundamental': 0.30, 'technical': 0.20},
                'unknown': {'kline_factor': 0.40, 'fundamental': 0.35, 'technical': 0.25}
            },
            'default': {'kline_factor': 0.50, 'fundamental': 0.30, 'technical': 0.20}
        }
    
    def get_current_regime(
        self,
        returns: pd.Series = None,
        prices: pd.Series = None,
        index_code: str = '000001'  # 默认使用上证指数
    ) -> str:
        """
        获取当前市场regime
        
        Args:
            returns: 收益率序列（可选，如果不提供则从数据加载器获取）
            prices: 价格序列（可选）
            index_code: 指数代码（用于获取市场数据）
            
        Returns:
            Regime名称 ('bull_market', 'bear_market', 'oscillating', 'unknown')
        """
        # 如果缓存有效，直接返回
        today = datetime.now().date()
        if self._current_regime and self._current_regime_date == today:
            return self._current_regime
        
        # 获取市场数据
        if returns is None and self.data_loader:
            try:
                # 获取指数数据
                index_df = self.data_loader.get_stock_data(index_code)
                if not index_df.empty and 'Close' in index_df.columns:
                    prices = index_df['Close']
                    returns = prices.pct_change().dropna()
            except Exception as e:
                print(f"⚠️ 获取市场数据失败: {e}")
        
        returns = self._normalize_returns(returns)
        if returns is None or len(returns) < 60:
            self._current_regime = 'unknown'
            self._current_regime_date = today
            return self._current_regime
        
        # 识别regime
        regimes = self.regime_detector.detect_regime(returns, prices)
        
        if len(regimes) > 0:
            current_regime = regimes.iloc[-1]
            if isinstance(current_regime, MarketRegime):
                self._current_regime = current_regime.value
            else:
                self._current_regime = str(current_regime)
        else:
            self._current_regime = 'unknown'
        
        self._current_regime_date = today
        
        return self._current_regime
    
    def get_regime_weights(
        self,
        current_regime: str = None,
        factor_ics: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        获取当前regime下的因子权重
        
        Args:
            current_regime: 当前regime（如果为None，则自动识别）
            factor_ics: 各因子的IC值 {'kline_factor': 0.05, 'fundamental': 0.03, ...}
            
        Returns:
            因子权重字典
        """
        # 获取当前regime
        if current_regime is None:
            current_regime = self.get_current_regime()
        
        # 获取基础权重（只保留有效因子字段）
        regime_weights = self.config.get('regime_weights', {})
        raw_base = regime_weights.get(current_regime, self.config.get('default', {})) or {}
        allowed = {"kline_factor", "fundamental", "technical"}
        base_weights = {k: self._to_float(v, 0.0) for k, v in raw_base.items() if k in allowed}
        if not base_weights:
            fallback = self.config.get('default', {}) or {}
            base_weights = {k: self._to_float(v, 0.0) for k, v in fallback.items() if k in allowed}
        
        # 如果提供了IC值，进行IC调整
        if factor_ics and self.config.get('ic_adjustment', {}).get('enabled', True):
            adjusted_weights = self._adjust_weights_by_ic(base_weights, factor_ics)
        else:
            adjusted_weights = base_weights.copy()
        
        # 归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _adjust_weights_by_ic(
        self,
        base_weights: Dict[str, float],
        factor_ics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        根据IC调整权重
        
        Args:
            base_weights: 基础权重
            factor_ics: 因子IC值
            
        Returns:
            调整后的权重
        """
        ic_config = self.config.get('ic_adjustment', {})
        adjustment_factor = ic_config.get('adjustment_factor', 0.5)
        max_adjustment = ic_config.get('max_adjustment', 0.3)
        
        adjusted_weights = {}
        
        for factor, base_weight in base_weights.items():
            base_weight = self._to_float(base_weight, 0.0)
            ic = self._to_float(factor_ics.get(factor, 0.0), 0.0)
            
            # IC调整：IC越高，权重越高
            adjustment = ic * adjustment_factor * 100  # 转换为百分比
            adjustment = max(-max_adjustment, min(max_adjustment, adjustment))
            
            adjusted_weight = base_weight * (1 + adjustment)
            adjusted_weights[factor] = adjusted_weight
        
        return adjusted_weights
    
    def calculate_dynamic_weights(
        self,
        factor_values: pd.Series = None,
        returns: pd.Series = None,
        factor_exposures: pd.DataFrame = None
    ) -> Dict:
        """
        计算动态权重（综合考虑regime和因子失效）
        
        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_exposures: 因子暴露度
            
        Returns:
            动态权重结果
        """
        # 1. 获取regime权重（基础）
        returns = self._normalize_returns(returns)
        current_regime = self.get_current_regime(returns)
        base_weights = self.get_regime_weights(current_regime)

        # 1.1 趋势/波动/回撤/尾部风险多因子动态调整（更统计化）
        trend_60 = None
        vol_60 = None
        drawdown_120 = None
        skew_60 = None
        kurt_60 = None
        trend_score = 0.5
        vol_score = 0.0
        dd_score = 0.0
        tail_score = 0.0
        skew_penalty = 0.0
        r = returns if returns is not None else None
        if r is not None and len(r) >= 20:
            try:
                if len(r) >= 60:
                    trend_60 = float(r.rolling(60).mean().iloc[-1] * 252)
                    vol_60 = float(r.rolling(60).std().iloc[-1] * np.sqrt(252))
                    trend_score = float(1 / (1 + np.exp(-(trend_60 / (vol_60 + 1e-8)))))
                    vol_score = min(max((vol_60 - 0.20) / 0.20, 0.0), 1.0)
                    skew_60 = float(r.iloc[-60:].skew())
                    kurt_60 = float(r.iloc[-60:].kurt())
                    skew_penalty = max(0.0, -skew_60)
                    tail_score = min(max((kurt_60 + 1.0) / 6.0, 0.0), 1.0)
                if len(r) >= 120:
                    window = r.iloc[-120:]
                    cum = (1 + window).cumprod()
                    peak = cum.cummax()
                    drawdown_120 = float((cum / peak - 1.0).min())
                    dd_score = min(max(-drawdown_120 / 0.25, 0.0), 1.0)
            except Exception:
                pass

        v = self._to_float(base_weights.get('kline_factor', 0.5), 0.5)
        f = self._to_float(base_weights.get('fundamental', 0.3), 0.3)
        q = self._to_float(base_weights.get('technical', 0.2), 0.2)

        # 趋势越强 -> 视觉/技术上调；波动/回撤/尾部风险越高 -> 基本面上调
        v_adj = v * (0.75 + 0.60 * trend_score) * (1 - 0.35 * vol_score) * (1 - 0.25 * dd_score)
        q_adj = q * (0.80 + 0.45 * trend_score) * (1 - 0.40 * vol_score) * (1 - 0.20 * dd_score)
        f_adj = f * (1.05 + 0.35 * vol_score + 0.30 * dd_score + 0.15 * tail_score) * (1 - 0.20 * trend_score) * (1 + 0.10 * skew_penalty)

        v_adj = max(v_adj, 1e-6)
        q_adj = max(q_adj, 1e-6)
        f_adj = max(f_adj, 1e-6)

        regime_weights = {'kline_factor': v_adj, 'fundamental': f_adj, 'technical': q_adj}
        
        # 2. 如果提供了因子数据，进行失效调整
        if factor_values is not None and returns is not None:
            invalidation_result = self.weight_manager.invalidation_detector.detect_invalidation(
                factor_values, returns, factor_exposures
            )
            
            # 根据失效程度调整K线因子权重
            invalidation_score = invalidation_result['invalidation_score']
            kline_base_weight = regime_weights.get('kline_factor', 0.5)
            
            # 失效调整
            decay_penalty = self.config.get('invalidation_handling', {}).get('decay_penalty', 0.2)
            adjustment = 1 - invalidation_score * decay_penalty
            
            adjusted_kline_weight = kline_base_weight * adjustment
            
            # 限制范围
            min_weight = self.config.get('invalidation_handling', {}).get('min_weight', 0.1)
            max_weight = self.config.get('invalidation_handling', {}).get('max_weight', 0.8)
            adjusted_kline_weight = max(min_weight, min(max_weight, adjusted_kline_weight))
            
            # 重新分配权重
            other_factors_total = regime_weights.get('fundamental', 0.3) + regime_weights.get('technical', 0.2)
            if other_factors_total > 0:
                scale = (1 - adjusted_kline_weight) / other_factors_total
                regime_weights['kline_factor'] = adjusted_kline_weight
                regime_weights['fundamental'] = regime_weights.get('fundamental', 0.3) * scale
                regime_weights['technical'] = regime_weights.get('technical', 0.2) * scale
        
        # 归一化
        total_w = sum(regime_weights.values())
        if total_w > 0:
            regime_weights = {k: v / total_w for k, v in regime_weights.items()}

        return {
            'weights': regime_weights,
            'regime': current_regime,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'explain': {
                'base_weights': base_weights,
                'trend_60': None if trend_60 is None else round(trend_60, 4),
                'vol_60': None if vol_60 is None else round(vol_60, 4),
                'trend_score': round(trend_score, 4),
                'vol_score': round(vol_score, 4),
                'dd_score': round(dd_score, 4),
                'max_drawdown_120': None if drawdown_120 is None else round(drawdown_120, 4),
                'skew_60': None if skew_60 is None else round(skew_60, 4),
                'kurt_60': None if kurt_60 is None else round(kurt_60, 4),
                'tail_score': round(tail_score, 4),
                'skew_penalty': round(skew_penalty, 4),
                'formula': "w_V=base_V*(0.75+0.60*trend_score)*(1-0.35*vol_score)*(1-0.25*dd_score); "
                           "w_Q=base_Q*(0.80+0.45*trend_score)*(1-0.40*vol_score)*(1-0.20*dd_score); "
                           "w_F=base_F*(1.05+0.35*vol_score+0.30*dd_score+0.15*tail_score)"
                           "*(1-0.20*trend_score)*(1+0.10*skew_penalty)"
            }
        }


if __name__ == "__main__":
    print("=== Regime管理器测试 ===")
    
    manager = RegimeManager()
    
    # 测试获取权重
    weights = manager.get_regime_weights('bull_market')
    print(f"\n牛市权重配置:")
    for factor, weight in weights.items():
        print(f"  {factor}: {weight:.2%}")
    
    # 测试IC调整
    factor_ics = {'kline_factor': 0.05, 'fundamental': 0.03, 'technical': 0.02}
    adjusted = manager.get_regime_weights('bull_market', factor_ics)
    print(f"\nIC调整后的权重:")
    for factor, weight in adjusted.items():
        print(f"  {factor}: {weight:.2%}")
