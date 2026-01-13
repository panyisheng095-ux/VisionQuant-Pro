"""
凯利公式仓位管理系统
Kelly Criterion Position Sizing System

凯利公式是由John Kelly Jr.在1956年提出的，用于确定最优投注比例的公式。
在量化交易中，它帮助我们确定：给定胜率和赔率，应该投入多少比例的资金。

核心公式:
f* = (p × b - q) / b

其中:
- f*: 最优仓位比例
- p: 胜率 (win probability)
- q: 亏损概率 = 1 - p
- b: 赔率 = 平均盈利 / 平均亏损

特点:
1. 数学上最优：长期来看能最大化资本增长率
2. 风险控制：自动避免破产风险
3. 需要准确的胜率和赔率估计

实际应用中的调整:
- 使用半凯利或1/3凯利降低波动
- 设置仓位上下限
- 与评分系统联动
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class KellyConfig:
    """凯利公式配置"""
    max_position: float = 0.25      # 最大单票仓位 25%
    min_position: float = 0.03      # 最小仓位 3%
    kelly_fraction: float = 0.5     # 半凯利（更保守）
    min_win_rate: float = 0.4       # 最低允许胜率
    min_win_loss_ratio: float = 0.8 # 最低允许盈亏比


class KellyPositionCalculator:
    """
    凯利公式仓位计算器
    
    用法:
    ```python
    calculator = KellyPositionCalculator()
    position = calculator.calculate(win_rate=0.6, win_loss_ratio=1.5)
    ```
    """
    
    def __init__(self, config: KellyConfig = None):
        """
        初始化凯利计算器
        
        Args:
            config: 配置参数，如果为None则使用默认配置
        """
        self.config = config or KellyConfig()
        
    def calculate(
        self,
        win_rate: float,
        win_loss_ratio: float,
        score: float = None
    ) -> Dict:
        """
        计算最优仓位
        
        Args:
            win_rate: 胜率 (0-1)
            win_loss_ratio: 盈亏比 (平均盈利/平均亏损)
            score: V+F+Q评分 (0-10)，用于调整仓位
            
        Returns:
            仓位建议字典
        """
        # 参数验证
        win_rate = max(0.01, min(0.99, win_rate))
        win_loss_ratio = max(0.1, win_loss_ratio)
        
        # 凯利公式计算
        p = win_rate
        q = 1 - p
        b = win_loss_ratio
        
        # f* = (p × b - q) / b
        kelly_raw = (p * b - q) / b
        
        # 应用半凯利（更保守）
        kelly_adjusted = kelly_raw * self.config.kelly_fraction
        
        # 限制仓位范围
        position = max(0, min(self.config.max_position, kelly_adjusted))
        
        # 如果胜率或盈亏比太低，直接返回0仓位
        if win_rate < self.config.min_win_rate or win_loss_ratio < self.config.min_win_loss_ratio:
            position = 0
            risk_level = "HIGH_RISK"
        else:
            risk_level = self._assess_risk(win_rate, win_loss_ratio, position)
        
        # 根据评分调整
        score_adjustment = 1.0
        if score is not None:
            score_adjustment = self._score_adjustment(score)
            position = position * score_adjustment
            position = max(0, min(self.config.max_position, position))
        
        # 应用最小仓位（如果有仓位的话）
        if position > 0 and position < self.config.min_position:
            position = self.config.min_position
        
        return {
            'position': round(position, 4),
            'position_pct': round(position * 100, 2),
            'kelly_raw': round(kelly_raw, 4),
            'kelly_adjusted': round(kelly_adjusted, 4),
            'win_rate': round(win_rate, 4),
            'win_loss_ratio': round(win_loss_ratio, 4),
            'risk_level': risk_level,
            'score_adjustment': round(score_adjustment, 2) if score else None,
            'recommendation': self._get_recommendation(position, risk_level)
        }
    
    def _assess_risk(
        self,
        win_rate: float,
        win_loss_ratio: float,
        position: float
    ) -> str:
        """评估风险等级"""
        # 计算期望收益
        expected_return = win_rate * win_loss_ratio - (1 - win_rate)
        
        if expected_return > 0.3 and win_rate >= 0.6:
            return "LOW"
        elif expected_return > 0.1 and win_rate >= 0.5:
            return "MEDIUM"
        elif expected_return > 0:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _score_adjustment(self, score: float) -> float:
        """
        根据V+F+Q评分调整仓位
        
        评分越高，仓位调整系数越大
        """
        if score >= 9:
            return 1.2  # 高评分，可以略微加仓
        elif score >= 7:
            return 1.0  # 正常仓位
        elif score >= 5:
            return 0.7  # 中等评分，减仓
        elif score >= 3:
            return 0.3  # 低评分，大幅减仓
        else:
            return 0.0  # 极低评分，不建议持仓
    
    def _get_recommendation(self, position: float, risk_level: str) -> str:
        """生成仓位建议文字"""
        if position <= 0:
            return "🚫 不建议买入：胜率或盈亏比不达标"
        elif position < 0.05:
            return f"⚠️ 轻仓试探：建议仓位 {position*100:.1f}%"
        elif position < 0.10:
            return f"📊 常规配置：建议仓位 {position*100:.1f}%"
        elif position < 0.15:
            return f"✅ 标准仓位：建议仓位 {position*100:.1f}%"
        elif position < 0.20:
            return f"💪 积极配置：建议仓位 {position*100:.1f}%"
        else:
            return f"🔥 重仓机会：建议仓位 {position*100:.1f}%（注意风险）"


class PositionManager:
    """
    综合仓位管理器
    
    结合凯利公式和固定规则，提供双重保障的仓位建议
    """
    
    def __init__(self, kelly_config: KellyConfig = None):
        """
        初始化仓位管理器
        
        Args:
            kelly_config: 凯利公式配置
        """
        self.kelly_calculator = KellyPositionCalculator(kelly_config)
        
        # 固定规则：评分→仓位映射
        self.score_position_map = {
            (9, 10): 0.20,   # 9-10分 → 20%
            (8, 9): 0.15,    # 8-9分 → 15%
            (7, 8): 0.12,    # 7-8分 → 12%
            (6, 7): 0.08,    # 6-7分 → 8%
            (5, 6): 0.05,    # 5-6分 → 5%
            (0, 5): 0.00,    # 0-5分 → 0%
        }
    
    def get_position(
        self,
        win_rate: float,
        win_loss_ratio: float,
        score: float,
        use_kelly: bool = True,
        use_fixed: bool = True
    ) -> Dict:
        """
        获取综合仓位建议
        
        Args:
            win_rate: 胜率
            win_loss_ratio: 盈亏比
            score: V+F+Q评分
            use_kelly: 是否使用凯利公式
            use_fixed: 是否使用固定规则
            
        Returns:
            综合仓位建议
        """
        result = {
            'win_rate': round(win_rate, 4),
            'win_loss_ratio': round(win_loss_ratio, 4),
            'score': round(score, 1)
        }
        
        # 凯利公式仓位
        if use_kelly:
            kelly_result = self.kelly_calculator.calculate(
                win_rate=win_rate,
                win_loss_ratio=win_loss_ratio,
                score=score
            )
            result['kelly_position'] = kelly_result['position']
            result['kelly_position_pct'] = kelly_result['position_pct']
            result['kelly_risk'] = kelly_result['risk_level']
        
        # 固定规则仓位
        if use_fixed:
            fixed_position = self._get_fixed_position(score)
            result['fixed_position'] = fixed_position
            result['fixed_position_pct'] = round(fixed_position * 100, 2)
        
        # 综合建议（取两者较小值，更保守）
        if use_kelly and use_fixed:
            final_position = min(
                result.get('kelly_position', 0),
                result.get('fixed_position', 0)
            )
            result['final_position'] = final_position
            result['final_position_pct'] = round(final_position * 100, 2)
            result['method'] = 'min(kelly, fixed)'
        elif use_kelly:
            result['final_position'] = result.get('kelly_position', 0)
            result['final_position_pct'] = result.get('kelly_position_pct', 0)
            result['method'] = 'kelly'
        elif use_fixed:
            result['final_position'] = result.get('fixed_position', 0)
            result['final_position_pct'] = result.get('fixed_position_pct', 0)
            result['method'] = 'fixed'
        else:
            result['final_position'] = 0
            result['final_position_pct'] = 0
            result['method'] = 'none'
        
        # 生成建议文字
        result['recommendation'] = self._generate_recommendation(result)
        
        return result
    
    def _get_fixed_position(self, score: float) -> float:
        """根据评分获取固定规则仓位"""
        for (low, high), position in self.score_position_map.items():
            if low <= score < high:
                return position
        return 0.0
    
    def _generate_recommendation(self, result: Dict) -> str:
        """生成综合建议"""
        position = result.get('final_position', 0)
        score = result.get('score', 0)
        win_rate = result.get('win_rate', 0)
        
        if position <= 0:
            return "🚫 暂不建议买入"
        
        # 根据评分生成建议
        if score >= 8:
            action = "强烈推荐"
            emoji = "🔥"
        elif score >= 7:
            action = "建议买入"
            emoji = "✅"
        elif score >= 6:
            action = "可以关注"
            emoji = "📊"
        else:
            action = "谨慎对待"
            emoji = "⚠️"
        
        return f"{emoji} {action}：建议仓位 {position*100:.1f}%（评分{score:.1f}分，历史胜率{win_rate*100:.0f}%）"


def calculate_position_from_matches(
    matches_results: list,
    score: float,
    position_manager: PositionManager = None
) -> Dict:
    """
    从历史匹配结果计算仓位建议
    
    Args:
        matches_results: 历史匹配结果列表
        score: V+F+Q评分
        position_manager: 仓位管理器
        
    Returns:
        仓位建议
    """
    if not matches_results:
        return {
            'valid': False,
            'message': '无有效匹配数据',
            'final_position': 0,
            'final_position_pct': 0
        }
    
    # 计算胜率和盈亏比
    wins = []
    losses = []
    
    for r in matches_results:
        final_ret = r.get('final_return', 0)
        if final_ret > 0:
            wins.append(final_ret)
        elif final_ret < 0:
            losses.append(abs(final_ret))
    
    total = len(wins) + len(losses)
    if total == 0:
        return {
            'valid': False,
            'message': '无盈亏数据',
            'final_position': 0,
            'final_position_pct': 0
        }
    
    win_rate = len(wins) / total
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 1
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
    
    # 使用仓位管理器
    if position_manager is None:
        position_manager = PositionManager()
    
    result = position_manager.get_position(
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        score=score
    )
    result['valid'] = True
    result['wins_count'] = len(wins)
    result['losses_count'] = len(losses)
    result['avg_win'] = round(avg_win, 2)
    result['avg_loss'] = round(avg_loss, 2)
    
    return result


if __name__ == "__main__":
    print("=== 凯利公式仓位计算测试 ===")
    
    # 创建计算器
    calculator = KellyPositionCalculator()
    
    # 测试不同场景
    test_cases = [
        {"win_rate": 0.70, "win_loss_ratio": 2.0, "score": 8},  # 高胜率高赔率
        {"win_rate": 0.55, "win_loss_ratio": 1.5, "score": 6},  # 中等
        {"win_rate": 0.45, "win_loss_ratio": 1.2, "score": 5},  # 偏低
        {"win_rate": 0.35, "win_loss_ratio": 0.8, "score": 3},  # 不推荐
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}:")
        print(f"  胜率: {case['win_rate']*100}%")
        print(f"  盈亏比: {case['win_loss_ratio']}")
        print(f"  评分: {case['score']}")
        
        result = calculator.calculate(**case)
        print(f"  凯利原始: {result['kelly_raw']*100:.1f}%")
        print(f"  半凯利: {result['kelly_adjusted']*100:.1f}%")
        print(f"  最终仓位: {result['position_pct']}%")
        print(f"  风险等级: {result['risk_level']}")
        print(f"  建议: {result['recommendation']}")
    
    print("\n\n=== 综合仓位管理器测试 ===")
    manager = PositionManager()
    
    result = manager.get_position(
        win_rate=0.65,
        win_loss_ratio=1.8,
        score=7.5
    )
    
    print(f"评分: {result['score']}")
    print(f"凯利仓位: {result['kelly_position_pct']}%")
    print(f"固定规则仓位: {result['fixed_position_pct']}%")
    print(f"最终建议仓位: {result['final_position_pct']}%")
    print(f"建议: {result['recommendation']}")
