"""
高级Transaction Cost模型
Advanced Transaction Cost Model

完整Transaction Cost = Commission + Slippage + Market Impact + Opportunity Cost

基于：
- Almgren & Chriss (2000): Optimal execution of portfolio transactions
- Kissell & Glantz (2003): Optimal Trading Strategies
- 业界标准实践

Author: VisionQuant Team
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TransactionCostConfig:
    """Transaction Cost配置"""
    commission_rate: float = 0.001      # 手续费率 0.1%
    slippage_rate: float = 0.001        # 滑点率 0.1%
    market_impact_coef: float = 0.0001  # 市场冲击系数
    opportunity_cost_rate: float = 0.0005  # 机会成本率
    min_commission: float = 5.0         # 最小手续费（元）
    max_commission_pct: float = 0.003   # 最大手续费率 0.3%


class AdvancedTransactionCost:
    """
    高级Transaction Cost计算器
    
    完整模型：
    Total Cost = Commission + Slippage + Market Impact + Opportunity Cost
    """
    
    def __init__(self, config: TransactionCostConfig = None):
        """
        初始化Transaction Cost计算器
        
        Args:
            config: 配置参数，如果为None则使用默认配置
        """
        self.config = config or TransactionCostConfig()
        
    def calculate_cost(
        self,
        trade_size: float,
        price: float,
        volume: float,
        volatility: float = None,
        is_buy: bool = True
    ) -> Dict:
        """
        计算完整Transaction Cost
        
        Args:
            trade_size: 交易金额（元）
            price: 交易价格
            volume: 当日成交量（股）
            volatility: 波动率（可选，用于计算市场冲击）
            is_buy: 是否买入（True=买入, False=卖出）
            
        Returns:
            成本明细字典
        """
        shares = trade_size / price
        
        # 1. Commission（手续费）
        commission = self._calculate_commission(trade_size)
        
        # 2. Slippage（滑点）
        slippage = self._calculate_slippage(trade_size, volatility)
        
        # 3. Market Impact（市场冲击）
        market_impact = self._calculate_market_impact(
            shares, price, volume, volatility, is_buy
        )
        
        # 4. Opportunity Cost（机会成本）
        opportunity_cost = self._calculate_opportunity_cost(trade_size, volatility)
        
        # 总成本
        total_cost = commission + slippage + market_impact + opportunity_cost
        total_cost_pct = (total_cost / trade_size) * 100 if trade_size > 0 else 0
        
        return {
            'total_cost': round(total_cost, 2),
            'total_cost_pct': round(total_cost_pct, 4),
            'commission': round(commission, 2),
            'commission_pct': round((commission / trade_size) * 100, 4) if trade_size > 0 else 0,
            'slippage': round(slippage, 2),
            'slippage_pct': round((slippage / trade_size) * 100, 4) if trade_size > 0 else 0,
            'market_impact': round(market_impact, 2),
            'market_impact_pct': round((market_impact / trade_size) * 100, 4) if trade_size > 0 else 0,
            'opportunity_cost': round(opportunity_cost, 2),
            'opportunity_cost_pct': round((opportunity_cost / trade_size) * 100, 4) if trade_size > 0 else 0,
            'trade_size': round(trade_size, 2),
            'shares': round(shares, 0)
        }
    
    def _calculate_commission(self, trade_size: float) -> float:
        """
        计算手续费
        
        规则：
        - 费率：0.1%
        - 最小：5元
        - 最大：0.3%
        """
        commission = trade_size * self.config.commission_rate
        commission = max(self.config.min_commission, commission)
        commission = min(commission, trade_size * self.config.max_commission_pct)
        return commission
    
    def _calculate_slippage(self, trade_size: float, volatility: float = None) -> float:
        """
        计算滑点成本
        
        滑点 = 交易金额 × 滑点率
        如果提供波动率，滑点率会随波动率增加
        """
        slippage_rate = self.config.slippage_rate
        
        # 波动率调整（波动率越高，滑点越大）
        if volatility is not None:
            # 假设基准波动率为20%，波动率每增加10%，滑点增加0.01%
            volatility_adjustment = (volatility - 0.20) / 0.10 * 0.0001
            slippage_rate = max(0.0005, min(0.002, slippage_rate + volatility_adjustment))
        
        return trade_size * slippage_rate
    
    def _calculate_market_impact(
        self,
        shares: float,
        price: float,
        volume: float,
        volatility: float = None,
        is_buy: bool = True
    ) -> float:
        """
        计算市场冲击成本
        
        基于Almgren-Chriss模型：
        Market Impact = α × (Trade Size / Daily Volume)^β × Volatility
        
        简化版本：
        Market Impact = Coef × (Trade Size / Daily Volume) × Price
        """
        if volume <= 0:
            return 0.0
        
        # 计算交易量占比
        trade_ratio = shares / volume
        
        # 基础市场冲击
        base_impact = self.config.market_impact_coef * trade_ratio * price * shares
        
        # 波动率调整（波动率越高，冲击越大）
        if volatility is not None:
            volatility_multiplier = 1 + (volatility - 0.20) / 0.10 * 0.5
            base_impact *= max(0.5, min(2.0, volatility_multiplier))
        
        # 买入和卖出的冲击可能不同（通常卖出冲击更大）
        if not is_buy:
            base_impact *= 1.1  # 卖出冲击增加10%
        
        return base_impact
    
    def _calculate_opportunity_cost(self, trade_size: float, volatility: float = None) -> float:
        """
        计算机会成本
        
        机会成本 = 交易金额 × 机会成本率 × 持有期（假设1天）
        
        机会成本率通常基于无风险利率和市场风险溢价
        """
        # 基础机会成本率
        cost_rate = self.config.opportunity_cost_rate
        
        # 如果提供波动率，机会成本随风险增加
        if volatility is not None:
            # 风险越高，机会成本越高
            risk_adjustment = (volatility - 0.20) / 0.10 * 0.0002
            cost_rate = max(0.0001, min(0.001, cost_rate + risk_adjustment))
        
        # 假设持有期为1天（实际应该根据策略调整）
        holding_days = 1
        return trade_size * cost_rate * holding_days
    
    def calculate_net_return(
        self,
        gross_return: float,
        trade_size: float,
        price: float,
        volume: float = None,
        volatility: float = None,
        is_buy: bool = True
    ) -> Dict:
        """
        计算净收益（扣除Transaction Cost后）
        
        Args:
            gross_return: 毛收益率（如0.05表示5%）
            trade_size: 交易金额
            price: 交易价格
            volume: 成交量（可选）
            volatility: 波动率（可选）
            is_buy: 是否买入
            
        Returns:
            净收益明细
        """
        # 计算成本
        cost_detail = self.calculate_cost(
            trade_size=trade_size,
            price=price,
            volume=volume or (trade_size / price * 10),  # 默认假设成交量是交易量的10倍
            volatility=volatility,
            is_buy=is_buy
        )
        
        # 毛收益
        gross_profit = trade_size * gross_return
        
        # 净收益
        net_profit = gross_profit - cost_detail['total_cost']
        net_return = (net_profit / trade_size) * 100 if trade_size > 0 else 0
        
        return {
            'gross_return_pct': round(gross_return * 100, 2),
            'gross_profit': round(gross_profit, 2),
            'total_cost': cost_detail['total_cost'],
            'total_cost_pct': cost_detail['total_cost_pct'],
            'net_profit': round(net_profit, 2),
            'net_return_pct': round(net_return, 2),
            'cost_breakdown': cost_detail
        }


def calculate_transaction_cost(
    trade_size: float,
    price: float,
    volume: float = None,
    volatility: float = None,
    is_buy: bool = True
) -> Dict:
    """
    便捷函数：计算Transaction Cost
    
    Args:
        trade_size: 交易金额（元）
        price: 交易价格
        volume: 成交量（股）
        volatility: 波动率
        is_buy: 是否买入
        
    Returns:
        成本明细
    """
    calculator = AdvancedTransactionCost()
    return calculator.calculate_cost(
        trade_size=trade_size,
        price=price,
        volume=volume or (trade_size / price * 10),
        volatility=volatility,
        is_buy=is_buy
    )


if __name__ == "__main__":
    print("=== 高级Transaction Cost模型测试 ===")
    
    calculator = AdvancedTransactionCost()
    
    # 测试场景
    test_cases = [
        {
            'trade_size': 100000,  # 10万元
            'price': 50.0,
            'volume': 1000000,  # 100万股
            'volatility': 0.25,  # 25%波动率
            'is_buy': True
        },
        {
            'trade_size': 50000,  # 5万元
            'price': 30.0,
            'volume': 500000,  # 50万股
            'volatility': 0.15,  # 15%波动率
            'is_buy': False
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}:")
        result = calculator.calculate_cost(**case)
        
        print(f"  交易金额: {result['trade_size']}元")
        print(f"  总成本: {result['total_cost']}元 ({result['total_cost_pct']}%)")
        print(f"  - 手续费: {result['commission']}元 ({result['commission_pct']}%)")
        print(f"  - 滑点: {result['slippage']}元 ({result['slippage_pct']}%)")
        print(f"  - 市场冲击: {result['market_impact']}元 ({result['market_impact_pct']}%)")
        print(f"  - 机会成本: {result['opportunity_cost']}元 ({result['opportunity_cost_pct']}%)")
        
        # 测试净收益
        net_result = calculator.calculate_net_return(
            gross_return=0.05,  # 5%毛收益
            **case
        )
        print(f"\n  净收益分析（假设5%毛收益）:")
        print(f"  毛收益: {net_result['gross_profit']}元")
        print(f"  扣除成本后净收益: {net_result['net_profit']}元 ({net_result['net_return_pct']}%)")
