"""
因子有效性分析报告生成模块
Factor Effectiveness Analysis Report Generator

整合所有分析模块，生成完整的因子有效性报告

Author: VisionQuant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import json

from .ic_analysis import ICAnalyzer
from .regime_detector import RegimeDetector, MarketRegime
from .decay_analysis import DecayAnalyzer
from .crowding_detector import CrowdingDetector
from .risk_compensation import RiskCompensationAnalyzer
from .industry_stratification import IndustryStratifier
from .factor_invalidation import FactorInvalidationDetector


class FactorReportGenerator:
    """
    因子有效性分析报告生成器
    
    整合所有分析模块，生成完整报告
    """
    
    def __init__(self):
        """初始化报告生成器"""
        self.ic_analyzer = ICAnalyzer(window=252)
        self.regime_detector = RegimeDetector()
        self.decay_analyzer = DecayAnalyzer()
        self.crowding_detector = CrowdingDetector()
        self.risk_analyzer = RiskCompensationAnalyzer()
        self.industry_stratifier = IndustryStratifier()
        self.invalidation_detector = FactorInvalidationDetector()
    
    def generate_report(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        prices: pd.Series = None,
        factor_exposures: pd.DataFrame = None,
        industry_mapping: Dict[str, str] = None
    ) -> Dict:
        """
        生成完整的因子有效性分析报告
        
        Args:
            factor_values: 因子值序列
            returns: 未来收益率序列
            prices: 价格序列（可选，用于Regime识别）
            factor_exposures: 因子暴露度DataFrame（可选，用于拥挤检测）
            industry_mapping: 行业映射（可选，用于行业分析）
            
        Returns:
            完整分析报告字典
        """
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'factor_name': 'K线学习因子',
            'analysis_period': {
                'start': str(factor_values.index[0]),
                'end': str(factor_values.index[-1]),
                'total_days': len(factor_values)
            }
        }
        
        # 1. IC分析
        print("📊 进行IC分析...")
        ic_result = self.ic_analyzer.analyze(factor_values, returns)
        report['ic_analysis'] = {
            'summary': ic_result['summary'],
            'ic_series': ic_result['ic_series'].to_dict() if hasattr(ic_result['ic_series'], 'to_dict') else None,
            'sharpe_series': ic_result['sharpe_series'].to_dict() if hasattr(ic_result['sharpe_series'], 'to_dict') else None
        }
        
        # 2. Regime分析
        print("📈 进行Regime识别...")
        regimes = self.regime_detector.detect_regime(returns, prices)
        regime_stats = self.regime_detector.get_regime_statistics(regimes, returns)
        report['regime_analysis'] = {
            'regime_distribution': regime_stats['regime_distribution'],
            'regime_returns': {k: v for k, v in regime_stats.items() if k.endswith('_returns')}
        }
        
        # 3. 衰减分析
        print("📉 进行衰减分析...")
        ic_series = ic_result['ic_series']
        decay_result = self.decay_analyzer.analyze_decay(ic_series)
        report['decay_analysis'] = {
            'is_decaying': decay_result['is_decaying'],
            'decay_start_date': str(decay_result['decay_start_date']) if decay_result['decay_start_date'] else None,
            'decay_rate': decay_result['decay_rate'],
            'predicted_invalidation_date': str(decay_result['predicted_invalidation_date']) if decay_result['predicted_invalidation_date'] else None
        }
        
        # 4. 拥挤检测
        if factor_exposures is not None:
            print("🔍 进行拥挤检测...")
            crowding_result = self.crowding_detector.detect_crowding(factor_exposures)
            report['crowding_analysis'] = {
                'is_crowded': crowding_result['is_crowded'],
                'concentration': crowding_result['concentration'],
                'herfindahl_index': crowding_result['herfindahl_index'],
                'crowding_score': crowding_result['crowding_score']
            }
        else:
            report['crowding_analysis'] = None
        
        # 5. 风险补偿分析
        print("💰 进行风险补偿分析...")
        risk_result = self.risk_analyzer.analyze_risk_compensation(returns, factor_values)
        report['risk_compensation'] = {
            'overall_metrics': risk_result['overall_metrics'],
            'quantile_metrics': risk_result['quantile_metrics']
        }
        
        # 6. 行业分析
        if industry_mapping is not None:
            print("🏭 进行行业分析...")
            # 需要将factor_values和returns转换为DataFrame格式
            # 这里简化处理，实际使用时需要根据数据格式调整
            try:
                # 假设factor_values和returns是单股票的序列
                # 实际应该传入多股票的DataFrame
                report['industry_analysis'] = {
                    'note': '行业分析需要多股票数据，当前为单股票序列'
                }
            except:
                report['industry_analysis'] = None
        else:
            report['industry_analysis'] = None
        
        # 7. 失效检测
        print("⚠️ 进行失效检测...")
        invalidation_result = self.invalidation_detector.detect_invalidation(
            factor_values, returns, factor_exposures
        )
        report['invalidation_detection'] = {
            'is_invalidated': invalidation_result['is_invalidated'],
            'invalidation_score': invalidation_result['invalidation_score'],
            'dimensions': invalidation_result['dimensions'],
            'warning': self.invalidation_detector.get_invalidation_warning(invalidation_result)
        }
        
        # 8. 综合评估
        report['overall_assessment'] = self._generate_overall_assessment(report)
        
        return report
    
    def _generate_overall_assessment(self, report: Dict) -> Dict:
        """
        生成综合评估
        """
        assessment = {
            'factor_quality': 'Unknown',
            'recommendation': 'Unknown',
            'key_strengths': [],
            'key_weaknesses': [],
            'risk_level': 'Medium'
        }
        
        # 评估因子质量
        ic_mean = report['ic_analysis']['summary'].get('mean_ic', 0)
        ic_ir = report['ic_analysis']['summary'].get('ir', 0)
        is_significant = report['ic_analysis']['summary'].get('significant', False)
        
        if ic_mean > 0.05 and ic_ir > 1.0 and is_significant:
            assessment['factor_quality'] = 'Excellent'
        elif ic_mean > 0.03 and ic_ir > 0.5:
            assessment['factor_quality'] = 'Good'
        elif ic_mean > 0.01:
            assessment['factor_quality'] = 'Fair'
        else:
            assessment['factor_quality'] = 'Poor'
        
        # 识别优势
        if ic_mean > 0.03:
            assessment['key_strengths'].append('IC表现良好')
        if ic_ir > 0.5:
            assessment['key_strengths'].append('信息比率较高')
        if report['risk_compensation']['overall_metrics']['sharpe_ratio'] > 1.0:
            assessment['key_strengths'].append('风险调整收益优秀')
        
        # 识别劣势
        if report['decay_analysis']['is_decaying']:
            assessment['key_weaknesses'].append('因子出现衰减')
        if report['invalidation_detection']['is_invalidated']:
            assessment['key_weaknesses'].append('因子可能失效')
        if report.get('crowding_analysis') and report['crowding_analysis']['is_crowded']:
            assessment['key_weaknesses'].append('检测到拥挤交易')
        
        # 推荐建议
        if assessment['factor_quality'] in ['Excellent', 'Good']:
            assessment['recommendation'] = '建议继续使用'
        elif assessment['factor_quality'] == 'Fair':
            assessment['recommendation'] = '建议谨慎使用，持续监控'
        else:
            assessment['recommendation'] = '建议暂停使用或降低权重'
        
        # 风险等级
        if report['invalidation_detection']['invalidation_score'] > 0.7:
            assessment['risk_level'] = 'High'
        elif report['invalidation_detection']['invalidation_score'] > 0.4:
            assessment['risk_level'] = 'Medium'
        else:
            assessment['risk_level'] = 'Low'
        
        return assessment
    
    def export_report(
        self,
        report: Dict,
        output_path: str,
        format: str = 'json'
    ):
        """
        导出报告
        
        Args:
            report: 报告字典
            output_path: 输出路径
            format: 格式 ('json', 'csv', 'html')
        """
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        elif format == 'csv':
            # 导出关键指标到CSV
            summary_data = {
                '指标': [
                    '平均IC', 'IC标准差', '信息比率', '平均Sharpe',
                    '是否衰减', '是否失效', '因子质量'
                ],
                '数值': [
                    report['ic_analysis']['summary']['mean_ic'],
                    report['ic_analysis']['summary']['std_ic'],
                    report['ic_analysis']['summary']['ir'],
                    report['risk_compensation']['overall_metrics']['sharpe_ratio'],
                    report['decay_analysis']['is_decaying'],
                    report['invalidation_detection']['is_invalidated'],
                    report['overall_assessment']['factor_quality']
                ]
            }
            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        else:
            raise ValueError(f"不支持的格式: {format}")


if __name__ == "__main__":
    print("=== 因子报告生成器测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    factor_values = pd.Series(np.random.randn(500).cumsum(), index=dates)
    returns = pd.Series(np.random.randn(500) * 0.01, index=dates)
    
    # 创建生成器
    generator = FactorReportGenerator()
    report = generator.generate_report(factor_values, returns)
    
    print(f"\n报告生成完成！")
    print(f"因子质量: {report['overall_assessment']['factor_quality']}")
    print(f"推荐建议: {report['overall_assessment']['recommendation']}")
