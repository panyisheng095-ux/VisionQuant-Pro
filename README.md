
<div align="center">

**AI驱动的K线形态智能投资系统 | AI-Powered K-Line Pattern Investment System**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/panyisheng095-ux/VisionQuant-Pro?style=social)](https://github.com/panyisheng095-ux/VisionQuant-Pro)

*K线视觉学习 | Top10历史形态对比 | 多因子评分 | 智能仓位建议 | 因子有效性分析*

</div>

---

## 📋 目录

- [项目概述](#项目概述)
- [核心创新：Magic Moment](#核心创新magic-moment)
- [系统架构](#系统架构)
- [核心功能](#核心功能)
- [技术实现](#技术实现)
- [理论基础](#理论基础)
- [快速开始](#快速开始)
- [版本历史](#版本历史)
- [引用与致谢](#引用与致谢)

---

## 项目概述

### 研究背景

传统量化投资主要依赖数值特征（如PE、ROE、技术指标），而忽略了K线图本身蕴含的丰富视觉信息。技术分析虽然广泛使用，但依赖人工识别，主观性强且难以量化。VisionQuant-Pro首次将**深度学习视觉学习**与**量化投资**深度融合，通过无监督学习从40万张历史K线图中自动提取形态特征，实现"让AI看懂K线图"。

### 核心价值

1. **可解释性**：Top10历史形态对比，让用户"亲眼看到"历史上相似形态的真实结果
2. **工业落地**：完整的因子有效性分析框架，支持动态权重调整和因子失效检测
3. **学术创新**：多理论融合（行为金融学+技术分析+市场微观结构+机器学习）

### 项目定位

**基于K线学习因子（视觉形态）的智能投研助手**，主因子为K线视觉学习，辅助因子为基本面和技术面（动态降权）。核心目标是“可解释、可落地、可复盘”，让使用者清楚知道每一步判断的来龙去脉。

---

## 🔥 最新更新（2026-01）

- **视觉检索更“像”**：适配FAISS内积索引的相似度映射 + 像素级重排兜底，Top10更贴近肉眼形态
- **查询对齐索引分布**：优先使用已有历史K线图作为查询图，减少风格偏移
- **行业对比更稳**：缓存全市场spot + 多层兜底逻辑，弱化外部接口波动
- **批量→单只更丝滑**：跳转时同步侧边栏输入与模块状态
- **多指标相似度合力**：Embedding + 像素/边缘 + 价格形态相关，多路信号融合排序
- **核心升级(DTW)**：视觉引擎全面升级为**DTW主导 + 趋势约束**检索，解决"形似神不似"痛点。
- **工业化加固**：财务数据抓取多源容错、批量分析增加异常保护、Web界面1:1迁移FastAPI准备中。

---

## 核心创新：Magic Moment

### 💡 想法是如何诞生的？

#### 问题1：为什么是K线图？

**观察**：技术分析在市场中广泛使用，说明投资者确实依赖形态识别做决策。但传统方法依赖人工经验，难以量化。

**灵感**：既然CNN能识别猫狗，为什么不能识别K线形态？K线图本质上是二维图像，包含丰富的空间信息（如头肩顶、双底等）。

#### 问题2：为什么是无监督学习？

**观察**：标注40万张K线图几乎不可能，且不同形态的定义主观性强。

**灵感**：自编码器（Autoencoder）可以无监督学习图像的潜在表示，通过重建误差学习特征。相似形态应该有相似的特征向量。

#### 问题3：为什么是Top10对比而不是直接预测？

**观察**：用户不信任"黑盒"预测，需要可解释性。

**灵感**：与其让AI说"我预测涨"，不如让AI说"历史上10个最相似的形态，7个涨了"。这样用户可以看到：
- 相似形态的K线图
- 后续真实走势
- 统计胜率

**这就是Magic Moment**：将"预测"转化为"历史参考"，既保持了AI的智能，又提供了人类可理解的解释。

#### 问题4：如何证明因子有效性？

**观察**：简单的"收益率>0"胜率计算太粗糙，无法证明因子在何时、为何有效。

**灵感**：借鉴量化因子研究的方法论：
- Rolling IC/Sharpe分析
- Regime识别（牛市/熊市/震荡）
- 因子衰减分析
- 拥挤交易检测
- 风险补偿分析

**Magic Moment 2**：将"K线学习"包装成"量化因子"，用专业的因子研究框架证明其有效性。

#### 问题5：如何应对因子失效？

**观察**：任何因子都可能失效，需要动态调整。

**灵感**：根据市场Regime和因子IC动态调整权重：
- 牛市：K线因子权重高（60%）
- 熊市：降低K线因子权重（40%），增加基本面权重
- 因子IC下降：自动降权

**Magic Moment 3**：让系统"自适应"，而不是"固定规则"。

### 🎯 核心设计哲学

1. **透明胜过准确**：宁可准确度略低，也要让用户理解AI的判断依据
2. **历史胜过预测**：用历史数据说话，而不是"AI预测"
3. **因子胜过模型**：将K线学习包装成因子，用因子研究框架证明有效性
4. **动态胜过静态**：根据市场环境动态调整，而不是固定规则

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    VisionQuant-Pro v2.0                      │
│               K线学习因子智能投研系统                          │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   数据层                模型层                  策略层
        │                     │                     │
┌───────┴───────┐   ┌─────────┴─────────┐   ┌──────┴──────┐
│ 数据源抽象层    │   │  AttentionCAE     │   │ 因子有效性   │
│ - akshare     │   │  (视觉特征提取)     │   │ 分析框架     │
│ - 聚宽/米筐    │   │  - 512/1024/2048维 │   │ - IC/Sharpe  │
│ - 数据质量检查  │   │  - 多尺度支持       │   │ - Regime识别 │
└───────┬───────┘   │  - SimCLR对比学习  │   │ - 衰减分析   │
        │           └─────────┬─────────┘   │ - 拥挤检测   │
        │                     │             │ - 风险补偿   │
        │           ┌─────────┴─────────┐   └──────┬──────┘
        │           │  FAISS 相似度检索  │          │
        │           │  (40万K线图索引)   │          │
        │           └─────────┬─────────┘          │
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Top10历史形态对比  │
                    │  (核心卖点)        │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  V+F+Q多因子评分  │
                    │  (动态权重)      │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  投资建议+仓位配置 │
                    └───────────────────┘
```

---

## 核心功能

https://github.com/user-attachments/assets/c13a0d82-1063-4dde-9e07-289fb1d64ac0

### 1. Top10历史形态对比（核心卖点）

**功能描述**：输入股票代码，系统在40万张历史K线图中搜索最相似的10个形态，展示其后续真实走势。相似度得分经过校准（距离→相似度），并支持相关性增强。

**价值**：
- 直观了解"这种形态历史上怎么走"
- 用历史数据说话，增强投资信心
- 完全透明，没有黑盒

**示例输出**：
```
当前形态: 600519 (贵州茅台)
         ↓
Top1相似: [K线图] → 后续+8.5% (2023-05-15)
Top2相似: [K线图] → 后续+12.3% (2022-11-08)
...
Top10相似: [K线图] → 后续-2.1% (2021-08-20)

统计结果:
- 胜率: 70% (10个中7个上涨)
- 平均收益: +6.2%
- 最大回撤: -3.1%
```

### 2. 多因子评分系统（V+F+Q）

**评分公式**：
```
总分 = V(视觉形态) × W_v + F(财务基本面) × W_f + Q(量化技术) × W_q
```

**动态权重**（根据市场Regime调整）：
- 牛市：W_v=0.60, W_f=0.20, W_q=0.20
- 熊市：W_v=0.40, W_f=0.40, W_q=0.20
- 震荡：W_v=0.50, W_f=0.30, W_q=0.20

**评分标准**：
- ≥8分 → 强烈买入
- 7分 → 买入
- 5-6分 → 观望
- <5分 → 卖出/回避

### 3. 因子有效性分析框架

**功能模块**：
1. **Rolling IC/Sharpe分析**：评估因子预测能力的稳定性
2. **Regime识别**：识别牛市/熊市/震荡市场
3. **因子衰减分析**：判断因子何时失效
4. **拥挤交易检测**：识别因子暴露度集中度
5. **风险补偿分析**：评估因子收益与风险的关系
6. **行业分层分析**：分析因子在不同行业的表现

### 4. 分层回测系统

**功能**：
- 按市值分层（大/中/小盘）
- 按行业分层
- 组合分层（市值×行业）
- Walk-Forward验证（可选）
- Stress Testing（历史危机 + 样本内压力窗口）
- A股约束（涨跌停/停牌不交易）

### 5. 组合构建与仓位设计

**组合优化**：
- Black–Litterman 融合观点（视觉因子作为“观点输入”）
- 组合结构：核心 + 备选增强（规则放宽，保证中小样本也能输出可用组合）
- 仓位约束：最小/最大仓位、最大持仓数

**输出**：
- 权重饼图、评分对比、胜率 vs 预期收益散点
- 单只股票一键跳转至深度分析

### 6. 动态权重管理

**功能**：
- 根据市场Regime自动调整因子权重
- 根据因子IC动态调整
- 因子失效检测和降权处理

---

## 技术实现

### 1. AttentionCAE - 带注意力机制的卷积自编码器

**架构**：
- CNN Encoder: 224×224×3 → 14×14×256
- Multi-Head Self-Attention: 8头，捕捉长距离依赖
- Latent Projection: 256 → 1024/2048维
- CNN Decoder: 重建图像

**特点**：
- 支持512/1024/2048维特征
- 支持多尺度K线图（日线/周线/月线）
- SimCLR对比学习增强

### 2. FAISS - 毫秒级相似度检索

**数据规模**：
- 40万张K线图
- 特征向量：~1.6GB (mmap)
- FAISS索引：~1.6GB

**性能**：
- Top10检索：<10ms
- 支持GPU加速

### 3. Triple Barrier标签系统

**定义**：
- 止盈线：+5%
- 止损线：-3%
- 最大持有期：20天

**存储**：
- HDF5格式，快速查询
- 支持批量计算和增量更新

### 4. 因子研究框架

**模块**：
- `src/factor_analysis/`: IC分析、Regime识别、衰减分析等
- `src/factor_research/`: 行为偏差、信息扩散、相关性分析等
- `src/backtest/`: 分层回测、Stress Testing

---

## 理论基础

### 多理论融合框架

1. **行为金融学**
   - 代表性启发：投资者基于相似形态做决策
   - 锚定偏差：K线形态作为价格锚点
   - 羊群效应：相似形态触发集体行为

2. **技术分析理论**
   - 形态识别：经典技术分析的基础
   - 支撑阻力：K线形态反映关键价位
   - 趋势延续：相似形态往往延续趋势

3. **市场微观结构理论**
   - 订单流与价格形成
   - 流动性影响
   - 信息扩散

4. **机器学习理论**
   - 无监督学习：CAE学习K线形态的潜在表示
   - 相似度匹配：FAISS实现高效检索
   - 迁移学习：历史形态知识迁移到新场景

详见：[理论基础文档](docs/theoretical_foundation.md)

---

## 快速开始

### 前置条件

**重要提示**：新代码（阶段1-8）大部分是框架代码，需要先运行/训练才能使用。但现有的Web应用（v1.5）可以直接运行（如果已有模型和索引）。

**检查现有资源**：
```bash
# 检查模型文件
ls data/models/attention_cae*.pth

# 检查FAISS索引
ls data/indices/cae_faiss.bin

# 检查K线图数据
ls data/images/ | wc -l  # 应该接近40万
```

### 安装

```bash
git clone https://github.com/panyisheng095-ux/VisionQuant-Pro.git
cd VisionQuant-Pro

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 运行Web应用（v1.5 - 现有功能）

```bash
# 如果已有模型和索引，直接运行
python run.py
# 或
PYTHONPATH=. streamlit run web/app.py
```

访问：`http://localhost:8501`

### 训练新模型（v2.0 - 新功能）

```bash
# 1. 训练AttentionCAE（如果还没有）
python scripts/train_attention_cae.py

# 2. 批量计算Triple Barrier标签
python scripts/batch_triple_barrier.py

# 3. 重建FAISS索引（如果使用新模型）
python scripts/rebuild_index_attention.py

# 4. 运行因子分析（可选）
python -c "from src.factor_analysis import *; ..."
```

### 使用新功能

**因子分析页面**（阶段8）：
```bash
# 访问因子分析页面
streamlit run web/pages/factor_analysis.py
```

**分层回测**（阶段6）：
```python
from src.backtest import StratifiedBacktester
# 使用示例见 src/backtest/stratified_backtester.py
```

---

## 版本历史

### v2.0 (2024-12) - K线学习因子深化

**核心改进**：将K线视觉学习包装成"量化因子"，建立完整的因子研究框架

#### 阶段1: 胜率计算重构
- ✅ 混合胜率计算（Triple Barrier 70% + 传统 30%）
- ✅ HDF5标签存储（快速查询）
- ✅ 40万数据处理优化（多进程并行）

#### 阶段2: 因子有效性分析框架
- ✅ Rolling IC/Sharpe分析
- ✅ Regime识别（牛市/熊市/震荡）
- ✅ 因子衰减分析
- ✅ 拥挤交易检测（HHI指数）
- ✅ 风险补偿分析
- ✅ 行业分层分析
- ✅ 因子失效多维度检测
- ✅ 动态权重管理系统

#### 阶段3: 专业数据源接入
- ✅ 数据源抽象层（统一接口）
- ✅ 聚宽/米筐适配器
- ✅ 数据源切换逻辑
- ✅ 数据质量检查器

#### 阶段4: K线学习精度提升
- ✅ 特征维度提升（支持2048维）
- ✅ 多尺度K线图生成（日线/周线/月线）
- ✅ 双流网络优化（视觉+时序融合）
- ✅ SimCLR对比学习
- ✅ 模型集成（多模型融合）

#### 阶段5: 动态权重管理系统
- ✅ Regime管理器
- ✅ 权重配置表（YAML）
- ✅ 评分系统重构（集成动态权重）
- ✅ 权重回测验证

#### 阶段6: 分层回测系统
- ✅ 股票分层逻辑（市值×行业）
- ✅ 分层回测引擎
- ✅ 结果汇总和可视化
- ✅ Walk-Forward可选模式
- ✅ Stress Testing（极端市场测试）

#### 阶段7: 因子研究框架完善
- ✅ 行为偏差分析
- ✅ 信息扩散分析
- ✅ 因子相关性分析
- ✅ 因子稳定性分析
- ✅ 因子组合优化
- ✅ 理论基础文档完善

#### 阶段8: Streamlit UI增强
- ✅ 因子分析主页面
- ✅ IC/Sharpe曲线图
- ✅ Regime识别图
- ✅ 拥挤交易热力图
- ✅ 风险补偿散点图
- ✅ 行业IC对比表
- ✅ PDF报告导出

**技术栈**：
- 新增：`src/factor_analysis/`, `src/factor_research/`, `src/backtest/`
- 新增：`config/factor_weights.yaml`
- 增强：`src/models/attention_cae.py`（支持更高维度）
- 增强：`src/strategies/factor_mining.py`（动态权重）

---

### v1.5 (2024-11) - 功能完善

**核心改进**：完善Top10对比、修复bug、增强用户体验

- ✅ 修复单股票分析锁定问题
- ✅ 修复回测功能
- ✅ 增强Top10统计信息（胜率、平均收益、最大回撤）
- ✅ 优化批量分析性能
- ✅ 完善组合优化（Markowitz模型）

**技术栈**：
- 修复：`web/app.py`（Session State管理）
- 增强：`src/utils/visualizer.py`（Top10统计）
- 新增：`src/strategies/portfolio_optimizer.py`

---

### v1.0 (2024-10) - 初始版本

**核心功能**：Top10历史形态对比、V+F+Q评分、凯利仓位

- ✅ AttentionCAE模型训练
- ✅ 40万张K线图索引构建
- ✅ FAISS相似度检索
- ✅ Top10历史形态对比
- ✅ V+F+Q多因子评分
- ✅ 凯利公式仓位建议
- ✅ Streamlit Web界面

**技术栈**：
- `src/models/attention_cae.py`
- `src/models/vision_engine.py`
- `src/strategies/factor_mining.py`
- `web/app.py`

---

## 引用与致谢

### 学术引用

```bibtex
@software{visionquant-pro,
  title = {VisionQuant-Pro: AI-Powered K-Line Pattern Investment System},
  author = {Pan, Yisheng},
  year = {2026},
  url = {https://github.com/panyisheng095-ux/VisionQuant-Pro},
  note = {基于K线视觉学习的智能投研系统，融合行为金融学、技术分析和机器学习理论}
}
```

### 参考文献

1. Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*.
2. Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*.
3. Lo, A. W. (2004). The adaptive markets hypothesis. *Journal of Portfolio Management*.

### 致谢

- PyTorch团队：深度学习框架
- Facebook AI Research：FAISS相似度检索
- Streamlit团队：Web应用框架
- 所有贡献者和用户

---

## ⚠️ 风险提示

1. **本项目仅供学习和研究使用，不构成任何投资建议**
2. 历史表现不代表未来收益
3. 量化交易存在显著风险
4. 请根据自身风险承受能力做出投资决策

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with ❤️ by [panyisheng095-ux](https://github.com/panyisheng095-ux)

</div>
