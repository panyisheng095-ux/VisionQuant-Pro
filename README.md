# VisionQuant-Pro

<div align="center">

**让AI学习人类交易员的"眼光" | Teaching AI to See Like a Trader**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*K线视觉学习 | 形态相似度检索 | 行为金融学支撑 | Top10历史对比*

</div>

---

## 🎯 项目定位

**VisionQuant-Pro 不是又一个量化交易系统，而是一个独特的"K线形态视觉智能"项目。**

### 核心创新

| 特性 | 说明 |
|------|------|
| **🖼️ Top10历史形态对比** | 用户能直观看到"当前K线和哪些历史形态相似"——这是GAF和纯数值模型做不到的 |
| **📚 行为金融学支撑** | 市场由人类行为驱动，人类是视觉动物。我们学习的是"集体视觉记忆" |
| **🔍 可解释的AI决策** | 不是黑盒预测"涨/跌"，而是"这个形态历史上70%会涨，看这10个例子" |

### 为什么用K线截图而不是GAF？

> **K线截图的"信息丢失"是一种有益的抽象，而非缺陷。**

人类交易员看K线图时，并不是在精确计算"今天涨了5.23%还是5.24%"。他们看的是**形态**——上涨是温和的还是强势的、支撑位在哪里、阻力位在哪里。

CAE学习的1024维特征正是这种"形态感知"，而非"数值拟合"。

详细论述见: [行为金融学理论基础](docs/behavioral_finance_rationale.md)

---

## 📊 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| **Web界面** | ✅ 可用 | 40万张K线图 + AttentionCAE |
| **AttentionCAE模型** | ✅ 已训练 | 8头注意力，5轮训练 |
| **FAISS索引** | ✅ 已构建 | 40万向量，毫秒级检索 |
| **Walk-Forward验证** | ✅ 可用 | 防止未来函数泄漏 |
| **学术分支 (GAF)** | 📦 独立分支 | 消融实验用，见 `academic` 分支 |

---

## 🇨🇳 架构说明

```
                    VisionQuant-Pro 架构
                           │
           ┌───────────────┴───────────────┐
           │                               │
      主线 (main)                    学术分支 (academic)
           │                               │
   K线截图 + AttentionCAE              GAF + ResNet
           │                               │
   - 40万张现有图片                   - 消融实验用
   - 已训练的模型                    - 论文对照组
   - Top10震撼对比                   - 按需启用
           │                               
     ↓ 核心流程                      
           │
   ┌───────┴───────┐
   │               │
 K线截图      AttentionCAE
 (matplotlib)   (CNN+8头注意力)
   │               │
   └───────┬───────┘
           │
     FAISS检索
     (毫秒级)
           │
    Top10相似形态
           │
  ┌────────┼────────┐
  │        │        │
胜率统计  轨迹推演  V+F+Q评分
  │        │        │
  └────────┼────────┘
           │
      AI投资建议
```

<details>
<summary>点击展开版本演进图</summary>

```
┌─────────────────────────────────────────────────────────────────┐
│                        VERSION EVOLUTION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  v1.0 (2026-01-05)          v1.5 (2026-01-10) [当前]             │
│  ─────────────────          ─────────────────                    │
│  K线截图                     K线截图                              │
│     │                           │                                 │
│     ↓                           ↓                                 │
│  QuantCAE                   AttentionCAE                          │
│  (4层CNN)                   (CAE + 8头注意力)                     │
│     │                           │                                 │
│     ↓                           ↓                                 │
│  FAISS检索                  FAISS检索                             │
│     │                           │                                 │
│     ↓                           ↓                                 │
│  胜率预测                    V+F+Q多因子评分                       │
│                                                                   │
│                                                                   │
│                     学术分支 (academic)                           │
│                     ─────────────────                             │
│                     GAF图像 + ResNet                              │
│                          ↓                                        │
│                     消融实验对照组                                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

</details>
│                               ↓                                   │
│                      Cross-Modal Attention                        │
│                               │                                   │
│                               ↓                                   │
│                      Triple Barrier预测                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 新增代码量统计

| 文件 | 功能 | 代码行数 |
|------|------|---------|
| `gaf_encoder.py` | GAF图像编码 | 491 |
| `triple_barrier.py` | Triple Barrier标签 | 549 |
| `walk_forward.py` | Walk-Forward验证 | 638 |
| `temporal_encoder.py` | TCN+Attention时序编码 | 579 |
| `dual_stream_network.py` | 双流融合网络 | 711 |
| `backtrader_strategy.py` | Backtrader策略集成 | 555 |
| `train_dual_stream.py` | 训练脚本 | 523 |
| `grad_cam.py` | Grad-CAM可视化 | 517 |
| **总计** | | **~4,600** |

</details>

---

## 🔥 核心功能

### 1. Top10 历史形态对比（独家功能）

当你输入一只股票，系统会：
1. 生成当前K线图
2. 在40万张历史K线中搜索最相似的10张
3. **直观展示这10张历史形态及其后续走势**

这是用户最喜欢的功能——不是告诉你"AI预测涨"，而是让你**亲眼看到**历史上相似形态的结果。

### 2. V+F+Q 多因子评分

```
总分 = V(视觉) + F(财务) + Q(量化)
     = [0-3分] + [0-4分] + [0-3分]
     = 0-10分

≥7分 → BUY（买入）
5-6分 → WAIT（观望）
<5分 → SELL（卖出）
```

### 3. AttentionCAE 形态学习

8头自注意力机制，能够捕捉：
- 头肩顶的"左肩"和"右肩"的对称性
- 双底形态中两个谷底的相似性
- 上升三角形中多个触顶点的关系

### 4. Walk-Forward 严格回测

防止未来函数泄漏的滚动窗口验证：

```
|------ 训练 (3年) ------|-- 验证 (6月) --|-- 测试 (6月) --|
                        |
                        ↓ 滚动
|------ 训练 (3年) ------|-- 验证 (6月) --|-- 测试 (6月) --|
```

---

## 📁 项目结构

```
VisionQuant-Pro/
├── src/
│   ├── models/
│   │   ├── attention_cae.py         # ⭐ 核心：AttentionCAE模型
│   │   ├── autoencoder.py           # 基础QuantCAE
│   │   └── vision_engine.py         # FAISS相似度搜索引擎
│   ├── data/
│   │   └── data_loader.py           # 股票数据加载器 (akshare)
│   ├── strategies/
│   │   ├── factor_mining.py         # V+F+Q多因子评分
│   │   ├── fundamental.py           # 财务数据获取
│   │   ├── portfolio_optimizer.py   # Markowitz组合优化
│   │   └── batch_analyzer.py        # 批量分析引擎
│   └── utils/
│       ├── visualizer.py            # Top10对比图生成
│       ├── walk_forward.py          # Walk-Forward验证框架
│       └── attention_visualizer.py  # 注意力权重可视化
├── web/
│   └── app.py                       # Streamlit Web界面
├── data/
│   ├── images/                      # 40万张K线截图 (1.5GB)
│   ├── models/                      # 训练好的模型
│   └── indices/                     # FAISS索引文件
├── docs/
│   ├── behavioral_finance_rationale.md  # 行为金融学理论
│   ├── 常见问题FAQ.md
│   └── 在线部署教程.md
└── requirements.txt

# 学术分支 (git checkout academic)
├── src/data/gaf_encoder.py          # GAF图像编码
├── src/models/dual_stream_network.py # 双流网络
└── scripts/train_dual_stream.py      # 训练脚本
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/panyisheng095-ux/VisionQuant-Pro.git
cd VisionQuant-Pro

python -m venv venv
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### Data Preparation

```bash
# Generate GAF images and labels
python scripts/prepare_data.py --symbols 600519 000858 601899 --window 60
```

### Training

```bash
# Train dual-stream network with Walk-Forward validation
python scripts/train_dual_stream.py \
    --data_dir data \
    --gaf_dir data/gaf_images \
    --batch_size 32 \
    --num_epochs 50
```

### Web Interface

```bash
python run.py  # or: PYTHONPATH=. streamlit run web/app.py
```

---

## 📊 与其他方法对比

| 维度 | 传统量化 | 纯数值CNN | RD-Agent | **VisionQuant** |
|------|---------|----------|----------|-----------------|
| 输入 | 数值因子 | OHLCV | 数值+文本 | **K线截图** |
| 可解释性 | 高 | 低 | 中 | **⭐ 高 (Top10对比)** |
| 人类直觉对齐 | 低 | 低 | 中 | **⭐ 高 (形态可视)** |
| 独特性 | 低 | 中 | 中 | **⭐ 高 (视觉震撼)** |
| 理论支撑 | 统计学 | 深度学习 | Agent | **行为金融学** |

---

## 📚 理论基础

### 行为金融学支撑

> "市场由人类行为驱动，人类是视觉动物。"

| 行为偏差 | 在K线图中的体现 | VisionQuant如何利用 |
|---------|----------------|-------------------|
| **锚定效应** | 历史高/低点成为心理锚点 | 学习视觉上显著的价位区域 |
| **羊群效应** | 突破形态触发集体行动 | 识别触发集体行动的视觉模式 |
| **代表性启发** | "这形态像头肩顶，会跌" | 形式化为相似度搜索 |

### 信息论视角

```
K线截图信息量:  224×224×3 = 150,528 像素值
原始数值信息量: 5 × N天 = 5N 个数值

K线截图 ≠ 信息丢失
K线截图 = 有益的抽象（专注形态，过滤噪声）
```

详细论述: [行为金融学理论基础](docs/behavioral_finance_rationale.md)

---

## Performance Notes

### Expected Results
- **Classification Accuracy**: 45-55% (3-class, beating random 33%)
- **Return Prediction MAE**: 2-4%
- **Alpha vs Buy-and-Hold**: Varies by market condition

### Disclaimer
- **This is a research project, NOT investment advice**
- Past performance does not guarantee future results
- Quantitative trading involves significant risk

---

## 🗺️ 路线图

### v1.6 (近期)
- [ ] Vision Transformer (ViT) 替换CNN骨干
- [ ] 对比学习 (SimCLR) 预训练
- [ ] 更多股票数据覆盖

### v2.0 (中期)
- [ ] 多时间框架融合 (日线 + 周线 + 月线)
- [ ] 实盘交易API集成
- [ ] 多市场支持 (A股 + 港股 + 美股)

### 学术方向 (academic分支)
- [ ] GAF vs K线截图消融实验
- [ ] 论文投稿准备
- [ ] 更多基线对比

---

## 📖 引用

```bibtex
@software{visionquant-pro,
  title = {VisionQuant-Pro: K-Line Visual Pattern Learning for Quantitative Trading},
  author = {Pan, Yisheng},
  year = {2026},
  url = {https://github.com/panyisheng095-ux/VisionQuant-Pro}
}
```

---

## 📚 参考文献

- Kahneman, D., & Tversky, A. (1979). Prospect Theory. *Econometrica*.
- Barberis, N., & Thaler, R. (2003). A Survey of Behavioral Finance. *Handbook of the Economics of Finance*.
- Lo, A. W. (2004). The Adaptive Markets Hypothesis. *Journal of Portfolio Management*.
- López de Prado, M. (2018). Advances in Financial Machine Learning. *Wiley*.

---

## Version History

### Detailed Changelog

---

### v2.0.0 (2026-01-13) - Major Architecture Overhaul

**This is a complete rewrite focused on academic rigor and industrial applicability.**

#### ⚡ Core Architecture Changes

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Image Encoding** | K-line screenshot (matplotlib) | **GAF (Gramian Angular Field)** | 数学严谨的时序→图像转换，保留时间依赖性 |
| **Network** | Single-stream CAE | **Dual-Stream (Vision+Temporal)** | 同时利用视觉空间信息和时序动态信息 |
| **Vision Encoder** | Custom 4-layer CNN | **ResNet18 (pretrained)** | ImageNet预训练，更强的特征提取能力 |
| **Temporal Encoder** | None | **TCN + Self-Attention** | 捕捉长距离时序依赖 |
| **Fusion Method** | None | **Cross-Modal Attention** | 可学习的模态融合权重 |

#### 📊 Data & Labels

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Input Data** | K线截图 (PNG) | **GAF 3通道图像 + 原始OHLCV** | 无信息丢失，精确数值保留 |
| **Label Definition** | 简单涨跌 (+5天收益率>0) | **Triple Barrier Method** | 业界标准，考虑止盈/止损/时间限制 |
| **Label Classes** | 2类 (涨/跌) | **3类 (看涨/震荡/看跌)** | 更符合实际交易决策 |

#### 🔬 Training & Validation

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Data Split** | 随机 90/10 | **Walk-Forward 滚动验证** | 防止未来函数泄露 |
| **Validation** | 单次验证集 | **滚动窗口多次验证** | 更可靠的泛化能力评估 |
| **Training Loss** | MSE重建损失 | **分类CE + 回归MSE + 对比损失** | 多任务联合优化 |

#### 📈 Backtesting

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Framework** | 自写简单回测 | **Backtrader 专业框架** | 工业级回测能力 |
| **Metrics** | 简单收益率 | **Sharpe/Calmar/MaxDD/胜率/盈亏比** | 完整绩效评估 |
| **Look-ahead Bias** | 未严格防范 | **严格时间隔离** | 可信的回测结果 |

#### 🎯 Explainability

| Component | v1.0 | v2.0 | Improvement |
|-----------|------|------|-------------|
| **Model Interpretation** | Attention权重热力图 | **Grad-CAM + Attention + 模态权重** | 多层次可解释性 |
| **Visualization** | 单一注意力图 | **GAF热力图 + 时序注意力 + 融合权重** | 完整的决策解释链 |

#### 📁 New Files Added (v2.0)

```
src/data/
├── gaf_encoder.py          # [NEW] GAF图像编码器 (491 lines)
└── triple_barrier.py       # [NEW] Triple Barrier标签 (549 lines)

src/models/
├── temporal_encoder.py     # [NEW] TCN+Attention时序编码器 (579 lines)
└── dual_stream_network.py  # [NEW] 双流融合网络 (711 lines)

src/strategies/
└── backtrader_strategy.py  # [NEW] Backtrader策略 (555 lines)

src/utils/
├── walk_forward.py         # [NEW] Walk-Forward验证 (638 lines)
└── grad_cam.py             # [NEW] Grad-CAM可视化 (517 lines)

scripts/
└── train_dual_stream.py    # [NEW] 双流网络训练脚本 (523 lines)
```

**Total new code: ~4,600 lines**

---

### v1.5.0 (2026-01-10) - Attention Enhancement

#### Changes from v1.0
- **AttentionCAE**: 在CAE末端添加8头自注意力机制
- **Multi-factor Scoring**: V(视觉)+F(财务)+Q(量化)三因子评分
- **Batch Analysis**: 支持30只股票批量分析
- **Portfolio Optimization**: Markowitz均值-方差优化
- **AI Agent**: 集成Google Gemini大模型辅助分析

#### Files Added (v1.5)
```
src/models/attention_cae.py        # 注意力增强CAE
src/strategies/batch_analyzer.py   # 批量分析引擎
src/strategies/portfolio_optimizer.py  # 组合优化器
src/utils/attention_visualizer.py  # 注意力可视化
```

---

### v1.0.0 (2026-01-05) - Initial Release

#### Core Features
- **QuantCAE**: 4层卷积自编码器，学习K线图形态
- **FAISS Search**: 向量相似度搜索，毫秒级检索
- **Streamlit Web**: 交互式Web界面
- **基础回测**: 简单的买入持有对比

#### Architecture (v1.0)
```
K线截图 (matplotlib)
    ↓
QuantCAE (4-layer CNN)
    ↓
FAISS Index (L2 distance)
    ↓
Top-K Similar Patterns
    ↓
Win Rate Prediction
```

#### Limitations Identified
1. ❌ K线截图丢失精确数值信息
2. ❌ 纯CNN无法捕捉长距离依赖
3. ❌ 简单涨跌标签不符合实际交易
4. ❌ 随机数据划分导致未来函数风险
5. ❌ 缺乏严谨的回测框架

---

### Version Comparison Summary

```
v1.0 Architecture:
──────────────────
K线截图 → CAE Encoder → FAISS → Win Rate → Simple Score

v1.5 Architecture:
──────────────────
K线截图 → AttentionCAE → FAISS → Win Rate → Multi-Factor Score
                ↑                              ↑
          + Attention                    + V+F+Q Factors

v2.0 Architecture:
──────────────────
         ┌→ GAF Image → ResNet18 ──────┐
OHLCV ───┤                              ├→ Cross-Modal Attention → Triple Barrier
         └→ Sequence  → TCN+Attention ─┘
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

**If you find this project useful, please give it a ⭐ Star!**

Made with ❤️ by [panyisheng095-ux](https://github.com/panyisheng095-ux)

</div>
