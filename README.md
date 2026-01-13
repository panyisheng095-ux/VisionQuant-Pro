# VisionQuant-Pro

<div align="center">

**AI驱动的K线形态智能投资系统 | AI-Powered K-Line Pattern Investment System**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/panyisheng095-ux/VisionQuant-Pro?style=social)](https://github.com/panyisheng095-ux/VisionQuant-Pro)

*K线视觉学习 | Top10历史形态对比 | 多因子评分 | 智能仓位建议*

</div>

---

## 🎯 这个项目能做什么？

**VisionQuant-Pro 是一个实用的AI投资辅助工具，帮助你做出更明智的投资决策。**

### 核心功能

| 功能 | 说明 | 实用价值 |
|------|------|---------|
| **🖼️ Top10历史形态对比** | 输入股票代码，展示历史上最相似的10个K线形态及其后续走势 | 直观了解"这种形态历史上怎么走" |
| **📊 多因子智能评分** | V(视觉)+F(财务)+Q(量化)三维度评分，0-10分 | 综合评估买入时机 |
| **💰 凯利公式仓位建议** | 基于胜率和赔率计算最优仓位 | 科学配置资金，避免过度集中 |
| **📰 舆情监控** | 自动抓取相关新闻，分析市场情绪 | 了解市场对该股的看法 |
| **🔬 可解释AI决策** | Grad-CAM热力图显示模型关注区域 | 知道AI为什么这么判断 |

---

## 🔥 独家卖点：Top10历史形态对比

这是VisionQuant-Pro的**独一无二**的功能：

```
输入: 600519 (贵州茅台)
      ↓
系统在40万张历史K线图中搜索最相似的10个形态
      ↓
输出: 
┌─────────────────────────────────────────────────────┐
│  当前形态        Top1相似      Top2相似      ...    │
│  [K线图]        [K线图]       [K线图]              │
│                 后续+8.5%     后续+12.3%           │
│                 2023-05-15    2022-11-08           │
└─────────────────────────────────────────────────────┘
      ↓
统计: 10个相似形态中，7个后续上涨，胜率70%
      平均收益+6.2%，最大回撤-3.1%
```

**这个功能的价值**：
- 不是告诉你"AI预测涨"，而是让你**亲眼看到**历史上相似形态的真实结果
- 用历史数据说话，增强投资信心
- 完全透明，没有黑盒

---

## 📊 多因子评分系统

### V+F+Q 三维度评分

```
总分 = V(视觉形态) + F(财务基本面) + Q(量化技术指标)
     = [0-3分]     + [0-4分]       + [0-3分]
     = 0-10分

评分解读:
≥8分 → 强烈买入信号
7分  → 买入信号
5-6分 → 观望
<5分 → 卖出/回避
```

### V - 视觉形态分 (0-3分)

| 胜率 | 得分 | 含义 |
|------|------|------|
| ≥65% | 3分 | 历史相似形态大概率上涨 |
| 55-65% | 2分 | 历史表现良好 |
| 45-55% | 1分 | 历史表现中性 |
| <45% | 0分 | 历史表现不佳 |

### F - 财务基本面分 (0-4分)

| 指标 | 条件 | 得分 |
|------|------|------|
| ROE | >15% | +2分 |
| ROE | 8-15% | +1分 |
| PE(TTM) | 0-20 | +2分 |
| PE(TTM) | 20-40 | +1分 |

### Q - 量化技术指标分 (0-3分)

| 指标 | 条件 | 得分 |
|------|------|------|
| MA60 | 股价>MA60 | +1分 |
| RSI | 30-70区间 | +1分 |
| MACD | 柱状>0 | +1分 |

---

## 💰 凯利公式仓位建议

### 什么是凯利公式？

凯利公式是专业交易员用于计算最优仓位的数学公式：

```python
f* = (p × b - q) / b

其中:
p = 胜率 (历史相似形态的上涨概率)
q = 1 - p (亏损概率)
b = 赔率 (平均盈利 / 平均亏损)

例如:
胜率 p = 70%
赔率 b = 2.0 (平均赚2块，亏1块)
最优仓位 f* = (0.7 × 2 - 0.3) / 2 = 55%

实际应用中限制最大25%，防止单票过度集中
```

### 仓位建议对照表

| 评分 | 凯利建议 | 实际建议 | 风险等级 |
|------|---------|---------|---------|
| 9-10分 | 20-25% | 20% | 可积极 |
| 7-8分 | 15-20% | 15% | 正常 |
| 5-6分 | 5-10% | 5% | 谨慎 |
| <5分 | 0% | 0% | 回避 |

---

## 🏗️ 系统架构

```
                        VisionQuant-Pro 系统架构
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              数据输入层                    AI分析层
                    │                           │
         ┌──────────┼──────────┐               │
         ↓          ↓          ↓               │
      K线数据    财务数据    新闻数据           │
         │          │          │               │
         ↓          ↓          ↓               │
    ┌────┴────┐     │          │               │
    │K线截图  │     │          │               │
    │(40万张) │     │          │               │
    └────┬────┘     │          │               │
         │          │          │               │
         ↓          │          │               │
  ┌──────┴──────┐   │          │               │
  │AttentionCAE │   │          │               │
  │(视觉特征)   │   │          │               │
  └──────┬──────┘   │          │               │
         │          │          │               │
         ↓          ↓          ↓               │
    ┌────┴──────────┴──────────┴────┐          │
    │      FAISS 相似度检索          │          │
    │      (毫秒级Top10匹配)         │          │
    └────────────┬──────────────────┘          │
                 │                              │
                 ↓                              │
    ┌────────────┴────────────┐                │
    │   V+F+Q 多因子评分系统   │←───────────────┘
    └────────────┬────────────┘
                 │
        ┌────────┼────────┐
        ↓        ↓        ↓
   凯利仓位   投资建议   风险提示
```

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/panyisheng095-ux/VisionQuant-Pro.git
cd VisionQuant-Pro

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 运行

```bash
python run.py
# 或
PYTHONPATH=. streamlit run web/app.py
```

### 使用

1. 打开浏览器访问 `http://localhost:8501`
2. 输入股票代码（如 600519）
3. 点击"开始分析"
4. 查看Top10历史形态对比、评分、仓位建议

---

## 📁 项目结构

```
VisionQuant-Pro/
├── src/
│   ├── models/
│   │   ├── attention_cae.py      # K线视觉学习模型
│   │   └── vision_engine.py      # FAISS相似度检索
│   ├── data/
│   │   ├── data_loader.py        # 股票数据加载
│   │   ├── news_harvester.py     # 新闻抓取
│   │   └── triple_barrier.py     # Triple Barrier标签
│   ├── strategies/
│   │   ├── factor_mining.py      # V+F+Q评分系统
│   │   ├── fundamental.py        # 财务数据分析
│   │   ├── kelly_position.py     # 凯利公式仓位
│   │   └── portfolio_optimizer.py # 组合优化
│   └── utils/
│       ├── visualizer.py         # Top10对比图生成
│       ├── grad_cam.py           # Grad-CAM可视化
│       └── walk_forward.py       # Walk-Forward回测
├── web/
│   └── app.py                    # Streamlit界面
├── data/
│   ├── images/                   # 40万张K线图
│   ├── models/                   # 训练好的模型
│   └── indices/                  # FAISS索引
└── scripts/
    └── train_dual_stream.py      # 双流网络训练
```

---

## 🔬 技术特点

### 1. AttentionCAE - 带注意力机制的卷积自编码器

- 8头自注意力机制，捕捉K线图中的长距离依赖
- 能识别"头肩顶"、"双底"等复杂形态
- 1024维特征向量，精确表示形态特征

### 2. FAISS - 毫秒级相似度检索

- 40万张K线图索引
- 基于L2距离的高效检索
- 毫秒级返回Top10最相似形态

### 3. Triple Barrier标签

- 业界标准的标签定义方法
- 止盈线(+5%)、止损线(-3%)、最大持有期(20天)
- 更贴近实际交易决策

### 4. Walk-Forward回测

- 滚动窗口验证，防止未来函数
- 模拟真实交易场景
- 提供统计显著性检验

---

## 📈 与其他项目对比

| 功能 | VisionQuant | 传统量化 | 其他AI项目 |
|------|------------|---------|-----------|
| Top10形态对比 | ✅ 独有 | ❌ | ❌ |
| K线视觉学习 | ✅ | ❌ | 部分 |
| 多因子评分 | ✅ V+F+Q | ✅ | 部分 |
| 凯利仓位 | ✅ | 部分 | ❌ |
| Grad-CAM解释 | ✅ | ❌ | 部分 |
| Walk-Forward | ✅ | 部分 | ❌ |

---

## ⚠️ 风险提示

1. **本项目仅供学习和研究使用，不构成任何投资建议**
2. 历史表现不代表未来收益
3. 量化交易存在显著风险
4. 请根据自身风险承受能力做出投资决策

---

## 🗺️ 路线图

### v1.6 (进行中)
- [x] Top10历史形态对比
- [x] V+F+Q多因子评分
- [ ] Triple Barrier标签系统
- [ ] 凯利公式仓位建议
- [ ] Grad-CAM可视化

### v2.0 (规划中)
- [ ] 双流网络（视觉+数值融合）
- [ ] Walk-Forward深度集成
- [ ] 强化Top10统计信息
- [ ] 多市场支持

---

## 📖 引用

```bibtex
@software{visionquant-pro,
  title = {VisionQuant-Pro: AI-Powered K-Line Pattern Investment System},
  author = {Pan, Yisheng},
  year = {2026},
  url = {https://github.com/panyisheng095-ux/VisionQuant-Pro}
}
```

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with ❤️ by [panyisheng095-ux](https://github.com/panyisheng095-ux)

</div>
