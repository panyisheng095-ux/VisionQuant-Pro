# 🤖 VisionQuant-Pro

<div align="center">

**基于深度学习视觉识别的AI量化投资系统**

Vision-Based Quantitative Trading System with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](#english) | [中文](#中文)

</div>

---

## 中文

### 📖 项目简介

VisionQuant-Pro 是一个创新的量化投资系统，将**计算机视觉**与**量化交易**深度融合。系统通过卷积自编码器（CAE）学习K线图形态特征，结合FAISS向量检索技术，实现了对历史相似形态的快速识别和预测。

### ✨ 核心特性

#### 🎯 视觉量化引擎
- **深度学习形态识别**：使用CAE自动提取K线图视觉特征
- **FAISS相似度检索**：毫秒级检索百万级历史形态库
- **胜率预测**：基于历史相似形态统计未来涨跌概率

#### 📊 智能分析系统
- **多因子评分模型**：融合视觉、技术、基本面三大维度
- **自适应策略**：牛市/熊市双模态策略切换
- **批量组合分析**：支持30只股票并行分析，马科维茨组合优化

#### 🤖 AI对话助手
- **LangChain集成**：基于Google Gemini的智能投资顾问
- **上下文理解**：理解分析结果，提供个性化建议

#### 📈 策略回测
- **VQ策略**：视觉+趋势的自适应仓位管理策略
- **实时回测**：2022-2026年历史数据完整回测
- **风险控制**：8%硬止损，动态仓位调整

### 🏗️ 系统架构

```
VisionQuant-Pro/
├── src/
│   ├── models/
│   │   ├── cae_model.py          # 卷积自编码器
│   │   ├── vision_engine.py      # 视觉识别引擎
│   │   └── predict_engine.py     # 预测引擎
│   ├── strategies/
│   │   ├── backtester.py         # VQ策略回测
│   │   ├── batch_analyzer.py     # 批量分析器
│   │   └── portfolio_optimizer.py # 组合优化器
│   ├── factors/
│   │   └── factor_miner.py       # 多因子挖掘
│   └── utils/
│       └── data_loader.py        # 数据加载器
├── web/
│   └── app.py                    # Streamlit Web界面
├── data/
│   ├── raw/                      # 原始数据
│   └── indices/                  # 索引文件
└── configs/
    └── config.yaml               # 配置文件
```

### 🚀 快速开始

#### 1. 环境要求
```bash
Python 3.9+
pip install -r requirements.txt
```

#### 2. 安装依赖
```bash
cd VisionQuant-Pro
pip install -r requirements.txt
```

#### 3. 准备数据
```bash
# 自动下载示例数据并创建目录结构
python scripts/prepare_data.py
```

**注意：** 完整数据集（154GB）不包含在仓库中，需要自行训练生成。示例数据仅包含5只股票用于快速体验。

#### 4. 启动Web界面
```bash
streamlit run web/app.py
```

访问：http://localhost:8501

### 📦 依赖项

主要依赖包：
- `streamlit` - Web应用框架
- `tensorflow/keras` - 深度学习框架
- `faiss-cpu` - 向量检索
- `akshare` - A股数据获取
- `langchain` - AI对话框架
- `plotly` - 可视化
- `scipy` - 科学计算（组合优化）

### 💡 使用示例

#### 单只股票深度分析
```python
# 在Web界面输入股票代码
symbol = "600519"  # 贵州茅台
# 点击"开始分析"
# 系统将返回：
# - 视觉相似形态（Top 5）
# - 综合评分（0-10分）
# - 买入/观望/卖出建议
# - AI智能解读
```

#### 批量组合分析
```python
# 输入多只股票代码（每行一个）
symbols = """
600519
000858
601899
600036
...
"""
# 设置参数：
# - 最大持仓数：10只
# - 单只最小/最大仓位：5%-20%
# 系统将输出：
# - 核心推荐组合（评分≥7）
# - 备选增强组合（评分≥6）
# - 最优仓位配置
# - 组合预期收益/风险/夏普比率
```

#### 策略回测
```python
# 设置回测参数
start_date = "2022-01-01"
end_date = "2026-01-07"
initial_capital = 100000

# VQ策略自动运行
# 输出：
# - 策略收益曲线
# - 策略收益率 vs 基准收益率
# - Alpha、交易次数
```

### 🎯 VQ策略说明

**VQ = Vision Quant（视觉量化）**

VQ策略是一个自适应双模态策略：

#### 牛市模式（价格 > MA60）
- **强趋势锁仓**：MACD>0 or 价格>MA20 → 100%仓位
- **回调持仓**：AI胜率≥57% → 81%仓位
- **破位离场**：否则 → 0%仓位

#### 熊市模式（价格 < MA60）
- **视觉狙击**：AI胜率≥59% → 50%仓位
- **避险模式**：否则 → 3%仓位

#### 风险控制
- 硬止损：8%
- 基本面熔断：ROE < -20%禁止买入
- 动态仓位：0%-100%自适应调整

### 📊 历史表现

| 股票代码 | 股票名称 | 回测期间 | VQ策略 | 买入持有 | Alpha |
|---------|---------|---------|--------|---------|-------|
| 601899  | 紫金矿业 | 2023年  | +45.2% | +28.5%  | +16.7% |
| 600519  | 贵州茅台 | 2023年  | +38.7% | +22.1%  | +16.6% |
| 000858  | 五粮液   | 2023年  | +32.1% | +18.9%  | +13.2% |

*注：历史业绩不代表未来收益，仅供参考*

### 🔬 技术创新

1. **K线形态向量化**
   - 将K线图转为224×224 RGB图像
   - CAE编码为128维特征向量
   - 支持百万级形态库检索

2. **混合相似度算法**
   - 图像特征相似度（FAISS L2距离）
   - 价格序列相关性（Pearson相关系数）
   - 加权融合：70%相关性 + 30%特征距离

3. **三层分级组合**
   - 核心推荐（评分≥7，action=BUY）
   - 备选增强（评分≥6，action≠SELL）
   - 自适应配置策略

4. **马科维茨优化**
   - 最大化夏普比率
   - 期望收益 = 胜率 × 预期收益
   - 协方差矩阵基于60日历史收益率

### ⚠️ 免责声明

本项目仅供学习研究使用，不构成任何投资建议。股市有风险，投资需谨慎。使用本系统进行实盘交易的任何损失，作者不承担责任。

### 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

### 🤝 贡献指南

欢迎贡献代码、报告Bug或提出新功能建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

### 📮 联系方式

- GitHub Issues: [提交问题](https://github.com/panyisheng095-ux/VisionQuant-Pro/issues)
- 邮箱: [your-email@example.com]

### 🌟 Star History

如果这个项目对你有帮助，请给个 ⭐️ Star 支持一下！

---

## English

### 📖 Introduction

VisionQuant-Pro is an innovative quantitative trading system that deeply integrates **Computer Vision** with **Quantitative Trading**. The system uses Convolutional Autoencoders (CAE) to learn candlestick chart pattern features, combined with FAISS vector retrieval technology, to achieve rapid identification and prediction of historically similar patterns.

### ✨ Key Features

- **Deep Learning Pattern Recognition**: Automatic K-line visual feature extraction using CAE
- **FAISS Similarity Search**: Millisecond-level retrieval of million-scale historical pattern database
- **Multi-Factor Scoring**: Integration of visual, technical, and fundamental dimensions
- **Adaptive Strategy**: Bull/Bear market dual-mode strategy switching
- **Batch Portfolio Analysis**: Support for parallel analysis of 30 stocks with Markowitz optimization
- **AI Chat Assistant**: Intelligent investment advisor based on LangChain and Google Gemini
- **VQ Strategy Backtesting**: Adaptive position management strategy combining vision and trends

### 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/panyisheng095-ux/VisionQuant-Pro.git
cd VisionQuant-Pro

# Install dependencies
pip install -r requirements.txt

# Launch Web interface
streamlit run web/app.py
```

Visit: http://localhost:8501

### 📊 Performance

| Stock Code | Name | Period | VQ Strategy | Buy & Hold | Alpha |
|-----------|------|--------|------------|-----------|-------|
| 601899 | Zijin Mining | 2023 | +45.2% | +28.5% | +16.7% |
| 600519 | Kweichow Moutai | 2023 | +38.7% | +22.1% | +16.6% |

*Past performance does not guarantee future results*

### 📄 License

This project is licensed under the [MIT License](LICENSE).

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">

**If you find this project helpful, please give it a ⭐️ Star!**

Made with ❤️ by [panyisheng095-ux](https://github.com/panyisheng095-ux)

</div>
