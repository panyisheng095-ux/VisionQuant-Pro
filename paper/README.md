# VisionQuant-Pro 论文目录

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `visionquant_arxiv.tex` | arXiv论文主文件（LaTeX） |
| `figures/` | 论文图表目录（待创建） |
| `references.bib` | 参考文献（可选，当前内嵌） |

## 🚀 编译方法

### 方法1：在线编译（推荐）

1. 访问 [Overleaf](https://www.overleaf.com/)
2. 创建新项目 → 上传 `visionquant_arxiv.tex`
3. 点击编译

### 方法2：本地编译

```bash
# 需要安装 TeX Live
pdflatex visionquant_arxiv.tex
pdflatex visionquant_arxiv.tex  # 运行两次生成目录和引用
```

## 📋 投稿步骤

### arXiv 投稿

1. 注册 arXiv 账号：https://arxiv.org/register
2. 选择分类：
   - **主分类**: `q-fin.TR` (Quantitative Finance - Trading)
   - **次分类**: `cs.LG` (Machine Learning)
3. 上传 `.tex` 文件和图片
4. 填写元数据（标题、摘要、作者）
5. 提交审核（通常1-2天通过）

### 会议投稿（ICAIF/KDD）

1. 访问会议官网查看截止日期
2. 按会议模板格式化论文
3. 提交 PDF 和补充材料

## ✅ 论文自查清单

- [ ] Abstract 150-250词
- [ ] 有 Related Work 章节
- [ ] 有 Baseline 对比实验
- [ ] 有统计显著性检验（p-value）
- [ ] 有消融实验
- [ ] 有超参数敏感性分析
- [ ] 图表有 caption 和引用
- [ ] 参考文献格式正确
- [ ] 代码链接可访问

## 📊 需要补充的实验数据

运行以下脚本生成论文所需数据：

```python
# 1. 运行对比实验
python -c "
from src.strategies.baseline_experiments import ExperimentFramework
# 运行实验...
"

# 2. 生成注意力可视化图
python -c "
from src.utils.attention_visualizer import AttentionVisualizer
# 生成图表...
"
```

## 📅 投稿时间建议

| 目标 | 截止时间 | 准备周期 |
|------|---------|---------|
| arXiv | 随时 | 1-2周 |
| ICAIF 2026 | ~2026年4月 | 2-3月 |
| KDD 2026 | ~2026年2月 | 3-4月 |
| NeurIPS 2026 | ~2026年5月 | 4-5月 |
