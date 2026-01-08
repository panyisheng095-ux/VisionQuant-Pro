# 贡献指南 | Contributing Guide

感谢你对 VisionQuant-Pro 的关注！我们欢迎任何形式的贡献。

## 如何贡献

### 报告Bug

如果你发现了Bug，请通过 [Issues](https://github.com/panyisheng095-ux/VisionQuant-Pro/issues) 报告，并提供：

1. Bug描述
2. 复现步骤
3. 预期行为
4. 实际行为
5. 环境信息（Python版本、操作系统等）

### 提出新功能

如果你有好的想法，请通过 Issues 提出，并说明：

1. 功能描述
2. 使用场景
3. 预期效果

### 提交代码

1. **Fork 仓库**
   ```bash
   # 点击页面右上角的 Fork 按钮
   ```

2. **克隆到本地**
   ```bash
   git clone https://github.com/你的用户名/VisionQuant-Pro.git
   cd VisionQuant-Pro
   ```

3. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **编写代码**
   - 遵循 PEP 8 代码规范
   - 添加必要的注释
   - 更新相关文档

5. **测试代码**
   ```bash
   # 确保代码能正常运行
   python -m pytest tests/
   ```

6. **提交更改**
   ```bash
   git add .
   git commit -m "feat: 添加XXX功能"
   ```

   提交信息规范：
   - `feat`: 新功能
   - `fix`: 修复Bug
   - `docs`: 文档更新
   - `style`: 代码格式调整
   - `refactor`: 代码重构
   - `test`: 测试相关
   - `chore`: 其他修改

7. **推送到GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **提交 Pull Request**
   - 在GitHub上创建 Pull Request
   - 描述你的修改内容
   - 等待审核

## 代码规范

### Python 代码风格

- 使用 4 个空格缩进
- 每行不超过 100 个字符
- 使用有意义的变量名
- 添加类型提示（Type Hints）

示例：
```python
def calculate_score(price: float, volume: int) -> float:
    """
    计算股票评分
    
    Args:
        price: 股票价格
        volume: 成交量
    
    Returns:
        评分 (0-10)
    """
    score = (price * volume) / 1000000
    return min(score, 10.0)
```

### 文档规范

- 所有函数都要有 docstring
- 使用中英文双语注释（重要部分）
- 更新 README.md 中的相关内容

## 项目结构

```
VisionQuant-Pro/
├── src/              # 核心代码
│   ├── models/       # 模型相关
│   ├── strategies/   # 策略相关
│   ├── factors/      # 因子相关
│   └── utils/        # 工具函数
├── web/              # Web界面
├── data/             # 数据文件
├── tests/            # 测试代码
├── docs/             # 文档
└── configs/          # 配置文件
```

## 开发环境设置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements-dev.txt

# 安装 pre-commit hooks
pre-commit install
```

## 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_vision_engine.py

# 生成覆盖率报告
pytest --cov=src tests/
```

## 问题求助

如果你在贡献过程中遇到任何问题，请：

1. 查看 [文档](docs/)
2. 搜索 [Issues](https://github.com/panyisheng095-ux/VisionQuant-Pro/issues)
3. 提出新的 Issue

## 行为准则

- 尊重所有贡献者
- 包容不同观点
- 专注于项目本身
- 保持友善和专业

---

再次感谢你的贡献！🎉
