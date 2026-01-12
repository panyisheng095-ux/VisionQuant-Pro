# VisionQuant-Pro 常见问题解答 (FAQ)

## 🔥 热门问题

### Q1: `ModuleNotFoundError: No module named 'src.data'`

**问题原因：** Python路径未正确设置，找不到src模块。

**解决方案：**

```bash
# 方法一：使用启动脚本（推荐）
python run.py

# 方法二：手动设置PYTHONPATH
cd VisionQuant-Pro  # 确保在项目根目录
PYTHONPATH=. streamlit run web/app.py

# 方法三：Windows用户
set PYTHONPATH=. && streamlit run web/app.py
```

---

### Q2: `ModuleNotFoundError: No module named 'streamlit_mic_recorder'`

**问题原因：** 依赖未完整安装。

**解决方案：**

```bash
pip install streamlit-mic-recorder
```

或重新安装所有依赖：

```bash
pip install -r requirements.txt
```

---

### Q3: AI对话功能不可用

**问题原因：** 未配置Google API Key。

**解决方案：**

1. 获取API Key：https://makersuite.google.com/app/apikey
2. 创建 `.env` 文件：
   ```bash
   echo "GOOGLE_API_KEY=your_key_here" > .env
   ```
3. 重启应用

**注意：** AI对话功能是可选的，不配置也可以使用其他功能。

---

### Q4: 数据目录为空 / 无法分析股票

**问题原因：** 完整数据集不包含在仓库中（154GB太大）。

**解决方案：**

```bash
# 运行数据准备脚本，下载示例数据
python scripts/prepare_data.py
```

示例数据包含5只股票用于快速体验。完整数据需自行训练生成。

---

### Q5: FAISS索引加载失败

**问题原因：** 索引文件不存在或损坏。

**解决方案：**

1. 检查 `data/indices/` 目录是否存在索引文件
2. 重新运行数据准备脚本
3. 如果仍有问题，尝试重新训练CAE模型

---

### Q6: `faiss-cpu` 安装失败

**问题原因：** 系统兼容性问题。

**解决方案：**

```bash
# 尝试指定版本
pip install faiss-cpu==1.7.4

# Mac M1/M2用户
pip install faiss-cpu --no-binary faiss-cpu

# 或使用conda
conda install -c conda-forge faiss-cpu
```

---

### Q7: Streamlit启动报错 `Address already in use`

**问题原因：** 8501端口被占用。

**解决方案：**

```bash
# 指定其他端口
streamlit run web/app.py --server.port 8502

# 或找到并杀死占用进程
lsof -i :8501  # 查看占用进程
kill -9 <PID>  # 杀死进程
```

---

### Q8: 语音识别不工作

**问题原因：** 
1. 浏览器未授权麦克风权限
2. Google API Key未配置

**解决方案：**

1. 检查浏览器是否允许麦克风访问
2. 确认 `.env` 文件中有 `GOOGLE_API_KEY`
3. 语音功能需要网络连接（调用Gemini API）

---

### Q9: 回测结果显示为直线

**问题原因：** VQ策略没有产生交易信号。

**可能原因：**
1. 历史数据不足（需要至少60天MA计算）
2. 股票数据缺失
3. AI胜率数据未正确传递

**解决方案：**
1. 确保选择的股票在数据库中有K线图像
2. 选择数据更完整的股票（如600519）
3. 检查控制台是否有错误日志

---

### Q10: 批量分析无法生成组合

**问题原因：** 所有股票评分都低于阈值。

**解决方案：**

系统采用三层分级机制：
- **核心推荐**：评分≥7 且 action=BUY
- **备选增强**：评分≥6 且 action≠SELL
- **观望列表**：其他

如果没有符合条件的股票，可以：
1. 输入更多股票代码
2. 调整市场预期
3. 查看"观望列表"中的潜力股

---

## 🛠️ 性能优化

### 提升分析速度

1. **减少相似形态数量**：默认Top 10，可改为Top 5
2. **使用GPU**：如有NVIDIA GPU，安装 `faiss-gpu`
3. **增加内存**：至少8GB RAM，推荐16GB

### 减少内存占用

1. 使用较小的批处理大小
2. 分析完成后手动清理缓存
3. 不要同时分析超过30只股票

---

## 📧 联系我们

如果以上方案无法解决你的问题：

1. **GitHub Issues**: [提交问题](https://github.com/panyisheng095-ux/VisionQuant-Pro/issues)
2. **邮箱**: panyisheng095@gmail.com

提交Issue时请包含：
- 操作系统和Python版本
- 完整的错误日志
- 复现步骤
