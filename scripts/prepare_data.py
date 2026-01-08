#!/usr/bin/env python3
"""
数据准备脚本
由于完整数据集过大（154GB），不包含在Git仓库中
运行此脚本可自动下载和生成所需数据
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader

def prepare_directories():
    """创建必要的目录结构"""
    dirs = [
        "data/raw",
        "data/indices",
        "data/indices/faiss_index",
        "data/cache",
        "data/images",
        "logs",
        "models"
    ]
    
    for d in dirs:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {d}")

def download_sample_data():
    """下载示例数据（部分股票）"""
    print("\n📥 开始下载示例数据...")
    print("提示：完整数据集需要自行运行训练流程生成")
    
    # 示例股票列表
    sample_symbols = [
        "600519",  # 贵州茅台
        "000858",  # 五粮液
        "601899",  # 紫金矿业
        "600036",  # 招商银行
        "000001",  # 平安银行
    ]
    
    loader = DataLoader()
    
    for symbol in sample_symbols:
        try:
            print(f"  下载 {symbol}...")
            df = loader.get_stock_data(symbol)
            if not df.empty:
                print(f"  ✅ {symbol}: {len(df)} 条数据")
            else:
                print(f"  ⚠️  {symbol}: 数据为空")
        except Exception as e:
            print(f"  ❌ {symbol}: {str(e)}")
    
    print("\n✅ 示例数据下载完成！")

def show_next_steps():
    """显示后续步骤"""
    print("\n" + "="*60)
    print("📋 后续步骤：")
    print("="*60)
    print("\n1️⃣  训练视觉模型（生成K线图特征）")
    print("   python src/models/train_cae.py")
    print("\n2️⃣  构建相似度索引")
    print("   python src/models/vision_engine.py --rebuild-index")
    print("\n3️⃣  启动Web界面")
    print("   streamlit run web/app.py")
    print("\n" + "="*60)
    print("⚠️  注意：完整训练需要较长时间（数小时到数天）")
    print("="*60)

if __name__ == "__main__":
    print("🚀 VisionQuant-Pro 数据准备脚本")
    print("="*60)
    
    # 1. 创建目录
    print("\n📁 步骤1：创建目录结构...")
    prepare_directories()
    
    # 2. 下载示例数据
    print("\n📥 步骤2：下载示例数据...")
    try:
        download_sample_data()
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print("提示：请检查网络连接和akshare库是否安装")
    
    # 3. 显示后续步骤
    show_next_steps()
    
    print("\n✅ 准备完成！")
