"""
多尺度模型训练脚本
Multi-Scale Model Training Script

训练日线/周线/月线K线图模型

Author: VisionQuant Team
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.attention_cae import AttentionCAE
from src.data.multi_scale_generator import MultiScaleChartGenerator
from src.data.data_loader import DataLoader as StockDataLoader


class MultiScaleDataset(Dataset):
    """
    多尺度K线图数据集
    """
    
    def __init__(
        self,
        stock_list: list,
        data_loader: StockDataLoader,
        chart_generator: MultiScaleChartGenerator,
        scale: str = 'daily'  # 'daily', 'weekly', 'monthly'
    ):
        """
        Args:
            stock_list: 股票代码列表
            data_loader: 数据加载器
            chart_generator: K线图生成器
            scale: 时间尺度
        """
        self.stock_list = stock_list
        self.data_loader = data_loader
        self.chart_generator = chart_generator
        self.scale = scale
        
        # 预生成所有图表（或按需生成）
        self.chart_paths = []
        self._prepare_data()
    
    def _prepare_data(self):
        """准备数据"""
        print(f"📊 准备{self.scale}尺度数据...")
        for symbol in tqdm(self.stock_list, desc="生成图表"):
            try:
                df = self.data_loader.get_stock_data(symbol)
                if df.empty:
                    continue
                
                # 生成对应尺度的图表
                if self.scale == 'daily':
                    chart_path = self.chart_generator.generate_daily_chart(df)
                elif self.scale == 'weekly':
                    chart_path = self.chart_generator.generate_weekly_chart(df)
                elif self.scale == 'monthly':
                    chart_path = self.chart_generator.generate_monthly_chart(df)
                else:
                    continue
                
                self.chart_paths.append({
                    'symbol': symbol,
                    'chart_path': chart_path
                })
            except Exception as e:
                print(f"⚠️ 处理 {symbol} 失败: {e}")
                continue
    
    def __len__(self):
        return len(self.chart_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        from torchvision import transforms
        
        item = self.chart_paths[idx]
        chart_path = item['chart_path']
        
        # 加载图像
        img = Image.open(chart_path).convert('RGB')
        
        # 预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        img_tensor = transform(img)
        
        return img_tensor, item['symbol']


def train_multi_scale_model(
    scale: str = 'daily',
    latent_dim: int = 2048,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    stock_list: list = None
):
    """
    训练多尺度模型
    
    Args:
        scale: 时间尺度 ('daily', 'weekly', 'monthly')
        latent_dim: 特征维度
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        stock_list: 股票列表（如果为None，使用默认列表）
    """
    print(f"🚀 开始训练{scale}尺度模型...")
    print(f"   特征维度: {latent_dim}")
    print(f"   批次大小: {batch_size}")
    print(f"   训练轮数: {epochs}")
    
    # 设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 使用 Apple MPS GPU 加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 使用 CUDA GPU 加速")
    else:
        device = torch.device("cpu")
        print("💻 使用 CPU")
    
    # 数据加载
    data_loader = StockDataLoader()
    chart_generator = MultiScaleChartGenerator()
    
    if stock_list is None:
        # 使用默认股票列表（Top 100）
        top_stocks = data_loader.get_top300_stocks()
        stock_list = top_stocks['code'].head(100).tolist()
    
    # 数据集
    dataset = MultiScaleDataset(
        stock_list=stock_list,
        data_loader=data_loader,
        chart_generator=chart_generator,
        scale=scale
    )
    
    if len(dataset) == 0:
        print("❌ 数据集为空，无法训练")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # 模型
    model = AttentionCAE(
        latent_dim=latent_dim,
        feature_dim=512 if latent_dim >= 2048 else 256  # 高维度时增加特征通道
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, symbols) in enumerate(progress_bar):
            images = images.to(device)
            
            # 前向传播
            recon, latent = model(images)
            
            # 重建损失
            loss = criterion(recon, images)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.4f}")
    
    # 保存模型
    model_dir = os.path.join(PROJECT_ROOT, "data", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(
        model_dir,
        f"attention_cae_{scale}_{latent_dim}d.pth"
    )
    
    torch.save(model.state_dict(), model_path)
    print(f"✅ 模型已保存: {model_path}")
    
    return model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练多尺度K线图模型')
    parser.add_argument('--scale', type=str, default='daily', choices=['daily', 'weekly', 'monthly'])
    parser.add_argument('--latent_dim', type=int, default=2048, choices=[512, 1024, 2048])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    train_multi_scale_model(
        scale=args.scale,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
