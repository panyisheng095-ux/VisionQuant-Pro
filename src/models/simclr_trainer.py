"""
SimCLR对比学习训练器
SimCLR Contrastive Learning Trainer

用于K线图的对比学习，提升特征表示质量

基于: Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations

Author: VisionQuant Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SimCLRTrainer:
    """
    SimCLR对比学习训练器
    
    核心思想：
    1. 对同一张K线图进行数据增强（得到两个view）
    2. 最大化两个view的相似度
    3. 最小化不同K线图的相似度
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 0.07,
        projection_dim: int = 128
    ):
        """
        初始化SimCLR训练器
        
        Args:
            model: 编码器模型（如AttentionCAE的encoder部分）
            temperature: 温度参数（控制softmax的平滑度）
            projection_dim: 投影层维度
        """
        self.model = model
        self.temperature = temperature
        
        # 投影头（将特征投影到对比学习空间）
        feature_dim = self._get_feature_dim()
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
    def _get_feature_dim(self) -> int:
        """获取模型特征维度"""
        # 尝试从模型获取
        if hasattr(self.model, 'latent_dim'):
            return self.model.latent_dim
        elif hasattr(self.model, 'feature_dim'):
            return self.model.feature_dim
        else:
            # 默认值
            return 1024
    
    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            z1: 第一个view的特征 [batch, projection_dim]
            z2: 第二个view的特征 [batch, projection_dim]
            
        Returns:
            对比学习损失
        """
        batch_size = z1.size(0)
        
        # L2归一化
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # 拼接所有特征
        all_features = torch.cat([z1, z2], dim=0)  # [2*batch, projection_dim]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(all_features, all_features.T)  # [2*batch, 2*batch]
        
        # 创建标签：正样本对（同一张图的两个view）的相似度应该高
        labels = torch.arange(batch_size).to(z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2*batch]
        
        # 计算正样本对的相似度（对角线上的元素）
        pos_sim = torch.diag(similarity_matrix, batch_size)  # 上对角线
        pos_sim = torch.cat([pos_sim, torch.diag(similarity_matrix, -batch_size)], dim=0)  # 下对角线
        
        # 计算负样本对的相似度（其他所有对）
        # 移除正样本对
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        mask = mask | torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device).roll(batch_size, dims=0)
        mask = mask | torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device).roll(-batch_size, dims=0)
        
        neg_sim = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # 计算InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.temperature
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x1: 第一个view的K线图 [batch, 3, 224, 224]
            x2: 第二个view的K线图 [batch, 3, 224, 224]
            
        Returns:
            loss: 对比学习损失
            z1: 第一个view的投影特征
            z2: 第二个view的投影特征
        """
        # 编码
        if hasattr(self.model, 'encode'):
            h1 = self.model.encode(x1)
            h2 = self.model.encode(x2)
        else:
            # 如果模型没有encode方法，使用forward
            h1, _ = self.model(x1)
            h2, _ = self.model(x2)
            # 如果返回的是特征图，需要pooling
            if len(h1.shape) > 2:
                h1 = F.adaptive_avg_pool2d(h1, (1, 1)).flatten(1)
                h2 = F.adaptive_avg_pool2d(h2, (1, 1)).flatten(1)
        
        # 投影
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        
        # 计算损失
        loss = self.contrastive_loss(z1, z2)
        
        return loss, z1, z2


class KLineAugmentation:
    """
    K线图数据增强
    
    用于SimCLR的对比学习
    """
    
    @staticmethod
    def augment(image: torch.Tensor) -> torch.Tensor:
        """
        对K线图进行增强
        
        增强方式：
        1. 随机水平翻转（模拟镜像形态）
        2. 轻微旋转（模拟视角变化）
        3. 颜色抖动（模拟不同显示风格）
        4. 随机裁剪（模拟不同时间窗口）
        """
        # 随机水平翻转（50%概率）
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[3])
        
        # 轻微旋转（-5度到+5度）
        angle = (torch.rand(1) - 0.5) * 10
        # 注意：这里简化处理，实际可以使用torchvision.transforms.RandomRotation
        
        # 颜色抖动（轻微调整亮度）
        brightness = 0.9 + torch.rand(1) * 0.2  # 0.9-1.1
        image = image * brightness
        image = torch.clamp(image, 0, 1)
        
        return image
    
    @staticmethod
    def get_augmented_pair(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一对增强后的图像
        
        Args:
            image: 原始K线图
            
        Returns:
            (aug1, aug2): 两个不同的增强版本
        """
        aug1 = KLineAugmentation.augment(image.clone())
        aug2 = KLineAugmentation.augment(image.clone())
        return aug1, aug2


if __name__ == "__main__":
    print("=== SimCLR对比学习训练器测试 ===")
    
    # 模拟模型
    from .attention_cae import AttentionCAE
    
    model = AttentionCAE(latent_dim=1024)
    trainer = SimCLRTrainer(model, temperature=0.07)
    
    # 模拟数据
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)
    
    # 前向传播
    loss, z1, z2 = trainer.forward(x1, x2)
    
    print(f"对比学习损失: {loss.item():.4f}")
    print(f"特征维度: z1={z1.shape}, z2={z2.shape}")
