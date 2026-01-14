"""
AttentionCAE: Self-Attention Enhanced Convolutional Autoencoder
用于K线图形态识别的注意力增强卷积自编码器

创新点:
1. 在CNN Encoder末端添加Multi-Head Self-Attention
2. 捕捉K线图中的长距离依赖（如头肩顶的三个峰值）
3. 注意力权重可视化，提供可解释性

Author: Yisheng Pan
Date: 2026-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力模块
    
    用于捕捉K线图中不同区域之间的全局依赖关系
    例如：
    - 头肩顶的"左肩"和"右肩"之间的对称性
    - 双底形态中两个谷底的相似性
    - 上升三角形中多个触顶点的关系
    """
    
    def __init__(self, in_channels: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            in_channels: 输入特征通道数
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        
        assert in_channels % num_heads == 0, \
            f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V 投影
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 输出投影
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Layer Norm 和 Dropout
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            out: 输出特征图 [B, C, H, W]
            attn_weights: 注意力权重 [B, num_heads, H*W, H*W]
        """
        B, C, H, W = x.shape
        
        # 生成 Q, K, V
        q = self.query(x)  # [B, C, H, W]
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        # [B, C, H, W] -> [B, num_heads, head_dim, H*W]
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)
        
        # 计算注意力分数
        # [B, num_heads, H*W, head_dim] @ [B, num_heads, head_dim, H*W]
        # -> [B, num_heads, H*W, H*W]
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        # [B, num_heads, head_dim, H*W] @ [B, num_heads, H*W, H*W]
        # -> [B, num_heads, head_dim, H*W]
        out = torch.matmul(v, attn_weights.transpose(-2, -1))
        
        # Reshape back
        out = out.view(B, C, H, W)
        out = self.out_proj(out)
        
        # 残差连接
        out = out + x
        
        # Layer Norm (需要permute)
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return out, attn_weights


class AttentionCAE(nn.Module):
    """
    带Self-Attention的卷积自编码器
    
    架构:
    ┌─────────────────────────────────────────┐
    │  Input: 224×224×3 (K线图)               │
    │           │                              │
    │           ▼                              │
    │  CNN Encoder (4层卷积)                   │
    │  224→112→56→28→14, channels: 32→64→128→256
    │           │                              │
    │           ▼                              │
    │  ★ Multi-Head Self-Attention ★          │
    │  14×14×256, 8 heads                      │
    │           │                              │
    │           ▼                              │
    │  Adaptive Pooling + Linear               │
    │  50176 → 1024 dim (latent vector)        │
    │           │                              │
    │           ▼                              │
    │  CNN Decoder (4层转置卷积)               │
    │  14→28→56→112→224                        │
    │           │                              │
    │           ▼                              │
    │  Output: 224×224×3 (重建图像)            │
    └─────────────────────────────────────────┘
    
    训练目标: 最小化重建损失 MSE(input, output)
    """
    
    def __init__(
        self, 
        latent_dim: int = 1024, 
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        use_attention: bool = True,
        feature_dim: int = 256
    ):
        """
        Args:
            latent_dim: 隐空间维度（支持512, 1024, 2048）
            num_attention_heads: 注意力头数
            dropout: Dropout比率
            use_attention: 是否使用注意力模块（用于消融实验）
            feature_dim: 编码器最后一层的特征通道数（默认256，可提升到512）
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.feature_dim = feature_dim
        
        # ========== Encoder ==========
        self.encoder = nn.Sequential(
            # Layer 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 28 -> 14
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # ========== Self-Attention ==========
        if use_attention:
            self.attention = MultiHeadSelfAttention(
                in_channels=feature_dim, 
                num_heads=num_attention_heads,
                dropout=dropout
            )
        
        # ========== Latent Projection ==========
        # 14×14×feature_dim -> latent_dim
        # 支持渐进式维度提升：256->512->1024->2048
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, feature_dim, 1, 1]
            nn.Flatten(),                   # [B, feature_dim]
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # ========== Decoder ==========
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, feature_dim * 14 * 14),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.decoder = nn.Sequential(
            # Layer 1: 14 -> 28
            nn.ConvTranspose2d(feature_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 28 -> 56
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 56 -> 112
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 112 -> 224
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        
        # 保存最近的注意力权重（用于可视化）
        self._last_attention_weights = None
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码：图像 -> 特征向量
        
        Args:
            x: 输入图像 [B, 3, 224, 224]
            
        Returns:
            latent: L2归一化的特征向量 [B, latent_dim]
        """
        # CNN编码
        features = self.encoder(x)  # [B, feature_dim, 14, 14]
        
        # 应用注意力
        if self.use_attention:
            features, attn_weights = self.attention(features)
            self._last_attention_weights = attn_weights
        
        # 投影到隐空间
        latent = self.to_latent(features)  # [B, latent_dim]
        
        # L2归一化（用于余弦相似度检索）
        latent = F.normalize(latent, p=2, dim=1)
        
        return latent
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码：特征向量 -> 重建图像
        
        Args:
            z: 特征向量 [B, latent_dim]
            
        Returns:
            recon: 重建图像 [B, 3, 224, 224]
        """
        x = self.from_latent(z)  # [B, feature_dim*14*14]
        x = x.view(-1, self.feature_dim, 14, 14)  # [B, feature_dim, 14, 14]
        recon = self.decoder(x)  # [B, 3, 224, 224]
        return recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, 224, 224]
            
        Returns:
            recon: 重建图像 [B, 3, 224, 224]
            latent: 特征向量 [B, latent_dim]
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入图像 [B, 3, 224, 224]
            
        Returns:
            attn_weights: 注意力权重 [B, num_heads, H*W, H*W]
        """
        if not self.use_attention:
            raise ValueError("Attention is disabled. Set use_attention=True")
        
        self.eval()
        with torch.no_grad():
            features = self.encoder(x)
            _, attn_weights = self.attention(features)
        
        return attn_weights
    
    def get_attention_map(self, x: torch.Tensor, head_idx: int = 0) -> np.ndarray:
        """
        获取单个图像的注意力热力图
        
        Args:
            x: 输入图像 [1, 3, 224, 224] 或 [3, 224, 224]
            head_idx: 选择哪个注意力头
            
        Returns:
            attention_map: 注意力热力图 [224, 224]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        attn = self.get_attention_weights(x)  # [1, num_heads, 196, 196]
        
        # 取中心点的注意力分布
        center_idx = 98  # 14*14/2 ≈ 98 (中心点)
        attn_map = attn[0, head_idx, center_idx, :].cpu().numpy()  # [196]
        attn_map = attn_map.reshape(14, 14)
        
        # 上采样到原图大小
        attn_map = np.kron(attn_map, np.ones((16, 16)))  # [224, 224]
        
        return attn_map


class AttentionCAETrainer:
    """
    AttentionCAE 训练器
    """
    
    def __init__(
        self,
        model: AttentionCAE,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(self.device)
            
            # 前向传播
            recon, latent = self.model(images)
            
            # 计算损失
            loss = self.criterion(recon, images)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0
        
        for images, _ in dataloader:
            images = images.to(self.device)
            recon, _ = self.model(images)
            loss = self.criterion(recon, images)
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"✅ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['loss']


# ============================================================
# 兼容性：保持与原有代码的接口一致
# ============================================================

class QuantCAE(AttentionCAE):
    """
    兼容性别名
    保持与原有 autoencoder.py 中 QuantCAE 的接口一致
    """
    
    def __init__(self, latent_dim: int = 1024, use_attention: bool = True):
        super().__init__(
            latent_dim=latent_dim,
            num_attention_heads=8,
            dropout=0.1,
            use_attention=use_attention
        )


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing AttentionCAE")
    print("=" * 60)
    
    # 创建模型
    model = AttentionCAE(latent_dim=1024, num_attention_heads=8)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试前向传播
    x = torch.randn(4, 3, 224, 224)
    recon, latent = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    
    # 测试注意力权重提取
    attn_weights = model.get_attention_weights(x)
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # 测试注意力热力图
    attn_map = model.get_attention_map(x[0])
    print(f"Attention map shape: {attn_map.shape}")
    
    # 测试无注意力版本（消融实验用）
    model_no_attn = AttentionCAE(latent_dim=1024, use_attention=False)
    recon_no_attn, latent_no_attn = model_no_attn(x)
    print(f"\nWithout Attention:")
    print(f"Latent shape: {latent_no_attn.shape}")
    
    print("\n✅ All tests passed!")
