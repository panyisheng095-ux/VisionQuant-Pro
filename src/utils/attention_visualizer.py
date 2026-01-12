"""
Attention Visualizer - 注意力可视化工具

用于生成论文中的可解释性分析图表:
1. 注意力热力图 (Attention Heatmap)
2. 多头注意力对比图
3. 形态识别案例分析

这对arXiv论文非常重要！可以展示:
- 模型在识别"双底"时关注两个底部
- 模型在识别"头肩顶"时关注三个峰值
- 增强论文的"Interpretability"（可解释性）

Author: Yisheng Pan
Date: 2026-01
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import Optional, List, Tuple
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AttentionVisualizer:
    """
    注意力可视化工具
    
    用于生成论文图表，展示模型的可解释性
    """
    
    def __init__(self, model, device: str = 'cpu'):
        """
        Args:
            model: AttentionCAE模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
    @torch.no_grad()
    def get_attention_weights(self, image: torch.Tensor) -> np.ndarray:
        """
        获取注意力权重
        
        Args:
            image: 输入图像 [3, 224, 224] 或 [1, 3, 224, 224]
            
        Returns:
            attention_weights: [num_heads, H*W, H*W]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        attn = self.model.get_attention_weights(image)
        
        return attn[0].cpu().numpy()  # [num_heads, 196, 196]
    
    def visualize_single_attention(
        self,
        image: torch.Tensor,
        head_idx: int = 0,
        query_pos: Tuple[int, int] = (7, 7),
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        可视化单个查询位置的注意力分布
        
        Args:
            image: 输入图像
            head_idx: 注意力头索引
            query_pos: 查询位置 (row, col) in 14x14 grid
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            fig: matplotlib图表对象
        """
        attn = self.get_attention_weights(image)  # [num_heads, 196, 196]
        
        # 获取指定位置的注意力分布
        query_idx = query_pos[0] * 14 + query_pos[1]
        attn_map = attn[head_idx, query_idx, :]  # [196]
        attn_map = attn_map.reshape(14, 14)
        
        # 上采样到原图大小
        attn_map_large = np.kron(attn_map, np.ones((16, 16)))
        
        # 转换图像格式
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(image)
        
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原图
        axes[0].imshow(img_np)
        # 标记查询位置
        rect = patches.Rectangle(
            (query_pos[1] * 16, query_pos[0] * 16), 
            16, 16, 
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        axes[0].add_patch(rect)
        axes[0].set_title("Original Chart (Query Position in Red)", fontsize=12)
        axes[0].axis('off')
        
        # 注意力热力图
        im = axes[1].imshow(attn_map_large, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f"Attention Map (Head {head_idx})", fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 叠加图
        axes[2].imshow(img_np)
        axes[2].imshow(attn_map_large, cmap='hot', alpha=0.6, interpolation='bilinear')
        axes[2].set_title("Overlay", fontsize=12)
        axes[2].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved to {save_path}")
        
        return fig
    
    def visualize_multi_head_attention(
        self,
        image: torch.Tensor,
        query_pos: Tuple[int, int] = (7, 7),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        可视化所有注意力头的对比
        
        用于论文展示不同头学习到不同的形态特征
        
        Args:
            image: 输入图像
            query_pos: 查询位置
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图表对象
        """
        attn = self.get_attention_weights(image)  # [num_heads, 196, 196]
        num_heads = attn.shape[0]
        
        # 转换图像
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(image)
        
        # 创建图表
        rows = 2
        cols = (num_heads + 2) // 2 + 1  # +1 for original image
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()
        
        # 原图
        axes[0].imshow(img_np)
        query_idx = query_pos[0] * 14 + query_pos[1]
        rect = patches.Rectangle(
            (query_pos[1] * 16, query_pos[0] * 16), 
            16, 16, 
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        axes[0].add_patch(rect)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis('off')
        
        # 各个头的注意力
        for i in range(num_heads):
            attn_map = attn[i, query_idx, :].reshape(14, 14)
            attn_map_large = np.kron(attn_map, np.ones((16, 16)))
            
            ax = axes[i + 1]
            ax.imshow(img_np)
            ax.imshow(attn_map_large, cmap='hot', alpha=0.6, interpolation='bilinear')
            ax.set_title(f"Head {i}", fontsize=10)
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_heads + 1, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle("Multi-Head Attention Visualization", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved to {save_path}")
        
        return fig
    
    def visualize_pattern_analysis(
        self,
        image: torch.Tensor,
        pattern_name: str,
        key_regions: List[Tuple[int, int]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        形态分析可视化
        
        用于论文展示模型如何识别特定形态（如双底、头肩顶）
        
        Args:
            image: 输入图像
            pattern_name: 形态名称（如"Double Bottom", "Head and Shoulders"）
            key_regions: 关键区域列表 [(row, col), ...]
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图表对象
        """
        attn = self.get_attention_weights(image)  # [num_heads, 196, 196]
        
        # 转换图像
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(image)
        
        # 创建图表
        num_regions = len(key_regions)
        fig, axes = plt.subplots(2, num_regions + 1, figsize=((num_regions + 1) * 4, 8))
        
        # 顶行：原图 + 关键区域标注
        axes[0, 0].imshow(img_np)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (r, c) in enumerate(key_regions):
            rect = patches.Rectangle(
                (c * 16, r * 16), 16, 16,
                linewidth=3,
                edgecolor=colors[i % len(colors)],
                facecolor='none'
            )
            axes[0, 0].add_patch(rect)
            axes[0, 0].text(
                c * 16 + 8, r * 16 - 5,
                f"R{i+1}",
                color=colors[i % len(colors)],
                fontsize=10,
                fontweight='bold',
                ha='center'
            )
        axes[0, 0].set_title(f"Pattern: {pattern_name}", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 顶行：各关键区域的注意力分布
        for i, (r, c) in enumerate(key_regions):
            query_idx = r * 14 + c
            # 平均所有头的注意力
            attn_map = attn.mean(axis=0)[query_idx, :].reshape(14, 14)
            attn_map_large = np.kron(attn_map, np.ones((16, 16)))
            
            axes[0, i + 1].imshow(img_np)
            axes[0, i + 1].imshow(attn_map_large, cmap='hot', alpha=0.6)
            axes[0, i + 1].set_title(f"Region {i+1} Attention", fontsize=10)
            axes[0, i + 1].axis('off')
        
        # 底行：区域间互注意力分析
        axes[1, 0].set_title("Cross-Region Attention", fontsize=12)
        axes[1, 0].axis('off')
        
        # 计算区域间注意力矩阵
        region_indices = [r * 14 + c for r, c in key_regions]
        cross_attn = np.zeros((num_regions, num_regions))
        for i, qi in enumerate(region_indices):
            for j, kj in enumerate(region_indices):
                cross_attn[i, j] = attn.mean(axis=0)[qi, kj]
        
        # 热力图
        im = axes[1, 1].imshow(cross_attn, cmap='YlOrRd')
        axes[1, 1].set_xticks(range(num_regions))
        axes[1, 1].set_yticks(range(num_regions))
        axes[1, 1].set_xticklabels([f"R{i+1}" for i in range(num_regions)])
        axes[1, 1].set_yticklabels([f"R{i+1}" for i in range(num_regions)])
        axes[1, 1].set_title("Cross-Region Attention Matrix", fontsize=10)
        plt.colorbar(im, ax=axes[1, 1])
        
        # 隐藏多余子图
        for i in range(2, num_regions + 1):
            axes[1, i].axis('off')
        
        fig.suptitle(f"Pattern Analysis: {pattern_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved to {save_path}")
        
        return fig
    
    def generate_paper_figures(
        self,
        images: List[torch.Tensor],
        patterns: List[str],
        output_dir: str
    ):
        """
        批量生成论文所需的所有图表
        
        Args:
            images: 图像列表
            patterns: 对应的形态名称列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (img, pattern) in enumerate(zip(images, patterns)):
            # 单头注意力图
            self.visualize_single_attention(
                img,
                head_idx=0,
                save_path=os.path.join(output_dir, f"attention_single_{i}.pdf")
            )
            
            # 多头对比图
            self.visualize_multi_head_attention(
                img,
                save_path=os.path.join(output_dir, f"attention_multihead_{i}.pdf")
            )
        
        print(f"✅ Generated {len(images) * 2} figures in {output_dir}")


def create_attention_comparison_figure(
    model_with_attn,
    model_without_attn,
    image: torch.Tensor,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    创建有/无注意力模型的对比图（用于消融实验）
    
    Args:
        model_with_attn: 带注意力的模型
        model_without_attn: 不带注意力的模型
        image: 输入图像
        save_path: 保存路径
        
    Returns:
        fig: matplotlib图表对象
    """
    # 转换图像
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # 获取重建结果
    model_with_attn.eval()
    model_without_attn.eval()
    
    with torch.no_grad():
        recon_with, _ = model_with_attn(image)
        recon_without, _ = model_without_attn(image)
    
    # 转换为numpy
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    recon_with_np = recon_with[0].permute(1, 2, 0).cpu().numpy()
    recon_without_np = recon_without[0].permute(1, 2, 0).cpu().numpy()
    
    if img_np.max() <= 1:
        img_np = (img_np * 255).astype(np.uint8)
        recon_with_np = (recon_with_np * 255).astype(np.uint8)
        recon_without_np = (recon_without_np * 255).astype(np.uint8)
    
    # 计算重建误差
    mse_with = np.mean((img_np.astype(float) - recon_with_np.astype(float)) ** 2)
    mse_without = np.mean((img_np.astype(float) - recon_without_np.astype(float)) ** 2)
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(recon_without_np)
    axes[1].set_title(f"CAE (no attention)\nMSE: {mse_without:.2f}", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(recon_with_np)
    axes[2].set_title(f"CAE + Attention\nMSE: {mse_with:.2f}", fontsize=12)
    axes[2].axis('off')
    
    fig.suptitle("Reconstruction Comparison: Effect of Self-Attention", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved to {save_path}")
    
    return fig


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    from attention_cae import AttentionCAE
    
    print("=" * 60)
    print("Testing Attention Visualizer")
    print("=" * 60)
    
    # 创建模型
    model = AttentionCAE(latent_dim=1024, use_attention=True)
    
    # 创建测试图像
    test_image = torch.randn(3, 224, 224)
    
    # 创建可视化器
    visualizer = AttentionVisualizer(model)
    
    # 测试单头注意力可视化
    print("\nTesting single attention visualization...")
    fig = visualizer.visualize_single_attention(test_image, head_idx=0)
    plt.close(fig)
    print("✅ Single attention visualization passed")
    
    # 测试多头注意力可视化
    print("\nTesting multi-head attention visualization...")
    fig = visualizer.visualize_multi_head_attention(test_image)
    plt.close(fig)
    print("✅ Multi-head attention visualization passed")
    
    # 测试形态分析
    print("\nTesting pattern analysis...")
    fig = visualizer.visualize_pattern_analysis(
        test_image,
        pattern_name="Double Bottom",
        key_regions=[(10, 4), (10, 10)]  # 假设的两个底部位置
    )
    plt.close(fig)
    print("✅ Pattern analysis visualization passed")
    
    print("\n✅ All tests passed!")
