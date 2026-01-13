"""
Grad-CAM 可视化模块
Gradient-weighted Class Activation Mapping

用于解释CNN模型的决策依据，在K线图上高亮显示模型关注的区域。

参考论文:
Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks 
via Gradient-based Localization" (ICCV 2017)

特点:
1. 不需要修改模型结构
2. 可应用于任何CNN模型
3. 生成热力图叠加在原图上
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional, List
import os


class GradCAM:
    """
    Grad-CAM 实现
    
    用法:
    ```python
    gradcam = GradCAM(model, target_layer='encoder_conv.9')
    heatmap = gradcam.generate(input_image)
    overlay = gradcam.overlay_heatmap(original_image, heatmap)
    ```
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        """
        初始化Grad-CAM
        
        Args:
            model: 目标模型（如AttentionCAE）
            target_layer: 目标层名称，用于计算梯度
        """
        self.model = model
        self.model.eval()
        
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self._register_hooks()
        
    def _register_hooks(self):
        """注册前向和反向钩子"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # 找到目标层
        if self.target_layer:
            target = self._get_layer(self.target_layer)
        else:
            # 默认使用最后一个卷积层
            target = self._find_last_conv()
            
        if target is not None:
            target.register_forward_hook(forward_hook)
            target.register_full_backward_hook(backward_hook)
        else:
            print("⚠️ 未找到目标层，Grad-CAM可能无法正常工作")
    
    def _get_layer(self, name: str):
        """根据名称获取层"""
        parts = name.split('.')
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                return None
        return module
    
    def _find_last_conv(self):
        """找到最后一个卷积层"""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None
    ) -> np.ndarray:
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: [1, C, H, W] 输入图像
            target_class: 目标类别，None表示使用最高概率类
            
        Returns:
            [H, W] 归一化的热力图
        """
        # 确保梯度计算
        input_tensor.requires_grad_(True)
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 处理不同的输出格式
        if isinstance(output, tuple):
            output = output[0]  # 取第一个输出（通常是重建或分类）
        
        # 如果是自编码器，使用编码特征
        if output.dim() == 4:
            # 重建输出，使用均值作为目标
            target = output.mean()
        elif output.dim() == 2:
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            target = output[0, target_class]
        else:
            target = output.mean()
        
        # 反向传播
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            print("⚠️ 未能捕获梯度或激活值")
            return np.zeros((224, 224))
        
        # 计算权重（全局平均池化梯度）
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU激活（只保留正向贡献）
        cam = F.relu(cam)
        
        # 上采样到输入尺寸
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # 归一化到[0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        将热力图叠加到原图上
        
        Args:
            original_image: 原始图像 [H, W, 3]
            heatmap: 热力图 [H, W]
            alpha: 热力图透明度
            colormap: matplotlib颜色映射
            
        Returns:
            叠加后的图像 [H, W, 3]
        """
        # 确保图像是uint8格式
        if original_image.max() <= 1:
            original_image = (original_image * 255).astype(np.uint8)
        
        # 调整热力图尺寸
        if heatmap.shape != original_image.shape[:2]:
            from PIL import Image as PILImage
            heatmap = np.array(
                PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
                    (original_image.shape[1], original_image.shape[0]),
                    PILImage.LANCZOS
                )
            ) / 255.0
        
        # 应用颜色映射
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # 去除alpha通道
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # 叠加
        overlay = (1 - alpha) * original_image + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay


class AttentionVisualizer:
    """
    注意力权重可视化器
    专门用于AttentionCAE的注意力可视化
    """
    
    def __init__(self, model):
        """
        Args:
            model: AttentionCAE模型
        """
        self.model = model
        self.model.eval()
        
    def get_attention_weights(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        获取注意力权重
        
        Args:
            input_tensor: [1, C, H, W] 输入图像
            
        Returns:
            [num_heads, seq_len, seq_len] 注意力权重
        """
        with torch.no_grad():
            _, attn_weights = self.model(input_tensor)
        
        return attn_weights.squeeze().cpu().numpy()
    
    def visualize_attention(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray = None,
        head_idx: int = 0
    ) -> np.ndarray:
        """
        可视化特定注意力头
        
        Args:
            input_tensor: [1, C, H, W] 输入图像
            original_image: 原始图像用于叠加
            head_idx: 注意力头索引
            
        Returns:
            注意力热力图或叠加图
        """
        attn_weights = self.get_attention_weights(input_tensor)
        
        if attn_weights.ndim == 3:
            # [num_heads, seq_len, seq_len]
            attn = attn_weights[head_idx]
        else:
            attn = attn_weights
        
        # 取对角线元素作为自注意力强度
        # 或者取行均值
        attn_map = attn.mean(axis=0)  # [seq_len]
        
        # 重塑为2D
        side = int(np.sqrt(len(attn_map)))
        attn_2d = attn_map[:side*side].reshape(side, side)
        
        # 上采样到原图尺寸
        from PIL import Image as PILImage
        if original_image is not None:
            h, w = original_image.shape[:2]
            attn_2d = np.array(
                PILImage.fromarray((attn_2d * 255).astype(np.uint8)).resize(
                    (w, h), PILImage.LANCZOS
                )
            ) / 255.0
        
        return attn_2d


def create_gradcam_overlay(
    model,
    image_path: str,
    output_path: str = None,
    device: str = 'cpu'
) -> np.ndarray:
    """
    便捷函数：为K线图生成Grad-CAM叠加图
    
    Args:
        model: 模型
        image_path: K线图路径
        output_path: 输出路径（可选）
        device: 计算设备
        
    Returns:
        叠加后的图像
    """
    from torchvision import transforms
    
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    original = np.array(img)
    
    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # 生成Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam.generate(input_tensor)
    
    # 叠加
    overlay = gradcam.overlay_heatmap(original, heatmap)
    
    # 保存
    if output_path:
        Image.fromarray(overlay).save(output_path)
        print(f"✅ Grad-CAM图已保存: {output_path}")
    
    return overlay


def visualize_top10_with_gradcam(
    model,
    query_image_path: str,
    matches: List[dict],
    output_path: str,
    device: str = 'cpu'
):
    """
    为Top10对比图添加Grad-CAM热力图
    
    Args:
        model: 模型
        query_image_path: 查询图像路径
        matches: Top10匹配结果
        output_path: 输出路径
        device: 计算设备
    """
    from torchvision import transforms
    import matplotlib.gridspec as gridspec
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 7, figure=fig)
    
    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    gradcam = GradCAM(model)
    
    # 1. 查询图像
    ax_main = fig.add_subplot(gs[:, :2])
    if os.path.exists(query_image_path):
        img = Image.open(query_image_path).convert('RGB')
        original = np.array(img)
        
        # 生成Grad-CAM
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        heatmap = gradcam.generate(input_tensor)
        overlay = gradcam.overlay_heatmap(original, heatmap)
        
        ax_main.imshow(overlay)
        ax_main.set_title("当前形态 (Grad-CAM)", fontsize=16, color='blue', fontweight='bold')
    ax_main.axis('off')
    
    # 2. Top10匹配
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    IMG_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "images")
    
    for i, match in enumerate(matches[:10]):
        row = i // 5
        col = 2 + (i % 5)
        ax = fig.add_subplot(gs[row, col])
        
        img_name = f"{match['symbol']}_{match['date']}.png"
        img_path = os.path.join(IMG_BASE_DIR, img_name)
        
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            ax.imshow(np.array(img))
            
            title = f"Top {i+1}\n{match['symbol']}\n{match['date']}\nSim: {match['score']:.3f}"
            ax.set_title(title, fontsize=9)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close('all')
    print(f"✅ Grad-CAM对比图已保存: {output_path}")


if __name__ == "__main__":
    print("=== Grad-CAM 测试 ===")
    
    # 创建简单的测试模型
    class SimpleConvNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
            )
            self.fc = torch.nn.Linear(64 * 224 * 224, 3)
            
        def forward(self, x):
            x = self.conv(x)
            return x  # 返回特征图用于测试
    
    model = SimpleConvNet()
    
    # 测试Grad-CAM
    gradcam = GradCAM(model)
    
    # 创建随机输入
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # 生成热力图
    heatmap = gradcam.generate(input_tensor)
    print(f"热力图形状: {heatmap.shape}")
    print(f"热力图范围: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    print("\n测试完成！")
