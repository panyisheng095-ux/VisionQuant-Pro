"""
双流融合网络 - 视觉+时序特征融合
Dual-Stream Fusion Network

将视觉流（AttentionCAE）和时序流（TemporalEncoder）的特征融合，
产生更全面的市场状态表示。

架构:
    K线图 → AttentionCAE → 视觉特征(512维)
                                        ↘
                                         → 融合层 → 预测头 → Triple Barrier概率
                                        ↗
    OHLCV → TemporalEncoder → 时序特征(256维)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    让视觉特征和时序特征互相关注
    """
    
    def __init__(self, visual_dim: int = 512, temporal_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        
        # 投影到相同维度
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        
        # 注意力计算
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, visual_feat: torch.Tensor, temporal_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feat: [batch, visual_dim]
            temporal_feat: [batch, temporal_dim]
        Returns:
            [batch, hidden_dim]
        """
        # 投影
        v = self.visual_proj(visual_feat).unsqueeze(1)  # [batch, 1, hidden]
        t = self.temporal_proj(temporal_feat).unsqueeze(1)  # [batch, 1, hidden]
        
        # 拼接作为序列
        combined = torch.cat([v, t], dim=1)  # [batch, 2, hidden]
        
        # 自注意力
        attn_out, _ = self.attention(combined, combined, combined)
        
        # 取平均
        fused = attn_out.mean(dim=1)  # [batch, hidden]
        
        return self.norm(fused)


class DualStreamNetwork(nn.Module):
    """
    双流融合网络
    
    输入:
        - visual_feat: 来自AttentionCAE的视觉特征 [batch, 512]
        - temporal_feat: 来自TemporalEncoder的时序特征 [batch, 256]
    
    输出:
        - Triple Barrier预测概率 [batch, 3] (看涨/震荡/看跌)
    """
    
    def __init__(
        self,
        visual_dim: int = 512,
        temporal_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 3,
        fusion_type: str = 'attention'  # 'concat', 'attention', 'gated'
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # 简单拼接
            self.fusion = nn.Sequential(
                nn.Linear(visual_dim + temporal_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        elif fusion_type == 'attention':
            # 跨模态注意力融合
            self.fusion = CrossModalAttention(visual_dim, temporal_dim, hidden_dim)
        elif fusion_type == 'gated':
            # 门控融合
            self.visual_gate = nn.Linear(visual_dim, hidden_dim)
            self.temporal_gate = nn.Linear(temporal_dim, hidden_dim)
            self.visual_proj = nn.Linear(visual_dim, hidden_dim)
            self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
        
        # 预测头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 回归头（预测收益率）
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(
        self,
        visual_feat: torch.Tensor,
        temporal_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            class_logits: [batch, 3] Triple Barrier分类logits
            return_pred: [batch, 1] 预测收益率
        """
        # 特征融合
        if self.fusion_type == 'concat':
            combined = torch.cat([visual_feat, temporal_feat], dim=1)
            fused = self.fusion(combined)
        elif self.fusion_type == 'attention':
            fused = self.fusion(visual_feat, temporal_feat)
        elif self.fusion_type == 'gated':
            # 门控融合
            gate = torch.sigmoid(
                self.visual_gate(visual_feat) + self.temporal_gate(temporal_feat)
            )
            v_proj = self.visual_proj(visual_feat)
            t_proj = self.temporal_proj(temporal_feat)
            fused = self.norm(gate * v_proj + (1 - gate) * t_proj)
        
        # 分类和回归
        class_logits = self.classifier(fused)
        return_pred = self.regressor(fused)
        
        return class_logits, return_pred
    
    def predict_proba(
        self,
        visual_feat: torch.Tensor,
        temporal_feat: torch.Tensor
    ) -> torch.Tensor:
        """返回概率"""
        logits, _ = self.forward(visual_feat, temporal_feat)
        return F.softmax(logits, dim=1)


class DualStreamPredictor:
    """
    双流预测器 - 整合视觉和时序编码器
    
    用法:
    ```python
    predictor = DualStreamPredictor()
    predictor.load_models(visual_path, temporal_path, fusion_path)
    result = predictor.predict(kline_image, ohlcv_data)
    ```
    """
    
    def __init__(self, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.visual_encoder = None
        self.temporal_encoder = None
        self.fusion_network = None
        self.loaded = False
        
    def load_models(
        self,
        visual_model_path: str,
        temporal_model_path: str = None,
        fusion_model_path: str = None
    ):
        """加载模型"""
        # 导入模型
        from .attention_cae import AttentionCAE
        from .temporal_encoder import TemporalEncoder
        
        # 加载视觉编码器
        self.visual_encoder = AttentionCAE().to(self.device)
        if visual_model_path:
            state_dict = torch.load(visual_model_path, map_location=self.device)
            self.visual_encoder.load_state_dict(state_dict)
        self.visual_encoder.eval()
        
        # 加载时序编码器
        self.temporal_encoder = TemporalEncoder().to(self.device)
        if temporal_model_path:
            state_dict = torch.load(temporal_model_path, map_location=self.device)
            self.temporal_encoder.load_state_dict(state_dict)
        self.temporal_encoder.eval()
        
        # 加载融合网络
        self.fusion_network = DualStreamNetwork().to(self.device)
        if fusion_model_path:
            state_dict = torch.load(fusion_model_path, map_location=self.device)
            self.fusion_network.load_state_dict(state_dict)
        self.fusion_network.eval()
        
        self.loaded = True
        
    def predict(
        self,
        kline_tensor: torch.Tensor,
        ohlcv_tensor: torch.Tensor
    ) -> Dict:
        """
        预测Triple Barrier结果
        
        Args:
            kline_tensor: [batch, 3, 224, 224] K线图
            ohlcv_tensor: [batch, 20, 5] OHLCV数据
            
        Returns:
            预测结果字典
        """
        if not self.loaded:
            raise RuntimeError("模型未加载，请先调用load_models()")
        
        with torch.no_grad():
            kline_tensor = kline_tensor.to(self.device)
            ohlcv_tensor = ohlcv_tensor.to(self.device)
            
            # 编码
            visual_feat = self.visual_encoder.encode(kline_tensor)
            temporal_feat = self.temporal_encoder.encode(ohlcv_tensor)
            
            # 预测
            class_logits, return_pred = self.fusion_network(visual_feat, temporal_feat)
            probs = F.softmax(class_logits, dim=1)
            
            # 解析结果
            pred_class = probs.argmax(dim=1).cpu().numpy()
            class_names = ['BEARISH', 'NEUTRAL', 'BULLISH']
            
            results = []
            for i in range(len(pred_class)):
                results.append({
                    'prediction': class_names[pred_class[i] + 1],  # -1,0,1 → 0,1,2
                    'bearish_prob': probs[i, 0].item(),
                    'neutral_prob': probs[i, 1].item(),
                    'bullish_prob': probs[i, 2].item(),
                    'expected_return': return_pred[i].item()
                })
            
            return results if len(results) > 1 else results[0]


if __name__ == "__main__":
    print("=== 双流融合网络测试 ===")
    
    # 创建模拟数据
    batch_size = 4
    visual_feat = torch.randn(batch_size, 512)
    temporal_feat = torch.randn(batch_size, 256)
    
    # 测试不同融合方式
    for fusion_type in ['concat', 'attention', 'gated']:
        print(f"\n融合方式: {fusion_type}")
        model = DualStreamNetwork(fusion_type=fusion_type)
        
        class_logits, return_pred = model(visual_feat, temporal_feat)
        probs = F.softmax(class_logits, dim=1)
        
        print(f"  分类输出: {class_logits.shape}")
        print(f"  预测概率: {probs[0].detach().numpy()}")
        print(f"  收益预测: {return_pred[0].item():.4f}")
    
    print("\n测试完成！")
