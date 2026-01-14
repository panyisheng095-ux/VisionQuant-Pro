"""
模型集成模块
Model Ensemble Module

集成多个模型的结果，提升预测稳定性

集成方法：
1. 投票法（Voting）
2. 加权平均（Weighted Average）
3. Stacking

Author: VisionQuant Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class ModelEnsemble:
    """
    模型集成器
    
    功能：
    1. 集成多个视觉模型（不同维度、不同架构）
    2. 集成多尺度模型（日线/周线/月线）
    3. 集成双流模型和单流模型
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        ensemble_method: str = 'weighted_average'
    ):
        """
        初始化模型集成器
        
        Args:
            models: 模型列表
            weights: 模型权重（如果为None，则使用均匀权重）
            ensemble_method: 集成方法 ('voting', 'weighted_average', 'stacking')
        """
        self.models = models
        self.ensemble_method = ensemble_method
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        # 确保模型在eval模式
        for model in self.models:
            model.eval()
    
    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> Dict:
        """
        集成预测
        
        Args:
            x: 输入数据 [batch, ...]
            return_probs: 是否返回概率
            
        Returns:
            预测结果字典
        """
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                # 获取模型预测
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(x)
                elif hasattr(model, 'forward'):
                    output = model(x)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    probs = F.softmax(logits, dim=1)
                else:
                    raise ValueError(f"模型 {type(model)} 不支持预测")
                
                # 加权
                weighted_probs = probs * weight
                all_probs.append(weighted_probs)
                
                # 预测类别
                pred = probs.argmax(dim=1)
                all_predictions.append(pred)
        
        # 集成
        if self.ensemble_method == 'weighted_average':
            ensemble_probs = torch.stack(all_probs).sum(dim=0)
            ensemble_pred = ensemble_probs.argmax(dim=1)
        elif self.ensemble_method == 'voting':
            # 投票
            all_preds = torch.stack(all_predictions)
            ensemble_pred = torch.mode(all_preds, dim=0)[0]
            # 计算投票概率
            ensemble_probs = torch.zeros_like(all_probs[0])
            for i, pred in enumerate(ensemble_pred):
                votes = (all_preds[:, i] == pred).sum().float()
                ensemble_probs[i, pred] = votes / len(self.models)
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
        
        result = {
            'prediction': ensemble_pred.cpu().numpy(),
            'confidence': ensemble_probs.max(dim=1)[0].cpu().numpy()
        }
        
        if return_probs:
            result['probabilities'] = ensemble_probs.cpu().numpy()
        
        return result
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        集成编码（用于特征提取）
        
        Args:
            x: 输入数据
            
        Returns:
            集成后的特征向量
        """
        all_features = []
        
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                if hasattr(model, 'encode'):
                    feat = model.encode(x)
                elif hasattr(model, 'forward'):
                    output = model(x)
                    if isinstance(output, tuple):
                        feat = output[1]  # 假设第二个是特征
                    else:
                        feat = output
                else:
                    continue
                
                # 加权特征
                weighted_feat = feat * weight
                all_features.append(weighted_feat)
        
        if not all_features:
            raise ValueError("没有模型支持编码")
        
        # 拼接或平均
        if self.ensemble_method == 'weighted_average':
            ensemble_feat = torch.stack(all_features).sum(dim=0)
        else:
            # 拼接
            ensemble_feat = torch.cat(all_features, dim=1)
        
        # L2归一化
        ensemble_feat = F.normalize(ensemble_feat, p=2, dim=1)
        
        return ensemble_feat


class MultiScaleEnsemble:
    """
    多尺度模型集成器
    
    集成日线/周线/月线模型的预测结果
    """
    
    def __init__(
        self,
        daily_model: nn.Module,
        weekly_model: Optional[nn.Module] = None,
        monthly_model: Optional[nn.Module] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        初始化多尺度集成器
        
        Args:
            daily_model: 日线模型
            weekly_model: 周线模型（可选）
            monthly_model: 月线模型（可选）
            weights: 各尺度权重 {'daily': 0.5, 'weekly': 0.3, 'monthly': 0.2}
        """
        self.daily_model = daily_model
        self.weekly_model = weekly_model
        self.monthly_model = monthly_model
        
        if weights is None:
            self.weights = {'daily': 0.5, 'weekly': 0.3, 'monthly': 0.2}
        else:
            # 归一化
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
        
        # 设置eval模式
        self.daily_model.eval()
        if self.weekly_model:
            self.weekly_model.eval()
        if self.monthly_model:
            self.monthly_model.eval()
    
    def predict(
        self,
        daily_image: torch.Tensor,
        weekly_image: Optional[torch.Tensor] = None,
        monthly_image: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        多尺度集成预测
        
        Args:
            daily_image: 日线K线图
            weekly_image: 周线K线图（可选）
            monthly_image: 月线K线图（可选）
            
        Returns:
            集成预测结果
        """
        all_probs = []
        total_weight = 0.0
        
        with torch.no_grad():
            # 日线预测
            if hasattr(self.daily_model, 'encode'):
                feat = self.daily_model.encode(daily_image)
                # 假设有分类头（这里简化处理）
                probs = torch.softmax(feat[:, :3], dim=1)  # 假设前3维是分类logits
            else:
                output = self.daily_model(daily_image)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                probs = F.softmax(logits, dim=1)
            
            all_probs.append(probs * self.weights['daily'])
            total_weight += self.weights['daily']
            
            # 周线预测（如果提供）
            if weekly_image is not None and self.weekly_model is not None:
                if hasattr(self.weekly_model, 'encode'):
                    feat = self.weekly_model.encode(weekly_image)
                    probs = torch.softmax(feat[:, :3], dim=1)
                else:
                    output = self.weekly_model(weekly_image)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    probs = F.softmax(logits, dim=1)
                
                all_probs.append(probs * self.weights['weekly'])
                total_weight += self.weights['weekly']
            
            # 月线预测（如果提供）
            if monthly_image is not None and self.monthly_model is not None:
                if hasattr(self.monthly_model, 'encode'):
                    feat = self.monthly_model.encode(monthly_image)
                    probs = torch.softmax(feat[:, :3], dim=1)
                else:
                    output = self.monthly_model(monthly_image)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    probs = F.softmax(logits, dim=1)
                
                all_probs.append(probs * self.weights['monthly'])
                total_weight += self.weights['monthly']
        
        # 加权平均
        ensemble_probs = torch.stack(all_probs).sum(dim=0) / total_weight
        ensemble_pred = ensemble_probs.argmax(dim=1)
        
        return {
            'prediction': ensemble_pred.cpu().numpy(),
            'probabilities': ensemble_probs.cpu().numpy(),
            'confidence': ensemble_probs.max(dim=1)[0].cpu().numpy()
        }


if __name__ == "__main__":
    print("=== 模型集成测试 ===")
    
    # 模拟多个模型
    from .attention_cae import AttentionCAE
    
    model1 = AttentionCAE(latent_dim=1024)
    model2 = AttentionCAE(latent_dim=2048)
    
    ensemble = ModelEnsemble(
        models=[model1, model2],
        weights=[0.6, 0.4],
        ensemble_method='weighted_average'
    )
    
    # 模拟输入
    x = torch.randn(2, 3, 224, 224)
    
    # 集成预测
    result = ensemble.predict(x)
    print(f"集成预测: {result['prediction']}")
    print(f"置信度: {result['confidence']}")
