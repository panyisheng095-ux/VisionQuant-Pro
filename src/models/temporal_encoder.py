"""
时序编码器 - 处理OHLCV数值数据
Temporal Encoder for OHLCV Numerical Data

与视觉编码器（AttentionCAE）配合使用，形成双流架构。
视觉流捕捉形态特征，时序流捕捉数值精度和动态变化。

特点:
1. LSTM/GRU处理序列依赖
2. 保留价格精确数值信息
3. 输出256维时序特征向量
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class TemporalEncoder(nn.Module):
    """
    时序编码器 - 使用LSTM处理OHLCV数据
    
    输入: [batch, seq_len, features] - 如 [32, 20, 5] (20天, 5个特征)
    输出: [batch, hidden_dim] - 如 [32, 256]
    """
    
    def __init__(
        self,
        input_dim: int = 5,           # OHLCV 5个特征
        hidden_dim: int = 256,        # 隐藏层维度
        num_layers: int = 2,          # LSTM层数
        dropout: float = 0.2,         # Dropout比率
        bidirectional: bool = True    # 是否双向
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 输入归一化层
        self.input_norm = nn.LayerNorm(input_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 输出维度（双向则翻倍）
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 注意力池化
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch, seq_len, input_dim]
            
        Returns:
            [batch, hidden_dim]
        """
        # 输入归一化
        x = self.input_norm(x)
        
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # 注意力加权池化
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权求和
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden*2]
        
        # 输出投影
        output = self.output_proj(context)  # [batch, hidden_dim]
        
        return output
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码接口（与AttentionCAE保持一致）"""
        return self.forward(x)


class GRUEncoder(nn.Module):
    """
    GRU时序编码器 - 更轻量的替代方案
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        # 取最后一层的双向隐藏状态
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.output_proj(hidden)


class OHLCVPreprocessor:
    """
    OHLCV数据预处理器
    
    将原始OHLCV数据转换为模型输入格式
    """
    
    def __init__(
        self,
        seq_len: int = 20,
        normalize: str = 'returns'  # 'returns', 'zscore', 'minmax'
    ):
        self.seq_len = seq_len
        self.normalize = normalize
        
    def preprocess(self, df) -> np.ndarray:
        """
        预处理DataFrame为模型输入
        
        Args:
            df: 包含OHLCV列的DataFrame
            
        Returns:
            [seq_len, 5] 的numpy数组
        """
        # 确保有足够数据
        if len(df) < self.seq_len:
            raise ValueError(f"数据长度({len(df)})不足{self.seq_len}")
        
        # 取最后seq_len天的数据
        df = df.tail(self.seq_len).copy()
        
        # 提取OHLCV
        features = []
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                features.append(df[col].values)
            else:
                # 如果缺少某列，用Close填充
                features.append(df['Close'].values if 'Close' in df.columns else np.zeros(self.seq_len))
        
        data = np.stack(features, axis=1)  # [seq_len, 5]
        
        # 归一化
        if self.normalize == 'returns':
            # 计算收益率（相对于第一天）
            data[:, :4] = (data[:, :4] / data[0, :4] - 1) * 100  # 价格列转为百分比收益
            data[:, 4] = data[:, 4] / data[:, 4].mean()  # 成交量相对于均值
        elif self.normalize == 'zscore':
            # Z-score标准化
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True) + 1e-8
            data = (data - mean) / std
        elif self.normalize == 'minmax':
            # Min-Max归一化
            min_val = data.min(axis=0, keepdims=True)
            max_val = data.max(axis=0, keepdims=True)
            data = (data - min_val) / (max_val - min_val + 1e-8)
        
        return data.astype(np.float32)
    
    def batch_preprocess(self, dfs: list) -> torch.Tensor:
        """
        批量预处理
        
        Args:
            dfs: DataFrame列表
            
        Returns:
            [batch, seq_len, 5] 的Tensor
        """
        batch = []
        for df in dfs:
            try:
                processed = self.preprocess(df)
                batch.append(processed)
            except:
                continue
        
        if not batch:
            raise ValueError("无有效数据")
        
        return torch.tensor(np.stack(batch), dtype=torch.float32)


if __name__ == "__main__":
    print("=== 时序编码器测试 ===")
    
    # 创建模拟数据
    batch_size = 4
    seq_len = 20
    input_dim = 5
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 测试LSTM编码器
    encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=256)
    output = encoder(x)
    print(f"LSTM编码器输出: {output.shape}")  # [4, 256]
    
    # 测试GRU编码器
    gru_encoder = GRUEncoder(input_dim=input_dim, hidden_dim=256)
    gru_output = gru_encoder(x)
    print(f"GRU编码器输出: {gru_output.shape}")  # [4, 256]
    
    print("\n测试完成！")
