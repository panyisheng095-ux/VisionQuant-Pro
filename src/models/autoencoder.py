import torch
import torch.nn as nn


class QuantCAE(nn.Module):
    """
    量化专用卷积自编码器 (Convolutional AutoEncoder)
    用于无监督学习 K 线图的形态特征
    """

    def __init__(self):
        super(QuantCAE, self).__init__()

        # =======================================
        # 1. 编码器 (Encoder): 图片 -> 特征向量
        # =======================================
        # 输入: [Batch, 3, 224, 224] (RGB图像)
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
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 此时输出形状为 [Batch, 256, 14, 14]
            # 这包含了丰富的空间特征（如顶背离、双底的位置信息）
        )

        # =======================================
        # 2. 解码器 (Decoder): 特征向量 -> 还原图片
        # =======================================
        # 目标是还原回 [Batch, 3, 224, 224]
        self.decoder = nn.Sequential(
            # Layer 4: 14 -> 28
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 28 -> 56
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 56 -> 112
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 1: 112 -> 224
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 输出层使用 Sigmoid，将像素值压缩到 0-1 之间
        )

    def forward(self, x):
        """
        前向传播：用于训练阶段
        返回：(特征, 还原图)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        """
        推理专用：只提取特征，不要还原图
        用于生成 FAISS 索引
        """
        x = self.encoder(x)
        # 将 [Batch, 256, 14, 14] 展平为 [Batch, 50176] 的一维向量
        # 虽然维度很高，但保留了极佳的形态细节
        return x.view(x.size(0), -1)


# === 简单的测试代码 ===
if __name__ == "__main__":
    # 测试一下模型结构是否正确，会不会报错
    print("正在测试模型结构...")
    model = QuantCAE()

    # 创建一个假的随机图片数据 [Batch=2, Channel=3, H=224, W=224]
    dummy_input = torch.randn(2, 3, 224, 224)

    encoded, decoded = model(dummy_input)

    print(f"输入尺寸: {dummy_input.shape}")
    print(f"编码特征尺寸: {encoded.shape}")  # 应该输出 [2, 256, 14, 14]
    print(f"解码还原尺寸: {decoded.shape}")  # 应该输出 [2, 3, 224, 224]

    feature_vec = model.encode(dummy_input)
    print(f"最终特征向量维度: {feature_vec.shape}")  # 应该输出 [2, 50176]

    print("✅ 模型定义无误！")