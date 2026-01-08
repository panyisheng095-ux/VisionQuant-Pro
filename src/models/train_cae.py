import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# === 1. Mac 系统防崩配置 ===
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === 2. 路径配置 ===
# 获取当前文件所在目录 src/models
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 VisionQuant-Pro
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# 图片读取路径
DATA_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "images")
# 模型保存路径
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "models")

# 确保能导入 src 下的模块
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.autoencoder import QuantCAE

# === 3. 超参数配置 ===
BATCH_SIZE = 64  # 一次训练 64 张图 (M1/M2 芯片建议 64-128)
LEARNING_RATE = 1e-3  # 学习率
EPOCHS = 5  # 训练轮数 (K线图比较简单，5轮通常能收敛)


# ==========================================
#  数据集加载器
# ==========================================
class KLineDataset(Dataset):
    def __init__(self, img_dir):
        print(f"🔍 正在扫描图片目录: {img_dir} ...")
        # 获取所有 png 文件
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        print(f"📦 训练集加载完毕: 共发现 {len(self.img_files)} 张 K 线图")

        # 预处理：调整大小 -> 转Tensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            img_path = self.img_files[idx]
            # 必须转为 RGB，防止部分图片是 RGBA
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            # 容错：如果某张图坏了，返回全黑图，防止训练中断
            return torch.zeros((3, 224, 224))


# ==========================================
#  核心训练流程
# ==========================================
def train():
    # 确保保存目录存在
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 1. 设备选择
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 [训练] 使用 Apple Metal (MPS) 显卡加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 [训练] 使用 NVIDIA CUDA 显卡加速")
    else:
        device = torch.device("cpu")
        print("🐢 [训练] 未检测到 GPU，使用 CPU (速度较慢)")

    # 2. 准备数据
    dataset = KLineDataset(DATA_IMG_DIR)
    if len(dataset) == 0:
        print("❌ 错误: data/images 目录下没有图片！请先运行 vision_engine 生成图片。")
        return

    # num_workers=0 是 Mac 上最稳的设置，防止多进程死锁
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 3. 初始化模型
    model = QuantCAE().to(device)

    # 损失函数：均方误差 (比较 原图 和 还原图 的像素差异)
    criterion = nn.MSELoss()
    # 优化器：Adam
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🔥 开始训练 (计划 {EPOCHS} 轮)...")
    print(f"💾 模型将保存在: {MODEL_SAVE_DIR}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # 进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for imgs in pbar:
            imgs = imgs.to(device)

            # --- A. 前向传播 ---
            # 这里的 labels 就是 imgs 本身，因为是自编码器
            encoded, decoded = model(imgs)

            # --- B. 计算损失 ---
            loss = criterion(decoded, imgs)

            # --- C. 反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新统计
            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        # 计算本轮平均 Loss
        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} 完成 | 平均 Loss: {avg_loss:.6f}")

        # --- D. 保存模型 ---
        # 保存两个版本：最新版和当前轮次版
        save_path_latest = os.path.join(MODEL_SAVE_DIR, "cae_best.pth")
        save_path_epoch = os.path.join(MODEL_SAVE_DIR, f"cae_epoch_{epoch + 1}.pth")

        torch.save(model.state_dict(), save_path_latest)
        torch.save(model.state_dict(), save_path_epoch)
        print(f"💾 模型参数已保存")

    print("\n🎉 训练全部完成！新大脑已就绪。")


if __name__ == "__main__":
    train()