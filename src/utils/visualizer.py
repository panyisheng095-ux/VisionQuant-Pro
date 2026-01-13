import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os
import matplotlib
import glob

# 强制后台模式
matplotlib.use('Agg')


def create_comparison_plot(query_img_path, search_results, output_path):
    """
    绘制 1 (Query) + 10 (Matches) 对比图
    
    增强版：
    1. 自动处理date格式差异（20200206 vs 2020-02-06）
    2. 支持子目录搜索
    3. 始终显示10个格子（即使部分图片缺失）
    """
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    IMG_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "images")

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 使用 GridSpec 进行复杂布局
    # 2行，7列。左边 2x2 的区域给大图，右边剩下的给小图
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 7, figure=fig)

    # 1. 左侧大图 (占 2行 x 2列)
    ax_main = fig.add_subplot(gs[:, :2])
    if os.path.exists(query_img_path):
        try:
            img = Image.open(query_img_path)
            ax_main.imshow(img)
            ax_main.set_title("当前目标形态 (Query)", fontsize=18, color='blue', fontweight='bold')
        except Exception as e:
            ax_main.text(0.5, 0.5, f"Query Image\nLoad Error", ha='center', va='center', fontsize=14)
    else:
        ax_main.text(0.5, 0.5, "Query Image\nNot Found", ha='center', va='center', fontsize=14)
    ax_main.axis('off')

    # 2. 右侧 10 张小图 (2行 x 5列)
    # 确保始终渲染10个格子
    for i in range(10):
        row = i // 5
        col = 2 + (i % 5)
        ax = fig.add_subplot(gs[row, col])

        if i < len(search_results):
            res = search_results[i]
            
            # 标准化date格式（移除连字符）
            date_str = str(res['date']).replace('-', '')
            symbol = str(res['symbol']).zfill(6)
            
            # 尝试多种路径查找图片
            hist_img_path = None
            possible_paths = [
                os.path.join(IMG_BASE_DIR, f"{symbol}_{date_str}.png"),  # 直接在images/下
                os.path.join(IMG_BASE_DIR, symbol, f"{symbol}_{date_str}.png"),  # 在images/symbol/下
                os.path.join(IMG_BASE_DIR, symbol, f"{date_str}.png"),  # 在images/symbol/下，无前缀
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    hist_img_path = path
                    break
            
            # 如果还没找到，尝试glob搜索
            if hist_img_path is None:
                pattern = os.path.join(IMG_BASE_DIR, "**", f"*{symbol}*{date_str}*.png")
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    hist_img_path = matches[0]

            if hist_img_path and os.path.exists(hist_img_path):
                try:
                    img_hist = Image.open(hist_img_path)
                    ax.imshow(img_hist)
                    
                    # 标题显示相似度和代码
                    score_val = res.get('score', 0)
                    corr_val = res.get('correlation')
                    if corr_val is not None:
                        title = f"Top {i + 1}\n{symbol}\n{date_str}\nSim:{score_val:.2f} Corr:{corr_val:.2f}"
                    else:
                        title = f"Top {i + 1}\n{symbol}\n{date_str}\nSim: {score_val:.3f}"
                    ax.set_title(title, fontsize=9)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Top {i+1}\nLoad Error", ha='center', va='center', fontsize=10, color='red')
            else:
                # 图片未找到，显示信息
                ax.text(0.5, 0.5, f"Top {i+1}\n{symbol}\n{date_str}\n(Image Missing)", 
                       ha='center', va='center', fontsize=9, color='gray')
        else:
            # 没有足够的搜索结果
            ax.text(0.5, 0.5, f"#{i+1}\nNo Match", ha='center', va='center', fontsize=10, color='lightgray')

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close('all')


def save_reconstruction_example(original, reconstructed, epoch, save_dir):
    """
    保存重建对比图（用于训练可视化）
    """
    import torch
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(min(4, original.shape[0])):
        # 原图
        if isinstance(original, torch.Tensor):
            orig_img = original[i].cpu().permute(1, 2, 0).numpy()
        else:
            orig_img = original[i].transpose(1, 2, 0)
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # 重建图
        if isinstance(reconstructed, torch.Tensor):
            recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
        else:
            recon_img = reconstructed[i].transpose(1, 2, 0)
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Epoch {epoch} Reconstruction')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'reconstruction_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=100)
    plt.close()
    return save_path