import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os
import matplotlib

# 强制后台模式
matplotlib.use('Agg')


def create_comparison_plot(query_img_path, search_results, output_path):
    """
    绘制 1 (Query) + 10 (Matches) 对比图
    """
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    IMG_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "images")

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 使用 GridSpec 进行复杂布局
    # 2行，6列。左边 2x2 的区域给大图，右边剩下的给小图
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 7, figure=fig)

    # 1. 左侧大图 (占 2行 x 2列)
    ax_main = fig.add_subplot(gs[:, :2])
    if os.path.exists(query_img_path):
        img = Image.open(query_img_path)
        ax_main.imshow(img)
        ax_main.set_title("当前目标形态 (Query)", fontsize=18, color='blue', fontweight='bold')
    ax_main.axis('off')

    # 2. 右侧 10 张小图 (2行 x 5列)
    # search_results 应该有 10 个
    for i, res in enumerate(search_results[:10]):
        # 计算网格位置: 行(0/1), 列(从第2列开始)
        row = i // 5
        col = 2 + (i % 5)

        ax = fig.add_subplot(gs[row, col])

        # 拼凑图片路径
        hist_img_name = f"{res['symbol']}_{res['date']}.png"
        hist_img_path = os.path.join(IMG_BASE_DIR, hist_img_name)

        if os.path.exists(hist_img_path):
            img_hist = Image.open(hist_img_path)
            ax.imshow(img_hist)

            # 标题显示相似度和代码
            title = f"Top {i + 1}\n{res['symbol']}\n{res['date']}\nSim: {res['score']:.3f}"
            ax.set_title(title, fontsize=10)
        else:
            ax.text(0.5, 0.5, "Image\nMissing", ha='center', va='center')

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close('all')