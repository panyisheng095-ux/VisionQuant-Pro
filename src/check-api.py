import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv

# === 1. 强制定位 .env 文件 ===
# 获取当前脚本所在目录 (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (VisionQuant-Pro)
project_root = os.path.dirname(current_dir)
# 拼接 .env 的绝对路径
env_path = os.path.join(project_root, ".env")

print(f"📂 正在尝试加载配置: {env_path}")

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("✅ .env 文件存在")
else:
    print("❌ 严重错误：找不到 .env 文件！请确认文件就在 VisionQuant-Pro 目录下。")
    sys.exit(1)

# 获取 Key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ 读取失败：.env 文件里没有 GOOGLE_API_KEY 这一行，或者没保存。")
else:
    # 隐去中间部分，只显示首尾
    masked_key = f"{api_key[:5]}...{api_key[-5:]}"
    print(f"🔑 成功读取 API Key: {masked_key}")

    # === 2. 测试连接与权限 ===
    genai.configure(api_key=api_key)

    print("\n📡 正在连接 Google 服务器查询可用模型...")
    try:
        # 列出所有可用模型
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
                print(f"   - 发现模型: {m.name}")

        print("\n✅ API 连接成功！")

        # 检查是否有 Pro 权限
        if "models/gemini-1.5-pro" in available_models:
            print("🎉 恭喜！你的 Key 支持【gemini-1.5-pro】(这就是目前最强的版本)")
        elif "models/gemini-pro" in available_models:
            print("👍 你的 Key 支持标准版【gemini-pro】")

    except Exception as e:
        print(f"❌ 连接被拒绝: {e}")
        print("原因可能是：Key 填错了 / 科学上网不稳定 / 额度耗尽")