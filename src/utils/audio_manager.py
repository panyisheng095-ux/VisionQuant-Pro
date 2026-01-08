import os
import google.generativeai as genai
from dotenv import load_dotenv
import datetime

# 加载配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GOOGLE_API_KEY")


class AudioManager:
    def __init__(self):
        if not API_KEY:
            self.model = None
            print("❌ [Audio] API Key 缺失")
            return

        genai.configure(api_key=API_KEY)

        # === 核心修复：定义模型候选列表 ===
        # 既然你的 Agent 能连上 2.5-pro，说明你的号很新
        # 我们优先尝试支持多模态的新模型
        self.candidate_models = [
            "gemini-2.0-flash-exp",  # 极速，支持音频
            "gemini-1.5-pro",  # 稳定，支持音频
            "gemini-1.5-flash",  # 备选
            "gemini-pro"  # 兜底 (可能不支持音频，但值得一试)
        ]

        print("🎤 [Audio] 语音引擎初始化...")

    def transcribe(self, audio_bytes):
        """
        语音转文字 (自动轮询模型版)
        """
        if not API_KEY: return None

        # 1. 检查数据大小
        data_size = len(audio_bytes)
        print(f"🎤 [Audio] 收到数据: {data_size} bytes")

        if data_size < 1000:
            print("⚠️ 录音时间太短，忽略")
            return None

        # 2. 强制保存调试文件 (保留这个好习惯)
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        debug_path = os.path.join(PROJECT_ROOT, f"debug_audio_{timestamp}.wav")
        try:
            with open(debug_path, "wb") as f:
                f.write(audio_bytes)
        except:
            pass

        # 3. 轮询模型进行识别
        prompt = "Please transcribe this audio to text. If Chinese, output Chinese directly. Do not translate. Output ONLY the text."

        for model_name in self.candidate_models:
            try:
                # 动态初始化模型
                model = genai.GenerativeModel(model_name)

                # print(f"   ⟳ 尝试使用 {model_name} 识别...")

                response = model.generate_content([
                    prompt,
                    {"mime_type": "audio/wav", "data": audio_bytes}
                ])

                text = response.text.strip()
                if text:
                    print(f"✅ [Audio] 识别成功 ({model_name}): {text}")
                    return text

            except Exception as e:
                # 如果是 404 Not Found，说明这个模型不可用，直接试下一个
                if "404" in str(e) or "not found" in str(e).lower():
                    continue
                else:
                    print(f"❌ {model_name} 报错: {e}")
                    continue

        print("❌ 所有模型均无法识别音频，请检查 API Key 权限或网络。")
        return None