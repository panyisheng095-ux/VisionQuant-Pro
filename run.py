#!/usr/bin/env python3
"""
VisionQuant-Pro 启动脚本
解决 src.data 导入问题
"""

import os
import sys
import subprocess

def main():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 将项目根目录添加到 Python 路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')
    
    # 检查依赖
    print("🔍 检查依赖...")
    try:
        import streamlit
        import torch
        import faiss
        from streamlit_mic_recorder import mic_recorder
        print("✅ 核心依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    # 检查 .env 文件
    env_path = os.path.join(project_root, '.env')
    if not os.path.exists(env_path):
        print("⚠️ 未找到 .env 文件，AI Agent 功能可能受限")
        print("   请创建 .env 文件并添加: GOOGLE_API_KEY=your_key_here")
    
    # 启动 Streamlit
    print("\n🚀 启动 VisionQuant-Pro Web 界面...")
    print("=" * 50)
    
    web_app = os.path.join(project_root, 'web', 'app.py')
    
    # 使用 subprocess 启动，确保环境变量传递
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', web_app,
        '--server.port', '8501',
        '--server.headless', 'true'
    ]
    
    try:
        subprocess.run(cmd, env={**os.environ, 'PYTHONPATH': project_root})
    except KeyboardInterrupt:
        print("\n👋 VisionQuant-Pro 已停止")

if __name__ == '__main__':
    main()
