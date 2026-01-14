"""
模型模块
Deep learning models for visual pattern recognition
"""

from .autoencoder import QuantCAE
from .vision_engine import VisionEngine

# PredictEngine已废弃，不再导出
# 如需使用，请直接导入: from src.models.predict_engine import IndustrialPredictorReduced

__all__ = ['QuantCAE', 'VisionEngine']
