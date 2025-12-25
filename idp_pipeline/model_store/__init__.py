"""
Model Store - Local AI Model Management
========================================
Handles downloading, caching, and loading of HuggingFace models.
"""

from .model_manager import ModelManager, get_model_manager

__all__ = ["ModelManager", "get_model_manager"]
