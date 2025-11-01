"""
LTX-Video Production-Ready ComfyUI Custom Nodes
Version 2.0 - Enhanced for hyper-realistic text-to-video generation

This module provides production-ready ComfyUI nodes for LTX-Video with:
- Text-to-Video generation with auto-prompt enhancement
- 8-10 second video duration support via frame interpolation
- 1080p and 4K output with intelligent upscaling
- Temporal consistency and realism optimization
- Low VRAM fallback modes (12GB+ supported)
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
