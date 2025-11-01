"""
LTX-Video Production-Ready ComfyUI Custom Nodes
Version 2.0.1 - Enhanced for hyper-realistic text-to-video generation

This module provides production-ready ComfyUI nodes for LTX-Video with:
- Text-to-Video generation with auto-prompt enhancement
- 8-10 second video duration support via frame interpolation
- 1080p and 4K output with intelligent upscaling
- Temporal consistency and realism optimization
- Enterprise GPU optimization (H100/H200/RTX Pro 6000)
- Automatic model downloading from HuggingFace
"""

from pathlib import Path

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Model auto-download configuration
MODELS_CONFIG = {
    "ltx_video": {
        "filename": "ltx-video-13b-v0.9.7-distilled.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-distilled.safetensors",
        "size_gb": 13,
        "path": "checkpoints",
    },
    "text_encoder": {
        "filename": "t5-v1_1-xxl-fp16.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-Video/resolve/main/text_encoders/t5-v1_1-xxl-fp16.safetensors",
        "size_gb": 4.7,
        "path": "clip",
    },
    "vae": {
        "filename": "vae-ft-mse-840000-ema-pruned.safetensors",
        "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
        "size_gb": 0.335,
        "path": "vae",
    },
}


def get_comfyui_models_dir():
    """Get ComfyUI models directory"""
    # Try to find ComfyUI root by going up from current directory
    current_dir = Path(__file__).resolve()

    # Look for ComfyUI root (contains 'models' directory)
    for parent in current_dir.parents:
        models_dir = parent / "models"
        if models_dir.exists() and models_dir.is_dir():
            return models_dir

    # Fallback: use current directory + ../../models
    return current_dir.parent.parent / "models"


def check_and_download_models():
    """Check for required models and provide download instructions"""
    try:
        models_dir = get_comfyui_models_dir()
        missing_models = []

        for model_name, config in MODELS_CONFIG.items():
            model_path = models_dir / config["path"] / config["filename"]
            if not model_path.exists():
                missing_models.append((model_name, config))

        if missing_models:
            print("\n" + "=" * 70)
            print("LTX-Video: Required models not found")
            print("=" * 70)
            print("\nThe following models are required but not found:")
            print(f"\nModels directory: {models_dir}")

            for model_name, config in missing_models:
                print(f"\n• {model_name.upper()}:")
                print(f"  File: {config['filename']}")
                print(f"  Size: ~{config['size_gb']} GB")
                print(f"  Install path: {models_dir / config['path']}")
                print(f"  Download: {config['url']}")

            print("\n" + "-" * 70)
            print("AUTOMATIC DOWNLOAD:")
            print("-" * 70)
            print("\nTo auto-download all models, run:")
            print(
                '  python -c "from model_downloader import download_all_models; download_all_models()"'
            )

            print("\n" + "-" * 70)
            print("MANUAL DOWNLOAD:")
            print("-" * 70)
            print("\nOr download manually using:")
            print(
                "  huggingface-cli download <model_repo> <filename> --local-dir <path>"
            )

            print("\n" + "=" * 70)
            print("NOTE: LTX-Video nodes will not function until models are installed.")
            print("=" * 70 + "\n")

    except Exception as e:
        print(f"[LTX-Video] Warning: Could not check for models: {e}")


# Check for models on import
check_and_download_models()
