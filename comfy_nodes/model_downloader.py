"""
LTX-Video Model Downloader
Automatic download of required models from HuggingFace
"""

from pathlib import Path
from typing import Dict


def get_comfyui_models_dir():
    """Get ComfyUI models directory"""
    current_dir = Path(__file__).resolve()

    # Look for ComfyUI root (contains 'models' directory)
    for parent in current_dir.parents:
        models_dir = parent / "models"
        if models_dir.exists() and models_dir.is_dir():
            return models_dir

    # Fallback
    return current_dir.parent.parent / "models"


def download_model(model_config: Dict, models_dir: Path) -> bool:
    """Download a single model using huggingface_hub"""
    try:
        from huggingface_hub import hf_hub_download

        target_dir = models_dir / model_config["path"]
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / model_config["filename"]

        print(f"\nDownloading {model_config['filename']}...")
        print(f"  Size: ~{model_config['size_gb']} GB")
        print(f"  Target: {target_file}")

        # Parse the URL to get repo and filename
        url = model_config["url"]
        if "huggingface.co/" in url:
            # Extract repo from URL
            parts = url.split("huggingface.co/")[1].split("/resolve/")
            repo_id = parts[0]
            file_parts = parts[1].split("/", 1)
            branch = file_parts[0]
            filename = (
                file_parts[1] if len(file_parts) > 1 else model_config["filename"]
            )

            # Download using hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=branch,
                cache_dir=None,
                force_download=False,
            )

            # Copy to target location if different
            if Path(downloaded_path) != target_file:
                import shutil

                shutil.copy2(downloaded_path, target_file)

            print(f"  ✓ Downloaded successfully to {target_file}")
            return True
        else:
            print(f"  ✗ Unsupported URL format: {url}")
            return False

    except ImportError:
        print("  ✗ Error: huggingface_hub not installed")
        print("    Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  ✗ Error downloading: {e}")
        return False


def download_all_models():
    """Download all required models"""
    from . import MODELS_CONFIG

    models_dir = get_comfyui_models_dir()

    print("=" * 70)
    print("LTX-Video: Downloading Required Models")
    print("=" * 70)
    print(f"\nModels directory: {models_dir}")

    total_size = sum(config["size_gb"] for config in MODELS_CONFIG.values())
    print(f"Total download size: ~{total_size:.1f} GB")
    print("\nThis may take a while depending on your internet connection...")

    results = []
    for model_name, config in MODELS_CONFIG.items():
        target_file = models_dir / config["path"] / config["filename"]

        if target_file.exists():
            print(f"\n✓ {model_name.upper()}: Already exists, skipping")
            results.append(True)
        else:
            success = download_model(config, models_dir)
            results.append(success)

    print("\n" + "=" * 70)
    if all(results):
        print("SUCCESS: All models downloaded successfully!")
    else:
        print("WARNING: Some models failed to download.")
        print("Please download them manually or check your internet connection.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    download_all_models()
