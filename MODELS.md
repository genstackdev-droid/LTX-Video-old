# LTX-Video Models Download Guide

This guide provides complete instructions for downloading all required models for the LTX-Video ComfyUI custom node.

## 📦 Required Models Overview

| Model | Size | Purpose | Required |
|-------|------|---------|----------|
| LTX-Video Core | ~13GB | Main video generation model | ✅ Yes |
| Text Encoder (T5-XXL) | ~4.7GB | Text encoding | ✅ Yes |
| VAE | ~335MB | Image encoding/decoding | ✅ Yes |

**Total Download Size**: ~18GB

## 🚀 Quick Setup (Automatic Download)

### Option 1: Let ComfyUI Auto-Download

Models will automatically download when you first use the node. Just:

1. Install the custom node
2. Restart ComfyUI
3. Add an LTX-Video node to your workflow
4. The models will download automatically on first use

### Option 2: Pre-Download Using Script

From your ComfyUI root directory or the LTX-Video-old custom node directory:

```bash
python -c "from model_downloader import download_all_models; download_all_models()"
```

This will download all models to the correct locations automatically.

## 📥 Manual Download Instructions

If automatic download fails or you prefer manual download, follow these steps:

### Prerequisites

Install huggingface-hub if not already installed:

```bash
pip install -U huggingface-hub
```

### Download Commands

Execute these commands from your ComfyUI root directory:

#### 1. LTX-Video Core Model (~13GB)

**Method 1: Using Python (Recommended - Modern API)**
```python
from huggingface_hub import hf_hub_download
import os

# Create directory
os.makedirs("models/checkpoints", exist_ok=True)

# Download model
hf_hub_download(
    repo_id="Lightricks/LTX-Video",
    filename="ltxv-13b-0.9.8-distilled.safetensors",
    local_dir="models/checkpoints",
    local_dir_use_symlinks=False,
    resume_download=True
)
```

**Method 2: Using Command Line**
```bash
# Create directory if it doesn't exist
mkdir -p models/checkpoints

# Download using Python one-liner
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Lightricks/LTX-Video', 'ltxv-13b-0.9.8-distilled.safetensors', local_dir='models/checkpoints', local_dir_use_symlinks=False)"
```

**Method 3: Direct Download with wget**
```bash
cd models/checkpoints/
wget https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-distilled.safetensors
```

#### 2. Text Encoder (~4.7GB)

**Note:** The text encoder (T5-XXL) is automatically downloaded from `PixArt-alpha/PixArt-XL-2-1024-MS` on first use. No manual download needed!

If you want to pre-download it:

```python
from huggingface_hub import snapshot_download

# Download entire model with text encoder
snapshot_download("PixArt-alpha/PixArt-XL-2-1024-MS")
```

#### 3. VAE (~335MB)

**Note:** The VAE is embedded in the main checkpoint file. No separate download needed!

## 📂 Directory Structure

After downloading, your ComfyUI models directory should look like this:

```
ComfyUI/
└── models/
    ├── checkpoints/
    │   └── ltx-video-13b-v0.9.7-distilled.safetensors (~13GB)
    ├── clip/
    │   └── t5-v1_1-xxl-fp16.safetensors (~4.7GB)
    └── vae/
        └── vae-ft-mse-840000-ema-pruned.safetensors (~335MB)
```

## ✅ Verify Installation

After downloading, verify the models are in place:

### On Linux/macOS:
```bash
# From ComfyUI root directory
ls -lh models/checkpoints/ltx-video-13b-v0.9.7-distilled.safetensors
ls -lh models/clip/t5-v1_1-xxl-fp16.safetensors
ls -lh models/vae/vae-ft-mse-840000-ema-pruned.safetensors
```

### On Windows:
```cmd
dir models\checkpoints\ltx-video-13b-v0.9.7-distilled.safetensors
dir models\clip\t5-v1_1-xxl-fp16.safetensors
dir models\vae\vae-ft-mse-840000-ema-pruned.safetensors
```

## 🔐 Authentication (For Private Models)

If you need to download private or gated models:

**Method 1: Using Python (Recommended)**
```python
from huggingface_hub import login

# Login with your token
login(token="hf_...")  # Get token from https://huggingface.co/settings/tokens

# Or login interactively
login()  # Will prompt for token
```

**Method 2: Using Environment Variable**
```bash
export HF_TOKEN="hf_..."
```

**Method 3: Using huggingface-cli (if available)**
```bash
huggingface-cli login
```

**Note:** For public models like Lightricks/LTX-Video, authentication is NOT required.

## 🔧 Troubleshooting

### Downloads Fail or Timeout

If downloads fail:

1. **Check internet connection**: Ensure stable connection
2. **Resume downloads**: The modern API automatically resumes interrupted downloads
3. **Try alternative download method**: Use wget/curl as fallback
4. **Check disk space**: Ensure you have at least 20GB free
5. **Update packages**: Run `pip install -U huggingface-hub transformers`

### Models Not Found

If ComfyUI says models are missing:

1. **Check file names**: Ensure exact names match
2. **Check paths**: Models must be in correct subdirectories
3. **Restart ComfyUI**: After downloading, restart completely
4. **Check permissions**: Ensure files are readable
5. **Clear cache**: Delete `~/.cache/huggingface/` and re-download

### Slow Download Speed

If downloads are slow:

1. **Use CDN**: HuggingFace automatically uses CDN
2. **Resume feature**: Interrupted downloads will resume automatically
3. **Download from another location**: Try from a different network
4. **Use aria2c for direct downloads**: Faster with multiple connections:
   ```bash
   aria2c -x 16 -s 16 <download_url>
   ```

### Authentication Errors

If you get authentication errors:

1. **Create token**: Go to https://huggingface.co/settings/tokens
2. **Login properly**: Use `login()` method from huggingface-hub
3. **Check token**: Ensure token has read permissions
4. **Clear old tokens**: Delete `~/.huggingface/token` and login again

## 🌐 Direct Download Links

If all else fails, download directly in your browser:

- **LTX-Video Core**: https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-distilled.safetensors
- **Text Encoder**: https://huggingface.co/Lightricks/LTX-Video/resolve/main/text_encoders/t5-v1_1-xxl-fp16.safetensors
- **VAE**: https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors

Then manually place them in the correct directories as shown in the Directory Structure section.

## 📞 Need Help?

- Check [Installation Guide](docs/INSTALL.md) for more details
- Open an issue on GitHub
- Join the [LTX Discord](https://discord.gg/ltxplatform)
