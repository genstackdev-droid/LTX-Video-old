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

Install huggingface-cli if not already installed:

```bash
pip install huggingface-hub
```

### Download Commands

Execute these commands from your ComfyUI root directory:

#### 1. LTX-Video Core Model (~13GB)

```bash
# Create directory if it doesn't exist
mkdir -p models/checkpoints

# Download the model
huggingface-cli download Lightricks/LTX-Video \
  ltxv-13b-0.9.8-distilled.safetensors \
  --local-dir models/checkpoints/ \
  --local-dir-use-symlinks False
```

**Alternative: Direct Download**
```bash
cd models/checkpoints/
wget https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-distilled.safetensors \
  -O ltx-video-13b-v0.9.7-distilled.safetensors
```

#### 2. Text Encoder (~4.7GB)

```bash
# Create directory if it doesn't exist
mkdir -p models/clip

# Download the model
huggingface-cli download Lightricks/LTX-Video \
  text_encoders/t5-v1_1-xxl-fp16.safetensors \
  --local-dir models/clip/ \
  --local-dir-use-symlinks False
```

**Alternative: Direct Download**
```bash
cd models/clip/
wget https://huggingface.co/Lightricks/LTX-Video/resolve/main/text_encoders/t5-v1_1-xxl-fp16.safetensors
```

#### 3. VAE (~335MB)

```bash
# Create directory if it doesn't exist
mkdir -p models/vae

# Download the model
huggingface-cli download stabilityai/sd-vae-ft-mse-original \
  vae-ft-mse-840000-ema-pruned.safetensors \
  --local-dir models/vae/ \
  --local-dir-use-symlinks False
```

**Alternative: Direct Download**
```bash
cd models/vae/
wget https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors
```

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

## 🔧 Troubleshooting

### Downloads Fail or Timeout

If downloads fail:

1. **Check internet connection**: Ensure stable connection
2. **Try alternative download method**: Use wget/curl if huggingface-cli fails
3. **Download in parts**: Use a download manager that supports resume
4. **Check disk space**: Ensure you have at least 20GB free

### Models Not Found

If ComfyUI says models are missing:

1. **Check file names**: Ensure exact names match
2. **Check paths**: Models must be in correct subdirectories
3. **Restart ComfyUI**: After downloading, restart completely
4. **Check permissions**: Ensure files are readable

### Slow Download Speed

If downloads are slow:

1. **Use closest mirror**: HuggingFace has CDN, but try different times
2. **Download from another location**: Try from a different network
3. **Use aria2c**: Faster download with multiple connections:
   ```bash
   aria2c -x 16 -s 16 <download_url>
   ```

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
