# LTX-Video ComfyUI Custom Node

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-Video)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-green)](https://github.com/comfyanonymous/ComfyUI)

Production-ready ComfyUI custom node for LTX-Video text-to-video generation with hyper-realistic output and temporal consistency.

## ✨ Features

- **🎬 Text-to-Video Generation**: Create videos from text descriptions
- **🖼️ Image-to-Video**: Animate static images with motion
- **📹 Multi-Keyframe Support**: Control video with multiple reference frames
- **⏱️ Extended Duration**: Generate 8-10 second videos
- **🎯 4K Output**: Support for 1080p and 4K resolution
- **✨ Auto-Prompt Enhancement**: Transform basic prompts into professional results
- **💾 VRAM Efficient**: Optimized for 12GB+ VRAM GPUs
- **📦 Auto-Download**: Models download automatically from HuggingFace

## 🚀 Quick Installation

### Step 1: Clone Repository

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/genstackdev-droid/LTX-Video-old
```

### Step 2: Install Dependencies

```bash
cd LTX-Video-old
pip install -r requirements_comfyui.txt
```

### Step 3: Restart ComfyUI

Restart ComfyUI and look for "LTX-Video" nodes in the node browser.

## 📦 Required Models

Models will automatically download on first use. Total size: ~18GB.

| Model | Size | Install Path | Download Command |
|-------|------|--------------|------------------|
| **LTX-Video Core** | ~13GB | `ComfyUI/models/checkpoints/` | See below |
| **Text Encoder (T5-XXL)** | ~4.7GB | `ComfyUI/models/clip/` | See below |
| **VAE** | ~335MB | `ComfyUI/models/vae/` | See below |

### Automatic Download

Models will download automatically when you first use the node. If you want to download them manually:

```bash
# Method 1: Using the downloader script (from ComfyUI root or custom_nodes/LTX-Video-old)
python -c "from model_downloader import download_all_models; download_all_models()"

# Method 2: Using huggingface-cli
cd ComfyUI/models

# Download LTX-Video core model
huggingface-cli download Lightricks/LTX-Video ltxv-13b-0.9.8-distilled.safetensors --local-dir checkpoints/

# Download Text Encoder
huggingface-cli download Lightricks/LTX-Video text_encoders/t5-v1_1-xxl-fp16.safetensors --local-dir clip/

# Download VAE
huggingface-cli download stabilityai/sd-vae-ft-mse-original vae-ft-mse-840000-ema-pruned.safetensors --local-dir vae/
```

### Manual Download (Alternative)

If automatic download fails, download manually from HuggingFace:

1. **LTX-Video Core Model** (~13GB)
   - URL: https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-distilled.safetensors
   - Save to: `ComfyUI/models/checkpoints/ltx-video-13b-v0.9.7-distilled.safetensors`

2. **Text Encoder** (~4.7GB)
   - URL: https://huggingface.co/Lightricks/LTX-Video/resolve/main/text_encoders/t5-v1_1-xxl-fp16.safetensors
   - Save to: `ComfyUI/models/clip/t5-v1_1-xxl-fp16.safetensors`

3. **VAE** (~335MB)
   - URL: https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors
   - Save to: `ComfyUI/models/vae/vae-ft-mse-840000-ema-pruned.safetensors`

## 💻 System Requirements

### Minimum
- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 16GB
- **Storage**: 20GB free space
- **OS**: Windows 10/11, Linux, macOS

### Recommended
- **GPU**: NVIDIA RTX 4070 Ti (16GB VRAM) or better
- **RAM**: 32GB
- **Storage**: 50GB SSD

### Optimal
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or A6000
- **RAM**: 64GB
- **Storage**: 100GB SSD

## 🎯 Usage

### Basic Text-to-Video

1. Add "LTX-Video Full Pipeline" node to your workflow
2. Enter your prompt: e.g., "A serene lake at sunset"
3. Select duration (8s or 10s) and resolution (720p/1080p/4K)
4. Click "Queue Prompt"

### Example Prompts

**Basic (Auto-Enhanced)**:
```
"a cat walking in a garden"
```

**Detailed**:
```
"A cinematic walk through a misty forest at dawn, volumetric fog filtering golden sunlight through ancient trees, smooth camera dolly forward, photorealistic, 8k quality"
```

## 📋 Available Nodes

1. **LTX-Video Full Pipeline** - Complete text-to-video generation
2. **LTX-Video Prompt Enhancer** - Automatic prompt improvement
3. **LTX-Video Frame Interpolator** - Extend video duration
4. **LTX-Video 4K Upscaler** - Intelligent resolution upscaling
5. **LTX-Video Advanced Sampler** - Fine-grained generation control

## 🛠️ Troubleshooting

### Node Not Appearing
- Ensure you installed dependencies: `pip install -r requirements_comfyui.txt`
- Restart ComfyUI completely
- Check ComfyUI console for error messages

### Out of Memory Errors
- Reduce resolution (use 720p instead of 1080p/4K)
- Reduce number of inference steps
- Close other GPU applications

### Models Not Loading
- Verify models are in correct directories
- Check file names match exactly
- Try manual download if auto-download fails

## 📚 Additional Documentation

- [Installation Guide](docs/INSTALL.md) - Detailed installation instructions
- [Quick Reference](docs/QUICK_REFERENCE.md) - Node parameters and workflows
- [ComfyUI Documentation](docs/README_COMFYUI.md) - Complete ComfyUI integration guide
- [Workflows](workflows/) - Example workflow JSON files
- [Original README](README_ORIGINAL.md) - LTX-Video project information

## 🔗 Links

- [LTX-Video Official Website](https://ltx.video)
- [HuggingFace Model](https://huggingface.co/Lightricks/LTX-Video)
- [Research Paper](https://arxiv.org/abs/2501.00103)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- LTX-Video by Lightricks
- ComfyUI by comfyanonymous
- Based on state-of-the-art research in video generation

## 📞 Support

- Open an issue on GitHub for bugs or feature requests
- Check existing issues for solutions to common problems
- Join the [LTX Discord](https://discord.gg/ltxplatform) for community support
