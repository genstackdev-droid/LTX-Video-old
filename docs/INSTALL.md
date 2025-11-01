# LTX-Video ComfyUI Installation Guide

Complete installation instructions for LTX-Video Production Nodes in ComfyUI.

## Prerequisites

### System Requirements

**Minimum**:
- GPU: NVIDIA RTX 3060 (12GB VRAM) or equivalent
- CPU: 4+ cores
- RAM: 16GB system memory
- Storage: 20GB free space for models
- OS: Windows 10/11, Linux, macOS (with Metal support)

**Recommended**:
- GPU: NVIDIA RTX 4070 Ti (16GB VRAM) or better
- CPU: 8+ cores
- RAM: 32GB system memory
- Storage: 50GB free space (SSD preferred)

**Optimal**:
- GPU: NVIDIA RTX 4090 (24GB VRAM) or A6000
- CPU: 12+ cores (for pre/post processing)
- RAM: 64GB system memory
- Storage: 100GB SSD for models and cache

### Software Requirements

- **Python**: 3.10 or 3.11 (3.12 supported but not fully tested)
- **CUDA**: 11.8 or 12.1 (for NVIDIA GPUs)
- **PyTorch**: 2.1.0 or later
- **ComfyUI**: Latest version (2024.11+ recommended)

## Installation Methods

### Method 1: Quick Install (Recommended for Most Users)

This method installs LTX-Video nodes directly into your ComfyUI custom_nodes folder.

#### Step 1: Navigate to ComfyUI Custom Nodes

**Windows**:
```cmd
cd C:\ComfyUI\custom_nodes
```

**Linux/macOS**:
```bash
cd ~/ComfyUI/custom_nodes
```

#### Step 2: Clone Repository

```bash
git clone https://github.com/genstackdev-droid/LTX-Video-old
cd LTX-Video-old
```

#### Step 3: Install Dependencies

**Windows**:
```cmd
python -m pip install -r comfy_nodes/requirements.txt
```

**Linux/macOS**:
```bash
pip install -r comfy_nodes/requirements.txt
```

#### Step 4: Restart ComfyUI

Close and restart ComfyUI completely. The LTX-Video nodes should appear under the "LTX-Video" category.

#### Step 5: Verify Installation

1. Open ComfyUI
2. Right-click on canvas → Add Node
3. Look for "LTX-Video" category
4. You should see 5 nodes:
   - LTX-Video Full Pipeline (Production)
   - LTX-Video Prompt Enhancer
   - LTX-Video Frame Interpolator
   - LTX-Video 4K Upscaler
   - LTX-Video Advanced Sampler

### Method 2: Development Install

For developers who want to modify the code or contribute.

#### Step 1: Clone and Navigate

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/genstackdev-droid/LTX-Video-old
cd LTX-Video-old
```

#### Step 2: Install in Editable Mode

```bash
pip install -e .[inference]
```

This allows you to make changes to the code without reinstalling.

#### Step 3: Restart ComfyUI

### Method 3: Manual Installation

If you prefer manual control or have a custom setup.

#### Step 1: Download Repository

Download ZIP from GitHub or clone:
```bash
git clone https://github.com/genstackdev-droid/LTX-Video-old
```

#### Step 2: Copy to ComfyUI

Copy the entire `LTX-Video-old` folder to:
- Windows: `C:\ComfyUI\custom_nodes\`
- Linux/macOS: `~/ComfyUI/custom_nodes/`

#### Step 3: Install Core Dependencies

```bash
cd LTX-Video-old
pip install torch>=2.1.0 diffusers>=0.28.2 transformers>=4.47.2
pip install sentencepiece>=0.1.96 huggingface-hub einops timm
pip install imageio[ffmpeg] av torchvision opencv-python
```

#### Step 4: Restart ComfyUI

## Post-Installation Setup

### Download Models (Required)

The nodes require LTX-Video model weights from HuggingFace.

#### Automatic Download (Recommended)

Models will auto-download on first use. Ensure you have:
- HuggingFace account (free)
- `huggingface-cli` logged in

```bash
huggingface-cli login
```

#### Manual Download

1. Visit [Lightricks/LTX-Video on HuggingFace](https://huggingface.co/Lightricks/LTX-Video)
2. Download model files to: `ComfyUI/models/ltx_video/`
3. Update `model_path` parameter in nodes

### Optional Enhancements

#### For ESRGAN Upscaling

```bash
pip install basicsr realesrgan
```

#### For RIFE Frame Interpolation

```bash
pip install rife-ncnn-vulkan-python
```

#### For FP8 Optimization

```bash
pip install bitsandbytes
```

### Import Example Workflows

1. Locate workflow files:
   - `comfy_nodes/workflows/ltx_production_workflow.json`
   - `comfy_nodes/workflows/ltx_advanced_workflow.json`

2. In ComfyUI:
   - Click "Load" button
   - Navigate to workflow file
   - Click "Open"

3. The workflow will load with all nodes connected

## Troubleshooting

### Issue: Nodes Don't Appear in ComfyUI

**Solutions**:
1. Verify installation path is correct
2. Check ComfyUI console for error messages
3. Ensure all dependencies are installed
4. Try clearing ComfyUI cache:
   - Delete `ComfyUI/custom_nodes/__pycache__/`
   - Restart ComfyUI

### Issue: Import Error on Node Load

**Error**: `"LTX-Video modules not available"`

**Solutions**:
```bash
cd ComfyUI/custom_nodes/LTX-Video-old
pip install -e .[inference]
```

### Issue: CUDA Out of Memory

**Solutions**:
1. Close other applications using GPU
2. In nodes, reduce:
   - Resolution (4K → 1080p → 720p)
   - Steps (80 → 60 → 40)
   - Duration (10s → 8s)
3. Enable gradient checkpointing (if available)
4. Use FP16 precision (automatic on modern GPUs)

### Issue: Slow Model Loading

**Solutions**:
1. Move models to SSD if on HDD
2. Increase system RAM
3. Use quantized models (FP8 versions)
4. Pre-load models before generation

### Issue: Poor Video Quality

**Solutions**:
1. Use "Basic" prompt mode for auto-enhancement
2. Increase steps to 80-100
3. Add detail to prompts
4. Adjust CFG scale (try 7-9)
5. Use negative prompts
6. Try different seeds

### Issue: Frame Stuttering

**Solutions**:
1. Enable frame interpolation
2. Increase target FPS
3. Add "smooth motion" to prompt
4. Use higher step count
5. Ensure consistent motion in prompt

### Issue: Python Version Conflicts

**Solutions**:
1. Create dedicated environment:
```bash
python -m venv ltxvideo_env
source ltxvideo_env/bin/activate  # Linux/macOS
ltxvideo_env\Scripts\activate     # Windows
pip install -r comfy_nodes/requirements.txt
```

2. Point ComfyUI to use this environment

## Verification Tests

### Test 1: Node Loading

1. Open ComfyUI
2. Add Node → LTX-Video category
3. All 5 nodes should be available

**Expected**: Success
**If fails**: Check installation and restart ComfyUI

### Test 2: Basic Generation

1. Import `ltx_production_workflow.json`
2. Set prompt: "a cat walking"
3. Queue prompt

**Expected**: Video generation starts
**If fails**: Check model download and dependencies

### Test 3: Memory Usage

1. Use 720p resolution
2. 40 steps
3. 8s duration

**Expected**: Generation completes without OOM
**If fails**: Reduce settings or close other apps

## Performance Optimization

### For 12GB VRAM (Minimum Setup)

**Recommended Settings**:
- Resolution: 720p or 1080p
- Steps: 40-60
- Duration: 8s
- CFG Scale: 7-8
- Disable 4K upscaling

**Expected Performance**: 1-2 minutes per 8s video

### For 16GB VRAM (Balanced Setup)

**Recommended Settings**:
- Resolution: 1080p
- Steps: 60-80
- Duration: 8-10s
- CFG Scale: 8-9
- Enable tiled upscaling to 4K

**Expected Performance**: 2-4 minutes per 8s video

### For 24GB+ VRAM (Optimal Setup)

**Recommended Settings**:
- Resolution: 4K
- Steps: 80-100
- Duration: 10s
- CFG Scale: 8-10
- Full quality upscaling

**Expected Performance**: 5-10 minutes per 10s video

## Updating

### Update to Latest Version

```bash
cd ComfyUI/custom_nodes/LTX-Video-old
git pull origin main
pip install -r comfy_nodes/requirements.txt --upgrade
```

Then restart ComfyUI.

### Check Current Version

Look at `CHANGELOG.md` or check git log:
```bash
cd ComfyUI/custom_nodes/LTX-Video-old
git log -1
```

## Uninstalling

### Complete Removal

1. Stop ComfyUI
2. Delete folder:
   ```bash
   rm -rf ComfyUI/custom_nodes/LTX-Video-old
   ```
3. Restart ComfyUI

### Keep Models (Reinstall Later)

Models are stored separately and won't be deleted. You can reinstall without re-downloading.

## Getting Help

### Resources

- **Documentation**: `comfy_nodes/README_COMFYUI.md`
- **Examples**: `comfy_nodes/workflows/`
- **Changelog**: `CHANGELOG.md`
- **Issues**: GitHub Issues page

### Community Support

- **Discord**: [LTX Platform Discord](https://discord.gg/ltxplatform)
- **Reddit**: r/StableDiffusion
- **GitHub Discussions**: Project discussions page

### Reporting Bugs

When reporting issues, include:
1. ComfyUI version
2. Python version
3. GPU model and VRAM
4. Error messages from console
5. Steps to reproduce
6. Workflow JSON (if applicable)

## FAQ

**Q: Do I need a HuggingFace account?**
A: Yes, for automatic model downloads. Manual downloads are also possible.

**Q: Can I use this with AMD GPUs?**
A: ROCm support is experimental. NVIDIA GPUs recommended.

**Q: Does this work on Apple Silicon?**
A: Yes, with MPS backend. Performance may vary.

**Q: How much disk space do I need?**
A: ~20GB for models, more for outputs (videos can be large).

**Q: Can I generate longer than 10 seconds?**
A: Yes, via manual frame extension or autoregressive generation (advanced).

**Q: Is commercial use allowed?**
A: Yes, under OpenRail-M license. Check LICENSE file for details.

**Q: Do I need the official ComfyUI-LTXVideo nodes?**
A: No, these nodes are standalone. But they can coexist.

---

**Installation complete!** Start creating amazing videos with LTX-Video in ComfyUI! 🎬✨
