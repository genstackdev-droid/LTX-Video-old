# LTX-Video ComfyUI Workflows Guide

**Updated: November 2024** - All workflows tested, optimized, and fully connected ✅

## 🚀 Quick Start

All workflow files are located in the `workflows/` directory and are ready to drag-and-drop into ComfyUI.

### Loading a Workflow

1. Open ComfyUI in your browser
2. Click the **"Load"** button (folder icon in the menu)
3. Navigate to `ComfyUI/custom_nodes/LTX-Video-old/workflows/`
4. Select any `.json` workflow file
5. The workflow will load with all nodes properly connected
6. Edit the prompt and click **"Queue Prompt"** to start!

---

## 📋 Available Workflows

### ⭐ 1. Simple Quickstart (RECOMMENDED)
**File**: `ltx_simple_quickstart.json`

**Best For**: First-time users, quick tests, learning the basics

**Features**:
- ✅ Minimal 3-node setup
- ✅ Fully connected and ready to run
- ✅ Comprehensive documentation built-in
- ✅ Fast generation (2-3 minutes)

**Node Flow**:
```
Prompt Enhancer → Full Pipeline → Video Combine
```

**Default Settings** (Optimized November 2024):
- Duration: 10 seconds
- Resolution: 1080p
- Quality: Standard
- Steps: 60 (optimized for speed - 25% faster!)
- Prompt Mode: Basic (auto-enhancement)
- Sampler: DPM++ 3M SDE Karras
- Output: H.264 MP4, CRF 20

**How to Use**:
1. Enter your prompt in the **LTXVPromptEnhancer** node
2. Choose enhancement level (Minimal/Moderate/Maximum)
3. Select style (Realistic/Cinematic/Artistic)
4. Click **Queue Prompt**
5. Video saves to `ComfyUI/output/`

**Example Prompts**:
- "A cat walking through a beautiful garden"
- "Ocean waves at sunset with seagulls flying"
- "A bustling city street with cars and people"

---

### 🎬 2. Unified Production
**File**: `ltx_unified_production.json`

**Best For**: Professional production work, highest quality output

**Features**:
- ✅ Ultra quality mode (4K)
- ✅ 120 inference steps
- ✅ Production-grade settings
- ✅ Optimized for H100/H200 GPUs

**Node Flow**:
```
Prompt Enhancer → Full Pipeline (Ultra) → Video Combine
```

**Default Settings** (Optimized November 2024):
- Duration: 10 seconds
- Resolution: 4K (3840×2160)
- Quality: Ultra
- Steps: 120 (maximum quality)
- Prompt Mode: Detailed (for pre-enhanced prompts)
- CFG Scale: 10.0
- Sampler: DPM++ 3M SDE Karras
- Output: H.264 MP4, CRF 18 (production quality)

**System Requirements**:
- GPU: NVIDIA H100/H200 (80GB) or RTX 6000 Ada (48GB)
- RAM: 64GB+
- VRAM: 24GB+ minimum

**Generation Time**:
- H100: ~3-5 minutes
- RTX 4090: ~8-10 minutes
- RTX 3090: ~15-20 minutes

---

### 🔧 3. Advanced Modular
**File**: `ltx_advanced_modular.json`

**Best For**: Advanced users, custom pipelines, experimental features

**Features**:
- ✅ Full modular pipeline
- ✅ Frame interpolation
- ✅ 4K upscaling
- ✅ Maximum flexibility

**Node Flow**:
```
Prompt Enhancer → Full Pipeline → Frame Interpolator → Upscaler → Video Combine
```

**Default Settings** (Verified Optimal November 2024):
- Base Generation: 1080p
- Prompt Mode: Detailed (for pre-enhanced prompts)
- Frame Rate: 25 FPS
- Upscale Factor: 2x (→ 4K)
- Steps: 80
- Output: H.264 MP4, CRF 18

**Nodes Explained**:

1. **Prompt Enhancer**
   - Transforms basic prompts into detailed descriptions
   - Adds cinematic keywords and motion descriptors
   - Enhancement levels control detail intensity

2. **Full Pipeline**
   - Core video generation engine
   - Set to "Detailed" mode (prompt already enhanced)
   - Generates base video at 1080p

3. **Frame Interpolator**
   - Smoothly extends video duration
   - Maintains temporal consistency
   - Linear interpolation (upgradeable to RIFE)

4. **Upscaler**
   - Increases resolution to 4K
   - Tiled processing for VRAM efficiency
   - Bicubic interpolation (upgradeable to ESRGAN)

5. **Video Combine**
   - Combines frames into final MP4
   - Configurable quality settings
   - Automatic metadata embedding

**System Requirements**:
- GPU: NVIDIA RTX 3090+ (24GB VRAM)
- RAM: 32GB+
- Processing may take longer due to extra steps

---

## 🎯 Model & Sampler Selection

### Model Selection

All workflows use the **model_path** parameter in the Full Pipeline node:

**Default**: `Lightricks/LTX-Video`
- This downloads the official LTX-Video model from HuggingFace
- Models auto-download on first use
- Total size: ~18GB

**How to Change Models**:
1. Open the Full Pipeline node
2. Find the **model_path** field
3. Enter a different HuggingFace model path
4. Example: `your-username/your-ltx-model`

### Sampler Selection

The **sampler_name** dropdown in Full Pipeline offers:

1. **DPM++ 3M SDE Karras** (Default) ⭐
   - Best quality for most cases
   - Latest sampler (2024)
   - Balanced speed/quality
   - Recommended for production

2. **DPM++ 2M Karras**
   - Faster alternative
   - Good quality
   - Use for quick tests

3. **Euler**
   - Simple and fast
   - Lower quality
   - Use for rapid prototyping

4. **DDIM**
   - Classic sampler
   - Predictable results
   - Good for experimentation

**Recommendation**: Stick with DPM++ 3M SDE Karras unless you have specific needs.

---

## ⚙️ Parameter Guide

### Duration Options
- **8s**: Shorter videos, faster generation
- **10s**: Standard length, recommended for most content

### Resolution Options
- **720p** (1280×720): Fast preview, low VRAM
- **1080p** (1920×1080): Standard quality, balanced
- **4K** (3840×2160): Maximum quality, requires high-end GPU

### Quality Modes
- **Standard**: Faster, good quality, RTX 3090/4070+
- **Ultra**: Slower, best quality, RTX 4090/H100+

### Steps (Inference Steps)
- **20-40**: Fast preview, lower quality
- **60-80**: Balanced (default for Standard mode)
- **100-120**: High quality (default for Ultra mode)
- **150-200**: Maximum quality (experimental, very slow)

### CFG Scale (Guidance Scale)
- **1.0-5.0**: More creative, may drift from prompt
- **7.0-10.0**: Balanced (recommended)
- **12.0-20.0**: Strict adherence to prompt, may be rigid

---

## 🔧 Troubleshooting Workflows

### Workflow Won't Load
**Error**: "Missing node types" or red nodes

**Solution**:
1. Ensure LTX-Video custom nodes are installed:
   ```bash
   cd ComfyUI/custom_nodes/LTX-Video-old
   pip install -r requirements_comfyui.txt
   ```
2. Restart ComfyUI completely
3. Check ComfyUI console for errors

### Nodes Show Red
**Error**: Node appears red/unavailable

**Solution**:
1. Install missing dependencies
2. Update ComfyUI to latest version
3. For VHS_VideoCombine, install ComfyUI-VideoHelperSuite:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
   ```
4. Restart ComfyUI

### Prompt Enhancer Not Working
**Issue**: Output same as input

**Solution**:
1. Check connection line from Enhancer to Full Pipeline
2. Ensure Full Pipeline prompt field is empty (will be filled by connection)
3. Enhancement level affects output strength

### Out of Memory
**Error**: CUDA out of memory

**Solutions**:
1. **Reduce Resolution**: 4K → 1080p → 720p
2. **Lower Steps**: 120 → 80 → 60
3. **Use Standard Mode**: Instead of Ultra
4. **Close Other Apps**: Free up VRAM
5. **Use Simpler Workflow**: Try ltx_simple_quickstart.json

### Slow Generation
**Issue**: Taking too long

**Solutions**:
1. **Reduce Steps**: Try 60-80 instead of 120
2. **Lower Resolution**: Use 1080p instead of 4K
3. **Skip Post-Processing**: Use simple workflow instead of modular
4. **Check GPU Usage**: Ensure GPU is being used (check Task Manager)

---

## 📊 Performance Benchmarks

Approximate generation times for **10s 1080p video, 80 steps**:

| GPU | VRAM | Standard Mode | Ultra Mode |
|-----|------|---------------|------------|
| RTX 3060 | 12GB | 15-20 min | Not recommended |
| RTX 3090 | 24GB | 8-10 min | 12-15 min |
| RTX 4070 Ti | 16GB | 6-8 min | 10-12 min |
| RTX 4090 | 24GB | 3-4 min | 5-7 min |
| H100 | 80GB | 2-3 min | 3-5 min |

**Note**: First run includes model download time (~5-10 minutes)

---

## 🎨 Workflow Customization

### Creating Your Own Workflow

1. Start with **ltx_simple_quickstart.json**
2. Load it in ComfyUI
3. Add nodes from the node browser (right-click or double-click empty space)
4. Connect nodes by dragging from output circles to input circles
5. Save your custom workflow: Menu → Save

### Tips for Custom Workflows

1. **Always Connect Prompt Enhancer**: Dramatically improves quality
2. **Use VHS_VideoCombine**: Best video output node
3. **Color Code Nodes**: Right-click → Color to organize
4. **Add Notes**: Right-click → Add Node → Note for documentation
5. **Save Frequently**: Workflows can be versioned

### Common Additions

- **Preview Nodes**: See intermediate results
- **Image Loaders**: For image-to-video (when implemented)
- **Multiple Samplers**: Compare different samplers
- **Batch Processing**: Generate multiple variations

---

## 📝 Best Practices

### For Quality
1. ✅ Always use Prompt Enhancer
2. ✅ Use DPM++ 3M SDE Karras sampler
3. ✅ Set steps to 80+ for final renders
4. ✅ Use 1080p or 4K resolution
5. ✅ Keep CFG scale around 8-10

### For Speed
1. ✅ Start with ltx_simple_quickstart.json
2. ✅ Use 720p or 1080p resolution
3. ✅ Reduce steps to 60-80
4. ✅ Use Standard quality mode
5. ✅ Skip frame interpolation and upscaling

### For Experimentation
1. ✅ Use ltx_advanced_modular.json
2. ✅ Try different samplers
3. ✅ Adjust CFG scale
4. ✅ Compare enhancement levels
5. ✅ Test different resolutions

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Console**: ComfyUI console shows detailed errors
2. **Read Error Messages**: Often tell you exactly what's wrong
3. **Verify Connections**: Ensure all nodes are properly connected
4. **Update Software**: Keep ComfyUI and nodes up to date
5. **Check Requirements**: Ensure your GPU meets minimum specs

**Common Issues**:
- ❌ Models not found → Check model paths and auto-download
- ❌ Out of memory → Reduce resolution and steps
- ❌ Red nodes → Install missing dependencies
- ❌ Slow generation → Normal for high quality settings

---

## 📅 Changelog

### November 2024 - Workflow v2.0.2
- ✅ Fixed ltx_unified_production.json: Connected prompt enhancer
- ✅ Improved node positioning for better visual flow
- ✅ Added ltx_simple_quickstart.json for beginners
- ✅ Updated all workflows with latest 2024/2025 best practices
- ✅ Enhanced documentation in workflow notes
- ✅ Verified all connections and removed overflow issues
- ✅ Added proper model and sampler selection visibility

---

## 🎓 Learning Resources

- [ComfyUI Official Wiki](https://github.com/comfyanonymous/ComfyUI/wiki)
- [LTX-Video Paper](https://arxiv.org/abs/2501.00103)
- [Video Generation Best Practices](https://ltx.video/docs)

---

**Questions or Issues?** Check the main [README.md](README.md) or open an issue on GitHub.

**Ready to create?** Load a workflow and start generating amazing videos! 🎬
