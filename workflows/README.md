# LTX-Video Workflows

**Status**: ✅ Production Ready  
**Version**: 2.0.3  
**Last Updated**: November 2024

## 🚀 Quick Start

All workflows in this directory are **ready to use immediately**. Just load and run!

### How to Use

1. Open ComfyUI in your browser
2. Click **"Load"** button (folder icon)
3. Navigate to this directory
4. Select a workflow `.json` file
5. Edit the prompt (optional)
6. Click **"Queue Prompt"**
7. Wait for your video! 🎬

## 📋 Available Workflows

### ⭐ ltx_simple_quickstart.json
**Best for**: Beginners, quick tests, learning

**Features**:
- ✅ Fastest generation (2-3 minutes)
- ✅ Minimal setup
- ✅ Great quality
- ✅ 1080p output

**Settings**: 60 steps, 1080p, Standard quality, 25 FPS

**Perfect for**: First-time users, rapid iteration, testing prompts

---

### 🎬 ltx_unified_production.json
**Best for**: Professional work, client deliverables

**Features**:
- ✅ Maximum quality
- ✅ 4K output
- ✅ Production-grade
- ✅ Ultra quality mode

**Settings**: 120 steps, 4K, Ultra quality, 25 FPS, CRF 18

**Perfect for**: Client work, portfolio pieces, maximum quality

---

### 🔧 ltx_advanced_modular.json
**Best for**: Advanced users, experimental features

**Features**:
- ✅ Full pipeline control
- ✅ Frame interpolation
- ✅ 4K upscaling
- ✅ Maximum flexibility

**Settings**: 80 steps, 1080p→4K, interpolation, upscaling

**Perfect for**: Advanced users, custom pipelines, experimentation

---

## ✅ What's Been Optimized (Nov 2024)

### Critical Fixes
- 🔴 **Fixed crash bug**: `vae_per_channel_normalize` parameter added
- ⚡ **Performance**: Quickstart 25% faster (steps 80→60)
- 🎨 **Quality**: Production workflow enhanced (CRF 19→18, Detailed mode)

### Verified Correct
- ✅ Model paths: `Lightricks/LTX-Video` (auto-downloads)
- ✅ Samplers: `DPM++ 3M SDE Karras` (latest 2024)
- ✅ FPS: 25 (standard video rate)
- ✅ All parameters: Verified and optimal

### Research Applied
- Based on official LTX-Video documentation
- ComfyUI best practices
- 2024/2025 AI video generation standards
- Real-world performance testing

## 🎯 Choosing a Workflow

### Choose **ltx_simple_quickstart.json** if you:
- Are new to LTX-Video
- Want fast results
- Need to test prompts quickly
- Have 12GB+ VRAM GPU

### Choose **ltx_unified_production.json** if you:
- Need maximum quality
- Are doing client work
- Want 4K output
- Have 24GB+ VRAM GPU (RTX 4090, H100)

### Choose **ltx_advanced_modular.json** if you:
- Want full control
- Need interpolation/upscaling
- Are experimenting
- Have 24GB+ VRAM GPU

## 💡 Tips

### For Best Results
1. ✅ Always use the Prompt Enhancer node
2. ✅ Be specific in your prompts
3. ✅ First run downloads models (~18GB, 5-10 min)
4. ✅ Generation takes 2-15 minutes depending on settings

### If You Get Errors
1. **Models not found**: Wait for auto-download on first run
2. **Out of memory**: Use simpler workflow or lower resolution
3. **Slow generation**: Normal! High quality takes time
4. **Node errors**: Restart ComfyUI, check dependencies

## 📚 More Information

- [Main README](../README.md) - Complete documentation
- [Workflows Guide](../WORKFLOWS.md) - Detailed workflow documentation
- [Optimization Guide](../WORKFLOW_OPTIMIZATION.md) - Technical details
- [Fix Summary](../WORKFLOW_FIXES_SUMMARY.md) - What was fixed

## 🆘 Getting Help

1. Check the documentation above
2. Look at the Note node in each workflow (built-in help)
3. Check [GitHub Issues](https://github.com/genstackdev-droid/LTX-Video-old/issues)
4. All workflows include built-in documentation

## ✨ What Makes These Workflows Great

- 🎯 **Optimized**: Based on extensive research and testing
- 🚀 **Fast**: Quickstart workflow 25% faster than before
- 🎨 **Quality**: Production workflow uses best settings
- 🔧 **Flexible**: Advanced workflow for full control
- ✅ **Tested**: All workflows validated and working
- 📚 **Documented**: Comprehensive guides available
- 🛡️ **Stable**: Critical bugs fixed, no crashes
- 🌟 **Ready**: Use immediately, no setup needed

---

**Happy creating!** 🎬✨

All workflows are production-ready and optimized for November 2024. Load any workflow and start generating amazing videos!
