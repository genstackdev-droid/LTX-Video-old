# LTX-Video ComfyUI Quick Reference

Fast reference guide for common tasks and settings.

## 🚀 Quick Start (30 seconds)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/genstackdev-droid/LTX-Video-old
cd LTX-Video-old
pip install -r requirements_comfyui.txt
# Restart ComfyUI
```

## 📊 Node Quick Reference

### LTX-Video Full Pipeline

**Best For**: Complete video generation in one node

| Parameter | Quick Settings | Notes |
|-----------|---------------|-------|
| **prompt** | Describe your video | Be specific about action & scene |
| **duration** | 8s (fast) or 10s (quality) | Longer = more processing time |
| **resolution** | 720p (preview), 1080p (production), 4K (max) | Higher = more VRAM |
| **prompt_mode** | Basic (auto-enhance) or Detailed (manual) | Basic adds quality keywords |
| **steps** | 40 (fast), 60 (balanced), 80 (quality) | More steps = better quality |
| **cfg_scale** | 8.0 (recommended) | 7-9 for most cases |
| **seed** | -1 (random) or fixed number | Same seed = reproducible |

**Quick Preset - Fast Preview**:
- Resolution: 720p | Duration: 8s | Steps: 40 | CFG: 7.5

**Quick Preset - Production Quality**:
- Resolution: 1080p | Duration: 8s | Steps: 60 | CFG: 8.0

**Quick Preset - Maximum Quality**:
- Resolution: 4K | Duration: 10s | Steps: 80 | CFG: 8.5

### LTX-Video Prompt Enhancer

**Best For**: Manual prompt optimization

| Level | When to Use | Result |
|-------|-------------|--------|
| **Minimal** | Already good prompts | Basic quality keywords |
| **Moderate** | Standard prompts | Quality + motion coherence |
| **Maximum** | Basic prompts | Full professional enhancement |

| Style | Best For | Keywords Added |
|-------|----------|----------------|
| **Realistic** | Photorealistic videos | Natural lighting, 8k uhd, detailed |
| **Cinematic** | Film-like videos | Dramatic lighting, depth of field |
| **Artistic** | Creative videos | Unique perspective, vibrant |

### LTX-Video Frame Interpolator

**Best For**: Smoother motion, extended duration

| FPS | Use Case | Notes |
|-----|----------|-------|
| **12-15** | Stylized/artistic | Lower data, stable |
| **24-25** | Standard (recommended) | Cinematic feel |
| **30** | Smooth motion | Higher quality needed |
| **60** | Ultra-smooth | Experimental, high compute |

### LTX-Video 4K Upscaler

**Best For**: Resolution enhancement

| Method | Speed | Quality | VRAM |
|--------|-------|---------|------|
| **Bicubic** | Fast | Good | Low |
| **Lanczos** | Medium | Better | Medium |
| **ESRGAN** | Slow | Best | High |

**Scale Factor**:
- 1.5x: Subtle enhancement
- 2.0x: Standard upscale (recommended)
- 4.0x: Maximum quality

## 🎯 Common Workflows

### Workflow 1: Quick Test

```
[Full Pipeline] → [Video Output]
```
**Settings**: 720p, 8s, 40 steps, Basic mode
**Time**: 30-60 seconds
**Use**: Quick preview, iteration

### Workflow 2: Production Video

```
[Full Pipeline] → [Video Output]
```
**Settings**: 1080p, 8s, 60 steps, Detailed mode
**Time**: 1-3 minutes
**Use**: Final outputs, client work

### Workflow 3: Maximum Quality

```
[Prompt Enhancer] → [Full Pipeline] → [Upscaler] → [Video Output]
```
**Settings**: 4K, 10s, 80 steps, Maximum enhancement
**Time**: 5-10 minutes
**Use**: Showcase, portfolio pieces

### Workflow 4: Extended Duration

```
[Full Pipeline] → [Frame Interpolator] → [Video Output]
```
**Settings**: 1080p, 10s, interpolate to 25 FPS
**Time**: 2-4 minutes
**Use**: Longer clips, smooth motion

## 🎨 Prompt Templates

### Template: Basic Scene
```
"[subject] [action] in [location], [time of day]"
Example: "A deer walking through forest, early morning"
```

### Template: Cinematic
```
"[camera movement] of [subject] [action], [environment details], [lighting], [quality keywords]"
Example: "Slow dolly shot of woman walking through rain, city lights reflecting, cinematic lighting, 8k"
```

### Template: Detailed
```
"[subject with details] [specific action], [environment with atmosphere], [camera angle and movement], [lighting and colors], [technical quality]"
Example: "A red sports car speeding through mountain roads at sunset, leaves swirling behind, aerial tracking shot, golden hour lighting with lens flares, photorealistic 8k uhd"
```

## 💾 VRAM Quick Guide

### 12GB VRAM (RTX 3060, 4060 Ti)

**Safe Settings**:
- Resolution: 720p or 1080p
- Steps: 40-60
- Duration: 8s
- Avoid: 4K generation, high step counts

**If OOM**:
1. Drop to 720p
2. Reduce steps to 40
3. Shorten to 8s
4. Close other apps

### 16GB VRAM (RTX 4070 Ti, A4000)

**Safe Settings**:
- Resolution: 1080p or 4K
- Steps: 60-80
- Duration: 8-10s
- Can use: Tiled 4K upscaling

### 24GB+ VRAM (RTX 4090, A6000)

**Safe Settings**:
- Resolution: Any (including 4K)
- Steps: 80-100
- Duration: 10s+
- Can use: All features maxed

## ⚡ Speed Optimization

### Faster Generation (2x-3x speedup)

1. **Use 720p** instead of 1080p
2. **Reduce steps** to 40-50
3. **Shorter duration** (8s instead of 10s)
4. **Basic prompt mode** (skip manual enhancement)
5. **Skip upscaling** (use direct output)

### Quality vs Speed Matrix

| Quality | Resolution | Steps | Duration | Time (RTX 4090) |
|---------|-----------|-------|----------|-----------------|
| Preview | 720p | 40 | 8s | ~45s |
| Good | 1080p | 60 | 8s | ~90s |
| Great | 1080p | 80 | 10s | ~3m |
| Amazing | 4K | 80 | 10s | ~5m |
| Perfect | 4K | 100 | 10s | ~8m |

## 🔧 Common Issues & Quick Fixes

### Problem: Blurry Output
**Quick Fix**: Increase steps to 80+, use Detailed mode

### Problem: Stuttering Motion
**Quick Fix**: Add "smooth motion" to prompt, use interpolator

### Problem: Low Quality
**Quick Fix**: Use Basic mode for auto-enhancement, increase CFG to 9

### Problem: OOM Error
**Quick Fix**: Reduce resolution one step, reduce steps to 40

### Problem: Artifacts
**Quick Fix**: Add to negative prompt: "artifacts, distorted, glitchy"

### Problem: Slow Generation
**Quick Fix**: Use 720p, 40 steps, Basic mode

## 📝 Prompt Keywords Cheat Sheet

### Quality Keywords
```
photorealistic, 8k uhd, high detail, professional quality, 
masterpiece, sharp focus, high resolution
```

### Motion Keywords
```
smooth motion, fluid movement, coherent, natural dynamics,
stable, flowing, seamless
```

### Lighting Keywords
```
cinematic lighting, natural light, golden hour, volumetric fog,
ray tracing, soft shadows, dramatic lighting
```

### Camera Keywords
```
tracking shot, dolly shot, pan, tilt, aerial view, 
close-up, wide shot, stabilized camera
```

### Style Keywords
```
cinematic, artistic, realistic, dramatic, epic, 
moody, vibrant, atmospheric
```

### Negative Keywords (Use in negative prompt)
```
blurry, low quality, distorted, watermark, text, 
artifacts, duplicate frames, stuttering, compression artifacts
```

## 🎯 Recommended Settings by Use Case

### Use Case: Social Media Content
- Resolution: 1080p
- Duration: 8s
- Steps: 50
- Mode: Basic
- **Why**: Fast, good quality, standard format

### Use Case: Client Work
- Resolution: 1080p or 4K
- Duration: 8-10s
- Steps: 80
- Mode: Detailed
- **Why**: Professional quality, reliable

### Use Case: Portfolio/Showcase
- Resolution: 4K
- Duration: 10s
- Steps: 100
- Mode: Maximum enhancement
- **Why**: Best possible quality

### Use Case: Rapid Iteration/Testing
- Resolution: 720p
- Duration: 8s
- Steps: 40
- Mode: Basic
- **Why**: Fast feedback, low resources

### Use Case: Motion Study
- Resolution: 1080p
- Duration: 10s
- Steps: 60
- Use: Frame Interpolator at 30 FPS
- **Why**: Smooth, detailed motion

## 🎬 Video Export Settings

### For YouTube
- Resolution: 1080p or 4K
- Format: MP4 (H.264)
- FPS: 25 or 30
- Bitrate: High

### For Instagram/TikTok
- Resolution: 1080p (1080x1920 portrait)
- Format: MP4
- FPS: 25 or 30
- Duration: 8-10s

### For Twitter/X
- Resolution: 720p or 1080p
- Format: MP4
- FPS: 25
- File size: <512MB

### For Professional Use
- Resolution: 4K
- Format: MP4 or ProRes
- FPS: 25 (cinematic) or 30 (broadcast)
- Bitrate: Maximum

## 📊 Performance Metrics

### Expected Generation Times

| GPU | 720p/8s/60 | 1080p/8s/60 | 4K/10s/80 |
|-----|-----------|-------------|-----------|
| RTX 4090 | 45s | 90s | 5m |
| RTX 4080 | 60s | 2m | 6m |
| RTX 4070 Ti | 75s | 2.5m | 8m |
| RTX 3090 | 60s | 2.5m | 7m |
| RTX 3080 | 90s | 3m | 10m |

*Times approximate, vary by system and settings*

## 🔗 Quick Links

- **Full Documentation**: `docs/README_COMFYUI.md`
- **Installation Guide**: `docs/INSTALL.md`
- **Changelog**: `CHANGELOG.md`
- **Workflows**: `workflows/`
- **Issues**: GitHub Issues page
- **Discord**: https://discord.gg/ltxplatform

---

**Pro Tip**: Save this file for offline reference! 📖
