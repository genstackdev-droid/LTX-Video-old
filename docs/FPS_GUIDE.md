# LTX-Video FPS Configuration Guide

## Overview

LTX-Video now supports adjustable frame rates from **12 FPS to 120 FPS**, allowing you to generate videos at various smoothness levels depending on your needs and hardware capabilities.

## FPS Parameter

The `fps` parameter is available in the **LTX-Video Full Pipeline** node and controls the frame rate of the generated video.

### Default Settings
- **Default FPS**: 25 (standard video frame rate)
- **Minimum FPS**: 12 (cinematic slow motion)
- **Maximum FPS**: 120 (ultra-smooth high frame rate)

## Usage

### In ComfyUI Node

1. Open the **LTX-Video Full Pipeline** node
2. Locate the `fps` parameter in the node settings
3. Set your desired frame rate (12-120)
4. Generate your video

### Frame Count Calculation

The total number of frames is calculated as:
```
total_frames = duration (seconds) × fps
```

Examples:
- 10 seconds @ 25 FPS = 250 frames
- 10 seconds @ 60 FPS = 600 frames
- 10 seconds @ 120 FPS = 1200 frames

## Recommended FPS Settings

### Standard Content (24-30 FPS)
- **Use Case**: General video content, storytelling, cinematic
- **FPS**: 24-30
- **VRAM**: ~8-12GB
- **Generation Time**: ~2-5 minutes on RTX 4090
- **Quality**: Excellent for most use cases

### Smooth Motion (60 FPS)
- **Use Case**: Action scenes, sports, gaming content
- **FPS**: 60
- **VRAM**: ~12-16GB
- **Generation Time**: ~4-8 minutes on RTX 4090
- **Quality**: Very smooth motion, great for fast-paced content

### Ultra-Smooth (120 FPS)
- **Use Case**: Professional production, slow-motion playback, high-end displays
- **FPS**: 120
- **VRAM**: ~16-24GB
- **Generation Time**: ~8-15 minutes on RTX 4090
- **Quality**: Exceptional smoothness, ideal for slow-motion effects

### Cinematic (12-24 FPS)
- **Use Case**: Artistic effects, retro look, resource-constrained environments
- **FPS**: 12-24
- **VRAM**: ~6-8GB
- **Generation Time**: ~1-3 minutes on RTX 4090
- **Quality**: Classic cinematic feel

## Hardware Requirements by FPS

### 25-30 FPS (Standard)
- **Minimum GPU**: RTX 3060 (12GB)
- **Recommended GPU**: RTX 4070 (12GB)
- **RAM**: 16GB

### 60 FPS (Smooth)
- **Minimum GPU**: RTX 4070 Ti (16GB)
- **Recommended GPU**: RTX 4080 (16GB)
- **RAM**: 32GB

### 120 FPS (Ultra)
- **Minimum GPU**: RTX 4090 (24GB)
- **Recommended GPU**: H100/H200 (80GB) for production
- **RAM**: 64GB

## Frame Interpolation

For extending duration or achieving higher FPS from lower FPS generations, use the **LTX-Video Frame Interpolator** node:

1. Generate at a lower FPS (e.g., 25 FPS)
2. Connect output to Frame Interpolator node
3. Set target_fps to desired rate (up to 120 FPS)
4. Choose interpolation mode:
   - **Linear**: Fast, good for smooth motion
   - **RIFE**: Better quality, motion-aware
   - **FILM**: Highest quality, slowest

## Best Practices

### For 60+ FPS Generation

1. **Start with lower resolution**: Generate at 720p or 1080p first
2. **Use shorter durations**: Start with 5-8 seconds instead of 10
3. **Monitor VRAM**: Enable VRAM monitoring in ComfyUI
4. **Batch carefully**: Avoid generating multiple high-FPS videos simultaneously

### Memory Optimization

- **Lower FPS during base generation**: Generate at 25-30 FPS, then interpolate to higher FPS
- **Use frame interpolation**: More memory-efficient than direct high-FPS generation
- **Reduce resolution**: Lower resolution uses less VRAM and allows higher FPS

### Quality Tips

- **60 FPS works best for**: Fast motion, camera pans, action sequences
- **120 FPS ideal for**: Slow-motion playback (play at 30 FPS for 4x slow-mo)
- **25-30 FPS sufficient for**: Most narrative content, talking heads, general scenes

## Troubleshooting

### Out of Memory Errors

If you encounter VRAM issues:
1. Reduce FPS (e.g., 60 → 30)
2. Reduce duration (e.g., 10s → 8s)
3. Lower resolution (e.g., 1080p → 720p)
4. Use frame interpolation instead of direct high-FPS generation

### Choppy Motion at High FPS

- Ensure your display supports the target FPS
- Check that video player can handle high frame rates
- Verify output video codec supports high FPS (use H.264 or H.265)

### Long Generation Times

- High FPS requires proportionally more computation
- Consider generating at standard FPS and interpolating
- Use Quality Mode: "Standard" instead of "Ultra" for faster generation

## Technical Details

### Frame Rate Parameter
- Type: Integer
- Range: 12-120
- Default: 25
- Passed to pipeline as float for precision

### Checkpoint Compatibility
- The FPS parameter is compatible with all LTX-Video checkpoints
- Uses explicit timesteps when checkpoint defines `allowed_inference_steps`
- Falls back to `num_inference_steps` for flexible checkpoints

## Examples

### Example 1: Standard Video
```
Duration: 10s
FPS: 25
Resolution: 1080p
Total Frames: 250
```

### Example 2: High Frame Rate
```
Duration: 8s
FPS: 60
Resolution: 720p
Total Frames: 480
```

### Example 3: Ultra Smooth
```
Duration: 5s
FPS: 120
Resolution: 1080p
Total Frames: 600
```

### Example 4: Interpolated High FPS
```
Step 1: Generate at 25 FPS (200 frames for 8s)
Step 2: Interpolate to 60 FPS (480 frames)
Result: 8s video at 60 FPS with optimized VRAM usage
```

## Conclusion

The adjustable FPS feature allows you to balance quality, smoothness, and resource usage based on your specific needs. Start with standard FPS settings and experiment with higher rates as your hardware and use case require.

For most users, **25-30 FPS** provides excellent results. Use **60 FPS** for action content, and reserve **120 FPS** for professional production or slow-motion effects.
