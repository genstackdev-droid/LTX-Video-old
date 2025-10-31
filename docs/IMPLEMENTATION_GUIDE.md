# LTX-Video Enhanced Implementation Guide

This guide provides step-by-step instructions for using the enhanced features of LTX-Video.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Audio-Video Generation](#audio-video-generation)
4. [4K Video Generation](#4k-video-generation)
5. [Multi-Keyframe Conditioning](#multi-keyframe-conditioning)
6. [Extended Video Generation](#extended-video-generation)
7. [Optimization Settings](#optimization-settings)
8. [Troubleshooting](#troubleshooting)

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- CUDA >= 11.8 (for GPU acceleration)
- 16GB+ VRAM recommended (12GB minimum with optimizations)

### Install from Source

```bash
git clone https://github.com/Lightricks/LTX-Video.git
cd LTX-Video
pip install -e .[inference]
```

### Optional Dependencies

For audio generation:
```bash
pip install soundfile librosa
```

For video I/O with audio:
```bash
pip install imageio[ffmpeg] av
```

## Basic Usage

### Text-to-Video Generation

```python
from ltx_video.inference import infer, InferenceConfig

# Basic T2V generation
infer(
    InferenceConfig(
        pipeline_config="configs/ltxv-13b-0.9.8-distilled.yaml",
        prompt="A serene lake at sunset with mountains in the background",
        height=512,
        width=768,
        num_frames=257,  # ~10 seconds at 24 FPS
        output_path="output.mp4",
    )
)
```

### Image-to-Video Generation

```python
infer(
    InferenceConfig(
        pipeline_config="configs/ltxv-13b-0.9.8-distilled.yaml",
        prompt="The woman smiles and waves at the camera",
        conditioning_media_paths=["input_image.jpg"],
        conditioning_start_frames=[0],
        height=512,
        width=768,
        num_frames=257,
        output_path="output.mp4",
    )
)
```

## Audio-Video Generation

### Setup

```python
from ltx_video.pipelines.ltx_audio_video_pipeline import LTXAudioVideoPipeline

# Initialize pipeline with audio support
pipeline = LTXAudioVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    audio_encoder_path="path/to/audio_encoder",  # Optional
)

# Enable audio generation
pipeline.enable_audio_generation(True)
```

### Generate Video with Audio

```python
output = pipeline(
    prompt="A guitarist performing on stage",
    audio_prompt="Acoustic guitar music with crowd ambience",  # Optional
    num_frames=257,
    height=512,
    width=768,
    generate_audio=True,
    audio_guidance_scale=3.0,
)

# Save output with embedded audio
save_video_with_audio(
    output.videos,
    output.audio,
    output.audio_sample_rate,
    "output_with_audio.mp4",
)
```

### Audio Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `audio_prompt` | Text description for audio | Same as video prompt | - |
| `audio_guidance_scale` | Guidance scale for audio | 3.0 | 1.0-10.0 |
| `audio_strength` | Strength of audio conditioning | 1.0 | 0.0-1.0 |
| `audio_sample_rate` | Output audio sample rate | 16000 | 16000-48000 |

### Tips for Audio Generation

1. **Prompt Engineering**:
   - Describe ambient sounds: "with birds chirping"
   - Specify music style: "upbeat jazz music"
   - Include dialogue: "person speaking clearly"

2. **Synchronization**:
   - Audio automatically syncs with video motion
   - Lip-sync works for close-up dialogue
   - Expected alignment: <100ms

3. **Quality vs Speed**:
   - Higher `audio_guidance_scale` = better quality, slower
   - Lower `audio_sample_rate` = faster, slightly lower quality

## 4K Video Generation

### Enable 4K Mode

```python
from ltx_video.models.autoencoders.vae_hierarchical import HierarchicalVAEEncoder
from ltx_video.attention.efficient_attention import SparseAttention

# Use hierarchical VAE for efficient 4K encoding
vae_encoder = HierarchicalVAEEncoder(
    in_channels=3,
    latent_channels=128,
    compression_levels=2,
    use_wavelet=True,
)

# Generate 4K video
output = pipeline(
    prompt="Aerial view of a pristine beach at sunset",
    height=2160,  # 4K
    width=3840,   # 4K
    num_frames=257,
    num_inference_steps=40,
    guidance_scale=7.5,
)
```

### 4K Performance Settings

```python
# For RTX 4090 (24GB VRAM)
pipeline_config = {
    "use_hierarchical_vae": True,
    "use_sparse_attention": True,
    "attention_block_size": 64,
    "enable_quantization": True,
    "quantization_mode": "fp8",
}

# For RTX 3090 (24GB VRAM) - reduce quality slightly
pipeline_config = {
    "use_hierarchical_vae": True,
    "use_sparse_attention": True,
    "attention_block_size": 64,
    "enable_quantization": True,
    "quantization_mode": "fp8",
    "num_inference_steps": 30,  # Reduce from 40
}
```

### Resolution Presets

| Preset | Resolution | Frames | RTX 4090 | RTX 3090 | RTX 4060 Ti |
|--------|-----------|--------|----------|----------|-------------|
| 1080p  | 1920x1080 | 257    | 8s       | 15s      | 30s         |
| 2K     | 2560x1440 | 257    | 12s      | 22s      | N/A         |
| 4K     | 3840x2160 | 257    | 15s      | 30s      | N/A         |

## Multi-Keyframe Conditioning

### Basic Multi-Keyframe Setup

```python
from ltx_video.conditioning.multi_keyframe_processor import (
    MultiKeyframeProcessor, Keyframe
)
from PIL import Image

# Initialize processor
processor = MultiKeyframeProcessor(
    latent_channels=128,
    num_frames=257,
    interpolation_mode="cubic",
)

# Load keyframe images
keyframe_images = [
    Image.open("keyframe_0.jpg"),
    Image.open("keyframe_128.jpg"),
    Image.open("keyframe_256.jpg"),
]

# Create keyframes (after VAE encoding)
keyframes = [
    Keyframe(frame_index=0, latent=encoded_kf0, strength=1.0),
    Keyframe(frame_index=128, latent=encoded_kf128, strength=0.9),
    Keyframe(frame_index=256, latent=encoded_kf256, strength=1.0),
]

# Process keyframes
conditioning_latents, conditioning_masks = processor.process_keyframes(keyframes)

# Generate video with keyframe conditioning
output = pipeline(
    prompt="Character walking through a forest",
    keyframes=keyframes,
    num_frames=257,
)
```

### Interpolation Modes

```python
# Linear interpolation (fast, simple)
processor = MultiKeyframeProcessor(interpolation_mode="linear")

# Cubic interpolation (smooth, recommended)
processor = MultiKeyframeProcessor(interpolation_mode="cubic")

# Learned interpolation (best quality, slower)
processor = MultiKeyframeProcessor(interpolation_mode="learned")
```

### Camera Control

```python
from ltx_video.conditioning.camera_controller import (
    CameraController, CameraParameters, CameraMotionType
)

# Define camera motion
camera_params = CameraParameters(
    position=(0.0, 0.0, 5.0),
    rotation=(0.0, 0.0, 0.0),
    fov=60.0,
    motion_type=CameraMotionType.DOLLY_FORWARD,
    motion_speed=0.5,
    motion_smoothness=0.8,
)

# Initialize controller
camera_controller = CameraController(
    latent_channels=128,
    num_frames=257,
)

# Generate motion constraints
motion_field, motion_masks = camera_controller(camera_params)

# Generate video with camera control
output = pipeline(
    prompt="A cinematic reveal shot",
    camera_parameters=camera_params,
    num_frames=257,
)
```

### Combining Keyframes and Camera

```python
# Use both keyframes and camera control
output = pipeline(
    prompt="Dynamic character animation",
    keyframes=keyframes,              # Maintain character consistency
    camera_parameters=camera_params,  # Add camera movement
    num_frames=257,
)
```

## Extended Video Generation

### Autoregressive Generation

```python
def generate_long_video(prompt, duration=15, chunk_size=5):
    """Generate extended video using autoregressive conditioning."""
    
    fps = 24
    total_frames = duration * fps
    chunk_frames = chunk_size * fps
    overlap_frames = 1 * fps  # 1 second overlap
    
    chunks = []
    previous_chunk = None
    
    for chunk_idx in range(0, total_frames, chunk_frames - overlap_frames):
        # Generate chunk with conditioning on previous
        chunk = pipeline(
            prompt=prompt,
            num_frames=min(chunk_frames, total_frames - chunk_idx),
            conditioning_frames=previous_chunk[-overlap_frames:] if previous_chunk else None,
            height=512,
            width=768,
        )
        
        chunks.append(chunk)
        previous_chunk = chunk
    
    # Concatenate chunks with smooth blending
    full_video = blend_chunks(chunks, overlap_frames)
    return full_video
```

### Chunk Blending

```python
def blend_chunks(chunks, overlap_frames):
    """Blend video chunks with smooth transitions."""
    
    blended = []
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk: use all frames
            blended.append(chunk)
        else:
            # Subsequent chunks: blend overlap region
            prev_overlap = blended[-1][-overlap_frames:]
            curr_overlap = chunk[:overlap_frames]
            
            # Weighted blending
            for j in range(overlap_frames):
                alpha = j / overlap_frames
                blended_frame = (1 - alpha) * prev_overlap[j] + alpha * curr_overlap[j]
                blended[-1][-(overlap_frames-j)] = blended_frame
            
            # Add non-overlapping frames
            blended.append(chunk[overlap_frames:])
    
    return torch.cat(blended, dim=0)
```

### Memory-Efficient Streaming

```python
def generate_and_stream(prompt, duration=30):
    """Generate long video with disk streaming."""
    
    import tempfile
    
    temp_dir = tempfile.mkdtemp()
    chunk_size = 5  # seconds
    
    for chunk_idx in range(0, duration, chunk_size):
        # Generate chunk
        chunk = generate_chunk(prompt, chunk_idx, chunk_size)
        
        # Save immediately to disk
        chunk_path = f"{temp_dir}/chunk_{chunk_idx:04d}.mp4"
        save_video(chunk, chunk_path)
        
        # Clear memory
        del chunk
        torch.cuda.empty_cache()
    
    # Concatenate chunks from disk
    concat_chunks(temp_dir, "final_output.mp4")
```

## Optimization Settings

### VRAM Optimization

```python
# Enable all optimizations for maximum VRAM savings
pipeline.enable_vram_optimization(
    use_multi_query_attention=True,      # 66% KV cache reduction
    use_gradient_checkpointing=True,     # 40% activation memory reduction
    use_flash_attention=True,            # 2-4x speedup, lower memory
    enable_cpu_offload=False,            # Keep on GPU for speed
)
```

### Quantization

```python
from ltx_video.quantization import quantize_model

# Quantize model to FP8
quantized_pipeline = quantize_model(
    pipeline,
    mode="fp8",                    # Options: 'fp8', 'fp16', 'bf16'
    preserve_layers=['vae'],       # Keep critical layers in FP32
)

# Expected memory savings: ~50%
# Expected quality loss: <2%
```

### Performance Presets

```python
# Fast Draft (speed priority)
preset_fast = {
    "num_inference_steps": 20,
    "quantization_mode": "fp8",
    "use_sparse_attention": True,
    "attention_block_size": 128,
}

# Balanced (recommended)
preset_balanced = {
    "num_inference_steps": 35,
    "quantization_mode": "bf16",
    "use_sparse_attention": True,
    "attention_block_size": 64,
}

# High Quality (quality priority)
preset_quality = {
    "num_inference_steps": 50,
    "quantization_mode": "fp32",
    "use_sparse_attention": False,
}
```

## Troubleshooting

### Out of Memory Errors

**Problem**: CUDA out of memory error during generation

**Solutions**:
1. Enable quantization:
   ```python
   pipeline = quantize_model(pipeline, mode="fp8")
   ```

2. Reduce resolution:
   ```python
   height = 512  # Instead of 1080
   width = 768   # Instead of 1920
   ```

3. Reduce batch size:
   ```python
   num_videos_per_prompt = 1  # Generate one at a time
   ```

4. Enable CPU offloading:
   ```python
   pipeline.enable_sequential_cpu_offload()
   ```

### Audio Sync Issues

**Problem**: Audio not syncing with video

**Solutions**:
1. Check frame rate matches audio calculation:
   ```python
   # Ensure FPS is consistent
   fps = 24
   audio_duration = num_frames / fps
   ```

2. Increase audio guidance scale:
   ```python
   audio_guidance_scale = 5.0  # Higher = tighter sync
   ```

3. Use audio conditioning:
   ```python
   # Provide reference audio for better sync
   audio_conditioning = encode_audio(reference_audio)
   ```

### Low Quality 4K Output

**Problem**: 4K video looks blurry or artifacted

**Solutions**:
1. Increase inference steps:
   ```python
   num_inference_steps = 50  # From default 40
   ```

2. Disable aggressive quantization:
   ```python
   quantization_mode = "bf16"  # Instead of "fp8"
   ```

3. Use hierarchical VAE properly:
   ```python
   use_hierarchical_vae = True
   use_wavelet = True  # Preserves high-frequency details
   ```

### Temporal Inconsistency

**Problem**: Flickering or inconsistent motion between frames

**Solutions**:
1. Increase keyframe count:
   ```python
   # Add more keyframes for stability
   keyframes = [
       Keyframe(0, ...),
       Keyframe(64, ...),
       Keyframe(128, ...),
       Keyframe(192, ...),
       Keyframe(256, ...),
   ]
   ```

2. Use temporal smoothing:
   ```python
   enable_temporal_smoothing = True
   temporal_window_size = 5
   ```

3. For long videos, increase overlap:
   ```python
   chunk_overlap = 2  # 2 seconds instead of 1
   ```

## Advanced Configuration

### Custom Pipeline Configuration

```python
from ltx_video.pipelines.ltx_audio_video_pipeline import LTXAudioVideoPipeline

pipeline = LTXAudioVideoPipeline(
    transformer=custom_transformer,
    vae=custom_vae,
    audio_encoder=custom_audio_encoder,
    text_encoder=text_encoder,
    scheduler=scheduler,
    # Optimization settings
    use_hierarchical_vae=True,
    use_sparse_attention=True,
    quantization_mode="fp8",
    # Feature flags
    enable_audio=True,
    enable_4k=True,
    enable_multi_keyframe=True,
)
```

### Training Custom Models

For fine-tuning or training custom control models, see:
- [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer)

## Best Practices

1. **Start Small**: Test with lower resolutions before generating 4K
2. **Use Presets**: Start with balanced preset, adjust as needed
3. **Monitor VRAM**: Use `nvidia-smi` to check memory usage
4. **Prompt Engineering**: Detailed prompts yield better results
5. **Keyframe Spacing**: Space keyframes evenly for smooth interpolation
6. **Audio Sync**: Describe audio explicitly in prompts
7. **Chunk Size**: Use 5-second chunks for long videos

## Getting Help

- **Issues**: https://github.com/Lightricks/LTX-Video/issues
- **Discussions**: https://github.com/Lightricks/LTX-Video/discussions
- **Discord**: https://discord.gg/ltxplatform

## What's Next

- Explore the [examples](../examples/) directory for complete code samples
- Read the [Architecture](./ARCHITECTURE.md) document for technical details
- Check the [Performance Report](./PERFORMANCE_REPORT.md) for benchmarks
