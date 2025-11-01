"""Example: Extended 10-15 second video generation.

Demonstrates autoregressive video generation with temporal coherence
for long-form video output.
"""

import torch


def main():
    """Generate extended video with autoregressive conditioning."""
    
    print("Extended Video Generation Example")
    print("=" * 50)
    
    # Parameters for long video
    params = {
        "prompt": (
            "A time-lapse of a flower blooming in a garden. The flower slowly "
            "opens its petals as the sun moves across the sky. Bees visit the "
            "flower. Shadows shift throughout the day. Cinematic nature documentary style."
        ),
        "duration": 15,  # 15 seconds
        "chunk_size": 5,  # Generate in 5-second chunks
        "overlap": 1,    # 1 second overlap for smooth transitions
        "fps": 24,
        "height": 512,
        "width": 768,
        "maintain_consistency": True,
    }
    
    total_frames = params["duration"] * params["fps"]
    chunk_frames = params["chunk_size"] * params["fps"]
    num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
    
    print(f"\nGeneration Settings:")
    print(f"  Duration: {params['duration']} seconds")
    print(f"  Total frames: {total_frames}")
    print(f"  Chunk size: {params['chunk_size']} seconds ({chunk_frames} frames)")
    print(f"  Number of chunks: {num_chunks}")
    print(f"  Overlap: {params['overlap']} second(s)")
    print(f"  Resolution: {params['width']}x{params['height']}")
    
    # Autoregressive generation workflow
    print("\n" + "=" * 50)
    print("Autoregressive Generation Workflow")
    print("=" * 50)
    
    for i in range(num_chunks):
        chunk_start = i * chunk_frames
        chunk_end = min(chunk_start + chunk_frames, total_frames)
        time_start = chunk_start / params["fps"]
        time_end = chunk_end / params["fps"]
        
        print(f"\nChunk {i+1}/{num_chunks}:")
        print(f"  Frames: {chunk_start} - {chunk_end}")
        print(f"  Time: {time_start:.1f}s - {time_end:.1f}s")
        
        if i == 0:
            print(f"  Conditioning: None (first chunk)")
        else:
            overlap_frames = params["overlap"] * params["fps"]
            print(f"  Conditioning: Last {overlap_frames} frames from previous chunk")
        
        # In practice, generate each chunk:
        # chunk_output = pipeline(
        #     prompt=params["prompt"],
        #     num_frames=chunk_frames,
        #     conditioning_frames=previous_chunk[-overlap_frames:] if i > 0 else None,
        #     height=params["height"],
        #     width=params["width"],
        # )
    
    print("\n" + "=" * 50)
    print("Temporal Coherence Techniques")
    print("=" * 50)
    print("1. Latent Feature Carryover")
    print("   - Last frame of chunk N → first frame of chunk N+1")
    print("   - Maintains subject consistency")
    print("\n2. Optical Flow Smoothing")
    print("   - Enforces smooth motion at chunk boundaries")
    print("   - Prevents sudden jumps or artifacts")
    print("\n3. Style Consistency Loss")
    print("   - Penalizes lighting/color changes")
    print("   - Maintains visual coherence")
    print("\n4. Overlapped Blending")
    print("   - Smooth transition in overlap region")
    print("   - Weighted blending of predictions")
    
    print("\n" + "=" * 50)
    print("Memory Management")
    print("=" * 50)
    print("Strategy: Stream-based processing")
    print("  - Generate one chunk at a time")
    print("  - Save to disk immediately")
    print("  - Only keep overlap frames in memory")
    print(f"\nMemory usage:")
    print(f"  Per chunk: ~8GB VRAM")
    print(f"  Total: ~8GB VRAM (constant, not cumulative)")
    print(f"  Disk space: ~{total_frames * 0.5 / 1024:.1f} MB")
    
    print("\n✓ Extended generation configured!")
    print("\nExpected Results:")
    print("  - 15 seconds of continuous video")
    print("  - No visible seams between chunks")
    print("  - Consistent subject appearance")
    print("  - Smooth temporal progression")
    print("  - Synchronized audio (if enabled)")


def batch_long_video_generation():
    """Example of generating multiple long videos efficiently."""
    
    print("\n" + "=" * 50)
    print("Batch Long Video Generation")
    print("=" * 50)
    
    prompts = [
        "A chef preparing a gourmet meal in a professional kitchen",
        "A painter creating a landscape painting in their studio",
        "A dancer performing a contemporary routine on stage",
    ]
    
    print(f"\nGenerating {len(prompts)} long videos...")
    print(f"Duration: 15 seconds each")
    print(f"Total generation time estimate: ~{len(prompts) * 2} minutes")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. {prompt[:50]}...")
        print(f"   Status: Queued")
        # output = generate_long_video(prompt, duration=15)
    
    print("\n✓ Batch generation configured!")


def quality_settings_guide():
    """Guide for balancing quality and speed in long videos."""
    
    print("\n" + "=" * 50)
    print("Quality vs Speed Settings")
    print("=" * 50)
    
    presets = [
        {
            "name": "Fast Draft",
            "inference_steps": 20,
            "chunk_size": 5,
            "overlap": 0,
            "time_per_chunk": "30s",
            "quality": "Good",
        },
        {
            "name": "Balanced",
            "inference_steps": 35,
            "chunk_size": 5,
            "overlap": 1,
            "time_per_chunk": "45s",
            "quality": "Great",
        },
        {
            "name": "High Quality",
            "inference_steps": 50,
            "chunk_size": 5,
            "overlap": 2,
            "time_per_chunk": "60s",
            "quality": "Excellent",
        },
    ]
    
    print(f"\n{'Preset':<15} {'Steps':<8} {'Overlap':<9} {'Time':<8} {'Quality':<10}")
    print("-" * 60)
    
    for preset in presets:
        print(f"{preset['name']:<15} {preset['inference_steps']:<8} "
              f"{preset['overlap']}s{' ':<7} {preset['time_per_chunk']:<8} "
              f"{preset['quality']:<10}")
    
    print("\nRecommendation:")
    print("  - Use 'Balanced' for most use cases")
    print("  - Use 'Fast Draft' for previews")
    print("  - Use 'High Quality' for final renders")


if __name__ == "__main__":
    main()
    batch_long_video_generation()
    quality_settings_guide()
    
    print("\n" + "=" * 50)
    print("Advanced Tips")
    print("=" * 50)
    print("1. Use consistent prompts across chunks")
    print("2. Maintain lighting continuity in descriptions")
    print("3. Use keyframes at chunk boundaries for control")
    print("4. Enable audio for better temporal coherence")
    print("5. Post-process with temporal smoothing if needed")
    print("6. Consider breaking very long videos (>30s) into scenes")
