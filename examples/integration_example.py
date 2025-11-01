"""Example: Complete integration of audio cross-attention, adaptive VAE decoder, and autoregressive pipeline.

This example demonstrates how to use all three integrated components together
for advanced video generation capabilities.
"""



def example_1_audio_cross_attention():
    """Example 1: Add audio cross-attention to existing transformer."""
    
    print("Example 1: Audio Cross-Attention Integration")
    print("=" * 60)
    
    # Load base pipeline (in practice, use from_pretrained)
    # pipeline = LTXVideoPipeline.from_pretrained("Lightricks/LTX-Video")
    
    # For demonstration, we'll show the API
    print("\n1. Add audio cross-attention to transformer layers 8-12:")
    print("""
    from ltx_video.models.transformers.audio_cross_attention import (
        add_audio_cross_attention_to_transformer
    )
    
    # Add audio cross-attention to transformer
    pipeline.transformer = add_audio_cross_attention_to_transformer(
        pipeline.transformer,
        audio_latent_dim=128,
        start_layer=8,
        end_layer=12,
    )
    """)
    
    print("\n2. Use in generation with audio conditioning:")
    print("""
    # Encode audio
    audio_encoder = AudioEncoder(latent_channels=128)
    audio_waveform = torch.randn(1, 1, 16000)  # 1 second of audio
    audio_latents = audio_encoder(audio_waveform)
    
    # Generate with audio conditioning
    output = pipeline(
        prompt="A guitarist performing on stage",
        audio_latents=audio_latents,  # Pass audio latents
        num_frames=257,
    )
    """)
    
    print("\n✓ Audio cross-attention enables synchronized audio-video generation")


def example_2_adaptive_vae_decoder():
    """Example 2: Use adaptive VAE decoder for 4K generation."""
    
    print("\n\nExample 2: Adaptive VAE Decoder for 4K")
    print("=" * 60)
    
    print("\n1. Create adaptive decoder with hierarchical encoder:")
    print("""
    from ltx_video.models.autoencoders.adaptive_vae_decoder import create_adaptive_decoder
    from ltx_video.models.autoencoders.vae_hierarchical import HierarchicalVAEEncoder
    
    # Create hierarchical encoder
    hierarchical_encoder = HierarchicalVAEEncoder(
        in_channels=3,
        latent_channels=128,
        compression_levels=2,
        use_wavelet=True,
    )
    
    # Create adaptive decoder
    adaptive_decoder = create_adaptive_decoder(
        vae=pipeline.vae,
        hierarchical_encoder=hierarchical_encoder,
        use_tiling=True,
    )
    
    # Replace VAE decoder in pipeline
    pipeline.vae.decoder = adaptive_decoder
    """)
    
    print("\n2. Generate 4K video with tiled decoding:")
    print("""
    output = pipeline(
        prompt="Aerial beach view at sunset",
        height=2160,  # 4K height
        width=3840,   # 4K width
        num_frames=257,
    )
    """)
    
    print("\n✓ Adaptive decoder enables efficient 4K generation with tiling")


def example_3_autoregressive_pipeline():
    """Example 3: Generate 10-15 second videos with autoregressive pipeline."""
    
    print("\n\nExample 3: Autoregressive Pipeline for Long Videos")
    print("=" * 60)
    
    print("\n1. Create autoregressive pipeline:")
    print("""
    from ltx_video.pipelines.autoregressive_pipeline import create_autoregressive_pipeline
    
    # Create autoregressive pipeline from base
    autoregressive_pipeline = create_autoregressive_pipeline(
        base_pipeline=pipeline,
        chunk_size=121,      # 5 seconds at 24 FPS
        overlap_frames=24,   # 1 second overlap
    )
    """)
    
    print("\n2. Generate 15-second video:")
    print("""
    output = autoregressive_pipeline.generate_long_video(
        prompt="Time-lapse of a flower blooming in a garden",
        total_frames=360,  # 15 seconds at 24 FPS
        height=512,
        width=768,
        num_inference_steps=50,
    )
    """)
    
    print("\n3. Access chunk boundaries:")
    print("""
    print(f"Video generated with {len(output.chunk_boundaries)} chunks")
    print(f"Chunk boundaries at frames: {output.chunk_boundaries}")
    """)
    
    print("\n✓ Autoregressive pipeline enables 10-15 second continuous videos")


def example_4_complete_integration():
    """Example 4: Use all three features together."""
    
    print("\n\nExample 4: Complete Integration")
    print("=" * 60)
    
    print("\nCombining all three features for maximum capability:")
    print("""
    # 1. Load base pipeline
    pipeline = LTXVideoPipeline.from_pretrained("Lightricks/LTX-Video")
    
    # 2. Add audio cross-attention to transformer
    from ltx_video.models.transformers.audio_cross_attention import (
        add_audio_cross_attention_to_transformer
    )
    pipeline.transformer = add_audio_cross_attention_to_transformer(
        pipeline.transformer,
        audio_latent_dim=128,
        start_layer=8,
        end_layer=12,
    )
    
    # 3. Set up hierarchical VAE for 4K
    from ltx_video.models.autoencoders.vae_hierarchical import HierarchicalVAEEncoder
    from ltx_video.models.autoencoders.adaptive_vae_decoder import create_adaptive_decoder
    
    hierarchical_encoder = HierarchicalVAEEncoder(
        in_channels=3,
        latent_channels=128,
        compression_levels=2,
        use_wavelet=True,
    )
    
    adaptive_decoder = create_adaptive_decoder(
        vae=pipeline.vae,
        hierarchical_encoder=hierarchical_encoder,
        use_tiling=True,
    )
    pipeline.vae.decoder = adaptive_decoder
    
    # 4. Create autoregressive pipeline
    from ltx_video.pipelines.autoregressive_pipeline import create_autoregressive_pipeline
    
    autoregressive_pipeline = create_autoregressive_pipeline(
        base_pipeline=pipeline,
        chunk_size=121,
        overlap_frames=24,
    )
    
    # 5. Generate 15-second 4K video with audio
    from ltx_video.models.audio.audio_encoder import AudioEncoder
    
    audio_encoder = AudioEncoder(latent_channels=128)
    audio_waveform = torch.randn(1, 1, 16000 * 15)  # 15 seconds of audio
    audio_latents = audio_encoder(audio_waveform)
    
    output = autoregressive_pipeline.generate_long_video(
        prompt="A cinematic journey through a futuristic city at night",
        total_frames=360,  # 15 seconds at 24 FPS
        height=2160,       # 4K
        width=3840,        # 4K
        audio_latents=audio_latents,  # Audio conditioning
        num_inference_steps=50,
    )
    """)
    
    print("\n✓ Complete integration enables:")
    print("  - 15-second continuous videos")
    print("  - Native 4K resolution")
    print("  - Synchronized audio-video generation")
    print("  - Efficient VRAM usage with tiling")


def example_5_configuration_options():
    """Example 5: Configuration options for each component."""
    
    print("\n\nExample 5: Configuration Options")
    print("=" * 60)
    
    print("\nAudio Cross-Attention Configuration:")
    print("""
    # Adjust which layers have audio conditioning
    add_audio_cross_attention_to_transformer(
        transformer,
        audio_latent_dim=128,
        start_layer=6,   # Start earlier for stronger audio influence
        end_layer=16,    # Extend to more layers
    )
    """)
    
    print("\nAdaptive VAE Decoder Configuration:")
    print("""
    # Adjust tiling parameters for different resolutions
    adaptive_decoder.enable_tiling(
        tile_size=1024,  # Larger tiles for faster processing
        overlap=128,     # More overlap for smoother blending
    )
    
    # Or disable tiling for smaller resolutions
    adaptive_decoder.disable_tiling()
    """)
    
    print("\nAutoregressive Pipeline Configuration:")
    print("""
    # Adjust chunk size and overlap for different durations
    autoregressive_pipeline = create_autoregressive_pipeline(
        base_pipeline=pipeline,
        chunk_size=241,      # 10 seconds at 24 FPS
        overlap_frames=48,   # 2 seconds overlap for smoother transitions
    )
    
    # Enable streaming for very long videos
    autoregressive_pipeline.enable_streaming(
        chunk_save_path="/tmp/video_chunks"
    )
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LTX-Video Integration Examples")
    print("=" * 60)
    
    example_1_audio_cross_attention()
    example_2_adaptive_vae_decoder()
    example_3_autoregressive_pipeline()
    example_4_complete_integration()
    example_5_configuration_options()
    
    print("\n" + "=" * 60)
    print("Integration Complete!")
    print("=" * 60)
    print("\nThese three components work together to enable:")
    print("  1. Audio cross-attention in transformer layers 8-12")
    print("  2. Efficient 4K decoding with adaptive VAE decoder")
    print("  3. Extended 10-15 second video generation")
    print("\nAll components maintain backward compatibility and can be used")
    print("independently or together based on your requirements.")
    print("=" * 60)
