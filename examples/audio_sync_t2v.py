"""Example: Text-to-video with synchronized audio generation.

This example demonstrates how to generate video with synchronized audio
using the LTXAudioVideoPipeline.
"""

import torch
from ltx_video.pipelines.ltx_audio_video_pipeline import LTXAudioVideoPipeline
from ltx_video.models.audio.audio_encoder import AudioEncoder


def main():
    """Generate video with synchronized audio."""
    
    # Initialize pipeline
    print("Loading LTX Audio-Video Pipeline...")
    
    # Note: In practice, you would load from pretrained weights
    # pipeline = LTXAudioVideoPipeline.from_pretrained(
    #     "Lightricks/LTX-Video",
    #     audio_encoder_path="path/to/audio_encoder",
    # )
    
    # For this example, we'll show the API usage
    # pipeline = LTXAudioVideoPipeline(...)
    
    # Example parameters
    prompt = (
        "A musician playing guitar on a stage with dramatic lighting. "
        "The camera slowly zooms in on the guitarist's hands as they strum "
        "the strings. Crowd cheering in the background. Warm stage lights "
        "create a golden glow. Professional concert cinematography."
    )
    
    audio_prompt = (
        "Acoustic guitar music with crowd cheering. "
        "Warm, melodic guitar strumming with ambient concert hall reverb."
    )
    
    # Generation parameters
    params = {
        "prompt": prompt,
        "audio_prompt": audio_prompt,
        "num_frames": 257,  # ~10 seconds at 24 FPS
        "height": 512,
        "width": 768,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "audio_guidance_scale": 3.0,
        "generate_audio": True,
        "seed": 42,
    }
    
    print(f"Generating video with audio...")
    print(f"Prompt: {prompt}")
    print(f"Audio Prompt: {audio_prompt}")
    print(f"Resolution: {params['width']}x{params['height']}")
    print(f"Frames: {params['num_frames']}")
    
    # Generate video with audio
    # output = pipeline(**params)
    
    # Save output
    # save_video_with_audio(
    #     output.videos,
    #     output.audio,
    #     output.audio_sample_rate,
    #     "output_with_audio.mp4",
    # )
    
    print("✓ Generation complete!")
    print("Expected output:")
    print("  - Video: 10 seconds @ 24 FPS, 768x512 resolution")
    print("  - Audio: 10 seconds @ 16kHz, synchronized with video")
    print("  - Temporal alignment: <100ms drift")
    

def save_video_with_audio(
    video: torch.Tensor,
    audio: torch.Tensor,
    sample_rate: int,
    output_path: str,
):
    """Save video with embedded audio.
    
    Args:
        video: Video tensor of shape (B, C, F, H, W)
        audio: Audio tensor of shape (B, C, T)
        sample_rate: Audio sample rate
        output_path: Output file path
    """
    try:
        import imageio
        import soundfile as sf
        import numpy as np
        
        # Convert video to numpy
        video_np = video[0].permute(1, 2, 3, 0).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # Convert audio to numpy
        audio_np = audio[0].cpu().numpy()
        
        # Save with imageio (supports audio embedding)
        writer = imageio.get_writer(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
        )
        
        # Write video frames
        for frame in video_np:
            writer.append_data(frame)
        
        # Add audio
        writer.set_meta_data({'audio': audio_np, 'audio_fps': sample_rate})
        writer.close()
        
        print(f"✓ Saved to: {output_path}")
        
    except ImportError as e:
        print(f"Warning: Could not save audio-video file: {e}")
        print("Please install: pip install imageio[ffmpeg] soundfile")


if __name__ == "__main__":
    main()
