"""Autoregressive pipeline for extended 10-15 second video generation.

Generates long videos by conditioning each chunk on previous chunks,
maintaining temporal coherence across chunk boundaries.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Callable, Tuple
from dataclasses import dataclass

from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from diffusers.utils import BaseOutput


@dataclass
class AutoregressiveVideoOutput(BaseOutput):
    """Output class for autoregressive video generation.
    
    Attributes:
        videos: Generated video frames
        chunk_boundaries: Frame indices where chunks were joined
    """
    videos: torch.Tensor
    chunk_boundaries: List[int]


class TemporalCoherenceLoss(nn.Module):
    """Loss for maintaining temporal coherence across chunk boundaries.
    
    Enforces smooth optical flow and consistent appearance at boundaries.
    """
    
    def __init__(self, flow_weight: float = 1.0, appearance_weight: float = 1.0):
        super().__init__()
        self.flow_weight = flow_weight
        self.appearance_weight = appearance_weight
    
    def forward(
        self,
        chunk1_end: torch.Tensor,
        chunk2_start: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal coherence loss.
        
        Args:
            chunk1_end: Last frames of first chunk (B, C, T, H, W)
            chunk2_start: First frames of second chunk (B, C, T, H, W)
        
        Returns:
            Temporal coherence loss
        """
        # Optical flow smoothness (approximate with frame difference)
        flow_loss = torch.mean(
            (chunk1_end[:, :, -1] - chunk2_start[:, :, 0]) ** 2
        )
        
        # Appearance consistency (feature similarity)
        appearance_loss = torch.mean(
            (chunk1_end.mean(dim=2) - chunk2_start.mean(dim=2)) ** 2
        )
        
        total_loss = (
            self.flow_weight * flow_loss +
            self.appearance_weight * appearance_loss
        )
        
        return total_loss


class AutoregressiveLTXVideoPipeline(LTXVideoPipeline):
    """Pipeline for autoregressive long video generation.
    
    Extends LTXVideoPipeline with capability to generate 10-15 second videos
    by generating in overlapping chunks and blending them seamlessly.
    
    Args:
        Same as LTXVideoPipeline, plus:
        chunk_size (int): Size of each chunk in frames (default: 121 = 5s @ 24fps)
        overlap_frames (int): Number of overlapping frames (default: 24 = 1s)
        use_coherence_loss (bool): Whether to use temporal coherence loss
    """
    
    def __init__(
        self,
        *args,
        chunk_size: int = 121,  # 5 seconds at 24 FPS
        overlap_frames: int = 24,  # 1 second overlap
        use_coherence_loss: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        self.use_coherence_loss = use_coherence_loss
        
        if use_coherence_loss:
            self.coherence_loss = TemporalCoherenceLoss()
    
    def generate_long_video(
        self,
        prompt: Union[str, List[str]],
        total_frames: int,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        **kwargs,
    ) -> Union[AutoregressiveVideoOutput, Tuple]:
        """Generate extended video using autoregressive conditioning.
        
        Args:
            prompt: Text prompt for video generation
            total_frames: Total number of frames to generate
            height: Video height
            width: Video width
            num_inference_steps: Number of denoising steps per chunk
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt
            generator: Random generator for reproducibility
            output_type: Output format ('pil', 'pt', 'np')
            return_dict: Whether to return dict or tuple
            callback: Optional callback function
            callback_steps: Frequency of callback calls
            **kwargs: Additional arguments
        
        Returns:
            AutoregressiveVideoOutput with generated video
        """
        # Calculate number of chunks needed
        effective_chunk_size = self.chunk_size - self.overlap_frames
        num_chunks = (total_frames - self.overlap_frames + effective_chunk_size - 1) // effective_chunk_size
        
        chunks = []
        chunk_boundaries = []
        previous_chunk_latents = None
        
        for chunk_idx in range(num_chunks):
            # Calculate frame range for this chunk
            start_frame = chunk_idx * effective_chunk_size
            end_frame = min(start_frame + self.chunk_size, total_frames)
            chunk_num_frames = end_frame - start_frame
            
            # Prepare conditioning from previous chunk
            conditioning_frames = None
            conditioning_start = None
            
            if chunk_idx > 0 and previous_chunk_latents is not None:
                # Use last overlap_frames from previous chunk as conditioning
                conditioning_frames = [previous_chunk_latents]
                conditioning_start = [0]  # Condition at start of this chunk
            
            # Generate chunk
            chunk_output = super().__call__(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=chunk_num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                output_type="pt",  # Always use tensor for intermediate chunks
                return_dict=True,
                callback=callback,
                callback_steps=callback_steps,
                conditioning_media_paths=conditioning_frames if conditioning_frames else None,
                conditioning_start_frames=conditioning_start if conditioning_start else None,
                **kwargs,
            )
            
            chunk_video = chunk_output.images
            
            # Store for next iteration (last overlap_frames)
            if chunk_idx < num_chunks - 1:
                previous_chunk_latents = chunk_video[:, :, -self.overlap_frames:]
            
            # Add chunk to list
            chunks.append(chunk_video)
            chunk_boundaries.append(start_frame)
        
        # Blend chunks together
        full_video = self._blend_chunks(chunks, self.overlap_frames)
        
        # Convert to requested output format if needed
        if output_type != "pt":
            # Use parent class's conversion logic
            full_video = self._convert_output_type(full_video, output_type)
        
        if not return_dict:
            return (full_video, chunk_boundaries)
        
        return AutoregressiveVideoOutput(
            videos=full_video,
            chunk_boundaries=chunk_boundaries,
        )
    
    def _blend_chunks(
        self,
        chunks: List[torch.Tensor],
        overlap_frames: int,
    ) -> torch.Tensor:
        """Blend video chunks with smooth transitions.
        
        Args:
            chunks: List of video chunks (each B, C, T, H, W)
            overlap_frames: Number of overlapping frames
        
        Returns:
            Blended full video tensor
        """
        if len(chunks) == 0:
            raise ValueError("No chunks to blend")
        
        if len(chunks) == 1:
            return chunks[0]
        
        # Initialize with first chunk
        blended_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            
            if overlap_frames > 0:
                # Get overlapping regions
                prev_overlap = blended_chunks[-1][:, :, -overlap_frames:]
                curr_overlap = current_chunk[:, :, :overlap_frames]
                
                # Create blending weights (linear fade)
                weights = torch.linspace(0, 1, overlap_frames, device=current_chunk.device)
                weights = weights.view(1, 1, -1, 1, 1)
                
                # Blend overlap region
                blended_overlap = (1 - weights) * prev_overlap + weights * curr_overlap
                
                # Replace overlap in previous chunk
                blended_chunks[-1] = torch.cat([
                    blended_chunks[-1][:, :, :-overlap_frames],
                    blended_overlap,
                ], dim=2)
                
                # Add non-overlapping part of current chunk
                if current_chunk.shape[2] > overlap_frames:
                    blended_chunks.append(current_chunk[:, :, overlap_frames:])
            else:
                # No overlap, just append
                blended_chunks.append(current_chunk)
        
        # Concatenate all chunks
        full_video = torch.cat(blended_chunks, dim=2)
        
        return full_video
    
    def _convert_output_type(self, video: torch.Tensor, output_type: str):
        """Convert video tensor to requested output type."""
        if output_type == "pil":
            # Convert to PIL images
            from PIL import Image
            import numpy as np
            
            video_np = video[0].permute(1, 2, 3, 0).cpu().numpy()
            video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
            
            pil_images = [Image.fromarray(frame) for frame in video_np]
            return pil_images
        
        elif output_type == "np":
            # Convert to numpy
            video_np = video.permute(0, 2, 3, 4, 1).cpu().numpy()
            return video_np
        
        else:
            return video
    
    def enable_streaming(self, chunk_save_path: Optional[str] = None):
        """Enable streaming mode for memory-efficient long video generation.
        
        Args:
            chunk_save_path: Optional path to save chunks to disk
        """
        self.streaming_enabled = True
        self.chunk_save_path = chunk_save_path
    
    def disable_streaming(self):
        """Disable streaming mode."""
        self.streaming_enabled = False


def create_autoregressive_pipeline(
    base_pipeline: LTXVideoPipeline,
    chunk_size: int = 121,
    overlap_frames: int = 24,
) -> AutoregressiveLTXVideoPipeline:
    """Create autoregressive pipeline from base pipeline.
    
    Args:
        base_pipeline: Base LTXVideoPipeline instance
        chunk_size: Size of each chunk in frames
        overlap_frames: Number of overlapping frames
    
    Returns:
        AutoregressiveLTXVideoPipeline instance
    """
    # Create new pipeline with same components
    autoregressive_pipeline = AutoregressiveLTXVideoPipeline(
        tokenizer=base_pipeline.tokenizer,
        text_encoder=base_pipeline.text_encoder,
        vae=base_pipeline.vae,
        transformer=base_pipeline.transformer,
        scheduler=base_pipeline.scheduler,
        patchifier=base_pipeline.patchifier,
        prompt_enhancer_image_caption_model=base_pipeline.prompt_enhancer_image_caption_model,
        prompt_enhancer_image_caption_processor=base_pipeline.prompt_enhancer_image_caption_processor,
        prompt_enhancer_llm_model=base_pipeline.prompt_enhancer_llm_model,
        prompt_enhancer_llm_tokenizer=base_pipeline.prompt_enhancer_llm_tokenizer,
        chunk_size=chunk_size,
        overlap_frames=overlap_frames,
    )
    
    return autoregressive_pipeline
