"""Multi-keyframe conditioning processor for precise video control.

Enables generation with multiple reference frames at different temporal positions,
maintaining consistency across the full video duration.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Keyframe:
    """Represents a single keyframe with its temporal position and content.
    
    Attributes:
        frame_index (int): Frame index in the target video (0-based)
        latent (torch.Tensor): Encoded latent representation of the keyframe
        strength (float): Conditioning strength (0.0 to 1.0)
        depth_map (Optional[torch.Tensor]): Optional depth map for the keyframe
    """
    frame_index: int
    latent: torch.Tensor
    strength: float = 1.0
    depth_map: Optional[torch.Tensor] = None


class MultiKeyframeProcessor(nn.Module):
    """Processes multiple keyframes for video generation with temporal interpolation.
    
    Args:
        latent_channels (int): Number of latent channels
        num_frames (int): Total number of frames in the video
        interpolation_mode (str): Interpolation mode ('linear', 'cubic', 'learned')
        use_temporal_masks (bool): Whether to use temporal masks for conditioning
    """
    
    def __init__(
        self,
        latent_channels: int = 128,
        num_frames: int = 257,
        interpolation_mode: str = "linear",
        use_temporal_masks: bool = True,
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.num_frames = num_frames
        self.interpolation_mode = interpolation_mode
        self.use_temporal_masks = use_temporal_masks
        
        # Learned interpolation weights (if using learned mode)
        if interpolation_mode == "learned":
            self.interpolation_network = nn.Sequential(
                nn.Linear(2, 64),  # Input: [distance_to_prev, distance_to_next]
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2),  # Output: [weight_prev, weight_next]
                nn.Softmax(dim=-1),
            )
        
        # Temporal conditioning strength modulator
        self.strength_modulator = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def process_keyframes(
        self,
        keyframes: List[Keyframe],
        num_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process keyframes into temporal conditioning tensors.
        
        Args:
            keyframes: List of Keyframe objects
            num_frames: Optional override for number of frames
        
        Returns:
            Tuple of (conditioning_latents, conditioning_masks) where:
                - conditioning_latents: (B, C, T, H, W) - interpolated latents
                - conditioning_masks: (B, T) - per-frame conditioning strength
        """
        if num_frames is None:
            num_frames = self.num_frames
        
        # Sort keyframes by frame index
        keyframes = sorted(keyframes, key=lambda k: k.frame_index)
        
        if len(keyframes) == 0:
            raise ValueError("At least one keyframe must be provided")
        
        # Get dimensions from first keyframe
        B, C, H, W = keyframes[0].latent.shape
        device = keyframes[0].latent.device
        
        # Initialize output tensors
        conditioning_latents = torch.zeros(B, C, num_frames, H, W, device=device)
        conditioning_masks = torch.zeros(B, num_frames, device=device)
        
        # Fill in keyframe positions
        for keyframe in keyframes:
            if keyframe.frame_index >= num_frames:
                continue
            conditioning_latents[:, :, keyframe.frame_index, :, :] = keyframe.latent
            conditioning_masks[:, keyframe.frame_index] = keyframe.strength
        
        # Interpolate between keyframes
        conditioning_latents, conditioning_masks = self._interpolate_keyframes(
            conditioning_latents,
            conditioning_masks,
            keyframes,
            num_frames,
        )
        
        return conditioning_latents, conditioning_masks
    
    def _interpolate_keyframes(
        self,
        conditioning_latents: torch.Tensor,
        conditioning_masks: torch.Tensor,
        keyframes: List[Keyframe],
        num_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interpolate latents and masks between keyframes."""
        
        for i in range(len(keyframes) - 1):
            start_frame = keyframes[i].frame_index
            end_frame = keyframes[i + 1].frame_index
            
            if end_frame <= start_frame or end_frame >= num_frames:
                continue
            
            # Get keyframe latents
            start_latent = keyframes[i].latent
            end_latent = keyframes[i + 1].latent
            start_strength = keyframes[i].strength
            end_strength = keyframes[i + 1].strength
            
            # Interpolate frames between keyframes
            for frame_idx in range(start_frame + 1, end_frame):
                alpha = (frame_idx - start_frame) / (end_frame - start_frame)
                
                if self.interpolation_mode == "linear":
                    weight_start = 1.0 - alpha
                    weight_end = alpha
                elif self.interpolation_mode == "cubic":
                    # Cubic interpolation (smoothstep)
                    alpha = alpha * alpha * (3 - 2 * alpha)
                    weight_start = 1.0 - alpha
                    weight_end = alpha
                elif self.interpolation_mode == "learned":
                    # Use learned interpolation network
                    dist_to_start = alpha
                    dist_to_end = 1.0 - alpha
                    distances = torch.tensor([[dist_to_start, dist_to_end]], 
                                            device=start_latent.device)
                    weights = self.interpolation_network(distances)
                    weight_start = weights[0, 0]
                    weight_end = weights[0, 1]
                else:
                    raise ValueError(f"Unknown interpolation mode: {self.interpolation_mode}")
                
                # Interpolate latent
                interpolated_latent = weight_start * start_latent + weight_end * end_latent
                conditioning_latents[:, :, frame_idx, :, :] = interpolated_latent
                
                # Interpolate conditioning strength
                interpolated_strength = weight_start * start_strength + weight_end * end_strength
                conditioning_masks[:, frame_idx] = interpolated_strength
        
        # Handle frames before first keyframe
        if keyframes[0].frame_index > 0:
            first_latent = keyframes[0].latent
            first_strength = keyframes[0].strength
            for frame_idx in range(keyframes[0].frame_index):
                # Decay strength based on distance
                decay = 1.0 - (keyframes[0].frame_index - frame_idx) / keyframes[0].frame_index
                decay = max(0.0, decay)
                conditioning_latents[:, :, frame_idx, :, :] = first_latent
                conditioning_masks[:, frame_idx] = first_strength * decay
        
        # Handle frames after last keyframe
        if keyframes[-1].frame_index < num_frames - 1:
            last_latent = keyframes[-1].latent
            last_strength = keyframes[-1].strength
            for frame_idx in range(keyframes[-1].frame_index + 1, num_frames):
                # Decay strength based on distance
                decay = 1.0 - (frame_idx - keyframes[-1].frame_index) / (num_frames - keyframes[-1].frame_index)
                decay = max(0.0, decay)
                conditioning_latents[:, :, frame_idx, :, :] = last_latent
                conditioning_masks[:, frame_idx] = last_strength * decay
        
        return conditioning_latents, conditioning_masks
    
    def create_temporal_attention_mask(
        self,
        keyframes: List[Keyframe],
        num_frames: int,
        attention_radius: int = 8,
    ) -> torch.Tensor:
        """Create temporal attention mask for keyframe-based generation.
        
        Args:
            keyframes: List of Keyframe objects
            num_frames: Total number of frames
            attention_radius: Radius around keyframes for strong attention
        
        Returns:
            Temporal attention mask of shape (num_frames, num_frames)
        """
        device = keyframes[0].latent.device
        
        # Initialize with uniform attention
        mask = torch.ones(num_frames, num_frames, device=device)
        
        # Boost attention near keyframes
        for keyframe in keyframes:
            frame_idx = keyframe.frame_index
            if frame_idx >= num_frames:
                continue
            
            # Define attention window around keyframe
            start_idx = max(0, frame_idx - attention_radius)
            end_idx = min(num_frames, frame_idx + attention_radius + 1)
            
            # Increase attention weights in this region
            mask[frame_idx, start_idx:end_idx] *= 2.0
            mask[start_idx:end_idx, frame_idx] *= 2.0
        
        # Normalize
        mask = mask / mask.sum(dim=-1, keepdim=True)
        
        return mask


class KeyframeConditioningAdapter(nn.Module):
    """Adapter module for integrating keyframe conditioning into the transformer.
    
    Args:
        hidden_dim (int): Hidden dimension of the transformer
        latent_channels (int): Number of latent channels from keyframes
        num_layers (int): Number of conditioning layers
    """
    
    def __init__(
        self,
        hidden_dim: int,
        latent_channels: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_channels = latent_channels
        
        # Project keyframe latents to transformer hidden dimension
        self.latent_proj = nn.Linear(latent_channels, hidden_dim)
        
        # Cross-attention layers for keyframe conditioning
        self.conditioning_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Gating mechanism for controllable conditioning strength
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        keyframe_conditioning: torch.Tensor,
        conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply keyframe conditioning to transformer features.
        
        Args:
            x: Transformer hidden states of shape (B, L, D)
            keyframe_conditioning: Keyframe latents of shape (B, K, C)
            conditioning_mask: Optional mask of shape (B, K)
        
        Returns:
            Conditioned features of shape (B, L, D)
        """
        # Project keyframe latents
        keyframe_features = self.latent_proj(keyframe_conditioning)
        
        # Apply cross-attention layers
        for i, (attn_layer, norm_layer) in enumerate(zip(self.conditioning_layers, self.layer_norms)):
            # Cross-attention: query from x, key/value from keyframes
            attn_output, _ = attn_layer(
                query=x,
                key=keyframe_features,
                value=keyframe_features,
                key_padding_mask=~conditioning_mask if conditioning_mask is not None else None,
            )
            
            # Gated residual connection
            gate = self.gate(x)
            x = x + gate * norm_layer(attn_output)
        
        return x
