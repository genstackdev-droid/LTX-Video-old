"""Audio encoder module for synchronized audio-video generation.

Based on DenseAV architecture for audio feature extraction and temporal alignment.
References:
- AV-DiT: Efficient Audio-Visual Diffusion Transformer (arXiv:2406.07686)
- SyncFlow: Temporally Aligned Joint Audio-Video Generation (arXiv:2412.15220)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config


class AudioEncoder(ModelMixin, ConfigMixin):
    """Audio encoder for converting audio waveforms to latent representations.
    
    This encoder processes 16kHz audio waveforms and outputs time-aligned audio tokens
    compatible with the video diffusion transformer.
    
    Args:
        input_channels (int): Number of input audio channels (default: 1 for mono)
        latent_channels (int): Number of output latent channels (default: 128)
        hidden_dim (int): Hidden dimension for intermediate layers (default: 512)
        num_layers (int): Number of encoder layers (default: 6)
        sample_rate (int): Audio sample rate in Hz (default: 16000)
        temporal_downsample_factor (int): Factor to downsample audio temporally (default: 320)
        use_rope (bool): Whether to use Rotary Position Embeddings (default: True)
    """
    
    @register_to_config
    def __init__(
        self,
        input_channels: int = 1,
        latent_channels: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 6,
        sample_rate: int = 16000,
        temporal_downsample_factor: int = 320,
        use_rope: bool = True,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sample_rate = sample_rate
        self.temporal_downsample_factor = temporal_downsample_factor
        self.use_rope = use_rope
        
        # Initial projection layer
        self.input_proj = nn.Conv1d(
            input_channels,
            hidden_dim,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        
        # Temporal downsampling layers
        self.downsample_layers = nn.ModuleList()
        current_dim = hidden_dim
        current_stride = 1
        
        while current_stride < temporal_downsample_factor:
            stride = min(4, temporal_downsample_factor // current_stride)
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        current_dim,
                        current_dim * 2,
                        kernel_size=stride * 2,
                        stride=stride,
                        padding=stride // 2,
                    ),
                    nn.GroupNorm(32, current_dim * 2),
                    nn.GELU(),
                )
            )
            current_dim = current_dim * 2
            current_stride *= stride
        
        # Transformer encoder layers for feature extraction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=current_dim,
            nhead=8,
            dim_feedforward=current_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection to latent space
        self.output_proj = nn.Linear(current_dim, latent_channels)
        
        # Rotary Position Embeddings for temporal alignment
        if use_rope:
            self.rope = RotaryPositionEmbedding(current_dim)
    
    def forward(
        self,
        audio: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio waveform to latent representation.
        
        Args:
            audio: Audio waveform tensor of shape (B, C, T) where:
                - B is batch size
                - C is number of channels (default: 1)
                - T is time dimension (number of samples)
            timestep: Optional timestep for conditioning (B,)
        
        Returns:
            Audio latent tensor of shape (B, L, D) where:
                - B is batch size
                - L is sequence length (T // temporal_downsample_factor)
                - D is latent_channels
        """
        # Initial projection
        x = self.input_proj(audio)  # (B, hidden_dim, T)
        
        # Temporal downsampling
        for downsample in self.downsample_layers:
            x = downsample(x)  # (B, current_dim, T')
        
        # Transpose for transformer (B, T', current_dim)
        x = x.transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            x = self.rope(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (B, T', current_dim)
        
        # Project to latent space
        x = self.output_proj(x)  # (B, T', latent_channels)
        
        return x
    
    def get_audio_token_count(self, audio_length: int) -> int:
        """Calculate number of audio tokens for given audio length.
        
        Args:
            audio_length: Length of audio in samples
        
        Returns:
            Number of audio tokens after encoding
        """
        return audio_length // self.temporal_downsample_factor


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for temporal alignment.
    
    Args:
        dim (int): Dimension of the embeddings
        max_seq_len (int): Maximum sequence length (default: 8192)
        theta (float): Base for rotation angles (default: 10000.0)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute rotation matrices
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings.
        
        Args:
            x: Input tensor of shape (B, L, D)
        
        Returns:
            Tensor with rotary position embeddings applied
        """
        seq_len = x.shape[1]
        device = x.device
        
        # Generate position indices
        position = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        
        # Compute rotation angles
        freqs = torch.outer(position, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Apply rotation
        cos_emb = emb.cos()[None, :, :]
        sin_emb = emb.sin()[None, :, :]
        
        # Split features for rotation
        x_rot = self._rotate_half(x)
        
        # Apply rotary transformation
        x = x * cos_emb + x_rot * sin_emb
        
        return x
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimensions."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
