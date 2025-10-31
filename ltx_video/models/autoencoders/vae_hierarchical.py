"""Hierarchical VAE encoder for efficient 4K video processing.

Based on LeanVAE architecture with progressive compression and neighborhood-aware processing.
References:
- LeanVAE: Ultra-Efficient Reconstruction VAE (arXiv:2503.14325)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

from ltx_video.models.autoencoders.conv_nd_factory import make_conv_nd


class HierarchicalVAEEncoder(ModelMixin, ConfigMixin):
    """Hierarchical VAE encoder for multi-resolution video encoding.
    
    Implements progressive spatial compression with adaptive patch sizes
    optimized for 4K video generation.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB)
        latent_channels (int): Number of latent channels (default: 128)
        hidden_dims (List[int]): Hidden dimensions for each level (default: [64, 128, 256, 512])
        compression_levels (int): Number of hierarchical compression levels (default: 2)
        use_wavelet (bool): Use wavelet-based compression for detail preservation (default: True)
        spatial_downsample_factor (int): Spatial downsampling per level (default: 2)
    """
    
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 128,
        hidden_dims: List[int] = None,
        compression_levels: int = 2,
        use_wavelet: bool = True,
        spatial_downsample_factor: int = 2,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims
        self.compression_levels = compression_levels
        self.use_wavelet = use_wavelet
        self.spatial_downsample_factor = spatial_downsample_factor
        
        # Multi-level encoder blocks
        self.encoder_levels = nn.ModuleList()
        
        for level in range(compression_levels):
            level_encoder = HierarchicalEncoderLevel(
                in_channels=in_channels if level == 0 else hidden_dims[level - 1],
                out_channels=hidden_dims[level],
                num_blocks=2,
                use_wavelet=use_wavelet,
                downsample=True,
            )
            self.encoder_levels.append(level_encoder)
        
        # Final projection to latent space
        final_dim = hidden_dims[compression_levels - 1]
        self.final_conv = make_conv_nd(
            dims=3,
            in_channels=final_dim,
            out_channels=latent_channels * 2,  # mean and logvar
            kernel_size=3,
            padding=1,
        )
        
        # Neighborhood-aware context aggregation
        self.context_aggregator = NeighborhoodContextAggregator(
            channels=latent_channels,
            kernel_size=3,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Encode video to hierarchical latent representation.
        
        Args:
            x: Input video tensor of shape (B, C, T, H, W)
            return_intermediates: Whether to return intermediate representations
        
        Returns:
            Tuple of (latent, intermediates) where:
                - latent: Latent tensor of shape (B, latent_channels*2, T', H', W')
                - intermediates: List of intermediate feature maps (if requested)
        """
        intermediates = [] if return_intermediates else None
        
        # Multi-level encoding
        for level_idx, encoder_level in enumerate(self.encoder_levels):
            x = encoder_level(x)
            if return_intermediates:
                intermediates.append(x)
        
        # Final projection
        latent = self.final_conv(x)
        
        # Apply neighborhood-aware context aggregation
        # Split mean and logvar
        mean, logvar = latent.chunk(2, dim=1)
        mean = self.context_aggregator(mean)
        
        # Recombine
        latent = torch.cat([mean, logvar], dim=1)
        
        return latent, intermediates
    
    def get_compression_ratio(self) -> int:
        """Calculate total spatial compression ratio."""
        return self.spatial_downsample_factor ** self.compression_levels


class HierarchicalEncoderLevel(nn.Module):
    """Single level of hierarchical encoder.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_blocks (int): Number of residual blocks
        use_wavelet (bool): Use wavelet-based downsampling
        downsample (bool): Whether to downsample spatially
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        use_wavelet: bool = True,
        downsample: bool = True,
    ):
        super().__init__()
        
        self.use_wavelet = use_wavelet
        self.downsample = downsample
        
        # Initial projection
        self.in_conv = make_conv_nd(
            dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock3D(out_channels, out_channels)
            for _ in range(num_blocks)
        ])
        
        # Downsampling
        if downsample:
            if use_wavelet:
                self.downsample_layer = WaveletDownsample3D(out_channels)
            else:
                self.downsample_layer = nn.Sequential(
                    make_conv_nd(
                        dims=3,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 4, 4),
                        stride=(1, 2, 2),
                        padding=(1, 1, 1),
                    ),
                    nn.GroupNorm(32, out_channels),
                )
        else:
            self.downsample_layer = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder level."""
        x = self.in_conv(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.downsample_layer(x)
        
        return x


class ResidualBlock3D(nn.Module):
    """3D residual block with group normalization.
    
    Args:
        channels (int): Number of channels
        groups (int): Number of groups for GroupNorm
    """
    
    def __init__(self, channels: int, out_channels: int = None, groups: int = 32):
        super().__init__()
        
        out_channels = out_channels or channels
        
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = make_conv_nd(
            dims=3,
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = make_conv_nd(
            dims=3,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        
        self.activation = nn.SiLU()
        
        if channels != out_channels:
            self.skip_conv = make_conv_nd(
                dims=3,
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        identity = self.skip_conv(x)
        
        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        return out + identity


class WaveletDownsample3D(nn.Module):
    """Wavelet-based spatial downsampling for detail preservation.
    
    Uses Haar wavelet transform for 2x spatial downsampling while preserving
    high-frequency details.
    
    Args:
        channels (int): Number of input/output channels
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Learnable fusion weights for wavelet subbands
        self.fusion = make_conv_nd(
            dims=3,
            in_channels=channels * 4,  # LL, LH, HL, HH subbands
            out_channels=channels,
            kernel_size=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply wavelet downsampling.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Downsampled tensor of shape (B, C, T, H//2, W//2)
        """
        B, C, T, H, W = x.shape
        
        # Apply 2D Haar wavelet transform on spatial dimensions
        # LL: low-pass both directions (approximation)
        # LH: low-pass horizontal, high-pass vertical
        # HL: high-pass horizontal, low-pass vertical
        # HH: high-pass both directions (details)
        
        # Spatial pooling for LL subband
        ll = (x[:, :, :, 0::2, 0::2] + x[:, :, :, 0::2, 1::2] + 
              x[:, :, :, 1::2, 0::2] + x[:, :, :, 1::2, 1::2]) / 4.0
        
        # High-pass filters for detail subbands
        lh = (x[:, :, :, 0::2, 0::2] + x[:, :, :, 0::2, 1::2] - 
              x[:, :, :, 1::2, 0::2] - x[:, :, :, 1::2, 1::2]) / 4.0
        
        hl = (x[:, :, :, 0::2, 0::2] - x[:, :, :, 0::2, 1::2] + 
              x[:, :, :, 1::2, 0::2] - x[:, :, :, 1::2, 1::2]) / 4.0
        
        hh = (x[:, :, :, 0::2, 0::2] - x[:, :, :, 0::2, 1::2] - 
              x[:, :, :, 1::2, 0::2] + x[:, :, :, 1::2, 1::2]) / 4.0
        
        # Concatenate all subbands
        wavelet_features = torch.cat([ll, lh, hl, hh], dim=1)
        
        # Learnable fusion
        out = self.fusion(wavelet_features)
        
        return out


class NeighborhoodContextAggregator(nn.Module):
    """Aggregates spatial context from neighboring regions.
    
    Implements neighborhood-aware compression from LeanVAE.
    
    Args:
        channels (int): Number of channels
        kernel_size (int): Kernel size for context aggregation
    """
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.context_conv = make_conv_nd(
            dims=3,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,  # Depthwise convolution
        )
        
        self.fusion = make_conv_nd(
            dims=3,
            in_channels=channels * 2,
            out_channels=channels,
            kernel_size=1,
        )
        
        self.norm = nn.GroupNorm(32, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate neighborhood context.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Context-aggregated tensor of same shape
        """
        # Extract local context
        context = self.context_conv(x)
        context = self.norm(context)
        
        # Fuse with original features
        fused = torch.cat([x, context], dim=1)
        out = self.fusion(fused)
        
        return out
