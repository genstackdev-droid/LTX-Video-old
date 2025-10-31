"""Tests for hierarchical VAE module."""

import pytest
import torch
from ltx_video.models.autoencoders.vae_hierarchical import (
    HierarchicalVAEEncoder,
    WaveletDownsample3D,
    NeighborhoodContextAggregator,
)


class TestHierarchicalVAEEncoder:
    """Test cases for HierarchicalVAEEncoder."""
    
    def test_encoder_initialization(self):
        """Test that encoder initializes correctly."""
        encoder = HierarchicalVAEEncoder(
            in_channels=3,
            latent_channels=128,
            compression_levels=2,
        )
        
        assert encoder.in_channels == 3
        assert encoder.latent_channels == 128
        assert encoder.compression_levels == 2
    
    def test_encoder_forward_shape(self):
        """Test that encoder produces correct output shape."""
        encoder = HierarchicalVAEEncoder(
            in_channels=3,
            latent_channels=128,
            compression_levels=2,
        )
        
        # Input: small video (B, C, T, H, W)
        batch_size = 1
        channels = 3
        frames = 8
        height = 64
        width = 64
        
        x = torch.randn(batch_size, channels, frames, height, width)
        output, intermediates = encoder(x, return_intermediates=False)
        
        # With 2 compression levels and 2x downsampling per level:
        # Spatial: 64 -> 32 -> 16
        # Output includes mean and logvar, so channels * 2
        expected_shape = (batch_size, 256, frames, 16, 16)  # 128*2 channels
        assert output.shape == expected_shape
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        encoder = HierarchicalVAEEncoder(
            compression_levels=2,
            spatial_downsample_factor=2,
        )
        
        ratio = encoder.get_compression_ratio()
        assert ratio == 4  # 2^2 = 4


class TestWaveletDownsample3D:
    """Test cases for WaveletDownsample3D."""
    
    def test_wavelet_initialization(self):
        """Test wavelet downsampler initialization."""
        wavelet = WaveletDownsample3D(channels=64)
        assert wavelet is not None
    
    def test_wavelet_downsampling(self):
        """Test that wavelet downsampling reduces spatial dimensions by 2x."""
        channels = 64
        wavelet = WaveletDownsample3D(channels=channels)
        
        # Input shape: (B, C, T, H, W)
        x = torch.randn(2, channels, 8, 32, 32)
        output = wavelet(x)
        
        # Output should be (B, C, T, H/2, W/2)
        assert output.shape == (2, channels, 8, 16, 16)


class TestNeighborhoodContextAggregator:
    """Test cases for NeighborhoodContextAggregator."""
    
    def test_context_aggregator_initialization(self):
        """Test context aggregator initialization."""
        aggregator = NeighborhoodContextAggregator(channels=128)
        assert aggregator is not None
    
    def test_context_aggregator_forward(self):
        """Test that context aggregator maintains shape."""
        channels = 128
        aggregator = NeighborhoodContextAggregator(channels=channels)
        
        x = torch.randn(2, channels, 8, 16, 16)
        output = aggregator(x)
        
        # Should maintain input shape
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
