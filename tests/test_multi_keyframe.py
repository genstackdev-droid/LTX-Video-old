"""Tests for multi-keyframe conditioning module."""

import pytest
import torch
from ltx_video.conditioning.multi_keyframe_processor import (
    MultiKeyframeProcessor,
    Keyframe,
    KeyframeConditioningAdapter,
)


class TestMultiKeyframeProcessor:
    """Test cases for MultiKeyframeProcessor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = MultiKeyframeProcessor(
            latent_channels=128,
            num_frames=257,
            interpolation_mode="cubic",
        )
        
        assert processor.latent_channels == 128
        assert processor.num_frames == 257
        assert processor.interpolation_mode == "cubic"
    
    def test_process_single_keyframe(self):
        """Test processing with a single keyframe."""
        processor = MultiKeyframeProcessor(
            latent_channels=128,
            num_frames=100,
        )
        
        # Create a single keyframe
        latent = torch.randn(1, 128, 16, 16)
        keyframe = Keyframe(frame_index=0, latent=latent, strength=1.0)
        
        conditioning_latents, conditioning_masks = processor.process_keyframes([keyframe], num_frames=100)
        
        # Check shapes
        assert conditioning_latents.shape == (1, 128, 100, 16, 16)
        assert conditioning_masks.shape == (1, 100)
    
    def test_process_multiple_keyframes(self):
        """Test processing with multiple keyframes."""
        processor = MultiKeyframeProcessor(
            latent_channels=128,
            num_frames=100,
            interpolation_mode="linear",
        )
        
        # Create keyframes at different positions
        latent1 = torch.randn(1, 128, 16, 16)
        latent2 = torch.randn(1, 128, 16, 16)
        latent3 = torch.randn(1, 128, 16, 16)
        
        keyframes = [
            Keyframe(frame_index=0, latent=latent1, strength=1.0),
            Keyframe(frame_index=50, latent=latent2, strength=0.9),
            Keyframe(frame_index=99, latent=latent3, strength=1.0),
        ]
        
        conditioning_latents, conditioning_masks = processor.process_keyframes(keyframes, num_frames=100)
        
        # Check that keyframes are at correct positions
        assert torch.allclose(conditioning_latents[:, :, 0, :, :], latent1, atol=1e-5)
        assert conditioning_masks[0, 0] == 1.0
        assert conditioning_masks[0, 50] == 0.9
    
    def test_interpolation_modes(self):
        """Test different interpolation modes."""
        for mode in ["linear", "cubic"]:
            processor = MultiKeyframeProcessor(
                latent_channels=64,
                num_frames=50,
                interpolation_mode=mode,
            )
            
            latent1 = torch.randn(1, 64, 8, 8)
            latent2 = torch.randn(1, 64, 8, 8)
            
            keyframes = [
                Keyframe(frame_index=0, latent=latent1, strength=1.0),
                Keyframe(frame_index=49, latent=latent2, strength=1.0),
            ]
            
            conditioning_latents, conditioning_masks = processor.process_keyframes(keyframes, num_frames=50)
            
            # Check that interpolation happens
            assert not torch.allclose(conditioning_latents[:, :, 25, :, :], latent1)
            assert not torch.allclose(conditioning_latents[:, :, 25, :, :], latent2)


class TestKeyframeConditioningAdapter:
    """Test cases for KeyframeConditioningAdapter."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = KeyframeConditioningAdapter(
            hidden_dim=512,
            latent_channels=128,
        )
        
        assert adapter.hidden_dim == 512
        assert adapter.latent_channels == 128
    
    def test_adapter_forward_shape(self):
        """Test that adapter maintains transformer feature shape."""
        adapter = KeyframeConditioningAdapter(
            hidden_dim=512,
            latent_channels=128,
        )
        
        # Transformer features
        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 512)
        
        # Keyframe conditioning
        num_keyframes = 3
        keyframe_latents = torch.randn(batch_size, num_keyframes, 128)
        
        output = adapter(x, keyframe_latents)
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
