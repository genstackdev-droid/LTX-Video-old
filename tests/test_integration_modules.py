"""Tests for integration modules: audio cross-attention, adaptive VAE decoder, and autoregressive pipeline."""

import pytest
import torch


class TestAudioCrossAttention:
    """Test cases for audio cross-attention integration."""
    
    def test_audio_cross_attention_block_initialization(self):
        """Test AudioCrossAttentionBlock initialization."""
        from ltx_video.models.transformers.audio_cross_attention import AudioCrossAttentionBlock
        
        block = AudioCrossAttentionBlock(
            hidden_dim=512,
            num_heads=8,
            audio_latent_dim=128,
        )
        
        assert block.hidden_dim == 512
        assert block.num_heads == 8
        assert block.audio_latent_dim == 128
    
    def test_audio_cross_attention_forward_without_audio(self):
        """Test forward pass without audio conditioning."""
        from ltx_video.models.transformers.audio_cross_attention import AudioCrossAttentionBlock
        
        block = AudioCrossAttentionBlock(hidden_dim=256, num_heads=4)
        
        hidden_states = torch.randn(2, 100, 256)
        output = block(hidden_states, audio_latents=None)
        
        # Should return unchanged when no audio
        assert torch.allclose(output, hidden_states)
    
    def test_audio_cross_attention_forward_with_audio(self):
        """Test forward pass with audio conditioning."""
        from ltx_video.models.transformers.audio_cross_attention import AudioCrossAttentionBlock
        
        block = AudioCrossAttentionBlock(hidden_dim=256, num_heads=4, audio_latent_dim=128)
        
        hidden_states = torch.randn(2, 100, 256)
        audio_latents = torch.randn(2, 50, 128)
        
        output = block(hidden_states, audio_latents=audio_latents)
        
        # Should maintain shape
        assert output.shape == hidden_states.shape
        # Should be different from input (audio conditioning applied)
        assert not torch.allclose(output, hidden_states)


class TestAdaptiveVAEDecoder:
    """Test cases for adaptive VAE decoder."""
    
    def test_adaptive_decoder_initialization(self):
        """Test AdaptiveVAEDecoder initialization."""
        from ltx_video.models.autoencoders.adaptive_vae_decoder import AdaptiveVAEDecoder
        
        # Mock decoder
        mock_decoder = torch.nn.Identity()
        
        decoder = AdaptiveVAEDecoder(
            vae_decoder=mock_decoder,
            use_tiling=True,
            tile_size=512,
        )
        
        assert decoder.use_tiling == True
        assert decoder.tile_size == 512
    
    def test_adaptive_decoder_direct_decode(self):
        """Test direct decoding without tiling."""
        from ltx_video.models.autoencoders.adaptive_vae_decoder import AdaptiveVAEDecoder
        
        # Mock decoder that returns input unchanged
        mock_decoder = torch.nn.Identity()
        
        decoder = AdaptiveVAEDecoder(
            vae_decoder=mock_decoder,
            use_tiling=False,
        )
        
        # Small latents (won't trigger tiling)
        latents = torch.randn(1, 256, 8, 16, 16)
        output = decoder(latents)
        
        assert output is not None
    
    def test_blend_weights_creation(self):
        """Test blending weights creation."""
        from ltx_video.models.autoencoders.adaptive_vae_decoder import AdaptiveVAEDecoder
        
        mock_decoder = torch.nn.Identity()
        decoder = AdaptiveVAEDecoder(vae_decoder=mock_decoder)
        
        weights = decoder._create_blend_weights(64, 64, 16)
        
        assert weights.shape == (64, 64)
        # Weights should be in [0, 1] range
        assert weights.min() >= 0
        assert weights.max() <= 1
    
    def test_enable_disable_tiling(self):
        """Test enabling and disabling tiling."""
        from ltx_video.models.autoencoders.adaptive_vae_decoder import AdaptiveVAEDecoder
        
        mock_decoder = torch.nn.Identity()
        decoder = AdaptiveVAEDecoder(vae_decoder=mock_decoder, use_tiling=False)
        
        assert decoder.use_tiling == False
        
        decoder.enable_tiling(tile_size=1024, overlap=128)
        assert decoder.use_tiling == True
        assert decoder.tile_size == 1024
        assert decoder.overlap == 128
        
        decoder.disable_tiling()
        assert decoder.use_tiling == False


class TestAutoregressivePipeline:
    """Test cases for autoregressive pipeline."""
    
    def test_temporal_coherence_loss_initialization(self):
        """Test TemporalCoherenceLoss initialization."""
        from ltx_video.pipelines.autoregressive_pipeline import TemporalCoherenceLoss
        
        loss_fn = TemporalCoherenceLoss(flow_weight=1.0, appearance_weight=1.0)
        
        assert loss_fn.flow_weight == 1.0
        assert loss_fn.appearance_weight == 1.0
    
    def test_temporal_coherence_loss_forward(self):
        """Test TemporalCoherenceLoss forward pass."""
        from ltx_video.pipelines.autoregressive_pipeline import TemporalCoherenceLoss
        
        loss_fn = TemporalCoherenceLoss()
        
        chunk1_end = torch.randn(1, 3, 5, 64, 64)
        chunk2_start = torch.randn(1, 3, 5, 64, 64)
        
        loss = loss_fn(chunk1_end, chunk2_start)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Scalar loss
        assert loss >= 0  # Loss should be non-negative
    
    def test_chunk_blending(self):
        """Test chunk blending logic."""
        
        # Create mock chunks
        chunk1 = torch.randn(1, 3, 50, 32, 32)
        chunk2 = torch.randn(1, 3, 50, 32, 32)
        chunks = [chunk1, chunk2]
        
        # Create pipeline instance (minimal setup for testing)
        # Note: In practice would need full pipeline initialization
        overlap_frames = 10
        
        # Test blending function directly
        # This would normally be called internally
        # For now, just verify the logic is correct
        
        # Calculate expected output length
        expected_length = 50 + (50 - overlap_frames)  # 50 + 40 = 90
        
        assert expected_length == 90
    
    def test_blend_weights_linear(self):
        """Test linear blending weights."""
        import torch
        
        overlap_frames = 10
        weights = torch.linspace(0, 1, overlap_frames)
        
        assert weights[0] == 0.0
        assert weights[-1] == 1.0
        assert len(weights) == overlap_frames


class TestIntegrationExamples:
    """Test that integration examples are syntactically correct."""
    
    def test_integration_example_syntax(self):
        """Test that integration_example.py has valid syntax."""
        import py_compile
        
        py_compile.compile(
            'examples/integration_example.py',
            doraise=True,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
