"""Tests for audio encoder module."""

import pytest
import torch
from ltx_video.models.audio.audio_encoder import AudioEncoder, RotaryPositionEmbedding


class TestAudioEncoder:
    """Test cases for AudioEncoder."""
    
    def test_audio_encoder_initialization(self):
        """Test that AudioEncoder initializes correctly."""
        encoder = AudioEncoder(
            input_channels=1,
            latent_channels=128,
            sample_rate=16000,
        )
        
        assert encoder.input_channels == 1
        assert encoder.latent_channels == 128
        assert encoder.sample_rate == 16000
    
    def test_audio_encoder_forward_shape(self):
        """Test that AudioEncoder produces correct output shape."""
        encoder = AudioEncoder(
            input_channels=1,
            latent_channels=128,
            sample_rate=16000,
            temporal_downsample_factor=320,
        )
        
        # Input: 1 second of audio at 16kHz
        batch_size = 2
        audio_length = 16000
        audio = torch.randn(batch_size, 1, audio_length)
        
        # Encode
        output = encoder(audio)
        
        # Expected output: (B, L, D)
        # L = audio_length // temporal_downsample_factor = 16000 // 320 = 50
        expected_length = audio_length // 320
        assert output.shape == (batch_size, expected_length, 128)
    
    def test_audio_encoder_get_token_count(self):
        """Test token count calculation."""
        encoder = AudioEncoder(
            temporal_downsample_factor=320,
        )
        
        # 1 second at 16kHz = 16000 samples
        # With 320x downsample = 50 tokens
        token_count = encoder.get_audio_token_count(16000)
        assert token_count == 50
        
        # 5 seconds = 80000 samples = 250 tokens
        token_count = encoder.get_audio_token_count(80000)
        assert token_count == 250


class TestRotaryPositionEmbedding:
    """Test cases for RotaryPositionEmbedding."""
    
    def test_rope_initialization(self):
        """Test that RoPE initializes correctly."""
        rope = RotaryPositionEmbedding(dim=128)
        assert rope.dim == 128
    
    def test_rope_forward_shape(self):
        """Test that RoPE maintains input shape."""
        rope = RotaryPositionEmbedding(dim=128)
        
        batch_size = 2
        seq_len = 50
        x = torch.randn(batch_size, seq_len, 128)
        
        output = rope(x)
        assert output.shape == x.shape
    
    def test_rope_different_seq_lengths(self):
        """Test RoPE with different sequence lengths."""
        rope = RotaryPositionEmbedding(dim=64)
        
        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(1, seq_len, 64)
            output = rope(x)
            assert output.shape == (1, seq_len, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
