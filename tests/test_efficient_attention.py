"""Tests for efficient attention modules."""

import pytest
import torch
from ltx_video.attention.efficient_attention import (
    EfficientAttention,
    SparseAttention,
)


class TestEfficientAttention:
    """Test cases for EfficientAttention."""
    
    def test_attention_initialization(self):
        """Test attention module initialization."""
        attn = EfficientAttention(
            dim=512,
            num_heads=8,
        )
        
        assert attn.dim == 512
        assert attn.num_heads == 8
        assert attn.head_dim == 64
    
    def test_attention_forward_shape(self):
        """Test that attention maintains input shape."""
        attn = EfficientAttention(
            dim=512,
            num_heads=8,
            use_multi_query=False,  # Disable MQA for basic test
        )
        
        batch_size = 2
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 512)
        
        output = attn(x)
        assert output.shape == x.shape
    
    def test_multi_query_attention(self):
        """Test multi-query attention variant."""
        attn = EfficientAttention(
            dim=512,
            num_heads=8,
            use_multi_query=True,
        )
        
        x = torch.randn(2, 100, 512)
        output = attn(x)
        assert output.shape == x.shape
    
    def test_windowed_attention(self):
        """Test windowed attention for long sequences."""
        attn = EfficientAttention(
            dim=256,
            num_heads=4,
            window_size=32,
        )
        
        # Long sequence
        x = torch.randn(1, 256, 256)
        output = attn(x)
        assert output.shape == x.shape


class TestSparseAttention:
    """Test cases for SparseAttention."""
    
    def test_sparse_attention_initialization(self):
        """Test sparse attention initialization."""
        attn = SparseAttention(
            dim=512,
            num_heads=8,
            block_size=64,
            sparsity_factor=4,
        )
        
        assert attn.block_size == 64
        assert attn.sparsity_factor == 4
    
    def test_sparse_attention_forward(self):
        """Test that sparse attention maintains shape."""
        attn = SparseAttention(
            dim=256,
            num_heads=4,
            block_size=32,
        )
        
        batch_size = 1
        seq_len = 128
        x = torch.randn(batch_size, seq_len, 256)
        
        output = attn(x)
        assert output.shape == x.shape
    
    def test_sparse_attention_large_sequence(self):
        """Test sparse attention with large sequence (4K tokens)."""
        attn = SparseAttention(
            dim=512,
            num_heads=8,
            block_size=64,
            sparsity_factor=4,
        )
        
        # Simulate 4K video tokens
        x = torch.randn(1, 1024, 512)
        output = attn(x)
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
