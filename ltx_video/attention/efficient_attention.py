"""Memory-efficient attention mechanisms for 4K video generation.

Implements optimized attention patterns including:
- Multi-query attention for reduced KV cache
- Sparse attention for long sequences
- Block-sparse attention for 4K tokens
- Flash attention integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


class EfficientAttention(nn.Module):
    """Memory-efficient attention with multiple optimization strategies.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in QKV projections (default: False)
        attn_drop (float): Attention dropout rate (default: 0.0)
        proj_drop (float): Output projection dropout rate (default: 0.0)
        use_flash_attn (bool): Use Flash Attention if available (default: True)
        use_multi_query (bool): Use multi-query attention (default: True)
        window_size (Optional[int]): Window size for local attention (default: None)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attn: bool = True,
        use_multi_query: bool = True,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn
        self.use_multi_query = use_multi_query
        self.window_size = window_size
        
        # Multi-query attention: shared K,V across heads
        if use_multi_query:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv_proj = nn.Linear(dim, 2 * self.head_dim, bias=qkv_bias)
            self.num_kv_heads = 1
        else:
            self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.num_kv_heads = num_heads
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Check for flash attention availability
        self._flash_attn_available = False
        if use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self._flash_attn_available = True
                self._flash_attn_func = flash_attn_func
            except ImportError:
                pass
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply efficient attention.
        
        Args:
            x: Input tensor of shape (B, L, D)
            attention_mask: Optional attention mask of shape (B, L) or (B, L, L)
        
        Returns:
            Output tensor of shape (B, L, D)
        """
        B, L, D = x.shape
        
        if self.use_multi_query:
            q = self.q_proj(x)
            kv = self.kv_proj(x)
            
            # Reshape Q
            q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
            
            # K, V are shared across heads
            k, v = kv.chunk(2, dim=-1)
            k = k.unsqueeze(1)  # (B, 1, L, head_dim)
            v = v.unsqueeze(1)  # (B, 1, L, head_dim)
            
            # Expand to all heads
            k = k.expand(-1, self.num_heads, -1, -1)
            v = v.expand(-1, self.num_heads, -1, -1)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = rearrange(qkv, 'b l (three h d) -> three b h l d',
                               three=3, h=self.num_heads)
        
        # Apply attention with appropriate method
        if self._flash_attn_available and self.training:
            # Use Flash Attention for training (more memory efficient)
            out = self._flash_attention(q, k, v, attention_mask)
        elif self.window_size is not None:
            # Use windowed attention for very long sequences
            out = self._windowed_attention(q, k, v, attention_mask)
        else:
            # Standard scaled dot-product attention
            out = self._standard_attention(q, k, v, attention_mask)
        
        # Reshape and project
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand mask to (B, 1, 1, L)
                attention_mask = attention_mask[:, None, None, :]
            attn = attn.masked_fill(~attention_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        return out
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flash Attention implementation (when available)."""
        # Rearrange for flash attention: (B, L, H, D)
        q = rearrange(q, 'b h l d -> b l h d')
        k = rearrange(k, 'b h l d -> b l h d')
        v = rearrange(v, 'b h l d -> b l h d')
        
        # Apply flash attention
        out = self._flash_attn_func(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        
        # Rearrange back
        out = rearrange(out, 'b l h d -> b h l d')
        
        return out
    
    def _windowed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Windowed local attention for long sequences."""
        B, H, L, D = q.shape
        window_size = self.window_size
        
        # Pad sequence to be divisible by window size
        padding = (window_size - L % window_size) % window_size
        if padding > 0:
            q = F.pad(q, (0, 0, 0, padding))
            k = F.pad(k, (0, 0, 0, padding))
            v = F.pad(v, (0, 0, 0, padding))
            L_padded = L + padding
        else:
            L_padded = L
        
        # Reshape into windows
        num_windows = L_padded // window_size
        q = rearrange(q, 'b h (n w) d -> (b n) h w d', n=num_windows, w=window_size)
        k = rearrange(k, 'b h (n w) d -> (b n) h w d', n=num_windows, w=window_size)
        v = rearrange(v, 'b h (n w) d -> (b n) h w d', n=num_windows, w=window_size)
        
        # Apply attention within windows
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = rearrange(out, '(b n) h w d -> b h (n w) d', b=B, n=num_windows)
        
        # Remove padding
        if padding > 0:
            out = out[:, :, :L, :]
        
        return out


class SparseAttention(nn.Module):
    """Sparse block attention for 4K video tokens.
    
    Implements block-sparse attention pattern to reduce computational complexity
    for high-resolution video generation.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        block_size (int): Size of attention blocks (default: 64)
        sparsity_factor (int): Sparsity factor for block selection (default: 4)
        qkv_bias (bool): Whether to use bias in QKV projections (default: False)
        attn_drop (float): Attention dropout rate (default: 0.0)
        proj_drop (float): Output projection dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        block_size: int = 64,
        sparsity_factor: int = 4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        self.sparsity_factor = sparsity_factor
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply sparse block attention.
        
        Args:
            x: Input tensor of shape (B, L, D)
            attention_mask: Optional attention mask
        
        Returns:
            Output tensor of shape (B, L, D)
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b l (three h d) -> three b h l d',
                           three=3, h=self.num_heads)
        
        # Apply block-sparse attention
        out = self._block_sparse_attention(q, k, v, attention_mask)
        
        # Reshape and project
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out
    
    def _block_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute block-sparse attention.
        
        Each block attends to:
        1. All tokens within its own block (local attention)
        2. Every sparsity_factor-th block (global attention)
        """
        B, H, L, D = q.shape
        block_size = self.block_size
        num_blocks = (L + block_size - 1) // block_size
        
        # Pad to block size
        padding = (block_size * num_blocks) - L
        if padding > 0:
            q = F.pad(q, (0, 0, 0, padding))
            k = F.pad(k, (0, 0, 0, padding))
            v = F.pad(v, (0, 0, 0, padding))
        
        # Reshape into blocks
        q_blocks = rearrange(q, 'b h (n s) d -> b h n s d', s=block_size)
        k_blocks = rearrange(k, 'b h (n s) d -> b h n s d', s=block_size)
        v_blocks = rearrange(v, 'b h (n s) d -> b h n s d', s=block_size)
        
        # Initialize output
        out_blocks = torch.zeros_like(q_blocks)
        
        for i in range(num_blocks):
            # Local attention within block
            q_local = q_blocks[:, :, i, :, :]  # (B, H, block_size, D)
            k_local = k_blocks[:, :, i, :, :]
            v_local = v_blocks[:, :, i, :, :]
            
            attn_local = torch.matmul(q_local, k_local.transpose(-2, -1)) * self.scale
            attn_local = F.softmax(attn_local, dim=-1)
            attn_local = self.attn_drop(attn_local)
            out_local = torch.matmul(attn_local, v_local)
            
            # Global attention to sparse blocks
            sparse_blocks = list(range(0, num_blocks, self.sparsity_factor))
            if i not in sparse_blocks:
                sparse_blocks.append(i)
            
            if len(sparse_blocks) > 1:
                k_global = k_blocks[:, :, sparse_blocks, :, :]  # (B, H, num_sparse, block_size, D)
                v_global = v_blocks[:, :, sparse_blocks, :, :]
                
                # Flatten sparse blocks
                k_global = rearrange(k_global, 'b h n s d -> b h (n s) d')
                v_global = rearrange(v_global, 'b h n s d -> b h (n s) d')
                
                attn_global = torch.matmul(q_local, k_global.transpose(-2, -1)) * self.scale
                attn_global = F.softmax(attn_global, dim=-1)
                attn_global = self.attn_drop(attn_global)
                out_global = torch.matmul(attn_global, v_global)
                
                # Combine local and global
                out_blocks[:, :, i, :, :] = (out_local + out_global) * 0.5
            else:
                out_blocks[:, :, i, :, :] = out_local
        
        # Reshape back
        out = rearrange(out_blocks, 'b h n s d -> b h (n s) d')
        
        # Remove padding
        if padding > 0:
            out = out[:, :, :L, :]
        
        return out
