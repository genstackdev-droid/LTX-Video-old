"""Audio cross-attention integration for transformer blocks.

This module adds audio cross-attention capability to transformer blocks (layers 8-12)
for synchronized audio-video generation.
"""

import torch
import torch.nn as nn
from typing import Optional
from diffusers.models.attention import Attention


class AudioCrossAttentionBlock(nn.Module):
    """Audio cross-attention block for transformer integration.
    
    This block is inserted into transformer layers 8-12 to enable audio conditioning
    through cross-attention between video latents (query) and audio latents (key/value).
    
    Args:
        hidden_dim (int): Hidden dimension of transformer
        num_heads (int): Number of attention heads
        audio_latent_dim (int): Dimension of audio latents (default: 128)
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        audio_latent_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.audio_latent_dim = audio_latent_dim
        
        # Project audio latents to transformer hidden dimension
        self.audio_proj = nn.Linear(audio_latent_dim, hidden_dim)
        
        # Cross-attention: video queries attend to audio keys/values
        self.cross_attention = Attention(
            query_dim=hidden_dim,
            cross_attention_dim=hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            bias=False,
        )
        
        # Layer norm for audio features
        self.audio_norm = nn.LayerNorm(hidden_dim)
        
        # Gating mechanism for controllable audio conditioning strength
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_latents: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply audio cross-attention to video hidden states.
        
        Args:
            hidden_states: Video hidden states (B, L_video, D)
            audio_latents: Audio latents (B, L_audio, D_audio)
            audio_mask: Optional mask for audio (B, L_audio)
        
        Returns:
            Audio-conditioned hidden states (B, L_video, D)
        """
        if audio_latents is None:
            # No audio conditioning, return unchanged
            return hidden_states
        
        # Project audio latents to hidden dimension
        audio_features = self.audio_proj(audio_latents)  # (B, L_audio, D)
        audio_features = self.audio_norm(audio_features)
        
        # Cross-attention: video queries, audio keys/values
        attn_output = self.cross_attention(
            hidden_states,
            encoder_hidden_states=audio_features,
            attention_mask=audio_mask,
        )
        
        # Gated residual connection
        gate = self.gate(hidden_states)
        output = hidden_states + gate * attn_output
        
        # Output normalization
        output = self.output_norm(output)
        
        return output


def add_audio_cross_attention_to_transformer(
    transformer: nn.Module,
    audio_latent_dim: int = 128,
    start_layer: int = 8,
    end_layer: int = 12,
) -> nn.Module:
    """Add audio cross-attention blocks to transformer layers 8-12.
    
    This function modifies the transformer in-place by wrapping specified layers
    with audio cross-attention capability.
    
    Args:
        transformer: Transformer3DModel to modify
        audio_latent_dim: Dimension of audio latents
        start_layer: First layer to add audio cross-attention (default: 8)
        end_layer: Last layer to add audio cross-attention (default: 12)
    
    Returns:
        Modified transformer with audio cross-attention
    """
    if not hasattr(transformer, 'transformer_blocks'):
        raise ValueError("Transformer must have 'transformer_blocks' attribute")
    
    num_layers = len(transformer.transformer_blocks)
    
    # Validate layer indices
    if start_layer < 0 or end_layer > num_layers or start_layer >= end_layer:
        raise ValueError(
            f"Invalid layer range: start_layer={start_layer}, end_layer={end_layer}, "
            f"num_layers={num_layers}"
        )
    
    # Get hidden dimension from first transformer block
    hidden_dim = transformer.inner_dim
    num_heads = transformer.num_attention_heads
    
    # Add audio cross-attention to specified layers
    for layer_idx in range(start_layer, end_layer):
        audio_cross_attn = AudioCrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            audio_latent_dim=audio_latent_dim,
        )
        
        # Store in transformer for later use
        if not hasattr(transformer, 'audio_cross_attention_blocks'):
            transformer.audio_cross_attention_blocks = nn.ModuleDict()
        
        transformer.audio_cross_attention_blocks[str(layer_idx)] = audio_cross_attn
    
    # Store configuration
    transformer.audio_cross_attention_config = {
        'start_layer': start_layer,
        'end_layer': end_layer,
        'audio_latent_dim': audio_latent_dim,
    }
    
    return transformer


def apply_audio_cross_attention(
    transformer: nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    audio_latents: Optional[torch.Tensor] = None,
    audio_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply audio cross-attention for a specific layer if available.
    
    This function should be called within the transformer's forward pass
    after each transformer block in layers 8-12.
    
    Args:
        transformer: Transformer with audio cross-attention
        layer_idx: Current layer index
        hidden_states: Current hidden states
        audio_latents: Audio latents for conditioning
        audio_mask: Optional audio mask
    
    Returns:
        Hidden states (with audio conditioning if applicable)
    """
    # Check if audio cross-attention is configured for this layer
    if not hasattr(transformer, 'audio_cross_attention_blocks'):
        return hidden_states
    
    if str(layer_idx) not in transformer.audio_cross_attention_blocks:
        return hidden_states
    
    # Apply audio cross-attention
    audio_cross_attn = transformer.audio_cross_attention_blocks[str(layer_idx)]
    hidden_states = audio_cross_attn(hidden_states, audio_latents, audio_mask)
    
    return hidden_states
