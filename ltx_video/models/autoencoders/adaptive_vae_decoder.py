"""Adaptive VAE decoder adapter for hierarchical VAE integration.

Connects the hierarchical VAE encoder with the existing VAE decoder,
enabling efficient 4K video decoding with tiling and blending.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from diffusers.models.autoencoders.vae import DecoderOutput


class AdaptiveVAEDecoder(nn.Module):
    """Adapter between hierarchical VAE encoder and existing VAE decoder.
    
    This adapter handles:
    1. Resolution-aware decoding (adapts to input resolution)
    2. Tiled decoding for 4K support
    3. Smooth blending of tile boundaries
    4. Integration with existing VAE decoder
    
    Args:
        vae_decoder: Existing VAE decoder module
        hierarchical_encoder: Optional hierarchical VAE encoder
        use_tiling: Whether to use tiled decoding for large resolutions
        tile_size: Size of tiles for tiled decoding (default: 512)
        overlap: Overlap between tiles in pixels (default: 64)
    """
    
    def __init__(
        self,
        vae_decoder: nn.Module,
        hierarchical_encoder: Optional[nn.Module] = None,
        use_tiling: bool = True,
        tile_size: int = 512,
        overlap: int = 64,
    ):
        super().__init__()
        
        self.vae_decoder = vae_decoder
        self.hierarchical_encoder = hierarchical_encoder
        self.use_tiling = use_tiling
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Adaptation layer for hierarchical latents
        if hierarchical_encoder is not None:
            latent_channels = hierarchical_encoder.latent_channels
            decoder_channels = self._get_decoder_input_channels(vae_decoder)
            
            # Projection from hierarchical latent to decoder input
            self.latent_adapter = nn.Conv3d(
                latent_channels * 2,  # mean and logvar from hierarchical encoder
                decoder_channels,
                kernel_size=1,
            )
        else:
            self.latent_adapter = nn.Identity()
    
    def _get_decoder_input_channels(self, decoder: nn.Module) -> int:
        """Extract expected input channels from decoder."""
        # Try to get from post_quant_conv if it exists
        if hasattr(decoder, 'post_quant_conv'):
            return decoder.post_quant_conv.in_channels
        # Otherwise, look at first conv layer
        for module in decoder.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                return module.in_channels
        return 128  # Default fallback
    
    def forward(
        self,
        latents: torch.Tensor,
        return_dict: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """Decode latents to video frames.
        
        Args:
            latents: Latent tensor (B, C, T, H, W)
            return_dict: Whether to return DecoderOutput
        
        Returns:
            Decoded video frames
        """
        B, C, T, H, W = latents.shape
        
        # Determine if we need tiling based on resolution
        latent_spatial_size = H * W
        needs_tiling = self.use_tiling and latent_spatial_size > (self.tile_size // 32) ** 2
        
        if needs_tiling and H * 32 > 1080:  # 4K or larger
            output = self._tiled_decode(latents)
        else:
            output = self._direct_decode(latents)
        
        if return_dict:
            return DecoderOutput(sample=output)
        return (output,)
    
    def _direct_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Direct decoding without tiling."""
        # Adapt hierarchical latents if needed
        adapted_latents = self.latent_adapter(latents)
        
        # Pass through VAE decoder
        if hasattr(self.vae_decoder, 'decode'):
            output = self.vae_decoder.decode(adapted_latents).sample
        else:
            output = self.vae_decoder(adapted_latents)
        
        return output
    
    def _tiled_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Tiled decoding for large resolutions (4K).
        
        Splits latents into overlapping tiles, decodes each tile,
        and blends them together smoothly.
        """
        B, C, T, H, W = latents.shape
        
        # Calculate tile parameters in latent space
        tile_latent_size = self.tile_size // 32  # VAE downsamples by 32
        overlap_latent = self.overlap // 32
        stride = tile_latent_size - overlap_latent
        
        # Calculate number of tiles
        num_tiles_h = (H + stride - 1) // stride
        num_tiles_w = (W + stride - 1) // stride
        
        # Initialize output tensor
        output_h = H * 32
        output_w = W * 32
        output = torch.zeros(
            B, 3, T, output_h, output_w,
            device=latents.device,
            dtype=latents.dtype,
        )
        weight_map = torch.zeros(
            B, 1, T, output_h, output_w,
            device=latents.device,
            dtype=latents.dtype,
        )
        
        # Process each tile
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # Extract tile
                h_start = i * stride
                h_end = min(h_start + tile_latent_size, H)
                w_start = j * stride
                w_end = min(w_start + tile_latent_size, W)
                
                tile_latent = latents[:, :, :, h_start:h_end, w_start:w_end]
                
                # Decode tile
                tile_output = self._direct_decode(tile_latent)
                
                # Calculate output position
                out_h_start = h_start * 32
                out_h_end = h_end * 32
                out_w_start = w_start * 32
                out_w_end = w_end * 32
                
                # Create blending weights (cosine window)
                tile_h = out_h_end - out_h_start
                tile_w = out_w_end - out_w_start
                weights = self._create_blend_weights(tile_h, tile_w, overlap_latent * 32)
                weights = weights.to(latents.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                
                # Accumulate with blending
                output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += (
                    tile_output * weights
                )
                weight_map[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += weights
        
        # Normalize by accumulated weights
        output = output / (weight_map + 1e-8)
        
        return output
    
    def _create_blend_weights(
        self,
        height: int,
        width: int,
        overlap: int,
    ) -> torch.Tensor:
        """Create smooth blending weights using cosine window.
        
        Args:
            height: Tile height
            width: Tile width
            overlap: Overlap size
        
        Returns:
            Weight tensor of shape (height, width)
        """
        # Create 1D windows
        h_window = torch.ones(height)
        w_window = torch.ones(width)
        
        # Apply cosine taper at edges if we have overlap
        if overlap > 0 and overlap < height:
            # Top edge
            h_window[:overlap] = 0.5 * (1 - torch.cos(
                torch.linspace(0, torch.pi, overlap)
            ))
            # Bottom edge
            h_window[-overlap:] = 0.5 * (1 + torch.cos(
                torch.linspace(0, torch.pi, overlap)
            ))
        
        if overlap > 0 and overlap < width:
            # Left edge
            w_window[:overlap] = 0.5 * (1 - torch.cos(
                torch.linspace(0, torch.pi, overlap)
            ))
            # Right edge
            w_window[-overlap:] = 0.5 * (1 + torch.cos(
                torch.linspace(0, torch.pi, overlap)
            ))
        
        # Create 2D weight map
        weights = h_window.unsqueeze(1) * w_window.unsqueeze(0)
        
        return weights
    
    def enable_tiling(self, tile_size: int = 512, overlap: int = 64):
        """Enable tiled decoding.
        
        Args:
            tile_size: Size of tiles in pixels (default: 512)
            overlap: Overlap between tiles in pixels (default: 64)
        """
        self.use_tiling = True
        self.tile_size = tile_size
        self.overlap = overlap
    
    def disable_tiling(self):
        """Disable tiled decoding."""
        self.use_tiling = False


def create_adaptive_decoder(
    vae: nn.Module,
    hierarchical_encoder: Optional[nn.Module] = None,
    use_tiling: bool = True,
) -> AdaptiveVAEDecoder:
    """Create an adaptive VAE decoder from existing VAE.
    
    Args:
        vae: Existing VAE model (AutoencoderKLWrapper or similar)
        hierarchical_encoder: Optional hierarchical encoder
        use_tiling: Whether to enable tiled decoding
    
    Returns:
        AdaptiveVAEDecoder instance
    """
    # Extract decoder from VAE
    if hasattr(vae, 'decoder'):
        decoder = vae.decoder
    else:
        raise ValueError("VAE must have 'decoder' attribute")
    
    # Create adaptive decoder
    adaptive_decoder = AdaptiveVAEDecoder(
        vae_decoder=decoder,
        hierarchical_encoder=hierarchical_encoder,
        use_tiling=use_tiling,
    )
    
    return adaptive_decoder
