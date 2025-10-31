"""Camera control module for 3D-aware video generation.

Enables precise camera movements (pan, zoom, orbit) through motion vector constraints
in the latent space.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import math


class CameraMotionType(Enum):
    """Types of camera motion."""
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    ORBIT_LEFT = "orbit_left"
    ORBIT_RIGHT = "orbit_right"
    DOLLY_FORWARD = "dolly_forward"
    DOLLY_BACKWARD = "dolly_backward"


@dataclass
class CameraParameters:
    """3D camera parameters for video generation.
    
    Attributes:
        position (Tuple[float, float, float]): Camera position (x, y, z)
        rotation (Tuple[float, float, float]): Camera rotation (pitch, yaw, roll) in degrees
        fov (float): Field of view in degrees
        focal_length (float): Focal length in mm (optional, alternative to FOV)
        motion_type (CameraMotionType): Type of camera motion
        motion_speed (float): Speed of camera motion (0.0 to 1.0)
        motion_smoothness (float): Smoothness of motion (0.0 to 1.0)
    """
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov: float = 60.0
    focal_length: Optional[float] = None
    motion_type: CameraMotionType = CameraMotionType.STATIC
    motion_speed: float = 0.5
    motion_smoothness: float = 0.8


class CameraController(nn.Module):
    """Controller for camera-based video generation.
    
    Converts 3D camera parameters into motion vectors in latent space
    that guide the video generation process.
    
    Args:
        latent_channels (int): Number of latent channels
        hidden_dim (int): Hidden dimension for camera parameter encoding
        num_frames (int): Number of frames in the video
        spatial_resolution (Tuple[int, int]): Spatial resolution (H, W)
    """
    
    def __init__(
        self,
        latent_channels: int = 128,
        hidden_dim: int = 256,
        num_frames: int = 257,
        spatial_resolution: Tuple[int, int] = (24, 40),
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.spatial_resolution = spatial_resolution
        
        # Camera parameter encoder
        # Encodes camera parameters to a latent representation
        self.camera_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),  # 10 camera params: pos(3), rot(3), fov(1), motion(3)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Motion vector generator
        # Generates spatiotemporal motion vectors for each frame
        self.motion_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_channels * 2),  # 2 for (dx, dy) motion
        )
        
        # Temporal motion smoother
        self.temporal_smoother = nn.Conv1d(
            in_channels=latent_channels * 2,
            out_channels=latent_channels * 2,
            kernel_size=5,
            padding=2,
            groups=latent_channels * 2,
        )
        
        # Spatial motion field generator
        self.spatial_field_generator = nn.Sequential(
            nn.Conv2d(
                in_channels=latent_channels * 2,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=latent_channels,
                kernel_size=3,
                padding=1,
            ),
        )
    
    def forward(
        self,
        camera_params: CameraParameters,
        num_frames: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate camera motion constraints for video generation.
        
        Args:
            camera_params: Camera parameters defining the motion
            num_frames: Optional override for number of frames
        
        Returns:
            Tuple of (motion_vectors, motion_masks) where:
                - motion_vectors: (1, C, T, H, W) - spatiotemporal motion field
                - motion_masks: (1, T, H, W) - per-frame motion strength
        """
        if num_frames is None:
            num_frames = self.num_frames
        
        H, W = self.spatial_resolution
        device = next(self.parameters()).device
        
        # Encode camera parameters
        camera_embedding = self._encode_camera_params(camera_params, device)
        
        # Generate base motion vectors for each frame
        motion_vectors = self._generate_temporal_motion(
            camera_embedding,
            camera_params,
            num_frames,
            device,
        )
        
        # Generate spatial motion field
        motion_field = self._generate_spatial_field(
            motion_vectors,
            camera_params,
            H,
            W,
        )
        
        # Generate motion masks (strength per frame)
        motion_masks = self._generate_motion_masks(
            camera_params,
            num_frames,
            H,
            W,
            device,
        )
        
        return motion_field, motion_masks
    
    def _encode_camera_params(
        self,
        camera_params: CameraParameters,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode camera parameters to a latent representation."""
        # Convert camera parameters to tensor
        params_list = [
            *camera_params.position,  # (x, y, z)
            *camera_params.rotation,  # (pitch, yaw, roll)
            camera_params.fov / 180.0,  # Normalized FOV
            camera_params.motion_speed,
            camera_params.motion_smoothness,
            float(camera_params.motion_type.value == "zoom_in"),  # Motion type encoding
        ]
        
        params_tensor = torch.tensor(params_list, device=device).unsqueeze(0)
        
        # Encode through network
        embedding = self.camera_encoder(params_tensor)
        
        return embedding
    
    def _generate_temporal_motion(
        self,
        camera_embedding: torch.Tensor,
        camera_params: CameraParameters,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate temporal motion vectors based on camera motion type."""
        # Generate base motion vector
        base_motion = self.motion_generator(camera_embedding)  # (1, C*2)
        
        # Reshape and expand to all frames
        base_motion = base_motion.reshape(1, self.latent_channels * 2, 1)
        motion_sequence = base_motion.expand(-1, -1, num_frames)
        
        # Apply temporal smoothing
        motion_sequence = self.temporal_smoother(motion_sequence)
        
        # Apply motion type-specific transformations
        motion_sequence = self._apply_motion_type(
            motion_sequence,
            camera_params.motion_type,
            camera_params.motion_speed,
            num_frames,
        )
        
        return motion_sequence
    
    def _apply_motion_type(
        self,
        motion_sequence: torch.Tensor,
        motion_type: CameraMotionType,
        motion_speed: float,
        num_frames: int,
    ) -> torch.Tensor:
        """Apply motion type-specific transformations."""
        # Generate time progression
        t = torch.linspace(0, 1, num_frames, device=motion_sequence.device)
        
        if motion_type == CameraMotionType.STATIC:
            motion_sequence = motion_sequence * 0.0
        
        elif motion_type == CameraMotionType.PAN_LEFT:
            # Horizontal motion to the left
            scale = -motion_speed * t.view(1, 1, -1)
            motion_sequence = motion_sequence * scale
        
        elif motion_type == CameraMotionType.PAN_RIGHT:
            # Horizontal motion to the right
            scale = motion_speed * t.view(1, 1, -1)
            motion_sequence = motion_sequence * scale
        
        elif motion_type == CameraMotionType.PAN_UP:
            # Vertical motion upward
            scale = -motion_speed * t.view(1, 1, -1)
            motion_sequence = motion_sequence * scale
        
        elif motion_type == CameraMotionType.PAN_DOWN:
            # Vertical motion downward
            scale = motion_speed * t.view(1, 1, -1)
            motion_sequence = motion_sequence * scale
        
        elif motion_type in [CameraMotionType.ZOOM_IN, CameraMotionType.ZOOM_OUT]:
            # Radial motion (zoom)
            direction = 1.0 if motion_type == CameraMotionType.ZOOM_IN else -1.0
            scale = direction * motion_speed * t.view(1, 1, -1)
            motion_sequence = motion_sequence * scale
        
        elif motion_type in [CameraMotionType.ORBIT_LEFT, CameraMotionType.ORBIT_RIGHT]:
            # Circular motion
            direction = -1.0 if motion_type == CameraMotionType.ORBIT_LEFT else 1.0
            angle = direction * motion_speed * t * 2 * math.pi
            # Apply rotation matrix in motion space
            cos_a = torch.cos(angle).view(1, 1, -1)
            sin_a = torch.sin(angle).view(1, 1, -1)
            motion_sequence = motion_sequence * torch.cat([cos_a, sin_a], dim=1)
        
        return motion_sequence
    
    def _generate_spatial_field(
        self,
        motion_vectors: torch.Tensor,
        camera_params: CameraParameters,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Generate spatial motion field from temporal motion vectors."""
        # motion_vectors: (1, C*2, T)
        T = motion_vectors.shape[2]
        
        # Expand spatially
        motion_field = motion_vectors.unsqueeze(-1).unsqueeze(-1)  # (1, C*2, T, 1, 1)
        motion_field = motion_field.expand(-1, -1, -1, H, W)  # (1, C*2, T, H, W)
        
        # Apply spatial variation based on FOV and position
        # Create spatial coordinate grids
        y_grid = torch.linspace(-1, 1, H, device=motion_field.device)
        x_grid = torch.linspace(-1, 1, W, device=motion_field.device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Apply FOV-based scaling (center vs edges)
        fov_scale = camera_params.fov / 90.0  # Normalize to typical FOV
        radial_dist = torch.sqrt(xx**2 + yy**2)
        spatial_scale = 1.0 + (radial_dist * fov_scale * 0.5)
        spatial_scale = spatial_scale.view(1, 1, 1, H, W)
        
        # Apply spatial scaling
        motion_field = motion_field * spatial_scale
        
        # Process through spatial field generator
        # Reshape for 2D convolution
        B, C2, T, H, W = motion_field.shape
        motion_field_2d = motion_field.permute(0, 2, 1, 3, 4).reshape(B * T, C2, H, W)
        
        # Generate refined spatial field
        refined_field = self.spatial_field_generator(motion_field_2d)
        
        # Reshape back
        refined_field = refined_field.reshape(B, T, self.latent_channels, H, W)
        refined_field = refined_field.permute(0, 2, 1, 3, 4)
        
        return refined_field
    
    def _generate_motion_masks(
        self,
        camera_params: CameraParameters,
        num_frames: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate per-frame motion strength masks."""
        # Start with uniform strength
        masks = torch.ones(1, num_frames, H, W, device=device)
        
        # Apply smoothness curve
        t = torch.linspace(0, 1, num_frames, device=device)
        
        # Smooth start and end (ease-in-out)
        if camera_params.motion_smoothness > 0:
            smoothness = camera_params.motion_smoothness
            temporal_curve = 0.5 * (1 - torch.cos(t * math.pi))
            temporal_curve = torch.pow(temporal_curve, smoothness)
            masks = masks * temporal_curve.view(1, -1, 1, 1)
        
        # Apply motion speed
        masks = masks * camera_params.motion_speed
        
        return masks
