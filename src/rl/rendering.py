import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class DifferentiableRenderer:
    """
    Differentiable canvas renderer for neural drawing.
    
    Implements soft rasterization to make drawing operations differentiable
    while maintaining GPU efficiency for batch processing.
    """
    
    def __init__(self, canvas_size: int = 128, device: str = 'cuda'):
        self.canvas_size = canvas_size
        self.device = device
        
        # Pre-compute coordinate grids for efficiency
        y_coords, x_coords = torch.meshgrid(
            torch.arange(canvas_size, device=device, dtype=torch.float32),
            torch.arange(canvas_size, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Shape: (canvas_size, canvas_size, 2)
        self.coord_grid = torch.stack([x_coords, y_coords], dim=-1)
        
    def draw_stroke_batch(
        self,
        canvas: torch.Tensor,
        start_pos: torch.Tensor,
        end_pos: torch.Tensor,
        pressure: torch.Tensor,
        pen_down: torch.Tensor,
        stroke_width: float = 2.0
    ) -> torch.Tensor:
        """
        Draw strokes on canvas batch using differentiable rendering.
        
        Args:
            canvas: (batch_size, 3, H, W) current canvas state
            start_pos: (batch_size, 2) start positions in [0,1] normalized coords
            end_pos: (batch_size, 2) end positions in [0,1] normalized coords  
            pressure: (batch_size,) pen pressure values [0,1]
            pen_down: (batch_size,) boolean mask for active strokes
            stroke_width: base stroke width in pixels
            
        Returns:
            Updated canvas tensor
        """
        batch_size = canvas.shape[0]
        
        # Convert normalized coordinates to pixel coordinates
        start_px = start_pos * self.canvas_size
        end_px = end_pos * self.canvas_size
        
        # Create stroke masks for entire batch
        stroke_masks = self._create_line_masks_batch(
            start_px, end_px, pressure, pen_down, stroke_width
        )
        
        # Apply strokes to canvas
        canvas_updated = canvas + stroke_masks * pressure.view(batch_size, 1, 1, 1)
        
        return torch.clamp(canvas_updated, 0, 1)
    
    def _create_line_masks_batch(
        self,
        start_pos: torch.Tensor,
        end_pos: torch.Tensor,
        pressure: torch.Tensor,
        pen_down: torch.Tensor,
        stroke_width: float
    ) -> torch.Tensor:
        """Create line masks for entire batch efficiently."""
        batch_size = start_pos.shape[0]
        
        # Initialize masks
        masks = torch.zeros(
            (batch_size, 3, self.canvas_size, self.canvas_size),
            device=self.device,
            dtype=torch.float32
        )
        
        # Only process active strokes
        active_mask = pen_down
        if not torch.any(active_mask):
            return masks
        
        # Get active stroke parameters
        active_start = start_pos[active_mask]  # (n_active, 2)
        active_end = end_pos[active_mask]      # (n_active, 2)  
        active_pressure = pressure[active_mask]  # (n_active,)
        
        # Create distance fields for active strokes
        distance_fields = self._line_distance_field_batch(
            active_start, active_end, active_pressure, stroke_width
        )
        
        # Assign back to full batch
        masks[active_mask] = distance_fields
        
        return masks
    
    def _line_distance_field_batch(
        self,
        start_pos: torch.Tensor,
        end_pos: torch.Tensor,
        pressure: torch.Tensor,
        base_width: float
    ) -> torch.Tensor:
        """
        Compute signed distance field for line segments.
        
        Uses the point-to-line-segment distance formula with soft thresholding
        for differentiable rendering.
        """
        n_active = start_pos.shape[0]
        
        # Expand coordinate grid for batch processing
        # coords: (canvas_size, canvas_size, 2)
        # expanded_coords: (n_active, canvas_size, canvas_size, 2)
        expanded_coords = self.coord_grid.unsqueeze(0).expand(n_active, -1, -1, -1)
        
        # Expand line parameters
        # start_expanded: (n_active, canvas_size, canvas_size, 2)
        start_expanded = start_pos.view(n_active, 1, 1, 2).expand(-1, self.canvas_size, self.canvas_size, -1)
        end_expanded = end_pos.view(n_active, 1, 1, 2).expand(-1, self.canvas_size, self.canvas_size, -1)
        
        # Vector from start to end
        line_vec = end_expanded - start_expanded  # (n_active, H, W, 2)
        
        # Vector from start to each pixel
        pixel_vec = expanded_coords - start_expanded  # (n_active, H, W, 2)
        
        # Project pixel vector onto line vector
        line_length_sq = torch.sum(line_vec ** 2, dim=-1, keepdim=True)  # (n_active, H, W, 1)
        line_length_sq = torch.clamp(line_length_sq, min=1e-8)  # Avoid division by zero
        
        projection_scalar = torch.sum(pixel_vec * line_vec, dim=-1, keepdim=True) / line_length_sq
        projection_scalar = torch.clamp(projection_scalar, 0, 1)  # Clamp to line segment
        
        # Closest point on line segment
        closest_point = start_expanded + projection_scalar * line_vec  # (n_active, H, W, 2)
        
        # Distance from pixel to closest point on line
        distance = torch.norm(expanded_coords - closest_point, dim=-1)  # (n_active, H, W)
        
        # Convert distance to soft mask using gaussian falloff
        stroke_width = base_width * pressure.view(n_active, 1, 1)  # Pressure affects width
        mask_intensity = torch.exp(-(distance ** 2) / (2 * stroke_width ** 2))
        
        # Expand to 3 channels for RGB
        masks = mask_intensity.unsqueeze(1).expand(-1, 3, -1, -1)  # (n_active, 3, H, W)
        
        return masks
    
    def draw_circle_batch(
        self,
        canvas: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        intensities: torch.Tensor,
        active_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Draw circles/dots on canvas batch.
        
        Useful for pen-down events or point-based drawing.
        """
        batch_size = canvas.shape[0]
        
        if not torch.any(active_mask):
            return canvas
        
        # Convert normalized coordinates to pixels
        centers_px = centers * self.canvas_size
        radii_px = radii * self.canvas_size
        
        # Create circle masks
        circle_masks = self._create_circle_masks_batch(
            centers_px[active_mask],
            radii_px[active_mask], 
            intensities[active_mask]
        )
        
        # Apply to canvas
        updated_canvas = canvas.clone()
        updated_canvas[active_mask] = torch.clamp(
            updated_canvas[active_mask] + circle_masks, 0, 1
        )
        
        return updated_canvas
    
    def _create_circle_masks_batch(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        intensities: torch.Tensor
    ) -> torch.Tensor:
        """Create circle masks for batch of circles."""
        n_active = centers.shape[0]
        
        # Expand coordinate grid
        expanded_coords = self.coord_grid.unsqueeze(0).expand(n_active, -1, -1, -1)
        
        # Expand circle parameters  
        centers_expanded = centers.view(n_active, 1, 1, 2).expand(-1, self.canvas_size, self.canvas_size, -1)
        radii_expanded = radii.view(n_active, 1, 1).expand(-1, self.canvas_size, self.canvas_size)
        
        # Distance from each pixel to circle center
        distances = torch.norm(expanded_coords - centers_expanded, dim=-1)
        
        # Soft circle mask using gaussian
        masks_single = torch.exp(-(distances ** 2) / (2 * radii_expanded ** 2))
        masks_single = masks_single * intensities.view(n_active, 1, 1)
        
        # Expand to 3 channels
        masks = masks_single.unsqueeze(1).expand(-1, 3, -1, -1)
        
        return masks