import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import cv2
from PIL import Image
import torchvision.transforms as transforms

class DrawingEnvironment:
    """
    Vectorized RL environment for learning to draw with pen strokes.
    
    State: [canvas, pen_pos, pen_orientation, target_image]
    Action: [delta_x, delta_y, pen_pressure, pen_up_down]
    """
    
    def __init__(
        self,
        batch_size: int = 256,
        canvas_size: int = 128,
        max_steps: int = 1000,
        device: str = 'cuda'
    ):
        self.batch_size = batch_size
        self.canvas_size = canvas_size
        self.max_steps = max_steps
        self.device = device
        
        # State components
        self.canvas = torch.zeros((batch_size, 3, canvas_size, canvas_size), device=device)
        self.pen_pos = torch.zeros((batch_size, 2), device=device)  # (x, y)
        self.pen_orientation = torch.zeros((batch_size, 1), device=device)  # angle
        self.target_images = None
        self.step_count = torch.zeros(batch_size, device=device)
        self.pen_down = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Action space bounds
        self.max_velocity = 0.1  # As fraction of canvas size
        self.max_pressure = 1.0
        
        # Stroke tracking
        self.stroke_points = []
        self.total_path_length = torch.zeros(batch_size, device=device)
        self.num_pen_lifts = torch.zeros(batch_size, device=device)
        
    def reset(self, target_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Reset environment with new target images."""
        self.canvas.zero_()
        self.pen_pos = torch.rand((self.batch_size, 2), device=self.device) * 0.8 + 0.1
        self.pen_orientation.zero_()
        self.target_images = target_images.to(self.device)
        self.step_count.zero_()
        self.pen_down.zero_()
        self.stroke_points = []
        self.total_path_length.zero_()
        self.num_pen_lifts.zero_()
        
        return self._get_state()
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Take environment step with vectorized actions.
        
        Args:
            actions: (batch_size, 4) - [delta_x, delta_y, pen_pressure, pen_up_down]
        
        Returns:
            state, reward, done, info
        """
        # Parse actions
        delta_pos = actions[:, :2] * self.max_velocity
        pen_pressure = torch.sigmoid(actions[:, 2]) * self.max_pressure
        pen_up_down = torch.sigmoid(actions[:, 3]) > 0.5
        
        # Update pen position
        old_pos = self.pen_pos.clone()
        self.pen_pos = torch.clamp(self.pen_pos + delta_pos, 0, 1)
        
        # Track path length
        move_distance = torch.norm(self.pen_pos - old_pos, dim=1)
        self.total_path_length += move_distance
        
        # Track pen lifts
        pen_lift_occurred = self.pen_down & ~pen_up_down
        self.num_pen_lifts += pen_lift_occurred.float()
        
        # Draw stroke if pen is down
        self._draw_stroke(old_pos, self.pen_pos, pen_pressure, pen_up_down)
        self.pen_down = pen_up_down
        
        # Update step count
        self.step_count += 1
        
        # Calculate rewards
        rewards = self._calculate_reward()
        
        # Check if done
        done = self.step_count >= self.max_steps
        
        return self._get_state(), rewards, done, {}
    
    def _draw_stroke(self, start_pos: torch.Tensor, end_pos: torch.Tensor, 
                    pressure: torch.Tensor, pen_down: torch.Tensor):
        """Draw stroke on canvas using differentiable rendering."""
        batch_size = start_pos.shape[0]
        
        for b in range(batch_size):
            if not pen_down[b]:
                continue
                
            # Convert normalized coordinates to pixel coordinates
            start_px = (start_pos[b] * self.canvas_size).int()
            end_px = (end_pos[b] * self.canvas_size).int()
            
            # Create line mask using Bresenham's algorithm approximation
            line_mask = self._create_line_mask(
                start_px, end_px, pressure[b], b
            )
            
            # Apply stroke to canvas
            self.canvas[b] = torch.clamp(
                self.canvas[b] + line_mask * pressure[b], 0, 1
            )
    
    def _create_line_mask(self, start: torch.Tensor, end: torch.Tensor, 
                         pressure: float, batch_idx: int) -> torch.Tensor:
        """Create differentiable line mask."""
        # Simple approach: create Gaussian blobs along line
        mask = torch.zeros((3, self.canvas_size, self.canvas_size), device=self.device)
        
        # Linear interpolation between start and end
        num_points = max(int(torch.norm((end - start).float()).item()), 1)
        t_values = torch.linspace(0, 1, num_points, device=self.device)
        
        for t in t_values:
            point = start.float() + t * (end - start).float()
            x, y = point.int()
            
            if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                # Create Gaussian blob
                radius = max(1, int(pressure * 3))
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(self.canvas_size, device=self.device),
                    torch.arange(self.canvas_size, device=self.device),
                    indexing='ij'
                )
                
                dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
                gaussian = torch.exp(-dist_sq / (2 * radius ** 2))
                mask += gaussian.unsqueeze(0) * 0.1
        
        return torch.clamp(mask, 0, 1)
    
    def _get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state representation."""
        return {
            'canvas': self.canvas,
            'pen_pos': self.pen_pos,
            'pen_orientation': self.pen_orientation,
            'target_image': self.target_images,
            'step_count': self.step_count / self.max_steps
        }
    
    def _calculate_reward(self) -> torch.Tensor:
        """Calculate multi-objective reward."""
        # Import reward functions to avoid circular imports
        from .rewards import RewardCalculator
        
        calculator = RewardCalculator(device=self.device)
        
        return calculator.compute_total_reward(
            canvas=self.canvas,
            target=self.target_images,
            pen_pos=self.pen_pos,
            total_path_length=self.total_path_length,
            num_pen_lifts=self.num_pen_lifts,
            step_count=self.step_count
        )