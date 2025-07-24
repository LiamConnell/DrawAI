import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Optional
import cv2
import numpy as np

class RewardCalculator:
    """
    Multi-objective reward calculation for drawing RL agent.
    
    Combines:
    1. Perceptual similarity (LPIPS + MSE) - 40%
    2. Stroke quality (smoothness, natural velocity) - 30% 
    3. Efficiency (path length, pen lifts) - 20%
    4. Edge/structure similarity - 10%
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Reward weights
        self.w_perceptual = 0.4
        self.w_stroke = 0.3
        self.w_efficiency = 0.2
        self.w_structure = 0.1
        
        # LPIPS model for perceptual similarity - disabled for now to avoid memory issues
        # TODO: Re-enable LPIPS with proper singleton pattern
        self.lpips_model = None
        if False:  # Temporarily disabled
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').to(device)
            except ImportError:
                print("Warning: LPIPS not available, using MSE only for perceptual reward")
                self.lpips_model = None
        
        # Edge detection kernels
        self.register_edge_kernels()
        
    def register_edge_kernels(self):
        """Register Sobel kernels for edge detection."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(self.device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(self.device)
    
    def compute_total_reward(
        self,
        canvas: torch.Tensor,
        target: torch.Tensor,
        pen_pos: torch.Tensor,
        total_path_length: torch.Tensor,
        num_pen_lifts: torch.Tensor,
        step_count: torch.Tensor,
        prev_pen_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute total multi-objective reward."""
        
        batch_size = canvas.shape[0]
        
        # 1. Perceptual similarity reward
        r_perceptual = self._perceptual_reward(canvas, target)
        
        # 2. Stroke quality reward  
        r_stroke = self._stroke_quality_reward(pen_pos, prev_pen_pos, step_count)
        
        # 3. Efficiency reward
        r_efficiency = self._efficiency_reward(total_path_length, num_pen_lifts, step_count)
        
        # 4. Structure/edge reward
        r_structure = self._structure_reward(canvas, target)
        
        # Combine rewards
        total_reward = (
            self.w_perceptual * r_perceptual +
            self.w_stroke * r_stroke +
            self.w_efficiency * r_efficiency +
            self.w_structure * r_structure
        )
        
        return total_reward
    
    def _perceptual_reward(self, canvas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual similarity reward using LPIPS + MSE."""
        batch_size = canvas.shape[0]
        
        # MSE component
        mse_loss = F.mse_loss(canvas, target, reduction='none')
        mse_reward = -torch.mean(mse_loss.view(batch_size, -1), dim=1)
        
        # LPIPS component (if available)
        if self.lpips_model is not None:
            # Normalize to [-1, 1] range for LPIPS
            canvas_norm = canvas * 2 - 1
            target_norm = target * 2 - 1
            
            lpips_distances = self.lpips_model(canvas_norm, target_norm)
            lpips_reward = -lpips_distances.squeeze()
        else:
            lpips_reward = torch.zeros(batch_size, device=self.device)
        
        # Combine MSE and LPIPS
        perceptual_reward = 0.7 * mse_reward + 0.3 * lpips_reward
        
        return perceptual_reward
    
    def _stroke_quality_reward(
        self, 
        pen_pos: torch.Tensor,
        prev_pen_pos: Optional[torch.Tensor],
        step_count: torch.Tensor
    ) -> torch.Tensor:
        """Calculate stroke quality reward based on smoothness and natural movement."""
        batch_size = pen_pos.shape[0]
        
        if prev_pen_pos is None or torch.all(step_count < 2):
            return torch.zeros(batch_size, device=self.device)
        
        # Calculate velocity
        velocity = pen_pos - prev_pen_pos
        velocity_magnitude = torch.norm(velocity, dim=1)
        
        # Penalize extreme velocities (too fast or too slow)
        optimal_velocity = 0.05  # Normalized canvas units per step
        velocity_penalty = torch.abs(velocity_magnitude - optimal_velocity)
        
        # Reward smooth, consistent movement
        smoothness_reward = -velocity_penalty
        
        # Bonus for staying within reasonable velocity range
        reasonable_range = (velocity_magnitude > 0.01) & (velocity_magnitude < 0.15)
        range_bonus = reasonable_range.float() * 0.1
        
        return smoothness_reward + range_bonus
    
    def _efficiency_reward(
        self,
        total_path_length: torch.Tensor,
        num_pen_lifts: torch.Tensor,
        step_count: torch.Tensor
    ) -> torch.Tensor:
        """Calculate efficiency reward penalizing excessive movement and pen lifts."""
        
        # Normalize path length by number of steps taken
        avg_step_length = total_path_length / torch.clamp(step_count, min=1)
        
        # Penalize very long average step lengths (inefficient drawing)
        length_penalty = torch.clamp(avg_step_length - 0.05, min=0) * 2.0
        
        # Penalize excessive pen lifts
        lift_penalty = num_pen_lifts * 0.1
        
        efficiency_reward = -length_penalty - lift_penalty
        
        return efficiency_reward
    
    def _structure_reward(self, canvas: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate structural similarity reward using edge detection."""
        batch_size = canvas.shape[0]
        
        # Convert to grayscale for edge detection
        canvas_gray = torch.mean(canvas, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)
        
        # Apply Sobel edge detection
        canvas_edges = self._detect_edges(canvas_gray)
        target_edges = self._detect_edges(target_gray)
        
        # Calculate IoU of edge maps
        intersection = torch.sum((canvas_edges > 0.1) & (target_edges > 0.1), dim=[1, 2, 3])
        union = torch.sum((canvas_edges > 0.1) | (target_edges > 0.1), dim=[1, 2, 3])
        
        # Avoid division by zero
        iou = intersection / torch.clamp(union, min=1)
        
        return iou
    
    def _detect_edges(self, image: torch.Tensor) -> torch.Tensor:
        """Detect edges using Sobel operator."""
        # Apply Sobel kernels
        edges_x = F.conv2d(image, self.sobel_x, padding=1)
        edges_y = F.conv2d(image, self.sobel_y, padding=1)
        
        # Combine edge magnitudes
        edge_magnitude = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        
        return edge_magnitude
    
    def get_reward_components(
        self,
        canvas: torch.Tensor,
        target: torch.Tensor,
        pen_pos: torch.Tensor,
        total_path_length: torch.Tensor,
        num_pen_lifts: torch.Tensor,
        step_count: torch.Tensor,
        prev_pen_pos: Optional[torch.Tensor] = None
    ) -> dict:
        """Get individual reward components for analysis."""
        
        return {
            'perceptual': self._perceptual_reward(canvas, target),
            'stroke_quality': self._stroke_quality_reward(pen_pos, prev_pen_pos, step_count),
            'efficiency': self._efficiency_reward(total_path_length, num_pen_lifts, step_count),
            'structure': self._structure_reward(canvas, target)
        }