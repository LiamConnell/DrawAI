import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
class PolicyNetwork(nn.Module):
    """
    Policy network for drawing agent.
    
    Takes canvas state and outputs continuous actions for pen movement,
    pressure, and discrete action for pen up/down.
    """
    
    def __init__(
        self,
        canvas_size: int = 128,
        hidden_dim: int = 256,
        action_dim: int = 4
    ):
        super().__init__()
        
        self.canvas_size = canvas_size
        self.action_dim = action_dim
        
        # CNN encoder for visual features
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1),  # canvas + target = 6 channels
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # Calculate CNN output size
        cnn_output_size = 256 * 4 * 4
        
        # MLP for pen state features
        pen_input_size = 4  # pen_pos(2) + pen_orientation(1) + step_count(1)
        self.pen_encoder = nn.Sequential(
            nn.Linear(pen_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined feature processing
        combined_size = cnn_output_size + 64
        self.feature_processor = nn.Sequential(
            nn.Linear(combined_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action heads
        self.continuous_mean = nn.Linear(hidden_dim, 3)  # delta_x, delta_y, pressure
        self.continuous_log_std = nn.Parameter(torch.zeros(3))
        self.discrete_head = nn.Linear(hidden_dim, 2)  # pen up/down
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action distribution parameters and value.
        
        Returns:
            continuous_params: (mean, std) for continuous actions
            discrete_logits: logits for discrete pen up/down action
            value: state value estimate
        """
        batch_size = state['canvas'].shape[0]
        
        # Encode visual features (canvas + target)
        visual_input = torch.cat([state['canvas'], state['target_image']], dim=1)
        visual_features = self.visual_encoder(visual_input)
        visual_features = visual_features.view(batch_size, -1)
        
        # Encode pen state features
        pen_input = torch.cat([
            state['pen_pos'],
            state['pen_orientation'], 
            state['step_count'].unsqueeze(-1)
        ], dim=1)
        pen_features = self.pen_encoder(pen_input)
        
        # Combine features
        combined_features = torch.cat([visual_features, pen_features], dim=1)
        features = self.feature_processor(combined_features)
        
        # Action distribution parameters
        continuous_mean = self.continuous_mean(features)
        continuous_std = torch.exp(self.continuous_log_std.expand_as(continuous_mean))
        
        discrete_logits = self.discrete_head(features)
        
        # State value
        value = self.value_head(features)
        
        return (continuous_mean, continuous_std), discrete_logits, value
    
    def get_action_and_value(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and get log probability and value."""
        (continuous_mean, continuous_std), discrete_logits, value = self.forward(state)
        
        # Sample continuous actions
        continuous_dist = Normal(continuous_mean, continuous_std)
        continuous_action = continuous_dist.sample()
        
        # Sample discrete action
        discrete_dist = Categorical(logits=discrete_logits)
        discrete_action = discrete_dist.sample()
        
        # Combine actions
        action = torch.cat([
            continuous_action,
            discrete_action.float().unsqueeze(-1)
        ], dim=1)
        
        # Calculate log probabilities
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        total_log_prob = continuous_log_prob + discrete_log_prob
        
        return action, total_log_prob, value.squeeze(-1)
    
    def get_log_prob_and_entropy(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get log probability and entropy for given state-action pair."""
        (continuous_mean, continuous_std), discrete_logits, _ = self.forward(state)
        
        # Split action into continuous and discrete parts
        continuous_action = action[:, :3]
        discrete_action = action[:, 3].long()
        
        # Continuous action log prob and entropy
        continuous_dist = Normal(continuous_mean, continuous_std)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1)
        
        # Discrete action log prob and entropy  
        discrete_dist = Categorical(logits=discrete_logits)
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        discrete_entropy = discrete_dist.entropy()
        
        total_log_prob = continuous_log_prob + discrete_log_prob
        total_entropy = continuous_entropy + discrete_entropy
        
        return total_log_prob, total_entropy

class PPOAgent:
    """
    Proximal Policy Optimization agent for drawing tasks.
    """
    
    def __init__(
        self,
        canvas_size: int = 128,
        config: PPOConfig = None,
        device: str = 'cuda'
    ):
        self.config = config or PPOConfig()
        self.device = device
        
        # Initialize networks
        self.policy = PolicyNetwork(canvas_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        
        # Training metrics
        self.training_step = 0
        
    def get_action(self, state: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action for given state."""
        with torch.no_grad():
            if deterministic:
                (continuous_mean, _), discrete_logits, value = self.policy(state)
                discrete_action = torch.argmax(discrete_logits, dim=-1)
                action = torch.cat([
                    continuous_mean,
                    discrete_action.float().unsqueeze(-1)
                ], dim=1)
                return action, torch.zeros_like(value), value.squeeze(-1)
            else:
                return self.policy.get_action_and_value(state)
    
    def update(self, rollout_data: Dict[str, torch.Tensor], num_epochs: int = 4) -> Dict[str, float]:
        """Update policy using PPO algorithm."""
        
        states = rollout_data['states']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['log_probs']
        returns = rollout_data['returns']
        advantages = rollout_data['advantages']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        
        for epoch in range(num_epochs):
            # Get current policy outputs
            new_log_probs, entropy = self.policy.get_log_prob_and_entropy(states, actions)
            _, _, values = self.policy.get_action_and_value(states)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (policy_loss + 
                   self.config.value_loss_coef * value_loss +
                   self.config.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy.mean().item()
        
        self.training_step += 1
        
        return {
            'total_loss': total_loss / num_epochs,
            'policy_loss': policy_loss_sum / num_epochs,
            'value_loss': value_loss_sum / num_epochs,
            'entropy': entropy_sum / num_epochs,
            'training_step': self.training_step
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']