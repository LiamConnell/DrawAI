import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import time
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from .environment import DrawingEnvironment
from .agent import PPOAgent, PPOConfig
from .rendering import DifferentiableRenderer
from .datasets import ImageDatasetManager

class CurriculumLearning:
    """
    Manages curriculum for drawing task complexity.
    
    Progressively increases task difficulty:
    1. Simple geometric shapes
    2. Line drawings  
    3. Complex sketches
    4. Natural images
    """
    
    def __init__(self, canvas_size: int = 128, dataset_name: str = 'cifar10'):
        self.canvas_size = canvas_size
        self.current_level = 0
        self.dataset_manager = ImageDatasetManager(canvas_size)
        self.dataset_name = dataset_name
        
        self.levels = [
            {"name": "circles", "complexity": 0.1, "max_steps": 200, "dataset": None},
            {"name": "lines", "complexity": 0.3, "max_steps": 400, "dataset": None},
            {"name": "simple_shapes", "complexity": 0.5, "max_steps": 600, "dataset": "quickdraw"},
            {"name": "sketches", "complexity": 0.8, "max_steps": 800, "dataset": "quickdraw"},
            {"name": "natural_images", "complexity": 1.0, "max_steps": 1000, "dataset": dataset_name}
        ]
        
    def get_current_level(self) -> Dict:
        """Get current curriculum level."""
        return self.levels[min(self.current_level, len(self.levels) - 1)]
    
    def should_advance(self, success_rate: float, episodes_at_level: int) -> bool:
        """Check if should advance to next level."""
        min_episodes = 1000
        min_success_rate = 0.6
        
        return (episodes_at_level >= min_episodes and 
                success_rate >= min_success_rate and
                self.current_level < len(self.levels) - 1)
    
    def advance_level(self):
        """Advance to next curriculum level."""
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            print(f"Advanced to curriculum level {self.current_level}: {self.get_current_level()['name']}")
    
    def generate_target_batch(self, batch_size: int, device: str) -> torch.Tensor:
        """Generate batch of target images for current level."""
        level = self.get_current_level()
        
        # Use dataset if specified for this level
        if level.get('dataset'):
            try:
                return self.dataset_manager.get_batch_for_curriculum(
                    dataset_name=level['dataset'],
                    batch_size=batch_size,
                    level_complexity=level['complexity']
                )
            except Exception as e:
                print(f"Failed to load dataset {level['dataset']}: {e}")
                print("Falling back to synthetic data...")
        
        # Fallback to synthetic generation
        if level['name'] == 'circles':
            return self._generate_circles(batch_size, device)
        elif level['name'] == 'lines':
            return self._generate_lines(batch_size, device)
        elif level['name'] == 'simple_shapes':
            return self._generate_simple_shapes(batch_size, device)
        elif level['name'] == 'sketches':
            return self._generate_sketches(batch_size, device)
        else:  # natural_images
            return self._generate_natural_images(batch_size, device)
    
    def _generate_circles(self, batch_size: int, device: str) -> torch.Tensor:
        """Generate simple circle targets."""
        targets = torch.zeros((batch_size, 3, self.canvas_size, self.canvas_size), device=device)
        
        for i in range(batch_size):
            # Random circle parameters (adaptive to canvas size)
            margin = min(self.canvas_size // 4, 20)
            cx = torch.randint(margin, max(margin + 1, self.canvas_size - margin), (1,)).item()
            cy = torch.randint(margin, max(margin + 1, self.canvas_size - margin), (1,)).item()
            radius = torch.randint(max(1, self.canvas_size // 8), max(2, self.canvas_size // 4), (1,)).item()
            
            # Create circle mask
            y, x = torch.meshgrid(
                torch.arange(self.canvas_size, device=device),
                torch.arange(self.canvas_size, device=device),
                indexing='ij'
            )
            
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            circle_mask = (dist <= radius).float()
            
            targets[i] = circle_mask.unsqueeze(0).expand(3, -1, -1)
            
        return targets
    
    def _generate_lines(self, batch_size: int, device: str) -> torch.Tensor:
        """Generate simple line targets."""
        targets = torch.zeros((batch_size, 3, self.canvas_size, self.canvas_size), device=device)
        
        for i in range(batch_size):
            # Random line parameters
            x1 = torch.randint(0, self.canvas_size, (1,)).item()
            y1 = torch.randint(0, self.canvas_size, (1,)).item()
            x2 = torch.randint(0, self.canvas_size, (1,)).item()
            y2 = torch.randint(0, self.canvas_size, (1,)).item()
            
            # Simple line drawing (Bresenham's algorithm approximation)
            line_mask = self._draw_line(x1, y1, x2, y2, device)
            targets[i] = line_mask.unsqueeze(0).expand(3, -1, -1)
            
        return targets
    
    def _draw_line(self, x1: int, y1: int, x2: int, y2: int, device: str) -> torch.Tensor:
        """Draw line between two points."""
        mask = torch.zeros((self.canvas_size, self.canvas_size), device=device)
        
        # Simple line drawing
        num_points = max(abs(x2 - x1), abs(y2 - y1), 1)
        for i in range(num_points + 1):
            t = i / num_points if num_points > 0 else 0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                mask[y, x] = 1.0
                
        return mask
    
    def _generate_simple_shapes(self, batch_size: int, device: str) -> torch.Tensor:
        """Generate simple geometric shapes."""
        # Placeholder - would implement rectangles, triangles, etc.
        return self._generate_circles(batch_size, device)
    
    def _generate_sketches(self, batch_size: int, device: str) -> torch.Tensor:
        """Generate sketch-like targets."""
        # Placeholder - would use actual sketch datasets
        return self._generate_lines(batch_size, device)
    
    def _generate_natural_images(self, batch_size: int, device: str) -> torch.Tensor:
        """Generate natural image targets using real datasets."""
        try:
            return self.dataset_manager.get_batch_for_curriculum(
                dataset_name=self.dataset_name,
                batch_size=batch_size,
                level_complexity=1.0
            )
        except Exception as e:
            print(f"Failed to load natural images: {e}")
            print("Using fallback synthetic images...")
            return self._generate_circles(batch_size, device)

class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, max_size: int, state_shapes: Dict[str, Tuple], action_dim: int, device: str):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Initialize buffers
        self.states = {}
        for key, shape in state_shapes.items():
            self.states[key] = torch.zeros((max_size, *shape), device=device)
            
        self.actions = torch.zeros((max_size, action_dim), device=device)
        self.rewards = torch.zeros(max_size, device=device)
        self.log_probs = torch.zeros(max_size, device=device)
        self.values = torch.zeros(max_size, device=device)
        self.dones = torch.zeros(max_size, dtype=torch.bool, device=device)
    
    def store(self, state: Dict[str, torch.Tensor], action: torch.Tensor, 
             reward: torch.Tensor, log_prob: torch.Tensor, value: torch.Tensor, done: torch.Tensor):
        """Store transition in buffer."""
        for key, tensor in state.items():
            self.states[key][self.ptr] = tensor
            
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get random batch from buffer."""
        indices = torch.randint(0, self.size, (batch_size,))
        
        batch_states = {}
        for key, tensor in self.states.items():
            batch_states[key] = tensor[indices]
            
        return {
            'states': batch_states,
            'actions': self.actions[indices],
            'rewards': self.rewards[indices], 
            'log_probs': self.log_probs[indices],
            'values': self.values[indices],
            'dones': self.dones[indices]
        }

class DrawingTrainer:
    """
    Main trainer class for RL drawing agent.
    
    Handles curriculum learning, training loops, and evaluation.
    """
    
    def __init__(
        self,
        config: PPOConfig = None,
        canvas_size: int = 128,
        batch_size: int = 256,
        device: str = 'cuda',
        log_dir: str = 'logs/drawing_rl',
        dataset_name: str = 'cifar10'
    ):
        self.config = config or PPOConfig()
        self.device = device
        self.batch_size = batch_size
        
        # Initialize components
        self.env = DrawingEnvironment(batch_size, canvas_size, device=device)
        self.agent = PPOAgent(canvas_size, self.config, device)
        self.curriculum = CurriculumLearning(canvas_size, dataset_name)
        
        # Training tracking
        self.episode = 0
        self.total_steps = 0
        self.writer = SummaryWriter(log_dir)
        
        # Success tracking for curriculum
        self.success_history = []
        self.episodes_at_level = 0
        
    def train(self, total_episodes: int = 100000, eval_freq: int = 1000, save_freq: int = 5000):
        """Main training loop."""
        print(f"Starting training for {total_episodes} episodes")
        
        while self.episode < total_episodes:
            # Generate curriculum targets
            targets = self.curriculum.generate_target_batch(self.batch_size, self.device)
            
            # Run episode
            episode_rewards, episode_length = self._run_episode(targets)
            
            # Update curriculum
            self._update_curriculum(episode_rewards.mean().item())
            
            # Logging
            if self.episode % 100 == 0:
                avg_reward = episode_rewards.mean().item()
                avg_length = episode_length.float().mean().item()
                
                self.writer.add_scalar('Train/AverageReward', avg_reward, self.episode)
                self.writer.add_scalar('Train/AverageLength', avg_length, self.episode)
                self.writer.add_scalar('Train/CurriculumLevel', self.curriculum.current_level, self.episode)
                
                print(f"Episode {self.episode}: Avg Reward = {avg_reward:.4f}, "
                      f"Avg Length = {avg_length:.1f}, Level = {self.curriculum.current_level}")
            
            # Evaluation
            if self.episode % eval_freq == 0 and self.episode > 0:
                self._evaluate()
            
            # Save checkpoint
            if self.episode % save_freq == 0 and self.episode > 0:
                self._save_checkpoint()
                
            self.episode += 1
            
        print("Training completed!")
        self._save_checkpoint()
    
    def _run_episode(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run single episode with given targets."""
        # Reset environment
        state = self.env.reset(targets)
        
        episode_rewards = torch.zeros(self.batch_size, device=self.device)
        episode_length = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
        
        # Collect rollout data
        rollout_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        # Episode loop
        for step in range(self.curriculum.get_current_level()['max_steps']):
            # Get action from agent
            action, log_prob, value = self.agent.get_action(state)
            
            # Take environment step
            next_state, reward, done, _ = self.env.step(action)
            
            # Store rollout data (detach to avoid gradient issues)
            rollout_data['states'].append({k: v.detach().clone() for k, v in state.items()})
            rollout_data['actions'].append(action.detach().clone())
            rollout_data['rewards'].append(reward.detach().clone())
            rollout_data['log_probs'].append(log_prob.detach().clone())
            rollout_data['values'].append(value.detach().clone())
            rollout_data['dones'].append(done.detach().clone())
            
            # Update episode tracking
            episode_rewards += reward
            episode_length += 1
            
            # Update state
            state = next_state
            
            # Early termination if all episodes done
            if torch.all(done):
                break
        
        # Convert rollout data to tensors and compute advantages
        processed_rollout = self._process_rollout(rollout_data)
        
        # Update agent
        if len(processed_rollout['returns']) > 0:
            train_stats = self.agent.update(processed_rollout)
            
            # Log training stats
            if self.episode % 100 == 0:
                for key, value in train_stats.items():
                    self.writer.add_scalar(f'Train/{key}', value, self.episode)
        
        return episode_rewards, episode_length
    
    def _process_rollout(self, rollout_data: Dict) -> Dict[str, torch.Tensor]:
        """Process rollout data and compute returns and advantages."""
        if not rollout_data['rewards']:
            return {}
            
        # Stack tensors
        rewards = torch.stack(rollout_data['rewards'])  # (time, batch)
        values = torch.stack(rollout_data['values'])    # (time, batch)
        dones = torch.stack(rollout_data['dones'])      # (time, batch)
        
        # Compute returns and advantages using GAE
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # Stack other tensors
        actions = torch.stack(rollout_data['actions'])
        log_probs = torch.stack(rollout_data['log_probs'])
        
        # Flatten batch and time dimensions
        T, B = rewards.shape
        
        # Process states
        states = {}
        for key in rollout_data['states'][0].keys():
            state_tensor = torch.stack([s[key] for s in rollout_data['states']])
            states[key] = state_tensor.view(T * B, *state_tensor.shape[2:])
        
        return {
            'states': states,
            'actions': actions.view(T * B, -1),
            'returns': returns.view(T * B),
            'advantages': advantages.view(T * B),
            'log_probs': log_probs.view(T * B)
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages using Generalized Advantage Estimation."""
        T, B = rewards.shape
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.config.gamma * next_value * (~dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (~dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        return returns, advantages
    
    def _update_curriculum(self, avg_reward: float):
        """Update curriculum based on performance."""
        # Simple success criterion based on reward
        success = avg_reward > -0.5  # Adjust threshold as needed
        
        self.success_history.append(success)
        self.episodes_at_level += 1
        
        # Keep recent history
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-100:]
        
        # Check if should advance curriculum
        if len(self.success_history) >= 50:
            success_rate = sum(self.success_history[-50:]) / 50
            
            if self.curriculum.should_advance(success_rate, self.episodes_at_level):
                self.curriculum.advance_level()
                self.episodes_at_level = 0
                self.success_history = []
    
    def _evaluate(self):
        """Run evaluation episodes."""
        print("Running evaluation...")
        eval_rewards = []
        
        for _ in range(10):  # Run 10 evaluation episodes
            targets = self.curriculum.generate_target_batch(self.batch_size, self.device)
            state = self.env.reset(targets)
            
            episode_reward = torch.zeros(self.batch_size, device=self.device)
            
            for step in range(self.curriculum.get_current_level()['max_steps']):
                action, _, _ = self.agent.get_action(state, deterministic=True)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                
                if torch.all(done):
                    break
            
            eval_rewards.append(episode_reward.mean().item())
        
        avg_eval_reward = np.mean(eval_rewards)
        self.writer.add_scalar('Eval/AverageReward', avg_eval_reward, self.episode)
        print(f"Evaluation: Average reward = {avg_eval_reward:.4f}")
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = f"checkpoints/rl_drawing_ep_{self.episode}.pt"
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.agent.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

def main():
    """Main training function."""
    config = PPOConfig()
    trainer = DrawingTrainer(config)
    trainer.train(total_episodes=100000)

if __name__ == "__main__":
    main()