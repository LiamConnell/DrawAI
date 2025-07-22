#!/usr/bin/env python3
"""
Real training with CIFAR-10 images
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from rl.trainer import DrawingTrainer
from rl.agent import PPOConfig

def train_with_real_images():
    """Train with real CIFAR-10 images."""
    print("Training with real CIFAR-10 images...")
    
    config = PPOConfig(
        lr=5e-4,
        clip_epsilon=0.2,
        entropy_coef=0.02
    )
    
    trainer = DrawingTrainer(
        config=config,
        canvas_size=64,    # Good size for CIFAR-10
        batch_size=16,     # Smaller batch for memory
        device='cpu',      # Use CPU for compatibility
        log_dir='logs/real_training',
        dataset_name='cifar10'  # Use real CIFAR-10 images
    )
    
    print("Starting real image training (50 episodes)...")
    trainer.train(
        total_episodes=50,
        eval_freq=20,
        save_freq=25
    )
    
    print("Real image training completed!")

if __name__ == "__main__":
    train_with_real_images()