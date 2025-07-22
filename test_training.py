#!/usr/bin/env python3
"""
Simple training test to validate the RL pipeline works
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from rl.trainer import DrawingTrainer
from rl.agent import PPOConfig

def test_minimal_training():
    """Test minimal training run."""
    print("Testing minimal RL training pipeline...")
    
    config = PPOConfig(
        lr=1e-3,
        clip_epsilon=0.3,
        entropy_coef=0.05
    )
    
    trainer = DrawingTrainer(
        config=config,
        canvas_size=32,  # Very small for speed
        batch_size=4,    # Very small batch
        device='cpu',    # CPU only for simplicity
        log_dir='logs/test',
        dataset_name='quickdraw'  # Use simple synthetic data
    )
    
    print("Starting minimal training (5 episodes)...")
    trainer.train(
        total_episodes=5,
        eval_freq=10,    # No eval
        save_freq=10     # No saving
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    test_minimal_training()