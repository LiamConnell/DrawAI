#!/usr/bin/env python3
"""
RL Drawing Training Demo

Example script showing how to train the RL drawing agent
with different configurations and curriculum levels.
"""

import torch
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rl.trainer import DrawingTrainer, CurriculumLearning
from rl.agent import PPOConfig
from rl.environment import DrawingEnvironment

def train_quick_demo():
    """Quick training demo with small parameters for testing."""
    print("Running quick training demo...")
    
    config = PPOConfig(
        lr=5e-4,
        clip_epsilon=0.2,
        entropy_coef=0.02
    )
    
    trainer = DrawingTrainer(
        config=config,
        canvas_size=64,  # Smaller for faster training
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='logs/quick_demo',
        dataset_name='cifar10'  # Use CIFAR-10 for real images
    )
    
    trainer.train(
        total_episodes=1000,
        eval_freq=200,
        save_freq=500
    )

def train_full_scale():
    """Full-scale training with production parameters."""
    print("Running full-scale training...")
    
    config = PPOConfig(
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01
    )
    
    trainer = DrawingTrainer(
        config=config,
        canvas_size=128,
        batch_size=256,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='logs/full_scale',
        dataset_name='cifar10'  # Use CIFAR-10 for real images
    )
    
    trainer.train(
        total_episodes=100000,
        eval_freq=1000,
        save_freq=5000
    )

def test_environment():
    """Test the drawing environment with random actions."""
    print("Testing drawing environment...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = DrawingEnvironment(batch_size=4, canvas_size=64, device=device)
    
    # Generate simple targets (circles)
    curriculum = CurriculumLearning(64)
    targets = curriculum.generate_target_batch(4, device)
    
    # Reset environment
    state = env.reset(targets)
    
    print(f"Initial state shapes:")
    for key, tensor in state.items():
        print(f"  {key}: {tensor.shape}")
    
    # Take random actions
    total_rewards = torch.zeros(4, device=device)
    
    for step in range(10):
        # Random actions: [delta_x, delta_y, pressure, pen_up_down]
        actions = torch.randn(4, 4, device=device) * 0.1
        
        state, rewards, done, _ = env.step(actions)
        total_rewards += rewards
        
        print(f"Step {step}: rewards = {rewards.detach().cpu().numpy()}")
        
        if torch.all(done):
            break
    
    print(f"Final total rewards: {total_rewards.detach().cpu().numpy()}")

def visualize_training_progress():
    """Visualize training progress with sample drawings."""
    print("Visualizing training progress...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load trained model if available
    checkpoint_path = "checkpoints/rl_drawing_ep_5000.pt"
    
    if Path(checkpoint_path).exists():
        from rl.agent import PPOAgent
        
        agent = PPOAgent(canvas_size=128, device=device)
        agent.load(checkpoint_path)
        
        # Create environment
        env = DrawingEnvironment(batch_size=1, canvas_size=128, device=device)
        curriculum = CurriculumLearning(128)
        
        # Generate target
        target = curriculum.generate_target_batch(1, device)
        state = env.reset(target)
        
        print("Generating sample drawing...")
        
        for step in range(500):
            action, _, _ = agent.get_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            
            if done or step % 100 == 0:
                print(f"Step {step}: reward = {reward.item():.4f}")
            
            if done:
                break
        
        # Save final canvas and target
        import torchvision.transforms as transforms
        from PIL import Image
        
        canvas = state['canvas'][0].cpu()
        target_img = target[0].cpu()
        
        to_pil = transforms.ToPILImage()
        
        canvas_pil = to_pil(canvas)
        target_pil = to_pil(target_img)
        
        canvas_pil.save('sample_drawing.png')
        target_pil.save('sample_target.png')
        
        print("Saved sample_drawing.png and sample_target.png")
        print("\nTo test photo-to-sketch with your own images:")
        print("python examples/photo_to_sketch.py your_photo.jpg")
        
    else:
        print(f"No trained model found at {checkpoint_path}")
        print("Run training first to generate visualizations")

def main():
    parser = argparse.ArgumentParser(description='RL Drawing Training Demo')
    parser.add_argument('--mode', choices=['quick', 'full', 'test', 'visualize'], 
                      default='quick', help='Demo mode to run')
    parser.add_argument('--device', choices=['cpu', 'cuda'], 
                      default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    print(f"Using device: {device}")
    
    if args.mode == 'quick':
        train_quick_demo()
    elif args.mode == 'full':
        train_full_scale() 
    elif args.mode == 'test':
        test_environment()
    elif args.mode == 'visualize':
        visualize_training_progress()

if __name__ == "__main__":
    main()