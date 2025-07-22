#!/usr/bin/env python3
"""
Test photo-to-sketch with a simple generated image
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from rl.agent import PPOAgent
from rl.environment import DrawingEnvironment

def create_test_image():
    """Create a simple test image."""
    canvas_size = 32
    
    # Create simple circle image
    image = torch.zeros(3, canvas_size, canvas_size)
    center = canvas_size // 2
    radius = canvas_size // 4
    
    y, x = torch.meshgrid(torch.arange(canvas_size), torch.arange(canvas_size), indexing='ij')
    dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
    circle_mask = (dist <= radius).float()
    
    # Make it white circle on black background
    image = circle_mask.unsqueeze(0).expand(3, -1, -1)
    
    # Save for reference
    transforms.ToPILImage()(image).save('test_target.png')
    print("Created test target image: test_target.png")
    
    return image.unsqueeze(0)  # Add batch dimension

def test_sketch_generation():
    """Test sketch generation from the trained model."""
    print("Testing photo-to-sketch...")
    
    # Create test target
    target_image = create_test_image()
    
    # Load trained model
    model_path = "checkpoints/rl_drawing_ep_5.pt"
    if not Path(model_path).exists():
        print(f"No trained model found at {model_path}")
        print("Run test_training.py first!")
        return
    
    # Load agent
    agent = PPOAgent(canvas_size=32, device='cpu')
    agent.load(model_path)
    print("Loaded trained agent")
    
    # Create environment
    env = DrawingEnvironment(batch_size=1, canvas_size=32, device='cpu')
    state = env.reset(target_image)
    
    print("Generating sketch...")
    
    # Generate sketch
    for step in range(100):  # Max 100 steps
        action, _, _ = agent.get_action(state, deterministic=True)
        state, reward, done, _ = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}: reward = {reward[0].item():.4f}")
        
        if done[0]:
            break
    
    # Save result
    final_canvas = state['canvas'][0].cpu()
    transforms.ToPILImage()(final_canvas).save('test_sketch.png')
    
    print(f"Sketch completed in {step + 1} steps")
    print(f"Final reward: {reward[0].item():.4f}")
    print("Sketch saved as: test_sketch.png")
    
    # Create comparison
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(transforms.ToPILImage()(target_image[0]))
    axes[0].set_title('Target')
    axes[0].axis('off')
    
    axes[1].imshow(transforms.ToPILImage()(final_canvas), cmap='gray')
    axes[1].set_title('Generated Sketch')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_comparison.png', dpi=150)
    print("Comparison saved as: test_comparison.png")

if __name__ == "__main__":
    test_sketch_generation()