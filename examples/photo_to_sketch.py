#!/usr/bin/env python3
"""
Photo to Sketch Evaluation Script

Takes arbitrary photos and generates sketches using trained RL agent.
Demonstrates the end-to-end photo-to-sketch capability.
"""

import torch
import argparse
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rl.agent import PPOAgent
from rl.environment import DrawingEnvironment
from rl.datasets import ImageDatasetManager

def load_and_preprocess_image(image_path: str, canvas_size: int = 128) -> torch.Tensor:
    """
    Load and preprocess an image for sketching.
    
    Args:
        image_path: Path to image file
        canvas_size: Target canvas size
        
    Returns:
        Preprocessed image tensor (1, 3, canvas_size, canvas_size)
    """
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((canvas_size, canvas_size)),
        transforms.ToTensor(),
        # Optional: convert to edges or simplify
        # transforms.Lambda(lambda x: apply_edge_detection(x))
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor

def apply_edge_detection(image_tensor: torch.Tensor) -> torch.Tensor:
    """Apply edge detection preprocessing."""
    
    # Convert to grayscale
    gray = torch.mean(image_tensor, dim=0, keepdim=True)
    
    # Simple edge detection using gradient
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    edges_x = torch.nn.functional.conv2d(gray.unsqueeze(0), sobel_x, padding=1)
    edges_y = torch.nn.functional.conv2d(gray.unsqueeze(0), sobel_y, padding=1)
    
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    edges = torch.clamp(edges, 0, 1).squeeze(0)
    
    # Convert back to 3 channels
    return edges.expand(3, -1, -1)

def generate_sketch(
    agent: PPOAgent,
    target_image: torch.Tensor,
    max_steps: int = 1000,
    device: str = 'cuda',
    deterministic: bool = True
) -> tuple[torch.Tensor, list]:
    """
    Generate sketch from target image using trained agent.
    
    Args:
        agent: Trained RL agent
        target_image: Target image tensor
        max_steps: Maximum drawing steps
        device: Device to run on
        deterministic: Use deterministic policy
        
    Returns:
        Final canvas tensor and list of intermediate canvases
    """
    
    # Create single-image environment
    env = DrawingEnvironment(batch_size=1, canvas_size=target_image.shape[-1], device=device)
    
    # Reset with target
    target_image = target_image.to(device)
    state = env.reset(target_image)
    
    # Track drawing progress
    canvas_history = [state['canvas'][0].cpu().clone()]
    rewards = []
    
    # Drawing loop
    for step in range(max_steps):
        # Get action from agent
        action, log_prob, value = agent.get_action(state, deterministic=deterministic)
        
        # Take step
        state, reward, done, info = env.step(action)
        
        # Track progress
        rewards.append(reward[0].item())
        
        # Save canvas every few steps
        if step % 50 == 0 or done[0]:
            canvas_history.append(state['canvas'][0].cpu().clone())
        
        # Early stopping
        if done[0]:
            break
    
    final_canvas = state['canvas'][0].cpu()
    
    return final_canvas, canvas_history, rewards

def save_comparison_image(
    original: torch.Tensor,
    sketch: torch.Tensor,
    output_path: str,
    canvas_history: list = None
):
    """Save comparison between original and sketch."""
    
    # Convert tensors to PIL images
    to_pil = transforms.ToPILImage()
    
    if canvas_history and len(canvas_history) > 1:
        # Create progression figure
        n_steps = min(6, len(canvas_history))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(to_pil(original.squeeze(0)))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Final sketch
        axes[0, 1].imshow(to_pil(sketch), cmap='gray')
        axes[0, 1].set_title('Final Sketch')
        axes[0, 1].axis('off')
        
        # Overlay
        orig_gray = transforms.Grayscale()(original.squeeze(0))
        overlay = torch.cat([sketch.mean(0, keepdim=True), orig_gray * 0.3, orig_gray * 0.3], dim=0)
        axes[0, 2].imshow(to_pil(overlay))
        axes[0, 2].set_title('Overlay')
        axes[0, 2].axis('off')
        
        # Drawing progression
        step_indices = np.linspace(0, len(canvas_history)-1, 3, dtype=int)
        for i, idx in enumerate(step_indices):
            axes[1, i].imshow(to_pil(canvas_history[idx]), cmap='gray')
            axes[1, i].set_title(f'Step {idx * 50}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    else:
        # Simple side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(to_pil(original.squeeze(0)))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(to_pil(sketch), cmap='gray')
        axes[1].set_title('Sketch')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate sketch from photo using RL agent')
    parser.add_argument('input', help='Path to input image')
    parser.add_argument('--model', help='Path to trained model checkpoint')
    parser.add_argument('--output', help='Output path for sketch comparison')
    parser.add_argument('--canvas-size', type=int, default=128, help='Canvas size')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum drawing steps')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='auto')
    parser.add_argument('--edge-mode', action='store_true', help='Apply edge detection preprocessing')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load image
    if not Path(args.input).exists():
        print(f"Error: Input image not found: {args.input}")
        return
    
    print(f"Loading image: {args.input}")
    target_image = load_and_preprocess_image(args.input, args.canvas_size)
    
    if args.edge_mode:
        print("Applying edge detection preprocessing...")
        target_image = apply_edge_detection(target_image[0]).unsqueeze(0)
    
    # Load trained model
    model_path = args.model or "checkpoints/rl_drawing_latest.pt"
    
    if not Path(model_path).exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        print("Train a model first using: python examples/rl_training_demo.py")
        return
    
    print(f"Loading model: {model_path}")
    agent = PPOAgent(canvas_size=args.canvas_size, device=device)
    
    try:
        agent.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate sketch
    print("Generating sketch...")
    sketch, canvas_history, rewards = generate_sketch(
        agent=agent,
        target_image=target_image,
        max_steps=args.max_steps,
        device=device,
        deterministic=args.deterministic
    )
    
    print(f"Sketch completed in {len(rewards)} steps")
    print(f"Final reward: {rewards[-1]:.4f}")
    print(f"Average reward: {np.mean(rewards):.4f}")
    
    # Save results
    output_path = args.output or f"sketch_{Path(args.input).stem}.png"
    save_comparison_image(target_image, sketch, output_path, canvas_history)
    print(f"Results saved to: {output_path}")
    
    # Save individual files
    sketch_path = f"sketch_only_{Path(args.input).stem}.png"
    transforms.ToPILImage()(sketch).save(sketch_path)
    print(f"Sketch saved to: {sketch_path}")

def test_with_sample_images():
    """Test with built-in sample images."""
    
    print("Testing photo-to-sketch with sample images...")
    
    # Create sample images if they don't exist
    samples_dir = Path("sample_images")
    samples_dir.mkdir(exist_ok=True)
    
    # Generate simple test images
    canvas_size = 128
    
    # Circle image
    circle_img = torch.zeros(3, canvas_size, canvas_size)
    center = canvas_size // 2
    radius = 30
    y, x = torch.meshgrid(torch.arange(canvas_size), torch.arange(canvas_size), indexing='ij')
    dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
    circle_mask = (dist <= radius).float()
    circle_img = circle_mask.unsqueeze(0).expand(3, -1, -1)
    
    circle_path = samples_dir / "circle.png"
    transforms.ToPILImage()(circle_img).save(circle_path)
    
    # Test with circle
    print(f"Testing with synthetic circle: {circle_path}")
    
    # Note: This would need a trained model to work
    print("Note: You need to train a model first before running photo-to-sketch")
    print("Run: python examples/rl_training_demo.py --mode quick")
    print("Then: python examples/photo_to_sketch.py sample_images/circle.png")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Run test mode if no arguments
        test_with_sample_images()
    else:
        main()