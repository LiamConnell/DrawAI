import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Optional, Tuple, List
import random

class ImageDatasetManager:
    """
    Manages real-world image datasets for RL drawing training.
    
    Supports CIFAR-10, CelebA, custom image folders, and provides
    preprocessing pipeline for converting photos to drawing targets.
    """
    
    def __init__(
        self,
        canvas_size: int = 128,
        device: str = 'cuda',
        data_dir: str = 'data'
    ):
        self.canvas_size = canvas_size
        self.device = device
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((canvas_size, canvas_size)),
            transforms.ToTensor(),
        ])
        
        # Available datasets
        self.datasets = {}
        self._initialize_datasets()
        
    def _initialize_datasets(self):
        """Initialize available datasets."""
        
        # CIFAR-10 - good for simple objects
        self.datasets['cifar10'] = {
            'loader': self._get_cifar10_loader,
            'description': 'CIFAR-10: 32x32 objects (animals, vehicles, etc.)',
            'num_classes': 10
        }
        
        # CelebA - faces dataset
        self.datasets['celeba'] = {
            'loader': self._get_celeba_loader,
            'description': 'CelebA: Celebrity faces dataset',
            'num_classes': 1
        }
        
        # Custom image folder
        self.datasets['custom'] = {
            'loader': self._get_custom_loader,
            'description': 'Custom images from folder',
            'num_classes': None
        }
        
        # Quick Web Images - pre-downloaded samples
        self.datasets['quickdraw'] = {
            'loader': self._get_quickdraw_loader,
            'description': 'Quick Draw! sketch dataset',
            'num_classes': 345
        }
    
    def _get_cifar10_loader(self, batch_size: int = 32, subset_size: Optional[int] = None) -> DataLoader:
        """Get CIFAR-10 dataset loader."""
        
        # Download CIFAR-10 if not available
        dataset = datasets.CIFAR10(
            root=self.data_dir / 'cifar10',
            train=True,
            download=True,
            transform=self.base_transform
        )
        
        if subset_size:
            # Create subset for faster training
            indices = torch.randperm(len(dataset))[:subset_size]
            dataset = torch.utils.data.Subset(dataset, indices)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    def _get_celeba_loader(self, batch_size: int = 32, subset_size: Optional[int] = None) -> Optional[DataLoader]:
        """Get CelebA dataset loader."""
        
        celeba_path = self.data_dir / 'celeba'
        
        if not celeba_path.exists():
            print(f"CelebA not found at {celeba_path}. Please download manually.")
            print("Download from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
            return None
        
        try:
            dataset = datasets.CelebA(
                root=self.data_dir,
                split='train',
                download=False,  # Must be downloaded manually
                transform=self.base_transform
            )
            
            if subset_size:
                indices = torch.randperm(len(dataset))[:subset_size]
                dataset = torch.utils.data.Subset(dataset, indices)
                
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            
        except Exception as e:
            print(f"Error loading CelebA: {e}")
            return None
    
    def _get_custom_loader(self, folder_path: str, batch_size: int = 32) -> Optional[DataLoader]:
        """Get custom image folder loader."""
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Custom folder not found: {folder_path}")
            return None
        
        dataset = datasets.ImageFolder(
            root=folder,
            transform=self.base_transform
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    def _get_quickdraw_loader(self, batch_size: int = 32, categories: List[str] = None) -> DataLoader:
        """Get Quick Draw! sketch dataset."""
        
        # Use simplified version - just generate sketch-like targets
        dataset = QuickDrawDataset(
            canvas_size=self.canvas_size,
            categories=categories or ['cat', 'dog', 'bird', 'car'],
            num_samples=1000
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_batch_for_curriculum(
        self,
        dataset_name: str,
        batch_size: int,
        level_complexity: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Get batch of images for curriculum training.
        
        Args:
            dataset_name: Name of dataset to use
            batch_size: Number of images in batch
            level_complexity: Complexity level (0-1) for preprocessing
            **kwargs: Additional arguments for dataset loader
        
        Returns:
            Batch of preprocessed images (batch_size, 3, canvas_size, canvas_size)
        """
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Get data loader
        loader_fn = self.datasets[dataset_name]['loader']
        loader = loader_fn(batch_size=batch_size, **kwargs)
        
        if loader is None:
            # Fallback to synthetic data
            return self._generate_fallback_images(batch_size)
        
        # Get one batch
        try:
            batch = next(iter(loader))
            if isinstance(batch, (list, tuple)):
                images = batch[0]  # Images are first element
            else:
                images = batch
                
            images = images.to(self.device)
            
            # Apply complexity-based preprocessing
            processed_images = self._apply_complexity_preprocessing(images, level_complexity)
            
            return processed_images[:batch_size]  # Ensure exact batch size
            
        except Exception as e:
            print(f"Error loading batch from {dataset_name}: {e}")
            return self._generate_fallback_images(batch_size)
    
    def _apply_complexity_preprocessing(
        self, 
        images: torch.Tensor, 
        complexity: float
    ) -> torch.Tensor:
        """
        Apply preprocessing based on curriculum complexity level.
        
        Lower complexity = more simplified/abstract images
        Higher complexity = more detailed/realistic images
        """
        
        if complexity < 0.3:
            # Low complexity: high blur, edge detection
            images = self._apply_edge_simplification(images)
        elif complexity < 0.6:
            # Medium complexity: moderate blur, some detail
            images = self._apply_moderate_simplification(images)
        elif complexity < 0.8:
            # High complexity: light processing
            images = self._apply_light_simplification(images)
        # else: keep original images (complexity >= 0.8)
        
        return images
    
    def _apply_edge_simplification(self, images: torch.Tensor) -> torch.Tensor:
        """Convert to edge-like representation for low complexity."""
        
        # Convert to grayscale
        gray = torch.mean(images, dim=1, keepdim=True)
        
        # Simple edge detection using gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        
        edges_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
        
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        edges = torch.clamp(edges, 0, 1)
        
        # Convert back to 3 channels
        return edges.expand(-1, 3, -1, -1)
    
    def _apply_moderate_simplification(self, images: torch.Tensor) -> torch.Tensor:
        """Apply moderate simplification."""
        
        # Light Gaussian blur
        from torch.nn.functional import conv2d
        
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 
                             dtype=torch.float32, device=images.device) / 16
        kernel = kernel.view(1, 1, 3, 3).expand(3, 1, -1, -1)
        
        blurred = conv2d(images, kernel, padding=1, groups=3)
        
        # Reduce color saturation
        gray = torch.mean(blurred, dim=1, keepdim=True)
        desaturated = 0.7 * blurred + 0.3 * gray.expand_as(blurred)
        
        return desaturated
    
    def _apply_light_simplification(self, images: torch.Tensor) -> torch.Tensor:
        """Apply light simplification."""
        
        # Just slight contrast adjustment
        return torch.clamp(images * 1.1 - 0.05, 0, 1)
    
    def _generate_fallback_images(self, batch_size: int) -> torch.Tensor:
        """Generate fallback synthetic images if real dataset fails."""
        
        images = torch.zeros((batch_size, 3, self.canvas_size, self.canvas_size), device=self.device)
        
        for i in range(batch_size):
            # Generate random simple shapes as fallback
            if random.random() < 0.5:
                # Circle
                cx, cy = random.randint(20, self.canvas_size-20), random.randint(20, self.canvas_size-20)
                radius = random.randint(10, 30)
                
                y, x = torch.meshgrid(
                    torch.arange(self.canvas_size, device=self.device),
                    torch.arange(self.canvas_size, device=self.device),
                    indexing='ij'
                )
                
                dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                mask = (dist <= radius).float()
                images[i] = mask.unsqueeze(0).expand(3, -1, -1)
            else:
                # Rectangle
                x1, y1 = random.randint(10, self.canvas_size//2), random.randint(10, self.canvas_size//2)
                x2, y2 = random.randint(self.canvas_size//2, self.canvas_size-10), random.randint(self.canvas_size//2, self.canvas_size-10)
                
                images[i, :, y1:y2, x1:x2] = 1.0
        
        return images
    
    def list_available_datasets(self):
        """Print available datasets."""
        print("Available datasets:")
        for name, info in self.datasets.items():
            print(f"  {name}: {info['description']}")

class QuickDrawDataset(Dataset):
    """
    Simplified Quick Draw!-style dataset for sketch training.
    
    Generates simple line drawings of common objects.
    """
    
    def __init__(
        self,
        canvas_size: int = 128,
        categories: List[str] = None,
        num_samples: int = 1000
    ):
        self.canvas_size = canvas_size
        self.categories = categories or ['circle', 'square', 'triangle', 'star']
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate simple sketch-like image
        image = torch.zeros((3, self.canvas_size, self.canvas_size))
        
        category = self.categories[idx % len(self.categories)]
        
        if category == 'circle':
            image = self._draw_circle(image)
        elif category == 'square':
            image = self._draw_square(image)
        elif category == 'triangle':
            image = self._draw_triangle(image)
        elif category == 'star':
            image = self._draw_star(image)
        
        return image, 0  # Return dummy label
    
    def _draw_circle(self, image):
        cx, cy = self.canvas_size // 2, self.canvas_size // 2
        radius = random.randint(20, self.canvas_size // 3)
        
        # Draw circle outline
        for angle in np.linspace(0, 2 * np.pi, 100):
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                image[:, y, x] = 1.0
        
        return image
    
    def _draw_square(self, image):
        size = random.randint(40, self.canvas_size // 2)
        x1 = (self.canvas_size - size) // 2
        y1 = (self.canvas_size - size) // 2
        x2, y2 = x1 + size, y1 + size
        
        # Draw square outline
        image[:, y1:y2, x1] = 1.0  # Left edge
        image[:, y1:y2, x2-1] = 1.0  # Right edge
        image[:, y1, x1:x2] = 1.0  # Top edge
        image[:, y2-1, x1:x2] = 1.0  # Bottom edge
        
        return image
    
    def _draw_triangle(self, image):
        cx, cy = self.canvas_size // 2, self.canvas_size // 2
        size = random.randint(30, self.canvas_size // 3)
        
        # Triangle vertices
        vertices = [
            (cx, cy - size),           # Top
            (cx - size, cy + size//2), # Bottom left
            (cx + size, cy + size//2)  # Bottom right
        ]
        
        # Draw triangle outline
        for i in range(3):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 3]
            
            # Simple line drawing
            steps = max(abs(x2 - x1), abs(y2 - y1))
            for j in range(steps + 1):
                if steps > 0:
                    t = j / steps
                    x = int(x1 + t * (x2 - x1))
                    y = int(y1 + t * (y2 - y1))
                    if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                        image[:, y, x] = 1.0
        
        return image
    
    def _draw_star(self, image):
        cx, cy = self.canvas_size // 2, self.canvas_size // 2
        outer_radius = random.randint(25, self.canvas_size // 3)
        inner_radius = outer_radius // 2
        
        # 5-pointed star
        vertices = []
        for i in range(10):
            angle = i * np.pi / 5
            if i % 2 == 0:
                # Outer vertex
                x = cx + outer_radius * np.cos(angle - np.pi/2)
                y = cy + outer_radius * np.sin(angle - np.pi/2)
            else:
                # Inner vertex
                x = cx + inner_radius * np.cos(angle - np.pi/2)
                y = cy + inner_radius * np.sin(angle - np.pi/2)
            vertices.append((int(x), int(y)))
        
        # Draw star outline
        for i in range(10):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 10]
            
            steps = max(abs(x2 - x1), abs(y2 - y1))
            for j in range(steps + 1):
                if steps > 0:
                    t = j / steps
                    x = int(x1 + t * (x2 - x1))
                    y = int(y1 + t * (y2 - y1))
                    if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                        image[:, y, x] = 1.0
        
        return image

def main():
    """Test dataset functionality."""
    
    manager = ImageDatasetManager(canvas_size=128)
    
    print("Testing dataset manager...")
    manager.list_available_datasets()
    
    # Test CIFAR-10
    print("\nTesting CIFAR-10...")
    try:
        batch = manager.get_batch_for_curriculum('cifar10', batch_size=4, subset_size=100)
        print(f"CIFAR-10 batch shape: {batch.shape}")
    except Exception as e:
        print(f"CIFAR-10 test failed: {e}")
    
    # Test QuickDraw
    print("\nTesting QuickDraw...")
    try:
        batch = manager.get_batch_for_curriculum('quickdraw', batch_size=4)
        print(f"QuickDraw batch shape: {batch.shape}")
    except Exception as e:
        print(f"QuickDraw test failed: {e}")

if __name__ == "__main__":
    main()