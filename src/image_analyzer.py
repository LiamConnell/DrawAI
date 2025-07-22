import cv2
import numpy as np
from skimage import feature, filters, measure
from scipy import ndimage
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class ImageAnalyzer:
    def __init__(self):
        self.edge_threshold_low = 50
        self.edge_threshold_high = 150
        self.min_contour_area = 100
        self.max_features = 200
    
    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(target_size[0] / width, target_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        pad_x = (target_size[0] - new_width) // 2
        pad_y = (target_size[1] - new_height) // 2
        
        padded = np.pad(resized, ((pad_y, target_size[1] - new_height - pad_y),
                                 (pad_x, target_size[0] - new_width - pad_x),
                                 (0, 0)), mode='constant', constant_values=255)
        
        return padded
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, self.edge_threshold_low, self.edge_threshold_high)
        return edges
    
    def extract_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                filtered_contours.append(approx.reshape(-1, 2))
        
        return filtered_contours
    
    def calculate_saliency(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(gray)
            
            if success:
                saliency_map = (saliency_map * 255).astype(np.uint8)
                return saliency_map
        except AttributeError:
            pass
        
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        saliency_map = np.sqrt(gradient_x**2 + gradient_y**2)
        saliency_map = (saliency_map * 255 / saliency_map.max()).astype(np.uint8)
        
        return saliency_map
    
    def extract_key_points(self, image: np.ndarray, saliency_map: np.ndarray, 
                          contours: List[np.ndarray]) -> List[Tuple[int, int, float]]:
        key_points = []
        
        for contour in contours:
            for point in contour[::max(1, len(contour) // 10)]:
                x, y = point
                if 0 <= x < saliency_map.shape[1] and 0 <= y < saliency_map.shape[0]:
                    importance = float(saliency_map[y, x]) / 255.0
                    key_points.append((x, y, importance))
        
        corners = cv2.goodFeaturesToTrack(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
            maxCorners=50,
            qualityLevel=0.01,
            minDistance=20
        )
        
        if corners is not None:
            for corner in corners:
                x, y = int(corner[0][0]), int(corner[0][1])
                if 0 <= x < saliency_map.shape[1] and 0 <= y < saliency_map.shape[0]:
                    importance = float(saliency_map[y, x]) / 255.0 + 0.3
                    key_points.append((x, y, importance))
        
        key_points.sort(key=lambda p: p[2], reverse=True)
        return key_points[:self.max_features]
    
    def analyze_image(self, image_path: str) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        image = self.load_image(image_path)
        processed_image = self.preprocess_image(image)
        
        edges = self.detect_edges(processed_image)
        contours = self.extract_contours(edges)
        saliency_map = self.calculate_saliency(processed_image)
        key_points = self.extract_key_points(processed_image, saliency_map, contours)
        
        return processed_image, key_points
    
    def visualize_analysis(self, image: np.ndarray, key_points: List[Tuple[int, int, float]], 
                          save_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(image)
        for x, y, importance in key_points:
            color = plt.cm.hot(importance)
            size = 20 + importance * 30
            axes[1].scatter(x, y, c=[color], s=size, alpha=0.7)
        
        axes[1].set_title('Key Points (colored by importance)')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()