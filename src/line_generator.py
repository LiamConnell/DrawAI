import numpy as np
import svgwrite
from scipy.interpolate import splprep, splev, interp1d
from typing import List, Tuple, Optional
import math
import random


class LineGenerator:
    def __init__(self):
        self.smoothing_factor = 0.0
        self.squiggle_amplitude = 2.0
        self.squiggle_frequency = 0.1
        self.curve_tension = 0.5
        self.min_segment_length = 5.0
        self.max_segment_length = 50.0
        
    def add_natural_variation(self, points: List[Tuple[int, int]], 
                            variation_strength: float = 1.0) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return [(float(p[0]), float(p[1])) for p in points]
        
        varied_points = []
        
        for i, (x, y) in enumerate(points):
            base_x, base_y = float(x), float(y)
            
            noise_x = (random.random() - 0.5) * self.squiggle_amplitude * variation_strength
            noise_y = (random.random() - 0.5) * self.squiggle_amplitude * variation_strength
            
            if i > 0 and i < len(points) - 1:
                prev_x, prev_y = points[i-1]
                next_x, next_y = points[i+1]
                
                direction_x = next_x - prev_x
                direction_y = next_y - prev_y
                length = math.sqrt(direction_x**2 + direction_y**2)
                
                if length > 0:
                    perp_x = -direction_y / length
                    perp_y = direction_x / length
                    
                    perp_noise = (random.random() - 0.5) * self.squiggle_amplitude * variation_strength
                    noise_x += perp_x * perp_noise
                    noise_y += perp_y * perp_noise
            
            varied_points.append((base_x + noise_x, base_y + noise_y))
        
        return varied_points
    
    def create_smooth_spline(self, points: List[Tuple[float, float]], 
                           num_points: int = 200) -> List[Tuple[float, float]]:
        if len(points) < 3:
            return points
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        try:
            tck, u = splprep([x_coords, y_coords], s=self.smoothing_factor, k=min(3, len(points)-1))
            
            u_new = np.linspace(0, 1, num_points)
            smooth_coords = splev(u_new, tck)
            
            return list(zip(smooth_coords[0], smooth_coords[1]))
        
        except Exception:
            return self.create_bezier_curve(points, num_points)
    
    def create_bezier_curve(self, points: List[Tuple[float, float]], 
                          num_points: int = 200) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return points
        
        if len(points) == 2:
            t_values = np.linspace(0, 1, num_points)
            x_vals = [(1-t) * points[0][0] + t * points[1][0] for t in t_values]
            y_vals = [(1-t) * points[0][1] + t * points[1][1] for t in t_values]
            return list(zip(x_vals, y_vals))
        
        bezier_points = []
        segment_size = max(2, len(points) // 3)
        
        for i in range(0, len(points) - 1, segment_size):
            segment_end = min(i + segment_size + 1, len(points))
            segment = points[i:segment_end]
            
            if len(segment) >= 2:
                t_values = np.linspace(0, 1, num_points // ((len(points) // segment_size) + 1))
                
                for t in t_values:
                    if len(segment) == 2:
                        x = (1-t) * segment[0][0] + t * segment[1][0]
                        y = (1-t) * segment[0][1] + t * segment[1][1]
                    elif len(segment) == 3:
                        x = (1-t)**2 * segment[0][0] + 2*(1-t)*t * segment[1][0] + t**2 * segment[2][0]
                        y = (1-t)**2 * segment[0][1] + 2*(1-t)*t * segment[1][1] + t**2 * segment[2][1]
                    else:
                        n = len(segment) - 1
                        x = sum(self.binomial_coefficient(n, k) * (1-t)**(n-k) * t**k * segment[k][0] 
                               for k in range(n+1))
                        y = sum(self.binomial_coefficient(n, k) * (1-t)**(n-k) * t**k * segment[k][1] 
                               for k in range(n+1))
                    
                    bezier_points.append((x, y))
        
        return bezier_points
    
    def binomial_coefficient(self, n: int, k: int) -> int:
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
    
    def add_artistic_squiggles(self, points: List[Tuple[float, float]], 
                             intensity: float = 1.0) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return points
        
        squiggled_points = []
        
        for i in range(len(points) - 1):
            current_point = points[i]
            next_point = points[i + 1]
            
            distance = math.sqrt((next_point[0] - current_point[0])**2 + 
                               (next_point[1] - current_point[1])**2)
            
            if distance < self.min_segment_length:
                squiggled_points.append(current_point)
                continue
            
            num_subdivisions = max(2, int(distance / 10))
            
            for j in range(num_subdivisions):
                t = j / num_subdivisions
                
                base_x = current_point[0] + t * (next_point[0] - current_point[0])
                base_y = current_point[1] + t * (next_point[1] - current_point[1])
                
                direction_x = next_point[0] - current_point[0]
                direction_y = next_point[1] - current_point[1]
                length = math.sqrt(direction_x**2 + direction_y**2)
                
                if length > 0:
                    perp_x = -direction_y / length
                    perp_y = direction_x / length
                    
                    squiggle_offset = (math.sin(t * distance * self.squiggle_frequency * 2 * math.pi) * 
                                     self.squiggle_amplitude * intensity)
                    
                    noise_factor = (random.random() - 0.5) * 0.5 * intensity
                    total_offset = squiggle_offset + noise_factor
                    
                    squiggled_x = base_x + perp_x * total_offset
                    squiggled_y = base_y + perp_y * total_offset
                    
                    squiggled_points.append((squiggled_x, squiggled_y))
                else:
                    squiggled_points.append((base_x, base_y))
        
        if points:
            squiggled_points.append(points[-1])
        
        return squiggled_points
    
    def generate_line_drawing(self, path_coordinates: List[Tuple[int, int]], 
                            image_size: Tuple[int, int] = (512, 512),
                            artistic_style: str = 'smooth') -> List[Tuple[float, float]]:
        if not path_coordinates:
            return []
        
        varied_points = self.add_natural_variation(path_coordinates, variation_strength=0.8)
        
        if artistic_style == 'smooth':
            smooth_points = self.create_smooth_spline(varied_points, num_points=len(varied_points) * 3)
            final_points = self.add_artistic_squiggles(smooth_points, intensity=0.3)
        
        elif artistic_style == 'sketchy':
            final_points = self.add_artistic_squiggles(varied_points, intensity=1.2)
        
        elif artistic_style == 'minimal':
            final_points = self.create_smooth_spline(varied_points, num_points=len(varied_points) * 2)
        
        else:
            final_points = varied_points
        
        final_points = self.ensure_connected_path(final_points)
        
        return final_points
    
    def ensure_connected_path(self, points: List[Tuple[float, float]], 
                            max_gap: float = 20.0) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return points
        
        connected_points = [points[0]]
        
        for i in range(1, len(points)):
            prev_point = connected_points[-1]
            current_point = points[i]
            
            distance = math.sqrt((current_point[0] - prev_point[0])**2 + 
                               (current_point[1] - prev_point[1])**2)
            
            if distance > max_gap:
                num_interpolations = int(distance / max_gap) + 1
                for j in range(1, num_interpolations):
                    t = j / num_interpolations
                    interp_x = prev_point[0] + t * (current_point[0] - prev_point[0])
                    interp_y = prev_point[1] + t * (current_point[1] - prev_point[1])
                    connected_points.append((interp_x, interp_y))
            
            connected_points.append(current_point)
        
        return connected_points
    
    def save_as_svg(self, points: List[Tuple[float, float]], 
                   filename: str, image_size: Tuple[int, int] = (512, 512),
                   stroke_width: float = 2.0, stroke_color: str = 'black'):
        dwg = svgwrite.Drawing(filename, size=image_size, profile='tiny')
        
        if len(points) < 2:
            dwg.save()
            return
        
        path_data = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
        
        for i in range(1, len(points)):
            if i == 1:
                path_data += f" L {points[i][0]:.2f},{points[i][1]:.2f}"
            else:
                prev_point = points[i-1]
                current_point = points[i]
                
                control_x = (prev_point[0] + current_point[0]) / 2
                control_y = (prev_point[1] + current_point[1]) / 2
                
                path_data += f" Q {control_x:.2f},{control_y:.2f} {current_point[0]:.2f},{current_point[1]:.2f}"
        
        dwg.add(dwg.path(d=path_data, 
                        stroke=stroke_color, 
                        stroke_width=stroke_width, 
                        fill='none',
                        stroke_linecap='round',
                        stroke_linejoin='round'))
        
        dwg.save()
    
    def save_as_points_file(self, points: List[Tuple[float, float]], filename: str):
        with open(filename, 'w') as f:
            f.write("# One-line drawing coordinates\n")
            f.write("# Format: x,y\n")
            for x, y in points:
                f.write(f"{x:.2f},{y:.2f}\n")