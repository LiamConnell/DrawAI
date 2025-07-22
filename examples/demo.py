#!/usr/bin/env python3

import sys
import os
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from draw_ai import DrawAI


def create_simple_test_image(filename: str = "test_face.png", size: Tuple[int, int] = (300, 300)):
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size[0] // 2, size[1] // 2
    
    face_radius = size[0] // 3
    draw.ellipse([center_x - face_radius, center_y - face_radius, 
                  center_x + face_radius, center_y + face_radius], 
                 outline='black', width=3)
    
    eye_radius = face_radius // 8
    eye_offset_x = face_radius // 3
    eye_offset_y = face_radius // 4
    
    draw.ellipse([center_x - eye_offset_x - eye_radius, center_y - eye_offset_y - eye_radius,
                  center_x - eye_offset_x + eye_radius, center_y - eye_offset_y + eye_radius],
                 outline='black', width=2)
    
    draw.ellipse([center_x + eye_offset_x - eye_radius, center_y - eye_offset_y - eye_radius,
                  center_x + eye_offset_x + eye_radius, center_y - eye_offset_y + eye_radius],
                 outline='black', width=2)
    
    mouth_width = face_radius // 2
    mouth_height = face_radius // 4
    mouth_y = center_y + face_radius // 3
    
    draw.arc([center_x - mouth_width, mouth_y - mouth_height,
              center_x + mouth_width, mouth_y + mouth_height],
             start=0, end=180, fill='black', width=2)
    
    img.save(filename)
    print(f"Created test image: {filename}")


def run_demo():
    print("DrawAI Demo - Creating One-Line Drawings")
    print("=" * 50)
    
    examples_dir = os.path.dirname(__file__)
    test_image_path = os.path.join(examples_dir, "test_face.png")
    
    if not os.path.exists(test_image_path):
        print("Creating simple test image...")
        create_simple_test_image(test_image_path)
    
    draw_ai = DrawAI()
    
    styles = ['smooth', 'sketchy', 'minimal']
    methods = ['genetic', 'nearest_neighbor']
    
    for style in styles:
        for method in methods:
            output_name = f"demo_{style}_{method}.svg"
            output_path = os.path.join(examples_dir, output_name)
            
            print(f"\nGenerating {style} style with {method} method...")
            
            success = draw_ai.create_one_line_drawing(
                test_image_path,
                output_path,
                style=style,
                planning_method=method,
                max_features=100,
                show_analysis=False,
                stroke_width=1.5
            )
            
            if success:
                print(f"✓ Created: {output_name}")
            else:
                print(f"✗ Failed to create: {output_name}")
    
    print(f"\nDemo complete! Check the {examples_dir} directory for output files.")
    print("\nTo use DrawAI with your own images:")
    print("python src/draw_ai.py your_image.jpg output.svg")
    print("python src/draw_ai.py your_image.jpg output.svg --style sketchy --show-analysis")


if __name__ == "__main__":
    run_demo()