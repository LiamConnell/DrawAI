import argparse
import sys
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from image_analyzer import ImageAnalyzer
from path_planner import PathPlanner
from line_generator import LineGenerator


class DrawAI:
    def __init__(self):
        self.analyzer = ImageAnalyzer()
        self.planner = PathPlanner()
        self.generator = LineGenerator()
        
    def create_one_line_drawing(self, 
                               image_path: str, 
                               output_path: str,
                               style: str = 'smooth',
                               planning_method: str = 'genetic',
                               stroke_width: float = 2.0,
                               stroke_color: str = 'black',
                               max_features: int = 150,
                               show_analysis: bool = False) -> bool:
        try:
            print(f"Analyzing image: {image_path}")
            
            self.analyzer.max_features = max_features
            processed_image, key_points = self.analyzer.analyze_image(image_path)
            
            if not key_points:
                print("No key points found in the image. Try adjusting the analysis parameters.")
                return False
            
            print(f"Found {len(key_points)} key points")
            
            if show_analysis:
                analysis_path = output_path.replace('.svg', '_analysis.png')
                self.analyzer.visualize_analysis(processed_image, key_points, analysis_path)
                print(f"Analysis visualization saved to: {analysis_path}")
            
            print(f"Planning optimal path using {planning_method} method...")
            optimal_path = self.planner.plan_path(key_points, method=planning_method)
            path_coordinates = self.planner.get_path_coordinates(key_points, optimal_path)
            
            print(f"Generating artistic line drawing with {style} style...")
            line_points = self.generator.generate_line_drawing(
                path_coordinates, 
                artistic_style=style
            )
            
            print(f"Saving SVG to: {output_path}")
            self.generator.save_as_svg(
                line_points, 
                output_path, 
                stroke_width=stroke_width,
                stroke_color=stroke_color
            )
            
            points_path = output_path.replace('.svg', '_points.txt')
            self.generator.save_as_points_file(line_points, points_path)
            
            print(f"One-line drawing completed successfully!")
            print(f"Files created:")
            print(f"  - SVG drawing: {output_path}")
            print(f"  - Point coordinates: {points_path}")
            
            return True
            
        except Exception as e:
            print(f"Error creating one-line drawing: {str(e)}")
            return False
    
    def batch_process(self, input_dir: str, output_dir: str, **kwargs) -> int:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
        
        if not image_files:
            print(f"No supported image files found in {input_dir}")
            return 0
        
        successful = 0
        
        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            output_name = os.path.splitext(image_file)[0] + '_line_drawing.svg'
            output_path = os.path.join(output_dir, output_name)
            
            print(f"\nProcessing {image_file}...")
            
            if self.create_one_line_drawing(image_path, output_path, **kwargs):
                successful += 1
            else:
                print(f"Failed to process {image_file}")
        
        print(f"\nBatch processing complete: {successful}/{len(image_files)} images processed successfully")
        return successful


def main():
    parser = argparse.ArgumentParser(
        description="DrawAI: Convert images to artistic one-line drawings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python draw_ai.py input.jpg output.svg
  python draw_ai.py input.png output.svg --style sketchy --method nearest_neighbor
  python draw_ai.py input.jpg output.svg --show-analysis --max-features 200
  python draw_ai.py --batch input_folder output_folder --style smooth
        """
    )
    
    parser.add_argument('input', help='Input image path or directory (for batch mode)')
    parser.add_argument('output', help='Output SVG path or directory (for batch mode)')
    
    parser.add_argument('--style', choices=['smooth', 'sketchy', 'minimal'], 
                       default='smooth', help='Artistic style for the line drawing')
    
    parser.add_argument('--method', choices=['genetic', 'nearest_neighbor'], 
                       default='genetic', help='Path planning algorithm')
    
    parser.add_argument('--stroke-width', type=float, default=2.0,
                       help='Line thickness in SVG output')
    
    parser.add_argument('--stroke-color', default='black',
                       help='Line color for SVG output')
    
    parser.add_argument('--max-features', type=int, default=150,
                       help='Maximum number of key points to extract')
    
    parser.add_argument('--show-analysis', action='store_true',
                       help='Save visualization of image analysis')
    
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in input directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    draw_ai = DrawAI()
    
    if args.batch:
        if not os.path.isdir(args.input):
            print("Error: Batch mode requires input to be a directory")
            sys.exit(1)
        
        successful = draw_ai.batch_process(
            args.input, 
            args.output,
            style=args.style,
            planning_method=args.method,
            stroke_width=args.stroke_width,
            stroke_color=args.stroke_color,
            max_features=args.max_features,
            show_analysis=args.show_analysis
        )
        
        sys.exit(0 if successful > 0 else 1)
    
    else:
        if os.path.isdir(args.input):
            print("Error: Single file mode requires input to be a file, not directory")
            print("Use --batch flag for directory processing")
            sys.exit(1)
        
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        success = draw_ai.create_one_line_drawing(
            args.input,
            args.output,
            style=args.style,
            planning_method=args.method,
            stroke_width=args.stroke_width,
            stroke_color=args.stroke_color,
            max_features=args.max_features,
            show_analysis=args.show_analysis
        )
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()