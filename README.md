# DrawAI

AI-powered one-line drawing generator that converts reference images into artistic continuous line drawings.

## Overview

DrawAI analyzes input images and generates beautiful one-line drawings that capture the essence of the subject using a single continuous stroke. Inspired by artists like Katie Acheson Wolford, the tool combines computer vision, path optimization, and artistic algorithms to create minimalist line art.

## Features

- **Image Analysis**: Extracts key visual features and important points from any input image
- **Path Optimization**: Uses genetic algorithms and TSP solving to find optimal drawing paths
- **Artistic Styles**: Multiple rendering styles (smooth, sketchy, minimal)
- **SVG Output**: Scalable vector graphics with customizable stroke properties
- **Batch Processing**: Process multiple images at once
- **Analysis Visualization**: See how the AI interprets your images

## Installation

```bash
git clone https://github.com/LiamConnell/DrawAI.git
cd DrawAI
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Generate a smooth one-line drawing
python src/draw_ai.py input.jpg output.svg

# Try different artistic styles
python src/draw_ai.py input.jpg output.svg --style sketchy
python src/draw_ai.py input.jpg output.svg --style minimal

# Show analysis visualization
python src/draw_ai.py input.jpg output.svg --show-analysis
```

### Advanced Options

```bash
# Customize the drawing
python src/draw_ai.py input.jpg output.svg \
  --style smooth \
  --method genetic \
  --stroke-width 2.5 \
  --stroke-color blue \
  --max-features 200

# Batch process multiple images
python src/draw_ai.py input_folder/ output_folder/ --batch --style sketchy
```

### Run Demo

```bash
cd examples
python demo.py
```

## How It Works

1. **Image Analysis**: 
   - Edge detection and contour extraction
   - Saliency mapping to identify important regions
   - Key point extraction with importance scoring

2. **Path Planning**:
   - Treats key points as nodes in a graph
   - Solves modified Traveling Salesman Problem
   - Balances drawing efficiency with artistic appeal

3. **Line Generation**:
   - Converts optimal path into smooth curves
   - Adds artistic variation and squiggles
   - Maintains single continuous line constraint

## Artistic Styles

- **smooth**: Clean curves with subtle artistic variation
- **sketchy**: Hand-drawn appearance with more squiggles
- **minimal**: Simplified lines focusing on essential features

## Path Planning Methods

- **genetic**: Evolutionary algorithm for global optimization (recommended)
- **nearest_neighbor**: Fast greedy approach with local optimization

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- SciPy
- scikit-image
- matplotlib
- Pillow
- NetworkX
- svgwrite

## Examples

The `examples/` directory contains demo scripts and sample outputs showing different artistic styles and techniques.

## Contributing

This project combines computer vision, optimization algorithms, and artistic AI. Contributions welcome in areas like:

- New artistic style algorithms
- Improved path optimization methods
- Better feature extraction techniques
- UI/web interface development

## License

MIT License