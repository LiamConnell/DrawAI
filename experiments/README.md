# DrawAI Experiments

This directory contains experiments and test results demonstrating the capabilities of the DrawAI one-line drawing generator.

## Experiments

### [Portrait Test](portrait_test/)
Tests the tool's ability to convert photographic portraits into artistic one-line drawings using different styles (smooth, sketchy, minimal).

**Key Results:**
- Successfully extracted 144 key points from 400x400 portrait
- Generated three distinct artistic styles
- Maintained perfect single-line continuity
- Preserved subject recognizability across all styles

## Running Your Own Experiments

To reproduce these experiments or create new ones:

```bash
# Basic usage
python src/draw_ai.py input_image.jpg output.svg --style smooth --show-analysis

# Try different styles
python src/draw_ai.py input_image.jpg output_sketchy.svg --style sketchy
python src/draw_ai.py input_image.jpg output_minimal.svg --style minimal

# Customize parameters
python src/draw_ai.py input_image.jpg output.svg \
  --max-features 200 \
  --stroke-width 2.5 \
  --stroke-color blue
```

## Contributing Experiments

When adding new experiments:
1. Create a new subdirectory with descriptive name
2. Include the original input image(s)
3. Generate outputs with different styles/parameters
4. Document results in a README.md with:
   - Input specifications
   - Generated outputs with embedded images
   - Technical parameters used
   - Analysis and observations