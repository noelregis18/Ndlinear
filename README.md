# NdLinear Project

## Overview

This project demonstrates and evaluates the NdLinear module, a next-generation replacement for PyTorch's nn.Linear layer. NdLinear preserves multi-dimensional structure of data, enhances representational power, and is parameter-efficient compared to standard linear layers.

## Project Structure

- **NdLinear/**: The core package containing the NdLinear implementation
  - `ndlinear.py`: Main implementation of the NdLinear module
  - `src/`: Source directory containing additional implementations
    - `vit.py`: Vision Transformer implementation using NdLinear
  - `README.md`: Detailed documentation for the NdLinear module
  - `LICENSE`: Apache 2.0 license

- **Root Directory Files**:
  - `nd_linear.py`: A simplified low-rank linear implementation for comparison
  - `model.py`: Example model implementations using NdLinear
  - `compare_models.py`: Script to compare ViT models with standard Linear vs NdLinear
  - `performance_comparison.py`: Performance benchmarks between standard and NdLinear models
  - `simple_comparison.py`: Simple comparison between standard Linear and NdLinear layers

## Key Features of NdLinear

- **Structure Preservation**: Retains the original data format and shape
- **Parameter Efficiency**: Reduces parameter count while improving performance
- **Minimal Overhead**: Maintains the same complexity as conventional linear layers
- **Flexible Integration**: Seamlessly replaces existing linear layers

## Usage Examples

The project includes several comparison scripts demonstrating how NdLinear can be used as a drop-in replacement for standard linear layers:

1. **Simple Comparison**: Run `simple_comparison.py` to see a basic comparison between standard Linear and NdLinear layers
2. **Performance Comparison**: Run `performance_comparison.py` to benchmark performance differences
3. **Model Comparison**: Run `compare_models.py` to see a detailed comparison using Vision Transformer models

## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -e NdLinear/
   ```
3. Run the comparison scripts to see NdLinear in action

## Implementation Details

The project includes two implementations of NdLinear:

1. A simplified low-rank implementation in `nd_linear.py` that demonstrates the core concept
2. The full NdLinear implementation in the `NdLinear/` package with additional features

## Results

The comparison scripts demonstrate that NdLinear models typically:
- Require fewer parameters (parameter reduction of up to 75%)
- Maintain or improve model performance
- Can provide inference speed improvements

## License

This project is distributed under the Apache 2.0 license. See the `NdLinear/LICENSE` file for more details.
