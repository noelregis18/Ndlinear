import torch
from model import compare_models

if __name__ == "__main__":
    # Compare parameter counts
    compare_models()
    
    # Test with random data
    input_dim, hidden_dim, output_dim = 784, 256, 10
    model = ModelWithNdLinear(input_dim, hidden_dim, output_dim)
    
    # Create random input
    x = torch.randn(32, input_dim)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")