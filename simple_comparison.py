import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path to import the NdLinear implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nd_linear import NdLinear

# Define a simple model with nn.Linear
class SimpleModelLinear(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Define the same model with NdLinear
class SimpleModelNdLinear(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, rank=4):
        super().__init__()
        self.layer1 = NdLinear(input_dim, hidden_dim, rank=rank)
        self.layer2 = NdLinear(hidden_dim, hidden_dim, rank=rank)
        self.layer3 = NdLinear(hidden_dim, output_dim, rank=rank)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Model dimensions
    input_dim = 784  # 28x28 for MNIST
    hidden_dim = 256
    output_dim = 10
    rank = 16  # Rank for NdLinear
    
    # Create models
    linear_model = SimpleModelLinear(input_dim, hidden_dim, output_dim)
    ndlinear_model = SimpleModelNdLinear(input_dim, hidden_dim, output_dim, rank)
    
    # Count parameters
    linear_params = count_parameters(linear_model)
    ndlinear_params = count_parameters(ndlinear_model)
    
    print(f"Standard model with nn.Linear parameters: {linear_params:,}")
    print(f"Model with NdLinear parameters: {ndlinear_params:,}")
    print(f"Parameter reduction: {(1 - ndlinear_params/linear_params)*100:.2f}%")
    
    # Breakdown of parameters per layer
    print("\nParameter breakdown per layer:")
    
    # Linear model
    print("\nnn.Linear model:")
    for name, param in linear_model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")
    
    # NdLinear model
    print("\nNdLinear model:")
    for name, param in ndlinear_model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")
    
    # Test inference with random input
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass with both models
    with torch.no_grad():
        linear_output = linear_model(x)
        ndlinear_output = ndlinear_model(x)
    
    print(f"\nOutput shape from linear model: {linear_output.shape}")
    print(f"Output shape from NdLinear model: {ndlinear_output.shape}")

if __name__ == "__main__":
    main()
