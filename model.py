import torch.nn as nn
from nd_linear import NdLinear

class ModelWithNdLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rank=4):
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

# Compare parameter counts
def compare_models(input_dim=784, hidden_dim=256, output_dim=10):
    # Standard model
    standard_model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    
    # NdLinear model
    nd_model = ModelWithNdLinear(input_dim, hidden_dim, output_dim)
    
    standard_params = sum(p.numel() for p in standard_model.parameters())
    nd_params = sum(p.numel() for p in nd_model.parameters())
    
    print(f"Standard model parameters: {standard_params:,}")
    print(f"NdLinear model parameters: {nd_params:,}")
    print(f"Parameter reduction: {(1 - nd_params/standard_params)*100:.2f}%")
    
    # Breakdown of parameters per layer
    print("\nParameter breakdown for standard model:")
    for name, param in standard_model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")
    
    print("\nParameter breakdown for NdLinear model:")
    for name, param in nd_model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")
    
    return standard_model, nd_model

# Main function to run the comparison
if __name__ == "__main__":
    compare_models()