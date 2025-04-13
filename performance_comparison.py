import torch
import torch.nn as nn
import time
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

def benchmark_inference(model, input_tensor, num_runs=1000):
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = model(input_tensor)
    
    end_time = time.time()
    
    return (end_time - start_time) / num_runs

def benchmark_training(model, input_tensor, target, num_runs=100):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Warm-up
    for _ in range(5):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_runs):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    return (end_time - start_time) / num_runs

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
    
    # Create random data for benchmarking
    batch_size = 64
    x = torch.randn(batch_size, input_dim)
    target = torch.randint(0, output_dim, (batch_size,))
    
    # Benchmark inference
    print("\n--- Inference Performance ---")
    linear_inference_time = benchmark_inference(linear_model, x)
    ndlinear_inference_time = benchmark_inference(ndlinear_model, x)
    
    print(f"nn.Linear inference time: {linear_inference_time*1000:.4f} ms per batch")
    print(f"NdLinear inference time: {ndlinear_inference_time*1000:.4f} ms per batch")
    
    if linear_inference_time > ndlinear_inference_time:
        speedup = linear_inference_time / ndlinear_inference_time
        print(f"NdLinear is {speedup:.2f}x faster for inference")
    else:
        slowdown = ndlinear_inference_time / linear_inference_time
        print(f"NdLinear is {slowdown:.2f}x slower for inference")
    
    # Benchmark training
    print("\n--- Training Performance ---")
    linear_training_time = benchmark_training(linear_model, x, target)
    ndlinear_training_time = benchmark_training(ndlinear_model, x, target)
    
    print(f"nn.Linear training time: {linear_training_time*1000:.4f} ms per batch")
    print(f"NdLinear training time: {ndlinear_training_time*1000:.4f} ms per batch")
    
    if linear_training_time > ndlinear_training_time:
        speedup = linear_training_time / ndlinear_training_time
        print(f"NdLinear is {speedup:.2f}x faster for training")
    else:
        slowdown = ndlinear_training_time / linear_training_time
        print(f"NdLinear is {slowdown:.2f}x slower for training")

if __name__ == "__main__":
    main()
