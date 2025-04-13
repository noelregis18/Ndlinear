import torch
import torch.nn as nn
import sys
import os
import time
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add the NdLinear directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NdLinear'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NdLinear', 'src'))

# Import the original ViT implementation
from vit import ViT as ViT_Modified

# Create a copy of the original ViT implementation with nn.Linear instead of NdLinear
class ViT_Original(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}
        self.to_patch_embedding = nn.Sequential(
            nn.Unflatten(1, (channels, image_height, image_width)),
            nn.Unfold(kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width)),
            nn.Flatten(start_dim=0, end_dim=1),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            nn.Unflatten(0, (-1, num_patches))
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Simple transformer implementation
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=depth
        )
        
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_inference_time(model, input_tensor, num_runs=100):
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(num_runs):
        _ = model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    return (end_time - start_time) / num_runs

def generate_synthetic_data(batch_size=32, image_size=224, channels=3):
    # Generate random images and labels
    images = torch.randn(batch_size, channels, image_size, image_size)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def main():
    # Model parameters
    image_size = 32  # Smaller image size for faster testing
    patch_size = 4
    num_classes = 10
    dim = 128
    depth = 2
    heads = 4
    mlp_dim = 256
    
    # Create models
    original_model = ViT_Original(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    )
    
    modified_model = ViT_Modified(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim
    )
    
    # Count parameters
    original_params = count_parameters(original_model)
    modified_params = count_parameters(modified_model)
    
    print(f"Original model parameters: {original_params:,}")
    print(f"Modified model parameters: {modified_params:,}")
    print(f"Parameter reduction: {(1 - modified_params/original_params)*100:.2f}%")
    
    # Generate synthetic data for inference benchmarking
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move models to device
    original_model = original_model.to(device)
    modified_model = modified_model.to(device)
    
    # Benchmark inference time
    input_tensor, _ = generate_synthetic_data(batch_size, image_size)
    input_tensor = input_tensor.to(device)
    
    original_time = benchmark_inference_time(original_model, input_tensor)
    modified_time = benchmark_inference_time(modified_model, input_tensor)
    
    print(f"Original model inference time: {original_time*1000:.2f} ms per batch")
    print(f"Modified model inference time: {modified_time*1000:.2f} ms per batch")
    print(f"Speedup: {original_time/modified_time:.2f}x")
    
    # Training comparison
    print("\nTraining comparison:")
    
    # Generate synthetic dataset
    num_samples = 500
    images, labels = generate_synthetic_data(num_samples, image_size)
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training settings
    criterion = nn.CrossEntropyLoss()
    epochs = 5
    
    # Train original model
    original_model.train()
    optimizer_original = torch.optim.Adam(original_model.parameters(), lr=0.001)
    
    print("Training original model:")
    for epoch in range(epochs):
        loss, acc = train_epoch(original_model, dataloader, criterion, optimizer_original, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
    
    # Train modified model
    modified_model.train()
    optimizer_modified = torch.optim.Adam(modified_model.parameters(), lr=0.001)
    
    print("\nTraining modified model:")
    for epoch in range(epochs):
        loss, acc = train_epoch(modified_model, dataloader, criterion, optimizer_modified, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
