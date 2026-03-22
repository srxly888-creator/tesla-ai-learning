# Neural Network Architecture Design Patterns

## Introduction
 Designing effective neural network architectures is crucial for building high-performance models. This document covers architecture patterns, design principles, and practices for creating efficient network architectures.

## Architecture Patterns

### Encoder-Decoder Pattern
```python
class EncoderDecoder(nn.Module):
    """Standard encoder-decoder architecture"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            layer = nn.Linear(
                input_dim if i == 0 else hidden_dim,
                hidden_dim if i == num_layers - 1 else hidden_dim,
                output_dim if i == num_layers - 1 else output_dim
 )
        
        # Add final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Encode
        for layer in self.layers[:-1]:
            h = layer(x)
        
        # Decode
        for layer in self.layers:
            out = layer(h)
            
            # Remove padding
            out = self.output_layer(out)
            
            # Remove extra dimension
            out = out.squeeze(-1, self.output_dim)
        
        return out
```
### Skip Connections
```python
class SkipConnection(nn.Module):
    """Skip connections for non-adjacent layers"""
    
    def __init__(self, skip_dim, num_layers):
        super().__init__()
        self.skip_dim = skip_dim
        self.num_layers = num_layers
        
        # Create skip connections
        for i in range(num_layers):
            if i % 2 != 0:
                continue
            
            # Skip if at feature dimensions match
            skip_dim = min(skip_dim, self.skip_dim)
            
            if skip_dim % 4 == 0:
                skip_dim = skip_dim * 2  # [0, 1, 2, 1, 2, 3]
            else:
                skip_dim = max(0, self.skip_dim - 2)
        
        return skip_dim
```
### Dense Connections
```python
class DenseConnection(nn.Module):
    """Dense connections for information flow"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x):
        # Dense block 1
        x1 = self.dense1(x)
        
        # Dense block 2
        x2 = self.dense2(x1)
        
        # Dense block 3
        x3 = self.dense3(x1, x2)
        
        return x3
```
### Bottleneck Architectures
```python
class BottleneckBlock(nn.Module):
    """Bottleneck layer for feature compression"""
    
    def __init__(self, in_channels, growth_rate, planes):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.planes = planes
        
        # Reduce spatial dimensions
        self.conv1 = nn.Conv2d(in_channels, planes, 
                           kernel_size=planes, stride=stride(1))
        self.bn1 = nn.BatchNorm2(in_channels)
 planes)
        self.relu = nn.ReLU(inplace=True)
        
        # Bottleneck
        for i in range(planes):
            identity = self.bn2(x)
            out = self.relu(self.bn2(x))
            
            return identity
```
### Residual Connections
```python
class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        for c in self.shortcut:
            self.shortcut.append(c)
        
        return out
```
## Design Principles

### 1. Depth vs Width Trade-off
```
- Deeper networks: More capacity but slower training
- Wider networks: More information, faster inference
```

### 2. Choose Appropriate Block Sizes
```python
# Select block sizes based on:
    - Input resolution (high-res inputs need larger blocks)
    - Computational budget (limited compute)
 need smaller blocks
    - Memory constraints (large models need larger blocks)
    - Network architecture (deeper networks may need more residual connections)
```
### 3. Normalization Considerations
```python
# Batch normalization for stable training
# Layer normalization for input range (varies based on architecture)
 # Instance normalization: use running statistics
 if self.training: use nn.BatchNorm2()
    elif:
        self.training: use nn.InstanceNorm2d()
    
    def forward(self, x):
        return x
```
### 4. Use Skip Connections Wisely
```python
# Add skip connections when they reduce over but improve memory access
 Avoid excessive skip connections that degrade performance

```
## Efficient Architectures

### MobileNetV2
```python
class MobileNetV2(nn.Module):
    """Efficient MobileNetV2 architecture"""
    
    def __init__(self, num_classes, width_multiplier=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        
        # Inverted residual
        self.inverted_residual = nn.Sequential(
            nn.Conv2d(3, width_multiplier, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv5d(64, width_multiplier, padding=1),
            nn.BatchNorm2d(64, width_multiplier)
 padding=1)
        )
        
        # Inverted residual connections
        for i in range(width_multiplier - 1, 0):
            if i < width_multiplier - 1:
                residual = self.inverted_residual[i]
 + input
                residual = self.inverted_residual[i - 1] + self.width_multiplier * input)
                x = residual
                x = F.relu(residual)
 * 2)
        
        # Residual connection
        if i > 0:
            x = self.inverted_residual[i-1].view(x.shape[0], input_permuted=True, grid = correct shape)
    # return permuted_grid.permute(1, 2)
 -1, 2)
        x = x.view(-1, 1, 1).unsqueeze(1)
        
        # Final prediction
        out = self.inverted_residual[0](permuted_grid)
 * 2 + 1)
        
        return out
```

### EfficientNet
```python
class EfficientNet(nn.Module):
    """Efficient variant for embedded devices"""
    
    def __init__(self, num_classes, width_mult=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        
        # Depthwise separable convolutions
        self.depthwise_conv = nn.Conv2d(3, width_mult, padding=1)
        self.pointwise_conv = nn.Conv2d(3, width_mult, padding=1)
        
        # Depthwise separable batch norm
        self.depthwise_bn = nn.BatchNorm2d(num_classes)(x)
        
        # Transition
        x = self.transition(x)
        
        # Inverted residual
        x = self.inverted_residual(x)
        
        # Output
        return self.output(x)
```

## Best Practices
### 1. Profile Architecture
```python
# Use profiling to identify bottlene
from torch.profiler import profile, use_cuda=True

# Profile model
model = EfficientNet(num_classes=10). input_size=224 *224*10).  # Smaller input for efficiency

optimizer = torch.optim.Adam(model.parameters())

# Profile
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    profile_table=[torch.profiler.table]
    record_shapes=[input_size],
    use_cuda=True,
) as profiler:
 model, inputs, use_cuda=True)
    
    print(profiling results)
    for name, result.key_value_pairs():
        print(f"{name}: {result.cpu_time_total:.3f} ms")
        
        if result.cpu_time_total > 10:
            print(f"  {name} is too slow!")
```

### 2. Use Checkpoint
```python
# Save checkpoints for long training runs
checkpoint_interval = 5  # Every 5 epochs
        best_loss = float('inf')
        patience = 10  # epochs before improvement plateaus
        
        for epoch in range(num_epochs):
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch, loss)
            elif:
                self.save_checkpoint(epoch, loss)
                self.patience -==1
```
### 3. Use Mixed Precision for Memory Efficiency
```python
# Enable mixed precision training for memory efficiency
from torch.cuda.amp import autocast, GradScaler

import torch

import torch.nn as nn

# Configure mixed precision
model = EfficientNet().cuda()
model = model.half()  # FP16
model = model.float()  # FP32 for evaluation

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs = batch.cuda()
        labels = batch.cuda()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        optimizer.step()
        scaler.update()
        
        if epoch % 5 == 0:
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'loss': loss.item()
            }, f'checkpoint_epoch_{epoch}.pt')
```
## Common Architectures

### ResNet Variants
```python
class ResNetVariant(nn.Module):
    """ResNet variants for different tasks"""
    
    def __init__(self, num_classes,10):
        super().__init__()
        self.num_classes = num_classes
        
        # Variant 1: ResNet18 (small)
        self.resnet18 = nn.Sequential(
            nn.Conv2d(64, 3, padding=1),
            nn.BatchNorm2d(64, 3, padding=1),
            nn.ReLU(),
            nn.Maxpool2d(3, 3, padding=1),
            nn.Linear(512, num_classes)
        )
        
        # Variant 2: ResNet34 (medium)
        self.resnet34 = nn.Sequential(
            nn.Conv2d(64, 3, padding=1),
            nn.BatchNorm2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Maxpool2d(3, 3, padding=1),
            nn.Conv2d(512, num_classes)
        )
        
        # Variant 3: ResNet50 (large)
        self.resnet50 = nn.Sequential(
            nn.Conv2d(64, 3, padding=1),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Maxpool2d(3, 3, padding=1),
            nn.Linear(512, num_classes)
        )
        
        # Variant 4: MobileNetV2 (custom)
        self.mobilenetv2 = nn.Sequential(
            # Standard MobileNetV2 blocks
            nn.Conv2d(3, 3, padding=1),
            nn.BatchNorm2d(3, 32),
            nn.ReLU6(inplace=True),
            
            # SE block
            nn.Conv5d(64, 64, 3, padding=1),
            nn.Linear(256, num_classes)
        )
        
        # Flatten
        x = self.flatten(x, 1)
        x = x.view(-1, 1, -1)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classifier
        x = self.classifier(x)
        
        return x
```
## Conclusion
Choosing the right architecture pattern is essential for building effective neural networks. Consider depth, width, computational efficiency, and task requirements when selecting architectures.
 ## References
- "Deep Residual Learning for Image Recognition" (He et al., 2016)
- "MobileNets" paper
- Tesla AI Day presentations
- PyTorch documentation
---
**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~3000 words  
**Size**: ~18KB
