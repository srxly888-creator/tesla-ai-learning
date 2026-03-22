# Convolutional Neural Networks - Feature Extraction and Classification
 ## Introduction
 Convolutional Neural Networks (CNNs) are the backbone of modern computer vision. This document covers CNN architectures, layer types, and applications in autonomous driving and from object detection to semantic segmentation.

 ## CNN Fundamentals

### Convolution Operation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

class Convolution(nn.Module):
    """2D convolution layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,1, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
 
    def forward(self, x):
        # Convolution
        x = self.conv(x)
        
        # Batch normalization
        x = self.bn(x)
        
        # Activation
        x = F.relu(x)
        
        return x
```
### Pooling Layers
```python
class MaxPool2d(nn.Module):
    """2D max pooling layer"""
    
    def __init__(self, kernel_size, stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        # Pooling
        x = self.pool(x)
        
        # Reshape
        x = x.view(-1, 1)
        
        return x
```

### Feature Pyramid Networks
```python
class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale features"""
    
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        # Bottom-up pathway
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Top-down pathway
        self.top_down = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Bottom-up
        features = self.features(x)
        
        # Top-down
        top_down = self.top_down(features)
        
        # Classifier
        output = self.classifier(top_down)
        
        return output
```
## Applications in Autonomous Driving

### Object Detection with CNNs
```python
class ObjectDetector(nn.Module):
    """CNN-based object detector"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extractor (backbone)
        self.backbone = ResNet50(pretrained=False)
        
        # Detection head
        self.detector = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, 1)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Detect objects
        detections = self.detector(features)
        
        return detections
```
### Semantic Segmentation
```python
class SemanticSegmentor(nn.Module):
    """CNN for semantic segmentation"""
    
    def __init__(self, num_classes=20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
        
    def forward(self, x):
        # Encode
        enc = self.encoder(x)
        
        # Decode
        dec = self.decoder(enc)
        
        return dec
```
## Best Practices
### 1. Use Batch Normalization
```python
# Always use batch normalization
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0, std=module.bias, mean=0)
```
### 2. Apply Data Augmentation
```python
# Data augmentation for robustness
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
```
### 3. Use Learning Rate Scheduling
```python
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)
```
## Conclusion
CNNs remain fundamental to computer vision, Understanding convolution operations, pooling mechanisms, and architecture patterns is essential for building effective models. From object detection to segmentation, CNNs enable autonomous vehicles to perceive and understand their environment.

 ## References
- "Deep Learning" book (Goodfellow et al.)
- Tesla AI Day presentations
- "Fully Convolutional Networks" paper
- PyTorch documentation
---
**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~2500 words  
**Size**: ~16KB
