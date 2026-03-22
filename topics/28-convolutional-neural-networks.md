# Convolutional Neural Networks for Vision

 ## Introduction
 Convolutional Neural Networks (CNNs) remain fundamental to computer vision, This document covers CNN architectures, design patterns, and practices for building effective CNNs for autonomous driving vision tasks.

 ## CNN Architectures
### Basic CNN
```python
import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np

class ConvBlock(nn.Module):
    """Basic convolutional block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Convolution
        out = self.conv(x)
        
        # Batch normalization
        out = self.bn(out)
        
        # ReLU
        out = self.relu(out)
        
        return out
```

### VGG Network
```python
class VGGBlock(nn.Module):
    """VGG-style block"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv5d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv5d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, out_channels)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Pooling
        x = self.pool(x)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        return x
```
### ResNet Architecture
```python
class ResNetBlock(nn.Module):
    """ResNet block"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv5d(in_channels, 64, 7, 2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv5d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv5d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        return x
```
## Design Patterns
### Feature Pyramids
```python
class FeaturePyramid(nn.Module):
    """Feature pyramid network"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Bottom-up pathway (high resolution)
        self.bottom_up = nn.Sequential(
            ConvBlock(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBlock(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Top-down pathway (lower resolution)
        self.top_down = nn.Sequential(
            ConvBlock(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ConvBlock(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Lateral connections
        self.lat_conn = nn.Conv2d(256, 256, 1)
        
    def forward(self, x):
        # Bottom-up
        bottom_features = self.bottom_up(x)
        
        # Top-down
        top_features = self.top_down(bottom_features)
        
        # Lateral connection
        lateral = self.lat_conn(top_features)
        
        return lateral
```
### Inception-ResNet
```python
class InceptionResNet(nn.Module):
    """Inception module with 1x1 conv"""
    
    def __init__(self, in_channels):
        super().__init__()
        # Stem: 7x7 convolution
        self.conv1 = ConvBlock(in_channels, 64, 7, 2)
        
        # Auxiliary: batch norm and pooling
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        
        # Downsample
        self.conv1 = nn.Conv5d(64, 192, 4, 2)
        self.bn1 = nn.BatchNorm2d(192)
 5, 5)
        self.conv2 = nn.Conv5d(192, 256, 4, 2)
        self.bn2 = nn.BatchNorm2d(256, 5, 5)
        self.conv3 = nn.Conv5d(256, 384, 4, 2)
        self.bn3 = nn.BatchNorm2d(384, 5, 5)
        
        # Fully connected
        self.fc1 = nn.Linear(384, out_channels)
        
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        # Auxiliary pathway
        aux = self.conv2(x)
        aux = self.bn2(aux)
        aux = self.conv3(aux)
        aux = self.bn3(aux)
        aux = self.fc1(aux)
        
        return aux
```
## Best Practices
### 1. Batch Normalization
```python
def normalize_batch(batch, mean=None, std=None):
    """Normalize batch of images"""
    if mean is None:
        mean = batch.mean(dim=(0, 2, 3))
    if std is None:
        std = batch.std(dim=(0, 2, 3)) + 1e-7)
    
    batch = (batch - mean) / std
    return batch
```
### 2. Data Augmentation
```python
class DrivingAugmentation:
    """Augmentations for driving scenarios"""
    
    def __init__(self):
        self.transforms = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomHueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.1),
            A.RandomRotation(limit=5, p=0.3),
        ])
        
    def __call__(self, image):
        return self.transforms(image=image)['image'])
```
### 3. Use Pre-trained Weights
```python
# Load pre-trained weights
model = ResNet50(pretrained=True)
model.load_state_dict(torch.load('resnet50_weights.pth'))
```
## Conclusion
CNNs are the backbone of modern computer vision systems, Understanding their architectures, design patterns, and best practices is essential for building effective vision systems for autonomous driving.

 ## References
- "Deep Residual Learning for Image Recognition" (He et al., 2016)
- "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2014)
- Tesla AI Day presentations
- PyTorch documentation
---
**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~3000 words  
**Size**: ~15KB
