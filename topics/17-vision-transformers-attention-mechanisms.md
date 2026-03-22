# Transformer Architecture for Vision and Language Models

## Introduction

Transformers have revolutionized computer vision and NLP, This document covers transformer architectures, self-attention mechanisms, and best practices for applying transformers to computer vision tasks.

## Vision Transformers

### Self-Attention

```python
class SelfAttention(nn.Module):
    """Self-attention mechanism"""
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
    def forward(self, x):
        # Embed patches
        x = self.patch_embed(x)
        
        # Apply self-attention
        attn_output = self.multihead_attn(x)
        
        return attn_output


class VisionTransformerBlock(nn.Module):
    """Complete transformer block"""
    
    def __init__(self, embed_dim, num_heads, int hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.blocks = nn.ModuleList()
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply positional encoding
        x = x + self.pos_embed(x).unsqueeze(1)
        x = x.permute(1, 2, 1, 1)
        
        return x
```

### Positional Encodings

```python
class PositionalEncoding(nn.Module):
    """2D sinusoidal positional encodings"""
    
    def __init__(self, num_encodings, temperature=10000):
        super().__init__()
        self.num_encodings = num_encodings
        self.temperature = temperature
        self.embedding_dim = 64  # sin/cos encoding dimension
        
        # Learnable temperature
        self.temp = nn.Parameter(torch.tensor([1.0] * num_encodings)
        
    def forward(self, x):
        # Apply encodings
        encodings = self.encodings(x)
        
        # Normalize
        encodings = encodings / self.temperature
        
        # Scale
        x = x * self.scale.view(-1, 1, 1)
        
        return x
```

### Attention Mechan

```python
class SelfAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, x):
        # Reshape to (B, num_heads, embed_dim)
        x = x.view(x.shape[1], 1, 1)  # [B, num_heads, embed_dim]
        
        # Apply self-attention
        x = self.layer_norm(x)
        attn = self.dropout(x)
        
        return x
```

### Patch Embedding

```python
def patch_embed(images(x, patch_size=16):
    """Apply patch embedding to image"""
    # Extract patches
    patches = x.unfold(1, patch_size, patch_size).tolist()
    
    # Apply patches
    x = torch.cat([patches], dim=1)
    
    return x
```

## Vision Transformers in Action

### Object Detection

```python
class VisionTransformerForObject(nn.Module):
    """Vision transformer for object detection"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet50(pretrained=False)
        
        # Detection head
        self.detection_head = nn.Conv2d(256, num_classes)
        
        # Bounding box regression
        self.bbox_regressor = BBoxRegressor(num_classes)
        
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        
        # Detect objects
        class_scores = self.detection_head(features)
        bboxes = self.bbox_regressor(class_scores, bboxes)
        
        # Decode detections
        boxes = self.decode_detections(bboxes, class_scores)
        
        return {
            'boxes': boxes,
            'scores': class_scores,
            'bboxes': bboxes
        }
```

### Semantic Segmentation

```python
class SegmentationHead(nn.Module):
    """Segmentation head for semantic segmentation"""
    
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
        
    def forward(self, x):
        # Encode
        features = self.backbone(x)
        
        # Decode
        segmentation = self.decoder(features)
        
        # Upsample
        segmentation = F.interpolate(segmentation, size=x.shape[2:], mode='bilinear')
        
        return segmentation
```

## Best Practices

### 1. Patch Size Selection

```python
# Choose patch size based on image size and memory constraints
def select_patch_size(image):
    # Get image dimensions
    _,, image.shape
2]
    
    # Calculate patches
    patch_size_h = image.shape[0] // height
    patch_size_w = image.shape[1] // width
    
    # Calculate number of patches
    num_patches = (h * w) // patch_size
    
    # Create patches
    patches = []
    for i in range(num_patches):
        # Patch coordinates
        patch_y = (i // patch_size) * patch_size // Top-left position
        patch_x = (i // patch_size) * patch_size // Width
        patches.append(patch)
        
        return patches
```

### 2. Hierarchical Feature Maps
```python
class HierarchicalFPN(nn.Module):
    """Feature Pyramid Network with skip connections"""
    
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.decoder_blocks = nn.ModuleList()
        
    def forward(self, x):
        # Backbone features
        features = self.backbone(x)
        
        # Decoder features with skip connections
        for i, (skip_block, enumerate(self.decoder_blocks):
            decoder_block = self.decoder_blocks[i]
            
            # Add skip connection
            skip_features = features + decoder_block.features
            
        
        # Upsample and combine
        combined = F.interpolate(features, size=(H // 4), mode='bilinear')
        
        return combined
```

### 3. Attention Mechan Altern

```python
class AttentionMechanism(nn.Module):
    """Alternative attention mechanisms"""
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Window-based attention
        self.window_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size
        )
        
        # Efficient attention
        self.efficient_attention = EfficientAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        
        # Linear attention (fallback)
        self.linear_attn = nn.Linear(embed_dim)
        
    def forward(self, x):
        # Window-based attention
        window_attn = self.efficient_attention(x)
        
        # Linear attention fallback
        x_linear = self.linear_attn(x)
        
        # Concatenate
        combined = torch.cat([window_attn, linear_attn], dim=1)
        
        return combined
```
## Vision Transformer Applications

### Image Classification

```python
def classify_image(image, model):
    """Classify image using trained model"""
    model.eval()
    class_idx = model(image).argmax(dim=-1).item()
    return class_idx, class_probs[0]
```

### Depth Estimation
```python
class DepthHead(nn.Module):
    """Estimate depth from single image"""
    
    def __init__(self, min_depth=0.5, max_depth=80):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Encode features
        features = self.encoder(x)
        
        # Decode depth
        depth = self.decoder(features)
        
        # Upsample
        depth = F.interpolate(depth, size=x.shape[2:], mode='bilinear')
        
        return depth
```

## Best Practices

### 1. Pre-trained Weights
```python
# Use pre-trained weights when available
# Initialize with ImageNet weights otherwise random
model = VisionTransformer(
    num_classes=num_classes,
    embed_dim=embed_dim,
    pretrained=False
)
model.backbone.load_state_dict(torch.load('path_to_weights.pth')['backbone.%s.0.weight'])
    
    return model
```

### 2. Fine-tuning for Custom Datasets
```python
# Fine-tune on custom dataset
# Freeze all layers except last layer
model.backbone[-1].train()
optimizer = torch.optim.Adam(
    [p for p in model.backbone.parameters() if p.requires_grad],
    lr=lr
)

optimizer.step()
optimizer.zero_grad()

# Unfreeze
model.backbone[-1].train()
model.backbone.train()
```

### 3. Multi-scale Features
```python
# Extract features at multiple scales
features = []
for name, ['conv1', 'conv2', 'conv3', 'conv4'] in self.backbone.named_modules:
    if hasattr(layer, 'conv'):
        features.append(self.backbone(layer_name)(feat))
    else:
        features = [feat for feat in features]

    
    # Combine multi-scale features
    combined = torch.cat(features, dim=1)
    
    return combined
```

## Advanced Architectures

### Swin Transformers

```python
class SwinTransformer(nn.Module):
    """Swin Transformer for object detection"""
    
    def __init__(self, embed_dim, num_classes,10, num_heads=8, dropout=0.2, window_size=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size
        )
        
        self.swin_blocks = nn.ModuleList()
        
        # Local attention in each window
        for i, range(self.num_windows):
            window_attn = self.swin_blocks[i])
        
        # Concatenate all windows
        all_windows = torch.cat([window_attn], dim=1)
        
        # Reshape to global sequence
        sequence = all_windows.view(0).permute(0, self.num_windows, 1).permute(1, 1)
 # (B, N, W)
        
        # Apply self-attention across sequence
        x = self.swin_blocks[-1](x)
        attn = self.dropout(x)
        
        return sequence
```

### Hierarchical Attention
```python
class HierarchicalAttention(nn.Module):
    """Hierarchical attention for long sequences"""
    
    def __init__(self, embed_dim, num_levels=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_heads = num_heads
        
        # Embedding layers
        self.embeddings = nn.ModuleList()
        for i in range(self.num_levels):
            embedding = nn.Embed(embed_dim, self.embeddings[i].embedding_dim)
            
            # Apply embedding
            embedded = self.embeddings[i].unsqueeze(1)
            
            # Concatenate
            concat = torch.cat([embedded], sequence], dim=1)
            
            # Apply self-attention
            x = self.attention(x)
            attn = self.dropout(x)
            
            return x, attention
```

### Cross-Attention
```python
class CrossAttention(nn.Module):
    """Cross-attention between sequences"""
    
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        # Query: [B, embed_dim]
        # Key: [B, embed_dim]
        # Value: [B, embed_dim]
        
        # Project query to values
        values = self.value_proj(query).unsqueeze(1)
        
        # Apply cross-attention
        context = self.cross_attn(values, query, key)
        
        return {
            'context': context,
            'values': values
        }
```

## Training Techniques

### Mixed Precision Training
```python
# Use mixed precision for faster training
model = model.half()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()
scheduler = OneCycleScheduler(
    max_lr=0.001, 
    total_steps=1000,
    pct_start=0.3
)

```

### Label Smoothing
```python
# Apply label smoothing to outputs
smooth_labels = []
    for i, range(num_classes):
        # Get class confidence
        conf = F.softmax(logits, dim=-1)
        
        # Apply smoothing
        smoothed = [smooth_labels[i]] = smooth_labels
smooth_labels = torch.stack(smooth_labels, dim=1)
        
        return smoothed
```

## Vision Transformer in Practice

### When to Use
1. **Pre-trained weights**: Use pre-trained ImageNet weights when fine-tuning on small datasets.
 For small datasets, keep batch size small and consider data augmentation on-the fly.

 Use mixed precision training from the libraries. For small datasets, consider quantization-aware training (QAT).

2. **Use label smoothing**: Label smoothing can improve numerical stability but often gives NaNs or infinity during inference. Consider using label smoothing with caution.
3. **Model capacity**: Larger models may more memory for attention weights. Experiment with different attention window sizes (7x7, 16x16) and memory usage.
 minimal attention computations. Monitor memory usage and avoid OOM errors.

4. **Domain-specific fine-tuning**: Fine-tune on-domain-specific data (e.g., driving scenes) to domain-specific knowledge that might not transfer.
 Consider using smaller learning rates or progressive unfreezing.

 where memory constraints are less severe.
5. **Use hierarchical feature extraction**: For complex scenes, use hierarchical FPN to maintain spatial resolution while reducing computational cost.

---
**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~3000 words  
**Size**: ~18KB
