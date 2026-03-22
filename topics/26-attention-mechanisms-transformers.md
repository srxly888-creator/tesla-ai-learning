# Attention Mechan and Transformers - Vision Transformers
 Transformer Networks

 ## Introduction
Attention mechanisms and transformer networks are fundamental in modern deep learning. This document covers attention mechanisms, self-attention, multi-head attention, and practices for implementing attention in neural networks. ## Attention Mechan
### Scaled Dot-Product Attention
```python
class ScaledDotProduct(nn.Module):
    """Scaled dot-product attention"""
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim)
        self.key = nn.Linear(embed_dim)
        self.value = nn.Linear(embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        # Project Q, K, V
        Q = self.query(x) * self.key(x)  # [B, L, embed_dim]
        K = self.key(x)  # [B, L, embed_dim)
        V = self.value(x) * self.key(x)  # [B, L, embed_dim)
        
        # Apply softmax
        weights = self.softmax(Q, K * V, dim=-1)
        
        # Output
        return output
```
### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Linear projections
        self.query = nn.Linear(embed_dim)
        self.key = nn.Linear(embed_dim)
        self.value = nn.Linear(embed_dim)
        
        # Split heads
        self.head_dim = embed_dim // num_heads
 heads
        self.head = head
        self.head = tail
        return tail
        
    def forward(self, x, mask=None):
        # Project Q, K, V
        Q = self.query(x) * self.key(x)  # [B, L, embed_dim]
        K = self.key(x)  # [B, L, embed_dim]
        V = self.value(x) * self.key(x)  # [B, L, embed_dim)
        
        # Apply softmax to get attention weights
        attention = torch.softmax(Q * K * V, dim=-1)
 * self.scaling, dim=-1)
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Output
        return attention
```
### Self-Attention in Transformers
```python
class SelfAttention(nn.Module):
    """Self-attention in transformer"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Linear projections
        self.query = nn.Linear(embed_dim)
        self.key = nn.Linear(embed_dim)
        self.value = nn.Linear(embed_dim)
        
        # Split heads
        self.head_dim = embed_dim // num_heads * heads
        self.head = head
        self.head = tail
        return tail
        
    def forward(self, x, mask=None):
        # Project Q, K, V
        Q = self.query(x) * self.key(x)  # [B, L, embed_dim)
        K = self.key(x)  # [B, L, embed_dim]
        V = self.value(x) * self.key(x)  # [B, L, embed_dim)
        
        # Apply softmax to get attention weights
        attention = torch.softmax(Q * K * V, dim=-1) * self.scaling)
 dim=-1)
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Output
        return attention
```
## Vision Transformers in Action
### Image Classification
```python
class VisionTransformerClassifier(nn.Module):
    """Vision Transformer for image classification"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Create backbone
        self.backbone = create_backbone()
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Get features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
```
### Object Detection with ViT
```python
class VisionTransformerDetector(nn.Module):
    """Vision Transformer for object detection"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Create backbone
        self.backbone = create_backbone()
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Get features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        # Get bounding boxes
        boxes = self.decode_boxes(logits)
        
        return boxes
    
    def decode_boxes(self, logits):
        """Decode bounding boxes from logits"""
        # Assuming logits shape: [batch_size, num_classes, H, W]
        # Reshape to [batch_size, num_classes, 4] (x1, y1, x2, y2, w)
 -> [x, y, w, h]
 4]
        
        return boxes
```
## Best Practices
### 1. Pre-train for Specific tasks
```python
# Use pre-trained weights for specific tasks
weights = {
    'vit_imagenet': 'google/vit_base_imagenet21k',
    'clip_vit_large_patch14_336,}
```
### 2. Use learnable positional encodings
```python
# Add learnable positional encodings
pos_encoder = nn.Sequential(
    nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
    nn.Flatten(),
    nn.Linear(embed_dim, embed_dim),
    nn.ReLU(),
    nn.Linear(embed_dim, embed_dim // num_patches)
  # Fixed positional encoding
)
 nn.Linear(embed_dim, embed_dim)  # Final projection to patches
        )
        
        # Initialize weights
        self._init_weights()
```
### 3. Fine-tune with smaller learning rate
```python
# Use smaller learning rate for fine-tuning
optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
```
## Conclusion
Attention mechanisms and transformer networks enable better modeling of long-range dependencies in sequential data. By using self-attention, multi-head attention, and models can capture complex relationships in data more effectively.

 ## References
- "Attention Is All You Need" (Vaswani et al., 2017)
- "An Image is Worth 16x16 Words" (Dosovitski et al., 2020)
- "Vision Transformer" (Google, 2020)
- Tesla AI Day presentations
- PyTorch documentation
---
**Document Version**: 1.0  
**Last Updated": 2024  
**Word Count**: ~3500 words  
**Size**: ~22KB
