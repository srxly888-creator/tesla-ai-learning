# Model Compression and Knowledge Distillation
 ## Introduction
 Model compression reduces model size while maintaining performance. This document covers pruning, quantization, and knowledge distillation techniques for deploying models on resource-constrained devices. ## Pruning
### Magnitude Pruning
```python
import torch
import torch.nn as nn
import numpy as np

class MagnitudePruner:
    """Prune by weight magnitude"""
    
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.masks = {}
        
    def prune(self):
        """Apply magnitude pruning"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Compute importance
                importance = torch.abs(param.data)
                
                # Determine threshold
                threshold = np.percentile(
                    importance.cpu().numpy(),
                    self.pruning_ratio * 100
                )
                
                # Create mask
                mask = importance > threshold
                
                # Apply mask
                param.data *= mask.float()
                self.masks[name] = mask
```
## Knowledge Distillation
### Teacher-Student Framework
```python
class KnowledgeDistillation(nn.Module):
    """Distill knowledge from teacher to student"""
    
    def __init__(self, teacher, student, temperature=3.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        
    def forward(self, x):
        """Distill knowledge"""
        with torch.no_grad():
            teacher_out = self.teacher(x)
        
        student_out = self.student(x)
        
        # Soft targets
        soft_teacher = F.softmax(teacher_out / self.temperature, dim=1)
        soft_student = F.log_softmax(student_out / self.temperature, dim=1)
        
        # KL divergence
        loss = F.kl_div(soft_teacher, soft_student, reduction='batchmean')
 * (self.temperature ** 2)
        
        return loss
```
## Best Practices
### 1. Iterative Pruning
```python
# Prune iteratively
for iteration in range(num_iterations):
    prune_ratio = initial_ratio * (1 + iteration) / num_iterations
    pruner.prune(prune_ratio)
    pruner.fine_tune()
```
### 2. Test Different Temperature Values
```python
# Try different temperatures for best results
for temp in [1.0, 2.0, 5.0, 10.0]:
    student = train_with_temperature(temp)
    evaluate(student)
```
## Conclusion
Model compression techniques are essential for deploying models on resource-constrained devices. By carefully applying pruning, quantization, and knowledge distillation, we can achieve significant model size reduction while maintaining accuracy. ## References
- "Model Compression for Autonomous Driving" papers
- Tesla AI Day presentations
- "Knowledge Distillation" papers
---
**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~2500 words  
**Size**: ~12KB
