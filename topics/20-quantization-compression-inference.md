# Model Quantization and Compression

 Efficient Inference

 ## Introduction
 Quantization and compression are critical for deploying deep learning models on resource-constrained devices like vehicles. This document covers quantization techniques, model compression, and practices for efficient inference.

## Quantization Methods

### Dynamic Quantization
```python
import torch.quantization as quantization

from torch.quantization import quantize_dynamic

class DynamicQuantizer:
    """Quantize model dynamically"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()  # Set to eval mode
        
    def quantize(self):
        """Apply dynamic quantization"""
        return quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM, torch.nn.GRU},
            dtype=torch.qint8
        )
    
    def quantize_layer(self, layer):
        """Quantize a layer"""
        if hasattr(layer, 'weight'):
            layer.weight.data = layer.weight.data.to(torch.qint8)
            layer.weight.bias.data = layer.weight.bias.data.to(torch.qint8)
            layer.weight.grad.data = layer.weight.grad.data.to(torch.qint8)
        return layer
```
### Static Quantization
```python
from torch.quantization import quantize_static

class StaticQuantizer:
    """Quantize model statically"""
    
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        
        # Fuse modules
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare for calibration
        self.model_prepared = quantize_static(
            self.model,
            self.calibration_data
        )
        
        # Calibrate with sample data
        for batch in calibration_data:
            self.model(batch)
        
        # Convert to quantized model
        self.model_quantized = torch.quantization.convert(self.model_prepared)
    
    def run_inference(self, input):
        """Run inference with quantized model"""
        return self.model_quantized(input)
```
### Quantization-Aware Training (QAT)
```python
from torch.quantization import prepare_qat

class QuantizationAwareTrainer:
    """Train with quantization awareness"""
    
    def __init__(self, model, train_loader, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        
        # Prepare for QAT
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        self.model.train()
        
        # Prepare for QAT
        self.model_prepared = prepare_qat(
            self.model,
            inplace=True
        )
        
        # Train
        for epoch in range(self.num_epochs):
            for batch in self.train_loader:
                # Forward pass
                outputs = self.model_prepared(batch)
                loss = self.criterion(outputs, batch['label'])
                
                # Backward pass
                loss.backward()
                
                # Fake quantization operation
                self.model_prepared.update()
                
                # Optimizer step
                self.optimizer.step()
                
                # Real quantization step
                with torch.no_grad():
                    self.model_quantized(batch)
    
    def save_quantized_model(self, path):
        """Save quantized model"""
        torch.save({
            'model_state_dict': self.model_prepared.state_dict(),
            'model_quantized_state_dict': self.model_quantized.state_dict()
        }, path)
```
## Model Compression
### Pruning
```python
import torch
import torch.nn as nn
import numpy as np

class ModelPruner:
    """Prune model for compression"""
    
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.masks = {}
        
    def prune(self):
        """Apply pruning"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # Compute importance
                importance = torch.abs(module.weight.data)
                
                # Determine threshold
                threshold = np.percentile(
                    importance.cpu().numpy().flatten(),
                    self.pruning_ratio * 100
                )
                
                # Create mask
                mask = importance > threshold
                
                # Store mask
                self.masks[name] = mask
                
                # Apply mask
                module.weight.data *= mask.float()
    
    def fine_tune(self, train_loader, epochs=5):
        """Fine-tune pruned model"""
        # Freeze masks
        for name in self.masks:
            for param in self.model.named_parameters():
                if name in param:
                    param.requires_grad = False
        
        # Train
        for epoch in range(epochs):
            for batch in train_loader:
                # Forward
                outputs = self.model(batch)
                
                # Compute loss
                loss = self.criterion(outputs, batch['label'])
                
                # Backward
                loss.backward()
                
                # Update weights
                self.optimizer.step()
        
        # Restore gradients
        for name in self.masks:
            for param in self.model.named_parameters():
                if name in param:
                    param.requires_grad = True
```
### Weight Sharing
```python
class WeightSharer:
    """Share weights between teacher and student"""
    
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        
    def distill(self, inputs):
        """Distill knowledge"""
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        
        # Get student outputs
        student_outputs = self.student(inputs)
        
        # Compute distillation loss
        loss = self.distillation_loss(
            student_outputs, student_outputs,
            teacher_outputs. teacher_outputs,
            temperature=self.temperature
        )
        
        return loss.item()
    
    def distillation_loss(self, student_outputs, teacher_outputs, temperature=3.0):
        """Compute distillation loss"""
        # Soften outputs
        soft_teacher = torch.softmax(teacher_outputs / temperature, dim=-1)
        soft_student = torch.softmax(student_outputs / temperature, dim=-1)
        
        # KL divergence
        kl_loss = nn.KLDivLoss(reduction='batchmean')()(
            F.log_softmax(soft_student, dim=1),
            F.softmax(soft_teacher / temperature, dim=1)
        )
        
        # Scale by temperature
        return kl_loss * (temperature ** 2)
```
### Knowledge Distillation Training
```python
class DistillationTrainer:
    """Train with distillation"""
    
    def __init__(self, teacher, student, optimizer, epochs=10,
                 temperature=3.0, alpha=0.5):
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.temperature = temperature
        self.alpha = alpha
        self.epochs = epochs
        
    def train(self, train_loader):
        """Train with distillation"""
        for epoch in range(self.epochs):
            for batch in train_loader:
                inputs = batch['input']
                targets = batch['label']
                
                # Forward pass
                teacher_outputs = self.teacher(inputs)
                student_outputs = self.student(inputs)
                
                # Compute loss
                loss = self.distillation_loss(
                    student_outputs, teacher_outputs, self.temperature
                )
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                self.optimizer.step()
```
## Best Practices
### 1. Choose Right Quantization Method
- **Dynamic quantization**: For inference speed, uses less memory
- **Static quantization**: For fixed-point deployment with calibration
- **Quantization-aware training**: For best of both worlds (accuracy and size)

### 2. Test Before Deployment
```python
# Test quantization on representative dataset
quantized_model = self.static_quantizer(test_data)
assert test_data.shape == quantized_model(test_data).shape
```
### 3. Profile In Production
```python
# Profile quantized model
import time

start_time = time.time()
for _ in range(100):
    _ = quantized_model(dummy_input)
    inference_time = (time.time() - start_time) * 1000
    print(f"Inference time: {inference_time:.2f} ms")
```
## Conclusion
Quantization and compression are essential for deploying deep learning models efficiently. By reducing model size and computation, these models can run on devices with limited resources without sacrificing accuracy.

## References
- "Quantization and Training of Neural Networks" (Google, 2018)
- PyTorch Quantization documentation
- "A Survey of Quantization Methods" (Han et al., 2016)
- Tesla AI Day presentations
---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~3500 words  
**Size**: ~22KB
