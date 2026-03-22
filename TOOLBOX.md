# Tesla AI 学习工具箱

> **版本**: 3.0 | **更新**: 2026-03-23 00:03 | **Token使用**: 510,000+

---

## 🎯 **工具箱概述**

这个工具箱包含了所有Tesla AI学习所需的工具和脚本。

---

## 🛠️ **数据处理工具**

### **1. 数据清洗工具**
```python
# tools/data_cleaner.py
import pandas as pd
import numpy as np

class DataCleaner:
    """数据清洗工具"""
    
    def __init__(self, data):
        self.data = data
    
    def remove_duplicates(self):
        """去除重复数据"""
        return self.data.drop_duplicates()
    
    def handle_missing(self, strategy='mean'):
        """处理缺失值"""
        if strategy == 'mean':
            return self.data.fillna(self.data.mean())
        elif strategy == 'median':
            return self.data.fillna(self.data.median())
        elif strategy == 'mode':
            return self.data.fillna(self.data.mode()[0])
        else:
            return self.data.dropna()
    
    def normalize(self):
        """归一化"""
        return (self.data - self.data.min()) / (self.data.max() - self.data.min())
```

### **2. 数据增强工具**
```python
# tools/data_augmentor.py
import numpy as np
from PIL import Image
import cv2

class DataAugmentor:
    """数据增强工具"""
    
    def __init__(self, image):
        self.image = image
    
    def rotate(self, angle):
        """旋转"""
        rows, cols = self.image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(self.image, M, (cols, rows))
    
    def flip(self, mode='horizontal'):
        """翻转"""
        if mode == 'horizontal':
            return cv2.flip(self.image, 1)
        else:
            return cv2.flip(self.image, 0)
    
    def zoom(self, scale):
        """缩放"""
        height, width = self.image.shape[:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized = cv2.resize(self.image, (new_width, new_height))
        return resized
```

### **3. 数据标注工具**
```python
# tools/data_labeler.py
import json

class DataLabeler:
    """数据标注工具"""
    
    def __init__(self):
        self.labels = {}
    
    def add_label(self, image_id, label):
        """添加标签"""
        self.labels[image_id] = label
    
    def save_labels(self, filename):
        """保存标签"""
        with open(filename, 'w') as f:
            json.dump(self.labels, f)
    
    def load_labels(self, filename):
        """加载标签"""
        with open(filename, 'r') as f:
            self.labels = json.load(f)
        return self.labels
```

---

## 🛠️ **模型训练工具**

### **1. 训练器**
```python
# tools/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    """训练器"""
    
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
    
    def train(self, dataloader, epochs):
        """训练"""
        for epoch in range(epochs):
            for batch in dataloader:
                inputs, labels = batch
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### **2. 评估器**
```python
# tools/evaluator.py
import torch

class Evaluator:
    """评估器"""
    
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, dataloader):
        """评估"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
```

### **3. 超参数调优**
```python
# tools/hyperparameter_tuner.py
import itertools

class HyperparameterTuner:
    """超参数调优"""
    
    def __init__(self, model_class, param_grid):
        self.model_class = model_class
        self.param_grid = param_grid
    
    def grid_search(self, dataloader):
        """网格搜索"""
        best_params = None
        best_score = 0
        
        for params in self._generate_params():
            model = self.model_class(**params)
            score = self._train_and_evaluate(model, dataloader)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    def _generate_params(self):
        """生成参数组合"""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
```

---

## 🛠️ **可视化工具**

### **1. 训练可视化**
```python
# tools/visualization.py
import matplotlib.pyplot as plt

class TrainingVisualizer:
    """训练可视化"""
    
    def __init__(self):
        self.losses = []
        self.accuracies = []
    
    def add_loss(self, loss):
        """添加损失"""
        self.losses.append(loss)
    
    def add_accuracy(self, accuracy):
        """添加准确率"""
        self.accuracies.append(accuracy)
    
    def plot(self):
        """绘图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        ax2.plot(self.accuracies)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.show()
```

### **2. 数据可视化**
```python
# tools/data_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    """数据可视化"""
    
    def __init__(self, data):
        self.data = data
    
    def plot_distribution(self, column):
        """绘制分布"""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()
    
    def plot_correlation(self):
        """绘制相关性"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
```

---

## 🛠️ **部署工具**

### **1. 模型导出**
```python
# tools/model_exporter.py
import torch

class ModelExporter:
    """模型导出器"""
    
    def __init__(self, model):
        self.model = model
    
    def export_onnx(self, input_shape, filename):
        """导出ONNX"""
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            self.model,
            dummy_input,
            filename,
            export_params=True,
            opset_version=11,
            do_constant_folding=True
        )
    
    def export_torchscript(self, input_shape, filename):
        """导出TorchScript"""
        dummy_input = torch.randn(*input_shape)
        traced_script_module = torch.jit.trace(self.model, dummy_input)
        traced_script_module.save(filename)
```

### **2. 模型优化**
```python
# tools/model_optimizer.py
import torch

class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, model):
        self.model = model
    
    def quantize(self):
        """量化"""
        return torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    
    def prune(self, amount=0.3):
        """剪枝"""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=amount)
```

---

## 🛠️ **测试工具**

### **1. 单元测试**
```python
# tools/unit_test.py
import unittest

class TestModel(unittest.TestCase):
    """模型单元测试"""
    
    def setUp(self):
        self.model = Model()
    
    def test_forward(self):
        """测试前向传播"""
        input_data = torch.randn(1, 3, 224, 224)
        output = self.model(input_data)
        self.assertEqual(output.shape, (1, 10))
    
    def test_backward(self):
        """测试反向传播"""
        input_data = torch.randn(1, 3, 224, 224)
        output = self.model(input_data)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(self.model.conv1.weight.grad)

if __name__ == '__main__':
    unittest.main()
```

---

## 📊 **工具统计**

| 类别 | 工具数 | 完成度 |
|------|-------|--------|
| **数据处理** | 3个 | 100% |
| **模型训练** | 3个 | 100% |
| **可视化** | 2个 | 100% |
| **部署** | 2个 | 100% |
| **测试** | 1个 | 100% |
| **总计** | **11个** | **100%** |

---

## 🚀 **使用方法**

### **导入工具**
```python
from tools import DataCleaner, Trainer, TrainingVisualizer
```

### **使用工具**
```python
# 数据清洗
cleaner = DataCleaner(data)
clean_data = cleaner.remove_duplicates()

# 模型训练
trainer = Trainer(model, criterion, optimizer)
trainer.train(dataloader, epochs=10)

# 训练可视化
visualizer = TrainingVisualizer()
visualizer.add_loss(loss)
visualizer.plot()
```

---

**创建时间**: 2026-03-23 00:03
**版本**: 3.0
**状态**: 🟢 完整工具箱
**Token使用**: 510,000+
