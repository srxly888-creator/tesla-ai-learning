# Tesla AI 学习额外补充内容

> **版本**: 3.0 | **更新**: 2026-03-23 01:06 | **Token使用**: 880,000+

---

## 🔧 **额外工具和脚本**

### **1. 数据处理工具**
```python
# data_tools.py
import numpy as np
import pandas as pd
from PIL import Image
import cv2

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config):
        self.config = config
    
    def load_image(self, path):
        """加载图像"""
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 调整大小
        image = cv2.resize(image, self.config['image_size'])
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image
    
    def augment_image(self, image):
        """数据增强"""
        # 随机翻转
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # 随机旋转
        angle = np.random.randint(-30, 30)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        
        # 随机亮度
        brightness = np.random.uniform(0.8, 1.2)
        image = image * brightness
        
        return np.clip(image, 0, 1)

# 使用示例
if __name__ == '__main__':
    config = {
        'image_size': (224, 224)
    }
    
    processor = DataProcessor(config)
    image = processor.load_image('test.jpg')
    processed = processor.preprocess_image(image)
    augmented = processor.augment_image(processed)
```

### **2. 模型评估工具**
```python
# model_evaluator.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        return metrics

# 使用示例
if __name__ == '__main__':
    model = MyModel()
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(test_dataloader)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
```

### **3. 可视化工具**
```python
# visualization_tools.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    """可视化工具"""
    
    def __init__(self):
        self.figsize = (12, 8)
        self.style = 'seaborn'
    
    def plot_training_history(self, history):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # 损失曲线
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # 准确率曲线
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm, classes):
        """绘制混淆矩阵"""
        plt.figure(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def plot_predictions(self, images, predictions, labels, num_samples=5):
        """绘制预测结果"""
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        for i in range(num_samples):
            axes[i].imshow(images[i])
            axes[i].set_title(f'Pred: {predictions[i]}, True: {labels[i]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == '__main__':
    visualizer = Visualizer()
    
    # 绘制训练历史
    history = {
        'train_loss': [0.5, 0.3, 0.2, 0.1],
        'val_loss': [0.6, 0.4, 0.3, 0.2],
        'train_acc': [0.7, 0.8, 0.9, 0.95],
        'val_acc': [0.65, 0.75, 0.85, 0.9]
    }
    visualizer.plot_training_history(history)
```

### **4. 日志工具**
```python
# logging_tools.py
import logging
import sys
from pathlib import Path

class Logger:
    """日志工具"""
    
    def __init__(self, name, log_file=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # 文件处理器
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)

# 使用示例
if __name__ == '__main__':
    logger = Logger('TeslaAI', log_file='tesla_ai.log')
    
    logger.info("Starting training...")
    logger.warning("Learning rate too high")
    logger.error("Training failed!")
```

---

## 📊 **额外工具统计**

| 工具 | 代码行数 | 功能 |
|------|---------|------|
| **数据处理** | 50+ | 数据加载、预处理、增强 |
| **模型评估** | 40+ | 评估指标计算 |
| **可视化** | 60+ | 图表绘制 |
| **日志** | 40+ | 日志记录 |
| **总计** | **190+** | **完整工具集** |

---

## 🚀 **使用建议**

### **1. 数据处理**
- 使用DataProcessor处理图像
- 应用数据增强
- 批量处理数据

### **2. 模型评估**
- 使用ModelEvaluator评估
- 计算多个指标
- 可视化结果

### **3. 日志记录**
- 使用Logger记录日志
- 保存到文件
- 便于调试

---

**创建时间**: 2026-03-23 01:06
**版本**: 3.0
**状态**: 🟢 额外补充内容
**Token使用**: 880,000+
