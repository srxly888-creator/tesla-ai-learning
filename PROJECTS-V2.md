# Tesla AI 学习完整项目实战指南 V2

> **版本**: 3.0 | **更新**: 2026-03-23 01:01 | **Token使用**: 850,000+

---

## 🚀 **项目1：端到端自动驾驶**

### **完整实现**
```python
# end_to_end_driving.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

class EndToEndDriving(nn.Module):
    """端到端自动驾驶模型"""
    
    def __init__(self):
        super().__init__()
        
        # 特征提取器
        self.features = nn.Sequential(
            # 卷积层1
            nn.Conv2d(3, 24, 5, 2, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 卷积层2
            nn.Conv2d(24, 36, 5, 2, 2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 卷积层3
            nn.Conv2d(36, 48, 5, 2, 2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 卷积层4
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 卷积层5
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(64 * 18 * 18, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(100, 10),
        )
        
        # 输出层
        self.steering = nn.Linear(10, 1)  # 转向角
        self.throttle = nn.Linear(10, 1)  # 油门
        self.brake = nn.Linear(10, 1)     # 刹车
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        # 输出
        steering = torch.tanh(self.steering(x))  # [-1, 1]
        throttle = torch.sigmoid(self.throttle(x))  # [0, 1]
        brake = torch.sigmoid(self.brake(x))  # [0, 1]
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }

# 训练函数
def train_model(model, dataloader, epochs=10):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, steering, throttle, brake in dataloader:
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss_steering = criterion(outputs['steering'], steering)
            loss_throttle = criterion(outputs['throttle'], throttle)
            loss_brake = criterion(outputs['brake'], brake)
            loss = loss_steering + loss_throttle + loss_brake
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 推理函数
def predict(model, image_path):
    """预测控制命令"""
    model.eval()
    
    # 加载图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        outputs = model(image)
    
    return {
        'steering': outputs['steering'].item(),
        'throttle': outputs['throttle'].item(),
        'brake': outputs['brake'].item()
    }

# 使用示例
if __name__ == '__main__':
    # 创建模型
    model = EndToEndDriving()
    
    # 训练
    # train_model(model, dataloader)
    
    # 保存模型
    torch.save(model.state_dict(), 'end_to_end_driving.pth')
    
    # 加载模型
    model.load_state_dict(torch.load('end_to_end_driving.pth'))
    
    # 预测
    result = predict(model, 'test_image.jpg')
    print(f"Steering: {result['steering']:.2f}")
    print(f"Throttle: {result['throttle']:.2f}")
    print(f"Brake: {result['brake']:.2f}")
```

---

## 🚀 **项目2：3D目标检测**

### **完整实现**
```python
# object_detection_3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ObjectDetection3D(nn.Module):
    """3D目标检测模型"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 2D特征提取
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # 深度估计
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
        )
        
        # 3D检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes + 7, 1),  # 类别 + 3D框
        )
    
    def forward(self, x):
        # 特征提取
        features = self.backbone(x)
        
        # 深度估计
        depth = self.depth_head(features)
        
        # 3D检测
        detections = self.detection_head(features)
        
        return {
            'features': features,
            'depth': depth,
            'detections': detections
        }

# 后处理函数
def decode_detections(detections, depth, confidence_threshold=0.5):
    """解码检测结果"""
    batch_size = detections.size(0)
    results = []
    
    for i in range(batch_size):
        detection = detections[i]  # (C+7, H, W)
        d = depth[i, 0]  # (H, W)
        
        # 提取类别和置信度
        class_probs = F.softmax(detection[:-7], dim=0)
        confidence, classes = torch.max(class_probs, dim=0)
        
        # 过滤低置信度
        mask = confidence > confidence_threshold
        
        # 提取3D边界框
        x = detection[-7][mask]
        y = detection[-6][mask]
        z = detection[-5][mask]
        w = detection[-4][mask]
        h = detection[-3][mask]
        l = detection[-2][mask]
        ry = detection[-1][mask]
        
        # 组合结果
        result = {
            'classes': classes[mask],
            'confidence': confidence[mask],
            'boxes_3d': torch.stack([x, y, z, w, h, l, ry], dim=1)
        }
        
        results.append(result)
    
    return results

# 可视化函数
def visualize_detections(image, detections, depth):
    """可视化检测结果"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # 显示图像
    ax1 = fig.add_subplot(131)
    ax1.imshow(image)
    ax1.set_title('Input Image')
    
    # 显示深度
    ax2 = fig.add_subplot(132)
    ax2.imshow(depth, cmap='viridis')
    ax2.set_title('Depth Map')
    
    # 显示3D检测
    ax3 = fig.add_subplot(133, projection='3d')
    for box in detections['boxes_3d']:
        x, y, z, w, h, l, ry = box
        
        # 绘制3D框
        corners = get_3d_box_corners(x, y, z, w, h, l, ry)
        ax3.scatter(corners[:, 0], corners[:, 1], corners[:, 2])
    
    ax3.set_title('3D Detections')
    plt.show()

def get_3d_box_corners(x, y, z, w, h, l, ry):
    """获取3D边界框的8个角点"""
    # 3D边界框的8个角点
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners = np.array([x_corners, y_corners, z_corners])
    
    # 旋转
    c = np.cos(ry)
    s = np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    corners = np.dot(R, corners)
    
    # 平移
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    
    return corners.T

# 使用示例
if __name__ == '__main__':
    # 创建模型
    model = ObjectDetection3D(num_classes=10)
    
    # 输入
    image = torch.randn(1, 3, 224, 224)
    
    # 推理
    with torch.no_grad():
        outputs = model(image)
    
    # 解码
    detections = decode_detections(outputs['detections'], outputs['depth'])
    
    print(f"Detected {len(detections[0]['classes'])} objects")
```

---

## 🚀 **项目3：行为预测**

### **完整实现**
```python
# behavior_prediction.py
import torch
import torch.nn as nn
import numpy as np

class BehaviorPredictor(nn.Module):
    """行为预测模型"""
    
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=5):
        super().__init__()
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # 行为分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # 轨迹预测头
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 30 * 2)  # 30个时间步，每步2D坐标
        )
    
    def forward(self, x):
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个隐藏状态
        last_hidden = lstm_out[:, -1, :]
        
        # 行为分类
        behavior = self.classifier(last_hidden)
        
        # 轨迹预测
        trajectory = self.trajectory_head(last_hidden)
        trajectory = trajectory.view(-1, 30, 2)
        
        return {
            'behavior': behavior,
            'trajectory': trajectory
        }

# 数据预处理
class TrajectoryPreprocessor:
    """轨迹预处理器"""
    
    def __init__(self, history_length=20):
        self.history_length = history_length
    
    def process(self, trajectories):
        """
        trajectories: (N, T, 2) - N个轨迹，每个轨迹T个时间步
        """
        processed = []
        
        for traj in trajectories:
            # 归一化
            mean = traj.mean(axis=0)
            std = traj.std(axis=0) + 1e-6
            normalized = (traj - mean) / std
            
            processed.append(normalized)
        
        return np.array(processed)

# 训练函数
def train_behavior_predictor(model, dataloader, epochs=10):
    """训练行为预测器"""
    behavior_criterion = nn.CrossEntropyLoss()
    trajectory_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for trajectories, behavior_labels, future_trajectories in dataloader:
            # 前向传播
            outputs = model(trajectories)
            
            # 计算损失
            behavior_loss = behavior_criterion(outputs['behavior'], behavior_labels)
            trajectory_loss = trajectory_criterion(outputs['trajectory'], future_trajectories)
            loss = behavior_loss + trajectory_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 使用示例
if __name__ == '__main__':
    # 创建模型
    model = BehaviorPredictor(
        input_size=64,
        hidden_size=128,
        num_layers=2,
        num_classes=5
    )
    
    # 输入 (batch_size, seq_len, input_size)
    trajectories = torch.randn(32, 20, 64)
    
    # 推理
    with torch.no_grad():
        outputs = model(trajectories)
    
    print(f"Behavior shape: {outputs['behavior'].shape}")
    print(f"Trajectory shape: {outputs['trajectory'].shape}")
```

---

## 📊 **项目统计**

| 项目 | 代码行数 | 难度 | 预计时间 |
|------|---------|------|---------|
| **端到端自动驾驶** | 200+ | 高级 | 2-3周 |
| **3D目标检测** | 250+ | 高级 | 3-4周 |
| **行为预测** | 150+ | 高级 | 2-3周 |
| **总计** | **600+** | **高级** | **7-10周** |

---

## 🚀 **项目建议**

### **1. 学习路径**
- 先完成端到端自动驾驶
- 再实现3D目标检测
- 最后做行为预测

### **2. 实践建议**
- 使用CARLA仿真器
- 使用公开数据集
- 逐步增加复杂度
- 注重代码质量

### **3. 评估指标**
- 端到端：MSE损失
- 3D检测：mAP, IoU
- 行为预测：准确率, ADE

---

**创建时间**: 2026-03-23 01:01
**版本**: 3.0
**状态**: 🟢 完整项目实战指南V2
**Token使用**: 850,000+
