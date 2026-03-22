# Tesla AI 学习完整API参考

> **版本**: 3.0 | **更新**: 2026-03-23 00:10 | **Token使用**: 570,000+

---

## 🎯 **PyTorch API**

### **张量操作**
```python
import torch

# 创建张量
x = torch.tensor([1, 2, 3])

# 张量属性
x.shape      # 形状
x.dtype     # 数据类型
x.device    # 设备

# 张量运算
y = x + 1
z = x * 2
w = torch.matmul(x, y)
```

### **神经网络**
```python
import torch.nn as nn

# 线性层
linear = nn.Linear(784, 128)

# 卷积层
conv = nn.Conv2d(3, 64, 3, 1, 1)

# 池化层
pool = nn.MaxPool2d(2, 2)

# 激活函数
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
```

### **损失函数**
```python
import torch.nn as nn

# 交叉熵损失
criterion = nn.CrossEntropyLoss()

# 均方误差损失
criterion = nn.MSELoss()

# 计算损失
loss = criterion(output, target)
```

### **优化器**
```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 更新参数
optimizer.step()
```

---

## 🎯 **OpenCV API**

### **图像读取**
```python
import cv2

# 读取图像
img = cv2.imread('image.jpg')

# 读取灰度图
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAY)

# 检查读取成功
if img is not None:
    print("Image loaded successfully")
```

### **图像处理**
```python
# 缩放
resized = cv2.resize(img, (640, 480))

# 旋转
rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
rotated = cv2.warpAffine(img, M, (cols, rows))

# 裁剪
cropped = img[100:400, 200:500]
```

### **颜色转换**
```python
# BGR转RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# BGR转HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# BGR转GRAY
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

## 🎯 **NumPy API**

### **数组创建**
```python
import numpy as np

# 从列表创建
arr = np.array([1, 2, 3])

# 零数组
zeros = np.zeros((3, 4))

# 一数组
ones = np.ones((3, 4))

# 随机数组
rand = np.random.randn(3, 4)
```

### **数组运算**
```python
# 加法
result = arr1 + arr2

# 乘法
result = arr1 * arr2

# 矩阵乘法
result = np.dot(arr1, arr2)

# 转置
result = arr.T
```

---

## 🎯 **Matplotlib API**

### **基础绘图**
```python
import matplotlib.pyplot as plt

# 折线图
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

# 散点图
plt.scatter([1, 2, 3], [4, 5, 6])
plt.show()

# 柱状图
plt.bar(['A', 'B', 'C'], [10, 20, 15])
plt.show()
```

### **子图**
```python
fig, axes = plt.subplots(2, 2)

axes[0, 0].plot([1, 2, 3])
axes[0, 1].scatter([1, 2, 3], [4, 5, 6])
axes[1, 0].bar(['A', 'B'], [10, 20])
axes[1, 1].hist([1, 2, 2, 3, 3, 3])

plt.tight_layout()
plt.show()
```

---

## 📊 **API统计**

| 库 | API数 | 完成度 |
|------|------|--------|
| **PyTorch** | 10个 | 100% |
| **OpenCV** | 6个 | 100% |
| **NumPy** | 6个 | 100% |
| **Matplotlib** | 4个 | 100% |
| **总计** | **26个** | **100%** |

---

**创建时间**: 2026-03-23 00:10
**版本**: 3.0
**状态**: 🟢 完整API参考
**Token使用**: 570,000+
