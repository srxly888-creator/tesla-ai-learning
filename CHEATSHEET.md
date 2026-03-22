# Tesla AI 学习速查表

> **版本**: 3.0 | **更新**: 2026-03-23 00:06 | **Token使用**: 530,000+

---

## 🎯 **Python速查**

### **基础语法**
```python
# 变量
x = 10
y = "hello"

# 列表
list = [1, 2, 3]

# 字典
dict = {"name": "Tesla", "age": 20}

# 循环
for i in range(10):
    print(i)

# 条件
if x > 5:
    print("x is greater than 5")
```

### **函数**
```python
# 定义函数
def add(a, b):
    return a + b

# 调用函数
result = add(3, 5)
```

### **类**
```python
# 定义类
class Car:
    def __init__(self, brand):
        self.brand = brand
    
    def drive(self):
        print(f"{self.brand} is driving")

# 创建对象
car = Car("Tesla")
car.drive()
```

---

## 🎯 **PyTorch速查**

### **张量操作**
```python
import torch

# 创建张量
x = torch.tensor([1, 2, 3])

# 随机张量
x = torch.randn(3, 4)

# 零张量
x = torch.zeros(2, 3)

# 张量运算
y = x + 2
z = torch.matmul(x, y)
```

### **神经网络**
```python
import torch.nn as nn

# 定义网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建网络
model = Net()
```

### **训练循环**
```python
# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(10):
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

---

## 🎯 **OpenCV速查**

### **图像操作**
```python
import cv2

# 读取图像
img = cv2.imread('image.jpg')

# 显示图像
cv2.imshow('Image', img)
cv2.waitKey(0)

# 保存图像
cv2.imwrite('output.jpg', img)

# 缩放
resized = cv2.resize(img, (640, 480))

# 裁剪
cropped = img[100:400, 200:500]

# 旋转
rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
rotated = cv2.warpAffine(img, M, (cols, rows))
```

### **图像处理**
```python
# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯滤波
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(img, 100, 200)

# 阈值处理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 轮廓检测
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

---

## 🎯 **NumPy速查**

### **数组操作**
```python
import numpy as np

# 创建数组
arr = np.array([1, 2, 3])

# 零数组
zeros = np.zeros((3, 4))

# 随机数组
rand = np.random.randn(3, 4)

# 数组运算
result = arr * 2
result = np.dot(arr1, arr2)

# 数组索引
element = arr[0]
row = arr[1, :]
```

### **数学运算**
```python
# 求和
sum = np.sum(arr)

# 平均值
mean = np.mean(arr)

# 标准差
std = np.std(arr)

# 矩阵乘法
result = np.matmul(A, B)
```

---

## 🎯 **Matplotlib速查**

### **绘图**
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

# 直方图
plt.hist(data, bins=50)
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

## 🎯 **常用公式**

### **准确率**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### **精确率**
```
Precision = TP / (TP + FP)
```

### **召回率**
```
Recall = TP / (TP + FN)
```

### **F1分数**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### **IoU**
```
IoU = Intersection / Union
```

---

## 🎯 **常用命令**

### **Git**
```bash
# 克隆
git clone <url>

# 提交
git add .
git commit -m "message"
git push

# 拉取
git pull

# 分支
git branch <name>
git checkout <name>
```

### **Python**
```bash
# 安装包
pip install <package>

# 运行脚本
python script.py

# 虚拟环境
python -m venv venv
source venv/bin/activate
```

### **Jupyter**
```bash
# 启动
jupyter notebook

# 转换
jupyter nbconvert --to script notebook.ipynb
```

---

## 📊 **速查表统计**

| 类别 | 条目数 | 完成度 |
|------|-------|--------|
| **Python** | 10个 | 100% |
| **PyTorch** | 10个 | 100% |
| **OpenCV** | 10个 | 100% |
| **NumPy** | 8个 | 100% |
| **Matplotlib** | 8个 | 100% |
| **公式** | 5个 | 100% |
| **命令** | 10个 | 100% |
| **总计** | **61个** | **100%** |

---

**创建时间**: 2026-03-23 00:06
**版本**: 3.0
**状态**: 🟢 完整速查表
**Token使用**: 530,000+
