"""
Python 数据科学教程 - 第4章

涵盖内容：
- NumPy 数组操作
- Pandas 数据处理
- Matplotlib 可视化
- Scikit-learn 机器学习
- 数据清洗
- 特征工程
- 模型评估
- 深度学习基础
"""

# NumPy 数组操作
import numpy as np

# 创建数组
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 数组属性
print(f"Shape: {arr2d.shape}")
print(f"Dimensions: {arr2d.ndim}")
print(f"Size: {arr2d.size}")
print(f"Dtype: {arr2d.dtype}")

# 数组创建
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
random = np.random.rand(3, 3)
arange = np.arange(0, 10, 2)
linspace = np.linspace(0, 10, 5)
identity = np.eye(3)

print(f"Zeros:\n{zeros}")
print(f"Ones:\n{ones}")
print(f"Random:\n{random}")
print(f"Arange: {arange}")
print(f"Linspace: {linspace}")
print(f"Identity:\n{identity}")

# 数组运算
arr = np.array([1, 2, 3, 4, 5])
print(f"Add 10: {arr + 10}")
print(f"Multiply by 2: {arr * 2}")
print(f"Square: {arr ** 2}")
print(f"Square root: {np.sqrt(arr)}")
print(f"Exponential: {np.exp(arr)}")

# 数组索引和切片
arr = np.array([10, 20, 30, 40, 50])
print(f"First element: {arr[0]}")
print(f"Last element: {arr[-1]}")
print(f"Slice [1:4]: {arr[1:4]}")
print(f"Slice [::2]: {arr[::2]}")

# 2D 数组索引
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Element [1,2]: {arr2d[1, 2]}")
print(f"Row 1: {arr2d[1, :]}")
print(f"Column 1: {arr2d[:, 1]}")
print(f"Submatrix: {arr2d[0:2, 1:3]}")

# 数组操作
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"Concatenate: {np.concatenate([arr1, arr2])}")
print(f"Stack: {np.stack([arr1, arr2])}")
print(f"Reshape: {np.arange(6).reshape(2, 3)}")
print(f"Flatten: {arr2d.flatten()}")

# 统计函数
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Mean: {np.mean(arr)}")
print(f"Median: {np.median(arr)}")
print(f"Std: {np.std(arr)}")
print(f"Var: {np.var(arr)}")
print(f"Min: {np.min(arr)}")
print(f"Max: {np.max(arr)}")
print(f"Sum: {np.sum(arr)}")

# 线性代数
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"Matrix multiplication: {np.dot(A, B)}")
print(f"Element-wise: {A * B}")
print(f"Transpose: {A.T}")
print(f"Inverse: {np.linalg.inv(A)}")
print(f"Eigenvalues: {np.linalg.eig(A)[0]}")
print(f"Eigenvectors: {np.linalg.eig(A)[1]}")

# Pandas 数据处理
import pandas as pd

# 创建 Series
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
s3 = pd.Series({'x': 100, 'y': 200, 'z': 300})

print(f"Series 1:\n{s1}")
print(f"Series 2:\n{s2}")
print(f"Series 3:\n{s3}")

# 创建 DataFrame
df1 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
})

df2 = pd.DataFrame([
    {'Name': 'Alice', 'Age': 25, 'City': 'New York'},
    {'Name': 'Bob', 'Age': 30, 'City': 'London'},
    {'Name': 'Charlie', 'Age': 35, 'City': 'Paris'}
])

data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}
df3 = pd.DataFrame(data)

print(f"DataFrame 1:\n{df1}")
print(f"DataFrame 2:\n{df2}")
print(f"DataFrame 3:\n{df3}")

# 读取数据
"""
# 从 CSV 读取
df = pd.read_csv('data.csv')

# 从 Excel 读取
df = pd.read_excel('data.xlsx')

# 从 JSON 读取
df = pd.read_json('data.json')

# 从数据库读取
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query('SELECT * FROM users', conn)
"""

# 数据探索
print(f"Head:\n{df1.head()}")
print(f"Tail:\n{df1.tail()}")
print(f"Info:\n{df1.info()}")
print(f"Describe:\n{df1.describe()}")
print(f"Shape: {df1.shape}")
print(f"Columns: {df1.columns}")
print(f"Index: {df1.index}")
print(f"Dtypes: {df1.dtypes}")

# 数据选择
print(f"Column 'Name':\n{df1['Name']}")
print(f"Column with dot notation:\n{df1.Name}")
print(f"Multiple columns:\n{df1[['Name', 'Age']]}")
print(f"Row by index (loc):\n{df1.loc[1]}")
print(f"Row by position (iloc):\n{df1.iloc[1]}")
print(f"Slice rows:\n{df1.loc[0:2]}")
print(f"Conditional:\n{df1[df1['Age'] > 30]}")

# 数据清洗
# 处理缺失值
df_with_nan = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, np.nan, 30, 40, np.nan]
})

print(f"With NaN:\n{df_with_nan}")

# 删除缺失值
print(f"Drop NaN:\n{df_with_nan.dropna()}")

# 填充缺失值
print(f"Fill NaN with 0:\n{df_with_nan.fillna(0)}")
print(f"Forward fill:\n{df_with_nan.ffill()}")
print(f"Backward fill:\n{df_with_nan.bfill()}")

# 处理重复值
df_with_duplicates = pd.DataFrame({
    'A': [1, 2, 2, 3, 3, 3],
    'B': [10, 20, 20, 30, 30, 30]
})

print(f"With duplicates:\n{df_with_duplicates}")
print(f"Drop duplicates:\n{df_with_duplicates.drop_duplicates()}")

# 数据转换
df1['Age_Double'] = df1['Age'] * 2
df1['Name_Length'] = df1['Name'].str.len()
df1['Is_Adult'] = df1['Age'] >= 30

print(f"Transformed:\n{df1}")

# 数据聚合
print(f"Mean age: {df1['Age'].mean()}")
print(f"Group by City:\n{df1.groupby('City')['Age'].mean()}")

# 数据合并
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'A': [7, 8, 9], 'C': [10, 11, 12]})

print(f"Merge:\n{pd.merge(df1, df2, on='A')}")
print(f"Concat:\n{pd.concat([df1, df2])}")

# 数据排序
print(f"Sort by Age:\n{df1.sort_values('Age', ascending=False)}")

# 数据应用函数
df1['Name_Upper'] = df1['Name'].apply(str.upper)
df1['Age_Squared'] = df1['Age'].apply(lambda x: x ** 2)

print(f"With applied functions:\n{df1}")

# Matplotlib 可视化
import matplotlib.pyplot as plt

# 创建图表
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 折线图
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine and Cosine Functions')
plt.legend()
plt.grid(True)
# plt.savefig('line_plot.png')
# plt.show()

# 散点图
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = np.random.rand(100) * 1000

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random Scatter Plot')
plt.colorbar()
# plt.show()

# 直方图
data = np.random.randn(1000)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
# plt.show()

# 箱线图
data = [np.random.randn(100) + i for i in range(5)]
plt.figure(figsize=(10, 6))
plt.boxplot(data)
plt.xlabel('Group')
plt.ylabel('Value')
plt.title('Box Plot')
# plt.show()

# 饼图
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Pie Chart')
# plt.show()

# 子图
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x, y1)
axes[0, 0].set_title('Sine')

axes[0, 1].plot(x, y2)
axes[0, 1].set_title('Cosine')

axes[1, 0].scatter(x, y1, alpha=0.5)
axes[1, 0].set_title('Scatter')

axes[1, 1].hist(data, bins=20)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
# plt.show()

# Scikit-learn 机器学习
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 逻辑回归
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

# 支持向量机
svm = SVC(kernel='rbf')
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

# 决策树
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")

# 随机森林
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# 分类报告
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_rf)}")

# 特征重要性
print(f"Feature Importance:\n{rf.feature_importances_}")

# 回归示例
X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

lr_reg = LinearRegression()
lr_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = lr_reg.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Coefficient: {lr_reg.coef_[0]:.4f}")
print(f"Intercept: {lr_reg.intercept_:.4f}")

# 特征工程
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# 特征选择
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
print(f"Selected features shape: {X_selected.shape}")

# 主成分分析
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# 交叉验证
from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.4f}")

# 模型调参
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# 深度学习基础（TensorFlow/Keras）
"""
import tensorflow as tf
from tensorflow import keras

# 简单神经网络
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# 预测
predictions = model.predict(X_test)

# 卷积神经网络（CNN）
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 循环神经网络（RNN）
model = keras.Sequential([
    keras.layers.SimpleRNN(64, input_shape=(10, 1)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# LSTM
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(10, 1)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
"""

print("=== Python 数据科学教程完成 ===")
