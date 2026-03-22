# Tesla AI 学习完整命令参考

> **版本**: 3.0 | **更新**: 2026-03-23 00:07 | **Token使用**: 540,000+

---

## 🎯 **Claude Code 命令**

### **基础命令**
```bash
# 启动
claude

# 帮助
claude --help

# 版本
claude --version
```

### **模式切换**
```bash
# Plan Mode
Shift + Tab

# Accept Edits
Shift + Tab

# Review Edits
Shift + Tab
```

### **配置**
```bash
# 设置API Key
export ANTHROPIC_API_KEY="sk-ant-..."

# 查看配置
claude config

# 设置模型
claude config set model claude-3-5-sonnet-20241022

# 设置Token限制
claude config set max_tokens 2000
```

---

## 🎯 **Git 命令**

### **基础操作**
```bash
# 初始化
git init

# 克隆
git clone <url>

# 状态
git status

# 添加
git add .

# 提交
git commit -m "message"

# 推送
git push

# 拉取
git pull
```

### **分支操作**
```bash
# 创建分支
git branch <name>

# 切换分支
git checkout <name>

# 合并分支
git merge <name>

# 删除分支
git branch -d <name>
```

### **远程操作**
```bash
# 添加远程
git remote add origin <url>

# 查看远程
git remote -v

# 推送到远程
git push -u origin master

# 拉取远程
git pull origin master
```

---

## 🎯 **Python 命令**

### **包管理**
```bash
# 安装包
pip install <package>

# 卸载包
pip uninstall <package>

# 列出包
pip list

# 冻结依赖
pip freeze > requirements.txt

# 安装依赖
pip install -r requirements.txt
```

### **虚拟环境**
```bash
# 创建虚拟环境
python -m venv venv

# 激活（Linux/Mac）
source venv/bin/activate

# 激活（Windows）
venv\Scripts\activate

# 退出
deactivate
```

### **运行脚本**
```bash
# 运行Python
python script.py

# 运行模块
python -m module

# 交互模式
python -i script.py
```

---

## 🎯 **PyTorch 命令**

### **张量操作**
```python
# 创建张量
x = torch.tensor([1, 2, 3])

# 查看形状
x.shape

# 查看类型
x.dtype

# 转换设备
x.to('cuda')
x.to('cpu')
```

### **模型操作**
```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model.load_state_dict(torch.load('model.pth'))

# 导出ONNX
torch.onnx.export(model, dummy_input, 'model.onnx')
```

---

## 🎯 **OpenCV 命令**

### **图像操作**
```python
# 读取
img = cv2.imread('image.jpg')

# 显示
cv2.imshow('Image', img)
cv2.waitKey(0)

# 保存
cv2.imwrite('output.jpg', img)

# 释放
cv2.destroyAllWindows()
```

### **视频操作**
```python
# 打开视频
cap = cv2.VideoCapture('video.mp4')

# 读取帧
ret, frame = cap.read()

# 释放
cap.release()
```

---

## 🎯 **Jupyter 命令**

### **启动**
```bash
# 启动Notebook
jupyter notebook

# 启动Lab
jupyter lab

# 指定端口
jupyter notebook --port 8889
```

### **转换**
```bash
# 转为Python
jupyter nbconvert --to script notebook.ipynb

# 转为HTML
jupyter nbconvert --to html notebook.ipynb

# 转为PDF
jupyter nbconvert --to pdf notebook.ipynb
```

---

## 🎯 **Docker 命令**

### **基础操作**
```bash
# 构建镜像
docker build -t name:tag .

# 运行容器
docker run -it name:tag

# 查看容器
docker ps

# 停止容器
docker stop <container_id>

# 删除容器
docker rm <container_id>
```

### **镜像操作**
```bash
# 查看镜像
docker images

# 删除镜像
docker rmi <image_id>

# 推送镜像
docker push name:tag
```

---

## 🎯 **Linux 命令**

### **文件操作**
```bash
# 列出文件
ls -la

# 创建目录
mkdir -p path/to/dir

# 删除文件
rm file.txt

# 删除目录
rm -rf directory

# 复制
cp source dest

# 移动
mv source dest
```

### **查找**
```bash
# 查找文件
find . -name "*.py"

# 搜索内容
grep -r "pattern" .

# 查看文件
cat file.txt
head -n 20 file.txt
tail -n 20 file.txt
```

### **进程管理**
```bash
# 查看进程
ps aux

# 杀死进程
kill -9 <pid>

# 查看端口
lsof -i :8000
```

---

## 🎯 **系统监控**

### **资源监控**
```bash
# CPU使用
top

# 内存使用
free -h

# 磁盘使用
df -h

# GPU使用
nvidia-smi
```

---

## 📊 **命令统计**

| 类别 | 命令数 | 完成度 |
|------|-------|--------|
| **Claude Code** | 10个 | 100% |
| **Git** | 15个 | 100% |
| **Python** | 12个 | 100% |
| **PyTorch** | 8个 | 100% |
| **OpenCV** | 8个 | 100% |
| **Jupyter** | 6个 | 100% |
| **Docker** | 8个 | 100% |
| **Linux** | 15个 | 100% |
| **总计** | **82个** | **100%** |

---

**创建时间**: 2026-03-23 00:07
**版本**: 3.0
**状态**: 🟢 完整命令参考
**Token使用**: 540,000+
