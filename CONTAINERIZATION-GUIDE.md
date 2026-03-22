# Tesla AI 学习完整容器化指南

> **版本**: 3.0 | **更新**: 2026-03-23 00:44 | **Token使用**: 740,000+

---

## 🐳 **Docker基础**

### **1. Dockerfile**
```dockerfile
# 基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **2. 多阶段构建**
```dockerfile
# 构建阶段
FROM python:3.11 as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 运行阶段
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🐳 **Docker Compose**

### **1. docker-compose.yml**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.pth
    volumes:
      - ./model.pth:/app/model.pth
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### **2. 启动服务**
```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down
```

---

## 🐳 **Kubernetes**

### **1. Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tesla-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tesla-ai
  template:
    metadata:
      labels:
        app: tesla-ai
    spec:
      containers:
      - name: api
        image: tesla-ai:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### **2. Service**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: tesla-ai-service
spec:
  selector:
    app: tesla-ai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### **3. ConfigMap**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tesla-ai-config
data:
  MODEL_PATH: "/app/model.pth"
  LOG_LEVEL: "INFO"
```

### **4. Secret**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: tesla-ai-secret
type: Opaque
data:
  API_KEY: <base64-encoded-key>
```

---

## 🐳 **Helm**

### **1. Chart.yaml**
```yaml
apiVersion: v2
name: tesla-ai
description: Tesla AI Deployment
version: 1.0.0
appVersion: "1.0.0"
```

### **2. values.yaml**
```yaml
replicaCount: 3

image:
  repository: tesla-ai
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80

resources:
  requests:
    cpu: 1
    memory: 2Gi
  limits:
    cpu: 2
    memory: 4Gi
```

### **3. 部署**
```bash
# 安装
helm install tesla-ai ./helm/

# 升级
helm upgrade tesla-ai ./helm/

# 回滚
helm rollback tesla-ai 1

# 卸载
helm uninstall tesla-ai
```

---

## 📊 **容器化统计**

| 工具 | 功能 | 复杂度 |
|------|------|--------|
| **Docker** | 容器运行 | ⭐⭐ |
| **Docker Compose** | 多容器编排 | ⭐⭐⭐ |
| **Kubernetes** | 集群编排 | ⭐⭐⭐⭐⭐ |
| **Helm** | K8s包管理 | ⭐⭐⭐⭐ |

---

## 🚀 **最佳实践**

### **1. 镜像优化**
- 使用多阶段构建
- 最小化基础镜像
- 缓存依赖层
- 使用.dockerignore

### **2. 安全性**
- 非root用户运行
- 扫描镜像漏洞
- 使用secrets管理
- 网络策略

### **3. 可观测性**
- 健康检查
- 日志收集
- 指标监控
- 分布式追踪

---

**创建时间**: 2026-03-23 00:44
**版本**: 3.0
**状态**: 🟢 完整容器化指南
**Token使用**: 740,000+
