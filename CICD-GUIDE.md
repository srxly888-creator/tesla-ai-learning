# Tesla AI 学习完整CI/CD指南

> **版本**: 3.0 | **更新**: 2026-03-23 00:43 | **Token使用**: 730,000+

---

## 🔄 **GitHub Actions**

### **1. 基础CI流程**
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### **2. 自动部署**
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          docker build -t tesla-ai:${{ github.sha }} .
      
      - name: Push to Registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push tesla-ai:${{ github.sha }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/tesla-ai tesla-ai=tesla-ai:${{ github.sha }}
```

---

## 🔄 **Jenkins**

### **1. Jenkinsfile**
```groovy
pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t tesla-ai .'
            }
        }
        
        stage('Test') {
            steps {
                sh 'pytest tests/ --cov=src'
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh 'docker push tesla-ai'
                sh 'kubectl apply -f k8s/'
            }
        }
    }
    
    post {
        always {
            junit 'test-results.xml'
            cobertura coberturaReportFile: 'coverage.xml'
        }
    }
}
```

### **2. 多分支流水线**
```groovy
pipeline {
    agent any
    
    stages {
        stage('Build') {
            parallel {
                stage('Python 3.11') {
                    steps {
                        sh 'python3.11 -m pytest'
                    }
                }
                stage('Python 3.10') {
                    steps {
                        sh 'python3.10 -m pytest'
                    }
                }
            }
        }
    }
}
```

---

## 🔄 **GitLab CI**

### **1. .gitlab-ci.yml**
```yaml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest tests/ --cov=src
  coverage: '/TOTAL.+?(\d+\%)$/'

build:
  stage: build
  image: docker:latest
  script:
    - docker build -t tesla-ai .
    - docker push tesla-ai
  only:
    - main

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl apply -f k8s/
  only:
    - main
```

---

## 🔄 **CircleCI**

### **1. .circleci/config.yml**
```yaml
version: 2.1

jobs:
  test:
    docker:
      - image: python:3.11
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: pip install -r requirements.txt
      - run:
          name: Run tests
          command: pytest tests/ --cov=src

workflows:
  version: 2
  test-and-deploy:
    jobs:
      - test
      - deploy:
          requires:
            - test
          filters:
            branches:
              only: main
```

---

## 📊 **CI/CD统计**

| 平台 | 功能 | 适用场景 |
|------|------|----------|
| **GitHub Actions** | ⭐⭐⭐⭐⭐ | GitHub项目 |
| **Jenkins** | ⭐⭐⭐⭐⭐ | 企业级 |
| **GitLab CI** | ⭐⭐⭐⭐⭐ | GitLab项目 |
| **CircleCI** | ⭐⭐⭐⭐ | 云原生 |

---

## 🚀 **最佳实践**

### **1. 测试自动化**
- 单元测试
- 集成测试
- 端到端测试
- 性能测试

### **2. 部署自动化**
- 自动构建
- 自动测试
- 自动部署
- 自动回滚

### **3. 监控自动化**
- 健康检查
- 性能监控
- 错误追踪
- 告警通知

---

**创建时间**: 2026-03-23 00:43
**版本**: 3.0
**状态**: 🟢 完整CI/CD指南
**Token使用**: 730,000+
