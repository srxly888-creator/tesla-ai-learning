# Tesla AI 学习完整前端指南

> **版本**: 3.0 | **更新**: 2026-03-23 00:48 | **Token使用**: 770,000+

---

## 🎨 **React前端**

### **1. 组件结构**
```
frontend/
├── src/
│   ├── components/
│   │   ├── ImageUpload.jsx      # 图像上传
│   │   ├── PredictionResult.jsx # 预测结果
│   │   └── ModelSelector.jsx    # 模型选择
│   ├── pages/
│   │   ├── Home.jsx             # 首页
│   │   ├── Predict.jsx          # 预测页面
│   │   └── Models.jsx           # 模型管理
│   ├── services/
│   │   └── api.js               # API调用
│   └── App.jsx
└── package.json
```

### **2. 核心组件**
```jsx
// ImageUpload.jsx
import React, { useState } from 'react';
import { Upload, message } from 'antd';

const ImageUpload = ({ onUpload }) => {
  const [loading, setLoading] = useState(false);

  const beforeUpload = (file) => {
    const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
    if (!isJpgOrPng) {
      message.error('只能上传 JPG/PNG 文件！');
    }
    const isLt10M = file.size / 1024 / 1024 < 10;
    if (!isLt10M) {
      message.error('图片大小不能超过 10MB！');
    }
    return isJpgOrPng && isLt10M;
  };

  const handleChange = (info) => {
    if (info.file.status === 'uploading') {
      setLoading(true);
      return;
    }
    if (info.file.status === 'done') {
      setLoading(false);
      onUpload(info.file.response);
    }
  };

  return (
    <Upload
      name="image"
      action="/api/v1/predict"
      beforeUpload={beforeUpload}
      onChange={handleChange}
      showUploadList={false}
    >
      <Button icon={<UploadOutlined />} loading={loading}>
        上传图片
      </Button>
    </Upload>
  );
};

export default ImageUpload;
```

---

## 🎨 **Vue前端**

### **1. 项目结构**
```
frontend/
├── src/
│   ├── components/
│   │   ├── ImageUpload.vue
│   │   ├── PredictionResult.vue
│   │   └── ModelSelector.vue
│   ├── views/
│   │   ├── Home.vue
│   │   ├── Predict.vue
│   │   └── Models.vue
│   ├── api/
│   │   └── index.js
│   └── App.vue
└── package.json
```

### **2. 核心组件**
```vue
<!-- ImageUpload.vue -->
<template>
  <div class="image-upload">
    <input
      type="file"
      @change="handleFileChange"
      accept="image/jpeg,image/png"
    />
    <div v-if="loading" class="loading">上传中...</div>
  </div>
</template>

<script>
export default {
  name: 'ImageUpload',
  data() {
    return {
      loading: false
    }
  },
  methods: {
    async handleFileChange(event) {
      const file = event.target.files[0];
      if (!file) return;

      this.loading = true;
      
      const formData = new FormData();
      formData.append('image', file);

      try {
        const response = await fetch('/api/v1/predict', {
          method: 'POST',
          body: formData
        });
        
        const result = await response.json();
        this.$emit('uploaded', result);
      } catch (error) {
        console.error('Upload failed:', error);
      } finally {
        this.loading = false;
      }
    }
  }
}
</script>

<style scoped>
.image-upload {
  padding: 20px;
  border: 2px dashed #ccc;
  border-radius: 8px;
}

.loading {
  margin-top: 10px;
  color: #1890ff;
}
</style>
```

---

## 🎨 **API集成**

### **1. Axios配置**
```javascript
// api.js
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
});

// 请求拦截器
api.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  error => Promise.reject(error)
);

// 响应拦截器
api.interceptors.response.use(
  response => response.data,
  error => {
    if (error.response?.status === 401) {
      // 跳转到登录页
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
```

### **2. API调用**
```javascript
// predictionApi.js
import api from './api';

export const predictionApi = {
  // 预测
  predict: async (image) => {
    const formData = new FormData();
    formData.append('image', image);
    
    return api.post('/api/v1/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  },

  // 获取模型列表
  getModels: async () => {
    return api.get('/api/v1/models');
  },

  // 获取预测历史
  getHistory: async (params) => {
    return api.get('/api/v1/predictions', { params });
  }
};
```

---

## 🎨 **状态管理**

### **1. Redux (React)**
```javascript
// store.js
import { configureStore } from '@reduxjs/toolkit';
import predictionReducer from './predictionSlice';

export const store = configureStore({
  reducer: {
    prediction: predictionReducer
  }
});

// predictionSlice.js
import { createSlice } from '@reduxjs/toolkit';

const predictionSlice = createSlice({
  name: 'prediction',
  initialState: {
    result: null,
    loading: false,
    error: null
  },
  reducers: {
    setLoading: (state, action) => {
      state.loading = action.payload;
    },
    setResult: (state, action) => {
      state.result = action.payload;
      state.loading = false;
    },
    setError: (state, action) => {
      state.error = action.payload;
      state.loading = false;
    }
  }
});

export const { setLoading, setResult, setError } = predictionSlice.actions;
export default predictionSlice.reducer;
```

### **2. Vuex (Vue)**
```javascript
// store.js
import { createStore } from 'vuex';

export default createStore({
  state: {
    result: null,
    loading: false,
    error: null
  },
  mutations: {
    SET_LOADING(state, loading) {
      state.loading = loading;
    },
    SET_RESULT(state, result) {
      state.result = result;
      state.loading = false;
    },
    SET_ERROR(state, error) {
      state.error = error;
      state.loading = false;
    }
  },
  actions: {
    async predict({ commit }, image) {
      commit('SET_LOADING', true);
      
      try {
        const formData = new FormData();
        formData.append('image', image);
        
        const response = await fetch('/api/v1/predict', {
          method: 'POST',
          body: formData
        });
        
        const result = await response.json();
        commit('SET_RESULT', result);
      } catch (error) {
        commit('SET_ERROR', error.message);
      }
    }
  }
});
```

---

## 🎨 **可视化组件**

### **1. 结果可视化**
```jsx
// PredictionResult.jsx
import React from 'react';
import { Card, Progress } from 'antd';

const PredictionResult = ({ result }) => {
  if (!result) return null;

  return (
    <Card title="预测结果">
      {result.objects.map((obj, index) => (
        <div key={index} style={{ marginBottom: 16 }}>
          <div>
            <strong>{obj.class}</strong>
          </div>
          <Progress
            percent={Math.round(obj.confidence * 100)}
            status="active"
          />
          <div>
            位置: [{obj.bbox.join(', ')}]
          </div>
        </div>
      ))}
    </Card>
  );
};

export default PredictionResult;
```

### **2. 图表可视化**
```jsx
// MetricsChart.jsx
import React from 'react';
import { Line } from 'react-chartjs-2';

const MetricsChart = ({ data }) => {
  const chartData = {
    labels: data.labels,
    datasets: [
      {
        label: '准确率',
        data: data.accuracy,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: '损失',
        data: data.loss,
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      }
    ]
  };

  return <Line data={chartData} />;
};

export default MetricsChart;
```

---

## 📊 **前端技术栈**

| 技术 | 用途 | 推荐度 |
|------|------|--------|
| **React** | UI框架 | ⭐⭐⭐⭐⭐ |
| **Vue** | UI框架 | ⭐⭐⭐⭐⭐ |
| **Ant Design** | UI组件库 | ⭐⭐⭐⭐⭐ |
| **Axios** | HTTP客户端 | ⭐⭐⭐⭐⭐ |
| **Redux** | 状态管理 | ⭐⭐⭐⭐ |
| **Chart.js** | 图表库 | ⭐⭐⭐⭐ |

---

## 🚀 **最佳实践**

### **1. 组件设计**
- 单一职责原则
- 可复用性
- 性能优化
- 类型检查

### **2. 状态管理**
- 合理拆分状态
- 避免过度设计
- 使用中间件
- 持久化存储

### **3. 性能优化**
- 代码分割
- 懒加载
- 虚拟滚动
- 缓存策略

---

**创建时间**: 2026-03-23 00:48
**版本**: 3.0
**状态**: 🟢 完整前端指南
**Token使用**: 770,000+
