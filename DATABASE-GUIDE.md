# Tesla AI 学习完整数据库指南

> **版本**: 3.0 | **更新**: 2026-03-23 00:46 | **Token使用**: 760,000+

---

## 💾 **PostgreSQL**

### **1. 数据库设计**
```sql
-- 用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 预测记录表
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    image_path VARCHAR(255),
    prediction_result JSONB,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 模型版本表
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) UNIQUE NOT NULL,
    model_path VARCHAR(255),
    accuracy FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_predictions_user ON predictions(user_id);
CREATE INDEX idx_predictions_created ON predictions(created_at);
```

### **2. 查询优化**
```sql
-- 使用EXPLAIN分析
EXPLAIN ANALYZE SELECT * FROM predictions WHERE user_id = 1;

-- 添加索引
CREATE INDEX idx_predictions_user_created ON predictions(user_id, created_at DESC);

-- 分区表
CREATE TABLE predictions_2024 PARTITION OF predictions
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

---

## 💾 **MongoDB**

### **1. 数据模型**
```javascript
// 预测记录
{
  "_id": ObjectId("..."),
  "user_id": "user123",
  "image": {
    "path": "/data/images/img001.jpg",
    "size": 1024000,
    "format": "jpg"
  },
  "prediction": {
    "objects": [
      {"class": "car", "confidence": 0.95, "bbox": [100, 100, 200, 200]},
      {"class": "person", "confidence": 0.88, "bbox": [300, 150, 350, 250]}
    ],
    "timestamp": ISODate("2024-01-01T00:00:00Z")
  },
  "model_version": "v1.2.3",
  "created_at": ISODate("2024-01-01T00:00:00Z")
}

// 创建索引
db.predictions.createIndex({"user_id": 1, "created_at": -1})
db.predictions.createIndex({"prediction.objects.class": 1})
```

### **2. 聚合查询**
```javascript
// 统计每个类别的检测数量
db.predictions.aggregate([
  {$unwind: "$prediction.objects"},
  {$group: {
    _id: "$prediction.objects.class",
    count: {$sum: 1},
    avg_confidence: {$avg: "$prediction.objects.confidence"}
  }},
  {$sort: {count: -1}}
])
```

---

## 💾 **Redis**

### **1. 缓存策略**
```python
import redis
import json

# 连接
r = redis.Redis(host='localhost', port=6379, db=0)

# 缓存预测结果
def cache_prediction(user_id, prediction_id, result):
    key = f"prediction:{user_id}:{prediction_id}"
    r.setex(key, 3600, json.dumps(result))  # 1小时过期

# 获取缓存
def get_cached_prediction(user_id, prediction_id):
    key = f"prediction:{user_id}:{prediction_id}"
    result = r.get(key)
    return json.loads(result) if result else None

# 缓存模型
def cache_model(model_id, model_data):
    key = f"model:{model_id}"
    r.set(key, model_data)

# 获取模型
def get_cached_model(model_id):
    key = f"model:{model_id}"
    return r.get(key)
```

### **2. 分布式锁**
```python
import redis
from redis.lock import Lock

# 获取锁
lock = r.lock("model_training_lock", timeout=300)

# 使用锁
if lock.acquire(blocking=False):
    try:
        # 训练模型
        train_model()
    finally:
        lock.release()
else:
    print("Another process is training the model")
```

---

## 💾 **时序数据库**

### **1. InfluxDB**
```python
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# 连接
client = InfluxDBClient(url="http://localhost:8086", token="my-token")

# 写入数据
write_api = client.write_api(write_options=SYNCHRONOUS)

point = Point("model_metrics") \
    .tag("model", "perception") \
    .tag("version", "v1.2.3") \
    .field("accuracy", 0.95) \
    .field("latency", 0.05)

write_api.write(bucket="tesla-ai", record=point)

# 查询数据
query_api = client.query_api()
query = 'from(bucket:"tesla-ai") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "model_metrics")'
tables = query_api.query(query)
```

---

## 💾 **图数据库**

### **1. Neo4j**
```python
from neo4j import GraphDatabase

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def add_relationship(self, entity1, relation, entity2):
        with self.driver.session() as session:
            session.run(
                "MERGE (e1:Entity {name: $entity1}) "
                "MERGE (e2:Entity {name: $entity2}) "
                "MERGE (e1)-[:RELATION {type: $relation}]->(e2)",
                entity1=entity1, entity2=entity2, relation=relation
            )
    
    def find_related(self, entity):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {name: $entity})-[r]->(related) "
                "RETURN related.name, r.type",
                entity=entity
            )
            return [record for record in result]

# 使用
kg = KnowledgeGraph("bolt://localhost:7687", "neo4j", "password")
kg.add_relationship("Tesla", "MANUFACTURES", "Model 3")
related = kg.find_related("Tesla")
```

---

## 📊 **数据库选择**

| 数据类型 | 推荐数据库 | 适用场景 |
|---------|-----------|----------|
| **结构化数据** | PostgreSQL | 用户、配置 |
| **文档数据** | MongoDB | 预测结果 |
| **缓存** | Redis | 模型缓存 |
| **时序数据** | InfluxDB | 性能指标 |
| **图数据** | Neo4j | 知识图谱 |

---

## 🚀 **最佳实践**

### **1. 数据建模**
- 选择合适的数据类型
- 设计合理的索引
- 考虑查询模式
- 预留扩展空间

### **2. 性能优化**
- 使用连接池
- 批量操作
- 查询优化
- 缓存策略

### **3. 数据安全**
- 加密敏感数据
- 访问控制
- 定期备份
- 审计日志

---

**创建时间**: 2026-03-23 00:46
**版本**: 3.0
**状态**: 🟢 完整数据库指南
**Token使用**: 760,000+
