# Tesla AI 学习完整后端指南

> **版本**: 3.0 | **更新**: 2026-03-23 00:49 | **Token使用**: 780,000+

---

## 🔧 **FastAPI后端**

### **1. 项目结构**
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # 主应用
│   ├── config.py            # 配置
│   ├── models/              # 数据模型
│   │   ├── user.py
│   │   └── prediction.py
│   ├── routers/             # 路由
│   │   ├── auth.py
│   │   ├── predict.py
│   │   └── models.py
│   ├── services/            # 业务逻辑
│   │   ├── auth_service.py
│   │   └── predict_service.py
│   └── utils/               # 工具函数
│       ├── security.py
│       └── logging.py
├── tests/
├── requirements.txt
└── Dockerfile
```

### **2. 主应用**
```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, predict, models
from app.config import settings

app = FastAPI(
    title="Tesla AI API",
    description="API for Tesla AI Learning",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["predict"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])

@app.get("/")
async def root():
    return {"message": "Tesla AI API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### **3. 认证路由**
```python
# routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from app.services.auth_service import AuthService
from app.models.user import User, Token

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/register", response_model=User)
async def register(user: User):
    """用户注册"""
    return await AuthService.create_user(user)

@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """用户登录"""
    user = await AuthService.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = AuthService.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=User)
async def read_users_me(token: str = Depends(oauth2_scheme)):
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials"
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await AuthService.get_user(username)
    if user is None:
        raise credentials_exception
    return user
```

### **4. 预测路由**
```python
# routers/predict.py
from fastapi import APIRouter, UploadFile, File, Depends
from app.services.predict_service import PredictService
from app.models.prediction import PredictionResult

router = APIRouter()

@router.post("/", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    model_id: str = None,
    token: str = Depends(oauth2_scheme)
):
    """图像预测"""
    # 验证文件类型
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # 读取图像
    image_bytes = await file.read()
    
    # 预测
    result = await PredictService.predict(image_bytes, model_id)
    
    return result

@router.get("/history")
async def get_history(
    skip: int = 0,
    limit: int = 10,
    token: str = Depends(oauth2_scheme)
):
    """获取预测历史"""
    user = await get_current_user(token)
    history = await PredictService.get_history(user.id, skip, limit)
    return history
```

### **5. 模型路由**
```python
# routers/models.py
from fastapi import APIRouter, HTTPException
from app.services.model_service import ModelService
from app.models.model import Model, ModelVersion

router = APIRouter()

@router.get("/", response_model=List[Model])
async def list_models():
    """获取模型列表"""
    return await ModelService.list_models()

@router.post("/", response_model=Model)
async def create_model(model: Model):
    """创建模型"""
    return await ModelService.create_model(model)

@router.get("/{model_id}", response_model=Model)
async def get_model(model_id: str):
    """获取模型详情"""
    model = await ModelService.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.get("/{model_id}/versions", response_model=List[ModelVersion])
async def list_model_versions(model_id: str):
    """获取模型版本列表"""
    return await ModelService.list_model_versions(model_id)
```

---

## 🔧 **业务逻辑**

### **1. 预测服务**
```python
# services/predict_service.py
import torch
from PIL import Image
import io
from app.models.prediction import PredictionResult

class PredictService:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    async def load_model(self, model_id: str):
        """加载模型"""
        model_path = f"models/{model_id}.pth"
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
    
    async def predict(self, image_bytes: bytes, model_id: str = None) -> PredictionResult:
        """预测"""
        # 加载模型
        if not self.model or (model_id and self.model.id != model_id):
            await self.load_model(model_id or "default")
        
        # 预处理
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))
        
        # 后处理
        result = self.postprocess(output)
        
        return result
    
    def preprocess(self, image):
        """预处理"""
        # 调整大小
        image = image.resize((224, 224))
        
        # 转换为张量
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0)
    
    def postprocess(self, output):
        """后处理"""
        # 获取预测结果
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return PredictionResult(
            prediction=predicted.item(),
            confidence=confidence.item(),
            objects=[]  # 如果是目标检测，返回物体列表
        )
```

### **2. 认证服务**
```python
# services/auth_service.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.models.user import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    SECRET_KEY = "your-secret-key"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """获取密码哈希"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        """创建访问令牌"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, AuthService.SECRET_KEY, algorithm=AuthService.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    async def authenticate_user(username: str, password: str) -> Optional[User]:
        """认证用户"""
        user = await AuthService.get_user(username)
        if not user:
            return False
        if not AuthService.verify_password(password, user.hashed_password):
            return False
        return user
```

---

## 📊 **后端技术栈**

| 技术 | 用途 | 推荐度 |
|------|------|--------|
| **FastAPI** | Web框架 | ⭐⭐⭐⭐⭐ |
| **Django** | Web框架 | ⭐⭐⭐⭐ |
| **PostgreSQL** | 数据库 | ⭐⭐⭐⭐⭐ |
| **Redis** | 缓存 | ⭐⭐⭐⭐⭐ |
| **Celery** | 任务队列 | ⭐⭐⭐⭐ |
| **Docker** | 容器化 | ⭐⭐⭐⭐⭐ |

---

## 🚀 **最佳实践**

### **1. API设计**
- RESTful风格
- 版本控制
- 统一响应格式
- 错误处理

### **2. 安全性**
- 认证授权
- 输入验证
- SQL注入防护
- XSS防护

### **3. 性能优化**
- 数据库优化
- 缓存策略
- 异步处理
- 连接池

---

**创建时间**: 2026-03-23 00:49
**版本**: 3.0
**状态**: 🟢 完整后端指南
**Token使用**: 780,000+
