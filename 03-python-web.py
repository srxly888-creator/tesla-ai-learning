"""
Python Web 开发教程 - 第3章

涵盖内容：
- HTTP 基础
- Flask 框架
- Django 框架
- FastAPI 框架
- RESTful API
- 数据库操作
- 认证授权
- WebSocket
- 异步 Web
- 部署
"""

# HTTP 基础
import requests
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

# 简单的 HTTP 服务器
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello, HTTP!')
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"message": "Received", "data": post_data.decode()}
        self.wfile.write(json.dumps(response).encode())

# Flask 框架示例
"""
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# 模型定义
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120), unique=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

# 路由定义
@app.route('/')
def home():
    return jsonify({"message": "Welcome to Flask!"})

@app.route('/users', methods=['GET', 'POST'])
def users():
    if request.method == 'GET':
        users = User.query.all()
        return jsonify([{"id": u.id, "username": u.username} for u in users])
    elif request.method == 'POST':
        data = request.get_json()
        user = User(username=data['username'], email=data['email'])
        db.session.add(user)
        db.session.commit()
        return jsonify({"message": "User created", "id": user.id}), 201

@app.route('/users/<int:user_id>', methods=['GET', 'PUT', 'DELETE'])
def user_detail(user_id):
    user = User.query.get_or_404(user_id)
    if request.method == 'GET':
        return jsonify({"id": user.id, "username": user.username})
    elif request.method == 'PUT':
        data = request.get_json()
        user.username = data.get('username', user.username)
        user.email = data.get('email', user.email)
        db.session.commit()
        return jsonify({"message": "User updated"})
    elif request.method == 'DELETE':
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": "User deleted"})

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# 运行应用
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
"""

# Django 框架示例
"""
# models.py
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    published_date = models.DateField()
    isbn = models.CharField(max_length=13, unique=True)
    
    def __str__(self):
        return self.title

# views.py
from django.http import JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from .models import Book

def book_list(request):
    books = Book.objects.all()
    data = [{"id": b.id, "title": b.title, "author": b.author} for b in books]
    return JsonResponse(data, safe=False)

def book_detail(request, book_id):
    try:
        book = Book.objects.get(pk=book_id)
        data = {"id": book.id, "title": book.title, "author": book.author}
        return JsonResponse(data)
    except Book.DoesNotExist:
        raise Http404("Book not found")

@csrf_exempt
def book_create(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        book = Book.objects.create(
            title=data['title'],
            author=data['author'],
            published_date=data['published_date'],
            isbn=data['isbn']
        )
        return JsonResponse({"message": "Book created", "id": book.id})

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('books/', views.book_list, name='book_list'),
    path('books/<int:book_id>/', views.book_detail, name='book_detail'),
    path('books/create/', views.book_create, name='book_create'),
]
"""

# FastAPI 框架示例
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="User Management API", version="1.0.0")

# Pydantic 模型
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

# 依赖注入
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 模拟数据库
fake_db: dict[int, User] = {}
fake_users_db = {
    "user1": {"username": "user1", "password": "password1"},
    "user2": {"username": "user2", "password": "password2"},
}

# 路由
@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": "fake_token", "token_type": "bearer"}

@app.get("/users/", response_model=List[User])
async def read_users(skip: int = 0, limit: int = 10):
    users = list(fake_db.values())[skip : skip + limit]
    return users

@app.get("/users/{user_id}", response_model=User)
async def read_user(user_id: int):
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail="User not found")
    return fake_db[user_id]

@app.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    user_id = len(fake_db) + 1
    new_user = User(
        id=user_id,
        username=user.username,
        email=user.email,
        created_at=datetime.now()
    )
    fake_db[user_id] = new_user
    return new_user

@app.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, user: UserCreate):
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail="User not found")
    fake_db[user_id] = User(
        id=user_id,
        username=user.username,
        email=user.email,
        created_at=datetime.now()
    )
    return fake_db[user_id]

@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    if user_id not in fake_db:
        raise HTTPException(status_code=404, detail="User not found")
    del fake_db[user_id]
    return {"message": "User deleted"}
"""

# RESTful API 设计
"""
API 端点设计原则：

GET    /users         - 获取用户列表
GET    /users/{id}    - 获取单个用户
POST   /users         - 创建新用户
PUT    /users/{id}    - 更新用户
PATCH  /users/{id}    - 部分更新用户
DELETE /users/{id}    - 删除用户

状态码：
200 OK - 成功
201 Created - 资源创建成功
204 No Content - 成功但无返回内容
400 Bad Request - 请求错误
401 Unauthorized - 未授权
403 Forbidden - 无权限
404 Not Found - 资源不存在
500 Internal Server Error - 服务器错误
"""

# 数据库操作示例（SQLAlchemy）
"""
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 创建引擎
engine = create_engine('sqlite:///example.db')

# 创建基类
Base = declarative_base()

# 定义模型
class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    department = Column(String(100))
    salary = Column(Integer)

# 创建表
Base.metadata.create_all(engine)

# 创建会话
Session = sessionmaker(bind=engine)
session = Session()

# CRUD 操作
# Create
emp1 = Employee(name="Alice", department="IT", salary=50000)
session.add(emp1)
session.commit()

# Read
employees = session.query(Employee).all()
it_employees = session.query(Employee).filter_by(department="IT").all()
high_salary = session.query(Employee).filter(Employee.salary > 40000).all()

# Update
emp = session.query(Employee).filter_by(name="Alice").first()
emp.salary = 55000
session.commit()

# Delete
emp = session.query(Employee).filter_by(name="Alice").first()
session.delete(emp)
session.commit()

# 关闭会话
session.close()
"""

# 认证授权示例（JWT）
"""
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT 配置
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 使用示例
password = "mypassword"
hashed_password = get_password_hash(password)

# 验证密码
if verify_password("mypassword", hashed_password):
    print("Password correct!")

# 创建 JWT token
data = {"sub": "user1"}
access_token = create_access_token(data, timedelta(minutes=30))
print(f"Token: {access_token}")
"""

# WebSocket 示例
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client {client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} left")
"""

# 异步 Web（aiohttp）
"""
from aiohttp import web
import asyncio

async def handle(request):
    text = "Hello, aiohttp!"
    return web.Response(text=text)

async def handle_post(request):
    data = await request.json()
    return web.json_response({"received": data})

app = web.Application()
app.add_routes([
    web.get('/', handle),
    web.post('/data', handle_post),
])

if __name__ == '__main__':
    web.run_app(app, host='localhost', port=8080)
"""

# 部署示例（Docker）
"""
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]

# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
    depends_on:
      - db
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=mydb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
"""

# 性能优化
"""
1. 数据库优化
   - 使用连接池
   - 添加索引
   - 使用 ORM 的 select_related/prefetch_related

2. 缓存
   - Redis 缓存
   - 内存缓存
   - CDN 缓存

3. 异步处理
   - 使用 Celery 处理后台任务
   - 使用 WebSocket 实现实时更新

4. 代码优化
   - 避免N+1查询
   - 使用批量操作
   - 压缩响应
"""

# 安全最佳实践
"""
1. 认证授权
   - 使用 JWT 或 OAuth2
   - 实现角色权限控制
   - 定期刷新 token

2. 数据验证
   - 使用 Pydantic 验证输入
   - 参数化 SQL 查询
   - XSS 过滤

3. HTTPS
   - 强制使用 HTTPS
   - 配置 HSTS
   - 使用安全的 Cookie

4. 安全头
   - X-Frame-Options
   - X-Content-Type-Options
   - Content-Security-Policy
"""

print("=== Python Web 开发教程完成 ===")
