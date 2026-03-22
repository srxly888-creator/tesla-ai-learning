# 项目1：基础自动驾驶模拟器 (Autonomous Driving Simulator)

## 项目简介
这是一个基础的自动驾驶决策模拟系统，模拟Tesla自动驾驶的核心决策逻辑。

## 学习目标
- 理解自动驾驶的感知-决策-控制循环
- 掌握传感器数据处理基础
- 学习路径规划和决策算法
- 实践强化学习在自动驾驶中的应用

## 技术栈
- Python 3.8+
- OpenCV - 图像处理
- NumPy - 数值计算
- PyTorch - 深度学习
- Gym - 强化学习环境
- Matplotlib - 可视化

## 项目结构
```
autonomous-driving-simulator/
├── src/
│   ├── sensors/          # 传感器模块
│   ├── perception/       # 感知模块
│   ├── planning/         # 规划模块
│   ├── control/          # 控制模块
│   └── simulation/       # 仿真环境
├── config/
│   ├── vehicle.yaml      # 车辆参数
│   ├── sensors.yaml      # 传感器配置
│   └── environment.yaml  # 环境参数
├── tests/
├── data/
├── models/
└── docs/
```

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 运行仿真
python src/main.py --config config/vehicle.yaml

# 运行测试
pytest tests/
```

## 核心功能
1. **传感器仿真**: 模拟摄像头、雷达、超声波传感器
2. **目标检测**: 基于深度学习的物体识别
3. **路径规划**: A*算法 + 动态窗口法
4. **决策控制**: PID控制器 + MPC控制

## 学习路径
1. Week 1-2: 理解传感器数据和预处理
2. Week 3-4: 实现基础的感知算法
3. Week 5-6: 路径规划和决策逻辑
4. Week 7-8: 控制算法和仿真测试

## 参考资料
- [Tesla Autopilot介绍](https://www.tesla.com/autopilot)
- [Udacity自动驾驶课程](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd0013)
