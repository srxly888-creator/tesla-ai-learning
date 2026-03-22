# 项目3：Tesla智能充电调度器 (Smart Charging Scheduler)

## 项目简介
基于电价、电网负载和用户习惯的Tesla智能充电调度系统，优化充电时间和成本。

## 学习目标
- 掌握优化算法（线性规划、动态规划）
- 理解电网负载均衡原理
- 学习时间序列预测
- 实践能源管理系统开发

## 技术栈
- Python 3.8+
- PuLP / CVXPY（优化）
- OR-Tools（调度）
- Pandas / NumPy
- FastAPI（API服务）
- Redis（缓存）

## 项目结构
```
smart-charging-scheduler/
├── src/
│   ├── optimizer/         # 优化算法
│   │   ├── linear_program.py
│   │   ├── dynamic_program.py
│   │   └── genetic_algorithm.py
│   ├── forecaster/        # 预测模块
│   │   ├── price_forecaster.py
│   │   └── load_forecaster.py
│   ├── scheduler/         # 调度器
│   │   ├── charger.py
│   │   └── manager.py
│   └── api/               # API服务
│       ├── routes.py
│       └── models.py
├── config/
│   ├── optimizer.yaml
│   ├── api.yaml
│   └── tesla_api.yaml
├── tests/
├── data/
│   ├── electricity_prices/
│   └── grid_load/
└── docs/
```

## 核心功能
1. **电价预测**: 预测未来24小时电价
2. **负载预测**: 预测电网负载
3. **充电优化**: 最小化充电成本
4. **智能调度**: 考虑用户需求和电网约束
5. **API服务**: RESTful API接口

## 优化目标
- 最小化充电成本
- 避免电网高峰时段
- 满足用户充电需求
- 延长电池寿命

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 启动API服务
uvicorn src.api.routes:app --reload

# 运行调度器
python src/scheduler/manager.py --config config/optimizer.yaml
```

## 使用示例
```python
from scheduler.manager import ChargingManager

manager = ChargingManager()
schedule = manager.optimize_charging(
    target_soc=90,
    deadline="2024-01-15 08:00",
    current_soc=20
)

print(schedule)
# 输出:
# [
#   {'start': '02:00', 'end': '03:30', 'power': 11},
#   {'start': '05:00', 'end': '06:00', 'power': 11}
# ]
```

## 参考资料
- [Tesla API Documentation](https://www.teslaapi.io/)
- [PuLP Documentation](https://coin-or.github.io/pulp/)
- [Time-of-Use Pricing](https://www.pge.com/en_US/residential/rate-plans/rate-plan-options/time-of-use-base-plan.page)
