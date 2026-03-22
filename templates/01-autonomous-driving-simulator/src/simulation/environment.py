"""
驾驶仿真环境
Driving simulation environment based on OpenAI Gym
"""

import numpy as np
from typing import Dict, Tuple, List
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    logger.warning("Gymnasium未安装，使用简化版本")
    gym = None


class DrivingEnvironment:
    """驾驶仿真环境"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.env_config = config['environment']
        
        # 环境参数
        self.world_size = self.env_config['world']['size']
        self.dt = 0.1  # 时间步长
        
        # 车辆状态
        self.state = {
            'x': 0.0,
            'y': 0.0,
            'theta': 0.0,
            'speed': 0.0,
            'acceleration': 0.0,
            'steering': 0.0,
        }
        
        # 其他车辆
        self.other_vehicles = []
        
        # 行人
        self.pedestrians = []
        
        # 道路信息
        self.lanes = []
        self.traffic_lights = []
        
        # 仿真计数器
        self.step_count = 0
        self.max_steps = 1000
        
        # 定义动作和观察空间
        if gym:
            self.action_space = spaces.Box(
                low=np.array([0, 0, -1]),  # [throttle, brake, steering]
                high=np.array([1, 1, 1]),
                dtype=np.float32
            )
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20,),  # 状态向量维度
                dtype=np.float32
            )
        
        logger.info(f"仿真环境初始化 | 世界大小: {self.world_size}")
    
    def reset(self) -> Dict:
        """重置环境"""
        # 重置车辆状态
        self.state = {
            'x': 0.0,
            'y': 0.0,
            'theta': 0.0,
            'speed': 0.0,
            'acceleration': 0.0,
            'steering': 0.0,
        }
        
        # 生成其他车辆
        self._spawn_vehicles()
        
        # 生成行人
        self._spawn_pedestrians()
        
        # 生成道路
        self._generate_roads()
        
        # 重置计数器
        self.step_count = 0
        
        logger.info("环境已重置")
        
        return self._get_observation()
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """执行一步仿真"""
        # 解析动作
        throttle = action.get('throttle', 0)
        brake = action.get('brake', 0)
        steering = action.get('steering', 0)
        
        # 更新车辆状态
        self._update_vehicle_state(throttle, brake, steering)
        
        # 更新其他物体
        self._update_other_objects()
        
        # 检查碰撞
        collision = self._check_collision()
        
        # 计算奖励
        reward = self._compute_reward(collision)
        
        # 检查是否结束
        done = collision or self.step_count >= self.max_steps
        
        # 额外信息
        info = {
            'collision': collision,
            'speed': self.state['speed'],
            'step': self.step_count
        }
        
        self.step_count += 1
        
        return self._get_observation(), reward, done, info
    
    def _update_vehicle_state(self, throttle: float, brake: float, steering: float):
        """更新车辆状态"""
        # 车辆参数
        max_accel = 5.8
        max_decel = 9.8
        max_speed = 250 / 3.6  # km/h -> m/s
        L = 2.875  # 轴距
        
        # 加速度
        if throttle > 0:
            accel = throttle * max_accel
        else:
            accel = -brake * max_decel
        
        # 更新速度
        self.state['speed'] += accel * self.dt
        self.state['speed'] = np.clip(self.state['speed'], 0, max_speed)
        
        # 更新位置和朝向（自行车模型）
        v = self.state['speed']
        theta = self.state['theta']
        delta = steering * 0.6  # 前轮转角
        
        self.state['x'] += v * np.cos(theta) * self.dt
        self.state['y'] += v * np.sin(theta) * self.dt
        self.state['theta'] += (v / L) * np.tan(delta) * self.dt
        
        # 保存控制输入
        self.state['acceleration'] = accel
        self.state['steering'] = steering
    
    def _spawn_vehicles(self):
        """生成其他车辆"""
        num_vehicles = self.env_config['traffic']['vehicle_count']
        self.other_vehicles = []
        
        for i in range(num_vehicles):
            vehicle = {
                'x': np.random.uniform(0, self.world_size[0]),
                'y': np.random.uniform(-20, 20),
                'speed': np.random.uniform(10, 30),
                'theta': 0,
                'id': i
            }
            self.other_vehicles.append(vehicle)
    
    def _spawn_pedestrians(self):
        """生成行人"""
        num_pedestrians = self.env_config['traffic']['pedestrian_count']
        self.pedestrians = []
        
        for i in range(num_pedestrians):
            pedestrian = {
                'x': np.random.uniform(0, self.world_size[0]),
                'y': np.random.uniform(-30, 30),
                'speed': np.random.uniform(0.5, 2.0),
                'direction': np.random.uniform(0, 2*np.pi),
                'id': i
            }
            self.pedestrians.append(pedestrian)
    
    def _generate_roads(self):
        """生成道路"""
        # 简化：生成一条直路
        lane_width = 3.5
        num_lanes = 2
        
        self.lanes = []
        for i in range(num_lanes):
            lane_y = (i - num_lanes/2 + 0.5) * lane_width
            self.lanes.append({'y': lane_y, 'width': lane_width})
        
        # 生成交通灯
        self.traffic_lights = [
            {'x': 100, 'state': 'green'},
            {'x': 200, 'state': 'red'},
        ]
    
    def _update_other_objects(self):
        """更新其他物体位置"""
        # 更新车辆
        for vehicle in self.other_vehicles:
            vehicle['x'] += vehicle['speed'] * self.dt
            
            # 循环
            if vehicle['x'] > self.world_size[0]:
                vehicle['x'] = 0
        
        # 更新行人
        for ped in self.pedestrians:
            ped['x'] += ped['speed'] * np.cos(ped['direction']) * self.dt
            ped['y'] += ped['speed'] * np.sin(ped['direction']) * self.dt
            
            # 边界检查
            ped['y'] = np.clip(ped['y'], -30, 30)
    
    def _check_collision(self) -> bool:
        """检查碰撞"""
        vehicle_radius = 2.0  # 米
        
        # 检查与其他车辆的碰撞
        for v in self.other_vehicles:
            dist = np.sqrt(
                (self.state['x'] - v['x'])**2 + 
                (self.state['y'] - v['y'])**2
            )
            if dist < vehicle_radius * 2:
                logger.warning(f"碰撞检测: 与车辆 {v['id']} 碰撞")
                return True
        
        # 检查与行人的碰撞
        for p in self.pedestrians:
            dist = np.sqrt(
                (self.state['x'] - p['x'])**2 + 
                (self.state['y'] - p['y'])**2
            )
            if dist < vehicle_radius:
                logger.warning(f"碰撞检测: 与行人 {p['id']} 碰撞")
                return True
        
        return False
    
    def _compute_reward(self, collision: bool) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 碰撞惩罚
        if collision:
            return -100.0
        
        # 速度奖励（保持目标速度）
        target_speed = 50 / 3.6
        speed_error = abs(self.state['speed'] - target_speed)
        reward += -0.1 * speed_error
        
        # 车道保持奖励
        lane_y = 0  # 目标车道
        lateral_error = abs(self.state['y'] - lane_y)
        reward += -0.5 * lateral_error
        
        # 平滑控制奖励
        reward += -0.1 * abs(self.state['steering'])
        reward += -0.05 * abs(self.state['acceleration'])
        
        # 前进奖励
        reward += 0.1 * self.state['speed']
        
        return reward
    
    def _get_observation(self) -> Dict:
        """获取观察"""
        obs = {
            'x': self.state['x'],
            'y': self.state['y'],
            'theta': self.state['theta'],
            'speed': self.state['speed'],
            'acceleration': self.state['acceleration'],
            'steering': self.state['steering'],
            'vehicles': self.other_vehicles[:10],  # 最近的10辆车
            'pedestrians': self.pedestrians[:10],
        }
        
        return obs
    
    def render(self):
        """渲染环境"""
        # 简化版本 - 实际应用中会使用pygame或matplotlib
        logger.info(
            f"[Step {self.step_count}] "
            f"位置: ({self.state['x']:.1f}, {self.state['y']:.1f}) | "
            f"速度: {self.state['speed']*3.6:.1f} km/h | "
            f"朝向: {np.degrees(self.state['theta']):.1f}°"
        )
    
    def close(self):
        """关闭环境"""
        logger.info("环境已关闭")


def test_environment():
    """测试仿真环境"""
    config = {
        'environment': {
            'world': {'size': [1000, 1000]},
            'traffic': {
                'vehicle_count': 10,
                'pedestrian_count': 5
            }
        }
    }
    
    env = DrivingEnvironment(config)
    obs = env.reset()
    
    logger.info("开始仿真测试...")
    
    for step in range(100):
        # 随机动作
        action = {
            'throttle': np.random.uniform(0.3, 0.7),
            'brake': 0,
            'steering': np.random.uniform(-0.1, 0.1)
        }
        
        obs, reward, done, info = env.step(action)
        
        if step % 10 == 0:
            env.render()
        
        if done:
            logger.info(f"仿真结束于步数 {step}")
            break
    
    env.close()
    logger.success("✅ 仿真环境测试完成")


if __name__ == "__main__":
    test_environment()
