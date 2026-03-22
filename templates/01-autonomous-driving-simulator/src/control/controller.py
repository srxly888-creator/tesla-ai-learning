"""
车辆控制模块
Vehicle control using PID and MPC controllers
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ControlSignal:
    """控制信号"""
    throttle: float  # 油门 [0, 1]
    brake: float  # 刹车 [0, 1]
    steering: float  # 转向 [-1, 1] (左负右正)
    timestamp: float


class PIDController:
    """PID控制器"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.1  # 时间步长
    
    def compute(self, setpoint: float, current: float) -> float:
        """计算控制输出"""
        error = setpoint - current
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        
        # 微分项
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative
        self.prev_error = error
        
        # 总输出
        output = p_term + i_term + d_term
        
        return output
    
    def reset(self):
        """重置控制器"""
        self.integral = 0.0
        self.prev_error = 0.0


class MPCController:
    """模型预测控制器（简化版）"""
    
    def __init__(self, horizon: int = 20, dt: float = 0.1):
        self.horizon = horizon
        self.dt = dt
        
        # 权重矩阵
        self.Q = np.diag([1.0, 1.0, 0.1])  # 状态误差权重
        self.R = np.diag([0.1, 0.1])  # 控制输入权重
        
        logger.info(f"MPC控制器初始化 | 预测时域: {horizon}")
    
    def compute(self, reference_path: List, current_state: Dict) -> ControlSignal:
        """计算最优控制输入"""
        # 简化的MPC实现
        # 实际应用中会使用优化求解器（如CVXPY, OSQP）
        
        # 提取参考轨迹的第一个点
        if reference_path:
            ref_point = reference_path[0]
            ref_x = ref_point.x
            ref_y = ref_point.y
            ref_theta = ref_point.theta
            ref_v = ref_point.velocity
        else:
            return ControlSignal(0, 0, 0, 0)
        
        # 当前状态
        curr_x = current_state.get('x', 0)
        curr_y = current_state.get('y', 0)
        curr_theta = current_state.get('theta', 0)
        curr_v = current_state.get('speed', 0)
        
        # 计算误差
        x_error = ref_x - curr_x
        y_error = ref_y - curr_y
        theta_error = self._normalize_angle(ref_theta - curr_theta)
        v_error = ref_v - curr_v
        
        # 简化的控制律
        # 转向控制（基于横向误差和航向误差）
        steering = -0.5 * y_error - 2.0 * theta_error
        steering = np.clip(steering, -1, 1)
        
        # 速度控制
        if v_error > 0:
            throttle = min(v_error / 30, 1.0)
            brake = 0
        else:
            throttle = 0
            brake = min(-v_error / 30, 1.0)
        
        return ControlSignal(throttle, brake, steering, 0)
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def predict_trajectory(self, state: Dict, 
                          control_sequence: List[ControlSignal]) -> List[Dict]:
        """预测未来轨迹"""
        trajectory = [state]
        
        for ctrl in control_sequence[:self.horizon]:
            next_state = self._vehicle_model(trajectory[-1], ctrl)
            trajectory.append(next_state)
        
        return trajectory
    
    def _vehicle_model(self, state: Dict, control: ControlSignal) -> Dict:
        """车辆动力学模型"""
        # 简化的自行车模型
        x = state['x']
        y = state['y']
        theta = state['theta']
        v = state['speed']
        
        # 车辆参数
        L = 2.875  # 轴距
        
        # 控制输入
        a = (control.throttle - control.brake) * 5.0  # 加速度
        delta = control.steering * 0.6  # 前轮转角
        
        # 更新状态
        new_v = v + a * self.dt
        new_theta = theta + (v / L) * np.tan(delta) * self.dt
        new_x = x + v * np.cos(theta) * self.dt
        new_y = y + v * np.sin(theta) * self.dt
        
        return {
            'x': new_x,
            'y': new_y,
            'theta': new_theta,
            'speed': new_v
        }


class VehicleController:
    """车辆控制器（整合PID和MPC）"""
    
    def __init__(self, pid_config: Dict, mpc_config: Dict):
        self.speed_controller = PIDController(
            kp=pid_config['kp'],
            ki=pid_config['ki'],
            kd=pid_config['kd']
        )
        
        self.mpc = MPCController(
            horizon=mpc_config['horizon'],
            dt=mpc_config['dt']
        )
        
        logger.info("车辆控制器初始化完成")
    
    def compute_control(self, planned_path: List, 
                       current_state: Dict) -> ControlSignal:
        """计算控制信号"""
        # 使用MPC进行轨迹跟踪
        mpc_signal = self.mpc.compute(planned_path, current_state)
        
        # 可选：使用PID进行速度微调
        # speed_signal = self.speed_controller.compute(...)
        
        return mpc_signal
    
    def emergency_brake(self) -> ControlSignal:
        """紧急制动"""
        return ControlSignal(0, 1.0, 0, 0)
    
    def adaptive_cruise_control(self, 
                                target_speed: float,
                                current_speed: float,
                                distance_to_lead: float,
                                min_distance: float = 2.0) -> ControlSignal:
        """自适应巡航控制"""
        # 速度控制
        speed_error = target_speed - current_speed
        
        # 距离控制
        if distance_to_lead < min_distance:
            # 太近，减速
            throttle = 0
            brake = (min_distance - distance_to_lead) / min_distance
        else:
            # 正常速度控制
            if speed_error > 0:
                throttle = self.speed_controller.compute(target_speed, current_speed)
                throttle = np.clip(throttle / 10, 0, 1)
                brake = 0
            else:
                throttle = 0
                brake = min(-speed_error / 30, 1.0)
        
        return ControlSignal(throttle, brake, 0, 0)


def test_controller():
    """测试控制器"""
    # 初始化
    pid_config = {'kp': 1.0, 'ki': 0.1, 'kd': 0.05}
    mpc_config = {'horizon': 20, 'dt': 0.1}
    
    controller = VehicleController(pid_config, mpc_config)
    
    # 模拟路径
    from planning.planner import PathPoint
    path = [
        PathPoint(0, 0, 0, 50, 0),
        PathPoint(1, 0, 0, 50, 0.1),
        PathPoint(2, 0, 0, 50, 0.2),
    ]
    
    # 当前状态
    state = {'x': 0, 'y': 0, 'theta': 0, 'speed': 30}
    
    # 计算控制
    signal = controller.compute_control(path, state)
    
    logger.info(
        f"控制信号: 油门={signal.throttle:.2f}, "
        f"刹车={signal.brake:.2f}, "
        f"转向={signal.steering:.2f}"
    )
    
    # 测试ACC
    acc_signal = controller.adaptive_cruise_control(60, 40, 5.0)
    logger.info(
        f"ACC信号: 油门={acc_signal.throttle:.2f}, "
        f"刹车={acc_signal.brake:.2f}"
    )
    
    logger.success("✅ 控制器测试完成")


if __name__ == "__main__":
    test_controller()
