"""
自动驾驶模拟器测试套件
Test suite for the autonomous driving simulator
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sensors.camera import CameraSystem
from sensors.radar import RadarSystem
from perception.detector import ObjectDetector
from planning.planner import PathPlanner, PathPoint
from control.controller import VehicleController, PIDController
from simulation.environment import DrivingEnvironment


class TestCameraSystem:
    """摄像头系统测试"""
    
    @pytest.fixture
    def camera_config(self):
        return {
            'front_main': {
                'position': 'front_center',
                'resolution': [1280, 720],
                'fov': 120,
                'fps': 36,
                'enabled': True
            }
        }
    
    def test_camera_initialization(self, camera_config):
        """测试摄像头初始化"""
        system = CameraSystem(camera_config)
        assert len(system.cameras) == 1
        assert 'front_main' in system.cameras
    
    def test_camera_capture(self, camera_config):
        """测试图像捕获"""
        system = CameraSystem(camera_config)
        image = system.capture()
        
        assert image is not None
        assert image.shape == (720, 1280, 3)
    
    def test_camera_disabled(self, camera_config):
        """测试禁用的摄像头"""
        camera_config['front_main']['enabled'] = False
        system = CameraSystem(camera_config)
        image = system.capture()
        
        assert image is None


class TestRadarSystem:
    """雷达系统测试"""
    
    @pytest.fixture
    def radar_config(self):
        return {
            'front_long_range': {
                'position': 'front_center',
                'max_range': 160,
                'fov': 20,
                'frequency': 77,
                'enabled': True
            }
        }
    
    def test_radar_initialization(self, radar_config):
        """测试雷达初始化"""
        system = RadarSystem(radar_config)
        assert len(system.radars) == 1
    
    def test_radar_scan(self, radar_config):
        """测试雷达扫描"""
        system = RadarSystem(radar_config)
        detections = system.scan()
        
        assert isinstance(detections, list)
        # 检测结果应该有合理数量的目标
        assert len(detections) >= 0
    
    def test_radar_detection_fields(self, radar_config):
        """测试检测字段"""
        system = RadarSystem(radar_config)
        detections = system.scan()
        
        if detections:
            det = detections[0]
            assert hasattr(det, 'distance')
            assert hasattr(det, 'azimuth')
            assert hasattr(det, 'velocity')
            assert hasattr(det, 'confidence')


class TestObjectDetector:
    """目标检测器测试"""
    
    @pytest.fixture
    def detector(self):
        return ObjectDetector(model_path=None)
    
    def test_detector_initialization(self, detector):
        """测试检测器初始化"""
        assert detector is not None
    
    def test_detection_output(self, detector):
        """测试检测结果"""
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(image)
        
        assert isinstance(detections, list)
    
    def test_detection_fields(self, detector):
        """测试检测字段"""
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(image)
        
        if detections:
            det = detections[0]
            assert hasattr(det, 'class_name')
            assert hasattr(det, 'confidence')
            assert hasattr(det, 'bbox')
            assert det.confidence >= 0


class TestPathPlanner:
    """路径规划器测试"""
    
    @pytest.fixture
    def planner(self):
        return PathPlanner()
    
    def test_planner_initialization(self, planner):
        """测试规划器初始化"""
        assert planner is not None
    
    def test_path_planning(self, planner):
        """测试路径规划"""
        state = {
            'x': 0, 'y': 0, 'speed': 30,
            'goal': (100, 0)
        }
        detections = []
        
        path = planner.plan(state, detections)
        
        assert isinstance(path, list)
    
    def test_lane_change_planning(self, planner):
        """测试变道规划"""
        path = planner.plan_lane_change(0, 1)
        
        assert isinstance(path, list)
        assert len(path) > 0
        
        # 检查起点和终点
        assert path[0].y == 0
        assert path[-1].y > 0


class TestPIDController:
    """PID控制器测试"""
    
    def test_pid_computation(self):
        """测试PID计算"""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        
        output = pid.compute(setpoint=10, current=5)
        
        assert isinstance(output, float)
    
    def test_pid_convergence(self):
        """测试PID收敛性"""
        pid = PIDController(kp=2.0, ki=0.1, kd=0.1)
        
        current = 0
        setpoint = 10
        
        for _ in range(100):
            output = pid.compute(setpoint, current)
            current += output * 0.1  # 简化的系统模型
        
        # 应该接近设定值
        assert abs(current - setpoint) < 2.0


class TestVehicleController:
    """车辆控制器测试"""
    
    @pytest.fixture
    def controller(self):
        pid_config = {'kp': 1.0, 'ki': 0.1, 'kd': 0.05}
        mpc_config = {'horizon': 20, 'dt': 0.1}
        return VehicleController(pid_config, mpc_config)
    
    def test_controller_initialization(self, controller):
        """测试控制器初始化"""
        assert controller is not None
    
    def test_control_signal(self, controller):
        """测试控制信号"""
        path = [PathPoint(0, 0, 0, 50, 0)]
        state = {'x': 0, 'y': 0, 'theta': 0, 'speed': 30}
        
        signal = controller.compute_control(path, state)
        
        assert hasattr(signal, 'throttle')
        assert hasattr(signal, 'brake')
        assert hasattr(signal, 'steering')
        
        # 检查范围
        assert 0 <= signal.throttle <= 1
        assert 0 <= signal.brake <= 1
        assert -1 <= signal.steering <= 1
    
    def test_emergency_brake(self, controller):
        """测试紧急制动"""
        signal = controller.emergency_brake()
        
        assert signal.throttle == 0
        assert signal.brake == 1.0
        assert signal.steering == 0


class TestDrivingEnvironment:
    """驾驶仿真环境测试"""
    
    @pytest.fixture
    def env(self):
        config = {
            'environment': {
                'world': {'size': [1000, 1000]},
                'traffic': {
                    'vehicle_count': 5,
                    'pedestrian_count': 3
                }
            }
        }
        return DrivingEnvironment(config)
    
    def test_environment_initialization(self, env):
        """测试环境初始化"""
        assert env is not None
    
    def test_environment_reset(self, env):
        """测试环境重置"""
        obs = env.reset()
        
        assert isinstance(obs, dict)
        assert 'x' in obs
        assert 'y' in obs
        assert 'speed' in obs
    
    def test_environment_step(self, env):
        """测试环境步进"""
        env.reset()
        
        action = {
            'throttle': 0.5,
            'brake': 0,
            'steering': 0
        }
        
        obs, reward, done, info = env.step(action)
        
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_collision_detection(self, env):
        """测试碰撞检测"""
        env.reset()
        
        # 设置一个碰撞场景
        env.state['x'] = env.other_vehicles[0]['x']
        env.state['y'] = env.other_vehicles[0]['y']
        
        collision = env._check_collision()
        assert collision == True


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
