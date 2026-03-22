"""
自动驾驶模拟器 - 主入口
Main entry point for the autonomous driving simulator
"""

import argparse
import yaml
from pathlib import Path
from loguru import logger
from simulation.environment import DrivingEnvironment
from sensors.camera import CameraSystem
from sensors.radar import RadarSystem
from perception.detector import ObjectDetector
from planning.planner import PathPlanner
from control.controller import VehicleController


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    """主函数"""
    logger.info("🚗 启动自动驾驶模拟器...")
    
    # 加载配置
    vehicle_config = load_config(args.vehicle_config)
    sensor_config = load_config(args.sensor_config)
    env_config = load_config(args.env_config)
    
    logger.info(f"车辆配置: {vehicle_config['vehicle']['name']}")
    logger.info(f"环境: {env_config['environment']['name']}")
    
    # 初始化传感器
    camera_system = CameraSystem(sensor_config['sensors']['cameras'])
    radar_system = RadarSystem(sensor_config['sensors']['radar'])
    
    # 初始化感知模块
    detector = ObjectDetector(model_path="models/yolov8n.pt")
    
    # 初始化规划模块
    planner = PathPlanner(config=vehicle_config['vehicle']['control'])
    
    # 初始化控制模块
    controller = VehicleController(
        pid_config=vehicle_config['vehicle']['control']['pid'],
        mpc_config=vehicle_config['vehicle']['control']['mpc']
    )
    
    # 创建仿真环境
    env = DrivingEnvironment(env_config)
    env.reset()
    
    # 主循环
    logger.info("开始仿真循环...")
    for episode in range(args.episodes):
        logger.info(f"\n{'='*50}")
        logger.info(f"Episode {episode + 1}/{args.episodes}")
        
        state = env.reset()
        done = False
        step = 0
        
        while not done:
            # 1. 感知 - 获取传感器数据
            camera_data = camera_system.capture()
            radar_data = radar_system.scan()
            
            # 2. 感知 - 目标检测
            detections = detector.detect(camera_data)
            
            # 3. 规划 - 路径规划
            planned_path = planner.plan(state, detections)
            
            # 4. 控制 - 执行动作
            control_signals = controller.compute_control(
                planned_path, 
                state
            )
            
            # 5. 执行动作
            next_state, reward, done, info = env.step(control_signals)
            
            # 日志记录
            if step % 10 == 0:
                logger.info(
                    f"Step {step} | "
                    f"Speed: {state['speed']:.1f} km/h | "
                    f"Steering: {control_signals['steering']:.2f} | "
                    f"Reward: {reward:.2f}"
                )
            
            state = next_state
            step += 1
            
            # 渲染
            if args.render and step % 5 == 0:
                env.render()
        
        logger.info(f"Episode {episode + 1} 完成 | 总步数: {step}")
    
    logger.info("✅ 仿真完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动驾驶模拟器")
    parser.add_argument(
        "--vehicle-config",
        type=str,
        default="config/vehicle.yaml",
        help="车辆配置文件路径"
    )
    parser.add_argument(
        "--sensor-config",
        type=str,
        default="config/sensors.yaml",
        help="传感器配置文件路径"
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="config/environment.yaml",
        help="环境配置文件路径"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="运行轮次"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="是否渲染环境"
    )
    
    args = parser.parse_args()
    main(args)
