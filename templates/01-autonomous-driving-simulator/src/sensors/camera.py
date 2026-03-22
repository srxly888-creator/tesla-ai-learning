"""
摄像头传感器系统
Camera sensor system for capturing visual data
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class CameraConfig:
    """摄像头配置"""
    name: str
    position: str
    resolution: Tuple[int, int]
    fov: float
    fps: int
    enabled: bool


class Camera:
    """单个摄像头类"""
    
    def __init__(self, config: Dict):
        self.name = config.get('position', 'unknown')
        self.resolution = config.get('resolution', [1280, 720])
        self.fov = config.get('fov', 90)
        self.fps = config.get('fps', 30)
        self.enabled = config.get('enabled', True)
        
        logger.info(f"初始化摄像头: {self.name} | "
                   f"分辨率: {self.resolution} | "
                   f"视场角: {self.fov}°")
    
    def capture(self) -> np.ndarray:
        """捕获图像"""
        if not self.enabled:
            logger.warning(f"摄像头 {self.name} 未启用")
            return None
        
        # 模拟图像捕获（实际应用中会从硬件或仿真环境获取）
        image = np.random.randint(
            0, 255, 
            (*self.resolution[::-1], 3), 
            dtype=np.uint8
        )
        
        return image
    
    def calibrate(self, calibration_data: Dict):
        """摄像头标定"""
        logger.info(f"标定摄像头: {self.name}")
        # 实际应用中会进行内参、外参标定
        self.intrinsic_matrix = calibration_data.get('intrinsic')
        self.distortion_coeffs = calibration_data.get('distortion')
    
    def undistort(self, image: np.ndarray) -> np.ndarray:
        """图像去畸变"""
        if hasattr(self, 'intrinsic_matrix'):
            return cv2.undistort(
                image, 
                self.intrinsic_matrix,
                self.distortion_coeffs
            )
        return image


class CameraSystem:
    """摄像头系统 - 管理多个摄像头"""
    
    def __init__(self, cameras_config: Dict):
        self.cameras: Dict[str, Camera] = {}
        
        for cam_name, cam_config in cameras_config.items():
            self.cameras[cam_name] = Camera(cam_config)
        
        logger.info(f"摄像头系统初始化完成 | 摄像头数量: {len(self.cameras)}")
    
    def capture_all(self) -> Dict[str, np.ndarray]:
        """捕获所有摄像头图像"""
        images = {}
        for name, camera in self.cameras.items():
            images[name] = camera.capture()
        return images
    
    def capture(self) -> Dict[str, np.ndarray]:
        """捕获主摄像头（前向）图像"""
        # 默认返回前向主摄像头
        if 'front_main' in self.cameras:
            return self.cameras['front_main'].capture()
        elif self.cameras:
            return list(self.cameras.values())[0].capture()
        return None
    
    def get_stereo_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取立体摄像头对（用于深度估计）"""
        left = self.cameras.get('front_left')
        right = self.cameras.get('front_right')
        
        if left and right:
            return left.capture(), right.capture()
        
        logger.warning("立体摄像头对不完整")
        return None, None
    
    def calibrate_all(self, calibration_dir: str):
        """标定所有摄像头"""
        import json
        from pathlib import Path
        
        calib_path = Path(calibration_dir)
        for cam_name, camera in self.cameras.items():
            calib_file = calib_path / f"{cam_name}_calibration.json"
            if calib_file.exists():
                with open(calib_file, 'r') as f:
                    calib_data = json.load(f)
                camera.calibrate(calib_data)


def test_camera_system():
    """测试摄像头系统"""
    config = {
        'front_main': {
            'position': 'front_center',
            'resolution': [1280, 720],
            'fov': 120,
            'fps': 36,
            'enabled': True
        }
    }
    
    system = CameraSystem(config)
    image = system.capture()
    
    if image is not None:
        logger.success(f"✅ 摄像头测试成功 | 图像尺寸: {image.shape}")
    else:
        logger.error("❌ 摄像头测试失败")


if __name__ == "__main__":
    test_camera_system()
