"""
雷达传感器系统
Radar sensor system for distance and velocity detection
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class RadarDetection:
    """雷达检测结果"""
    distance: float  # 米
    azimuth: float  # 方位角（度）
    elevation: float  # 仰角（度）
    velocity: float  # 相对速度 (m/s)
    rcs: float  # 雷达散射截面 (Radar Cross Section)
    confidence: float  # 置信度


class Radar:
    """单个雷达传感器"""
    
    def __init__(self, config: Dict):
        self.name = config.get('position', 'unknown')
        self.max_range = config.get('max_range', 160)
        self.fov = config.get('fov', 20)
        self.frequency = config.get('frequency', 77)  # GHz
        self.enabled = config.get('enabled', True)
        
        logger.info(f"初始化雷达: {self.name} | "
                   f"最大距离: {self.max_range}m | "
                   f"视场角: {self.fov}°")
    
    def scan(self, num_points: int = 100) -> List[RadarDetection]:
        """执行雷达扫描"""
        if not self.enabled:
            logger.warning(f"雷达 {self.name} 未启用")
            return []
        
        # 模拟雷达数据
        detections = []
        for _ in range(np.random.randint(0, 20)):  # 随机目标数量
            detection = RadarDetection(
                distance=np.random.uniform(5, self.max_range),
                azimuth=np.random.uniform(-self.fov/2, self.fov/2),
                elevation=np.random.uniform(-10, 10),
                velocity=np.random.uniform(-30, 30),
                rcs=np.random.uniform(-20, 20),
                confidence=np.random.uniform(0.5, 1.0)
            )
            detections.append(detection)
        
        return detections
    
    def filter_noise(self, detections: List[RadarDetection], 
                    min_confidence: float = 0.6) -> List[RadarDetection]:
        """过滤低置信度检测"""
        return [d for d in detections if d.confidence >= min_confidence]
    
    def cluster_detections(self, detections: List[RadarDetection],
                          distance_threshold: float = 2.0) -> List[RadarDetection]:
        """聚类检测点（合并相近目标）"""
        if not detections:
            return []
        
        # 简单的距离聚类
        clusters = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            cluster = [det]
            used.add(i)
            
            for j, other in enumerate(detections):
                if j in used:
                    continue
                
                dist = np.sqrt(
                    (det.distance - other.distance)**2 +
                    (det.azimuth - other.azimuth)**2
                )
                
                if dist < distance_threshold:
                    cluster.append(other)
                    used.add(j)
            
            # 合并聚类
            if cluster:
                avg_det = RadarDetection(
                    distance=np.mean([d.distance for d in cluster]),
                    azimuth=np.mean([d.azimuth for d in cluster]),
                    elevation=np.mean([d.elevation for d in cluster]),
                    velocity=np.mean([d.velocity for d in cluster]),
                    rcs=np.mean([d.rcs for d in cluster]),
                    confidence=np.max([d.confidence for d in cluster])
                )
                clusters.append(avg_det)
        
        return clusters


class RadarSystem:
    """雷达系统 - 管理多个雷达"""
    
    def __init__(self, radar_config: Dict):
        self.radars: Dict[str, Radar] = {}
        
        for radar_name, config in radar_config.items():
            self.radars[radar_name] = Radar(config)
        
        logger.info(f"雷达系统初始化完成 | 雷达数量: {len(self.radars)}")
    
    def scan_all(self) -> Dict[str, List[RadarDetection]]:
        """扫描所有雷达"""
        all_detections = {}
        for name, radar in self.radars.items():
            detections = radar.scan()
            detections = radar.filter_noise(detections)
            detections = radar.cluster_detections(detections)
            all_detections[name] = detections
        
        return all_detections
    
    def scan(self) -> List[RadarDetection]:
        """扫描主雷达（前向长距离）"""
        if 'front_long_range' in self.radars:
            radar = self.radars['front_long_range']
            detections = radar.scan()
            return radar.filter_noise(detections)
        elif self.radars:
            radar = list(self.radars.values())[0]
            detections = radar.scan()
            return radar.filter_noise(detections)
        return []
    
    def get_target_list(self) -> List[Dict]:
        """获取目标列表（格式化输出）"""
        detections = self.scan()
        targets = []
        
        for det in detections:
            target = {
                'distance': det.distance,
                'azimuth': det.azimuth,
                'velocity': det.velocity,
                'confidence': det.confidence,
                'position': self._spherical_to_cartesian(
                    det.distance, 
                    det.azimuth, 
                    det.elevation
                )
            }
            targets.append(target)
        
        return targets
    
    @staticmethod
    def _spherical_to_cartesian(r: float, azimuth: float, 
                                elevation: float) -> Tuple[float, float, float]:
        """球坐标转笛卡尔坐标"""
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        x = r * np.cos(el_rad) * np.cos(az_rad)
        y = r * np.cos(el_rad) * np.sin(az_rad)
        z = r * np.sin(el_rad)
        
        return (x, y, z)


def test_radar_system():
    """测试雷达系统"""
    config = {
        'front_long_range': {
            'position': 'front_center',
            'max_range': 160,
            'fov': 20,
            'frequency': 77,
            'enabled': True
        }
    }
    
    system = RadarSystem(config)
    detections = system.scan()
    
    logger.info(f"检测到 {len(detections)} 个目标")
    for i, det in enumerate(detections[:5]):  # 只显示前5个
        logger.info(
            f"目标 {i+1}: 距离={det.distance:.1f}m, "
            f"方位={det.azimuth:.1f}°, "
            f"速度={det.velocity:.1f}m/s"
        )
    
    logger.success("✅ 雷达系统测试完成")


if __name__ == "__main__":
    test_radar_system()
