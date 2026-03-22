"""
路径规划模块
Path planning using A* algorithm and dynamic window approach
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class PathPoint:
    """路径点"""
    x: float
    y: float
    theta: float  # 朝向角度
    velocity: float
    timestamp: float


class PathPlanner:
    """路径规划器"""
    
    def __init__(self, config: Dict = None):
        self.resolution = 0.5  # 地图分辨率（米/格）
        self.safety_margin = 2.0  # 安全距离（米）
        self.max_acceleration = 5.0  # m/s^2
        self.max_steering_rate = 30.0  # 度/秒
        
        logger.info("路径规划器初始化完成")
    
    def plan(self, state: Dict, detections: List) -> List[PathPoint]:
        """规划路径"""
        # 提取当前状态
        current_pos = (state['x'], state['y'])
        goal_pos = state.get('goal', (100, 0))
        
        # 创建占用栅格
        occupancy_grid = self._create_occupancy_grid(detections, state)
        
        # A*搜索
        path = self._astar(current_pos, goal_pos, occupancy_grid)
        
        if path is None:
            logger.warning("未找到有效路径")
            return []
        
        # 路径平滑
        smooth_path = self._smooth_path(path)
        
        # 添加速度信息
        trajectory = self._add_velocity(smooth_path, state['speed'])
        
        return trajectory
    
    def _create_occupancy_grid(self, detections: List, 
                               state: Dict) -> np.ndarray:
        """创建占用栅格地图"""
        grid_size = 200  # 200x200栅格
        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # 标记障碍物
        for det in detections:
            if hasattr(det, 'bbox'):
                # 从检测结果计算位置
                # 简化：假设障碍物在正前方
                obs_x = grid_size // 2 + int(det.distance / self.resolution)
                obs_y = grid_size // 2
                
                # 标记障碍物区域（带安全距离）
                margin = int(self.safety_margin / self.resolution)
                for dx in range(-margin, margin+1):
                    for dy in range(-margin, margin+1):
                        gx, gy = obs_x + dx, obs_y + dy
                        if 0 <= gx < grid_size and 0 <= gy < grid_size:
                            grid[gy, gx] = 1
        
        return grid
    
    def _astar(self, start: Tuple[float, float], 
               goal: Tuple[float, float],
               grid: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """A*算法"""
        start_cell = self._world_to_grid(start, grid.shape)
        goal_cell = self._world_to_grid(goal, grid.shape)
        
        # 优先队列: (f_score, g_score, position, path)
        open_set = [(0, 0, start_cell, [start_cell])]
        closed_set = set()
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # 到达目标
            if current == goal_cell:
                logger.info(f"✅ 找到路径 | 长度: {len(path)} 个点")
                return path
            
            # 扩展邻居
            for neighbor in self._get_neighbors(current, grid):
                if neighbor in closed_set:
                    continue
                
                # 检查碰撞
                if grid[neighbor[1], neighbor[0]] == 1:
                    continue
                
                new_g = g + self._distance(current, neighbor)
                new_f = new_g + self._heuristic(neighbor, goal_cell)
                
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))
        
        return None
    
    def _get_neighbors(self, pos: Tuple[int, int], 
                      grid: np.ndarray) -> List[Tuple[int, int]]:
        """获取8邻域"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    neighbors.append((nx, ny))
        return neighbors
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """启发式函数（欧几里得距离）"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """两点距离"""
        return self._heuristic(a, b)
    
    def _world_to_grid(self, pos: Tuple[float, float], 
                      shape: Tuple[int, int]) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        gx = int(shape[1] // 2 + pos[0] / self.resolution)
        gy = int(shape[0] // 2 + pos[1] / self.resolution)
        return (gx, gy)
    
    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[PathPoint]:
        """路径平滑"""
        if len(path) < 3:
            return [PathPoint(p[0], p[1], 0, 0, 0) for p in path]
        
        smooth_path = []
        
        for i in range(len(path)):
            # 计算朝向
            if i < len(path) - 1:
                dx = path[i+1][0] - path[i][0]
                dy = path[i+1][1] - path[i][1]
                theta = np.arctan2(dy, dx)
            else:
                theta = smooth_path[-1].theta if smooth_path else 0
            
            point = PathPoint(
                x=path[i][0] * self.resolution,
                y=path[i][1] * self.resolution,
                theta=theta,
                velocity=0,
                timestamp=i * 0.1
            )
            smooth_path.append(point)
        
        return smooth_path
    
    def _add_velocity(self, path: List[PathPoint], 
                     current_speed: float) -> List[PathPoint]:
        """添加速度信息"""
        target_speed = 60  # km/h
        
        for i, point in enumerate(path):
            # 简化：线性加速
            if i < len(path) * 0.3:
                point.velocity = min(current_speed + i * 2, target_speed)
            elif i > len(path) * 0.7:
                point.velocity = max(target_speed - (i - len(path) * 0.7) * 2, 10)
            else:
                point.velocity = target_speed
        
        return path
    
    def plan_lane_change(self, current_lane: int, 
                        target_lane: int) -> List[PathPoint]:
        """规划变道轨迹"""
        lane_width = 3.5  # 米
        points = []
        
        for i in range(50):
            t = i / 50.0
            # 使用正弦曲线平滑变道
            x = t * 50
            y = (target_lane - current_lane) * lane_width * np.sin(np.pi * t / 2)
            theta = np.arctan2(
                (target_lane - current_lane) * lane_width * np.pi / 2 * np.cos(np.pi * t / 2),
                50
            )
            
            points.append(PathPoint(x, y, theta, 50, t))
        
        return points


def test_planner():
    """测试路径规划器"""
    planner = PathPlanner()
    
    state = {
        'x': 0,
        'y': 0,
        'speed': 30,
        'goal': (100, 0)
    }
    
    # 模拟检测结果
    detections = []
    
    path = planner.plan(state, detections)
    
    logger.info(f"规划路径包含 {len(path)} 个点")
    for i, point in enumerate(path[:5]):
        logger.info(
            f"  点 {i}: x={point.x:.2f}, y={point.y:.2f}, "
            f"theta={np.degrees(point.theta):.1f}°, v={point.velocity:.1f}km/h"
        )
    
    logger.success("✅ 路径规划测试完成")


if __name__ == "__main__":
    test_planner()
