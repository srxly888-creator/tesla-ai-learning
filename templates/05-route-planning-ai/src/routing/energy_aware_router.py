"""
能耗感知路由器
Energy-aware routing considering battery constraints
"""

import numpy as np
import heapq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Node:
    """图节点"""
    id: str
    lat: float
    lon: float
    is_charger: bool = False
    charger_power: float = 0  # kW


@dataclass
class Edge:
    """图边"""
    source: str
    target: str
    distance: float  # km
    speed_limit: float  # km/h
    elevation_gain: float  # meters
    energy_cost: float  # kWh


@dataclass
class Route:
    """路线"""
    nodes: List[str]
    total_distance: float
    total_time: float
    total_energy: float
    charging_stops: List[Dict]
    segments: List[Dict]


class EnergyAwareRouter:
    """能耗感知路由器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 车辆参数
        self.battery_capacity = config.get('battery_capacity', 75)  # kWh
        self.usable_capacity = config.get('usable_capacity', 70)  # kWh
        self.energy_efficiency = config.get('efficiency', 0.15)  # kWh/km
        
        # 图数据
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self.adjacency: Dict[str, List[str]] = {}
        
        # 缓存
        self.energy_cache = {}
        
        logger.info("能耗感知路由器初始化")
    
    def add_node(self, node: Node):
        """添加节点"""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
    
    def add_edge(self, edge: Edge):
        """添加边"""
        self.edges[(edge.source, edge.target)] = edge
        self.adjacency[edge.source].append(edge.target)
        
        # 计算能耗
        edge.energy_cost = self._calculate_edge_energy(edge)
    
    def _calculate_edge_energy(self, edge: Edge) -> float:
        """计算边的能耗"""
        # 基础能耗
        base_energy = edge.distance * self.energy_efficiency
        
        # 速度因子（高速更耗电）
        speed_factor = 1.0
        if edge.speed_limit > 100:
            speed_factor = 1.0 + (edge.speed_limit - 100) / 100 * 0.3
        
        # 海拔因子（上坡更耗电）
        elevation_factor = 1.0
        if edge.elevation_gain > 0:
            # 每上升100m增加约1.5kWh
            elevation_energy = edge.elevation_gain / 100 * 1.5
            elevation_factor = (base_energy + elevation_energy) / base_energy
        
        # 总能耗
        total_energy = base_energy * speed_factor * elevation_factor
        
        # 下坡可回收能量（简化）
        if edge.elevation_gain < -50:
            regen = abs(edge.elevation_gain) / 100 * 0.5  # 回收0.5kWh/100m
            total_energy = max(0, total_energy - regen)
        
        return total_energy
    
    def find_route(self,
                   origin: str,
                   destination: str,
                   initial_soc: float = 100,
                   min_soc: float = 10,
                   target_soc: float = 20) -> Optional[Route]:
        """
        查找能耗感知路线
        
        Args:
            origin: 起点ID
            destination: 终点ID
            initial_soc: 初始SOC (%)
            min_soc: 最低SOC (%)
            target_soc: 目标SOC (%)
        
        Returns:
            路线对象，如果无法到达则返回None
        """
        if origin not in self.nodes or destination not in self.nodes:
            logger.error("起点或终点不存在")
            return None
        
        # 初始化能量（kWh）
        initial_energy = initial_soc / 100 * self.usable_capacity
        min_energy = min_soc / 100 * self.usable_capacity
        target_energy = target_soc / 100 * self.usable_capacity
        
        # A*搜索
        # 状态: (节点ID, 当前能量)
        # 优先级: (预估总成本, 实际成本)
        
        # 优先队列: (f_score, g_score, node_id, energy, path, charging_stops)
        start_state = (0, 0, origin, initial_energy, [origin], [])
        open_set = [start_state]
        
        # 已访问状态
        visited = set()
        
        # 最佳路线
        best_route = None
        best_cost = float('inf')
        
        while open_set:
            f_score, g_score, current_id, current_energy, path, charging_stops = \
                heapq.heappop(open_set)
            
            state_key = (current_id, int(current_energy))
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # 到达终点
            if current_id == destination:
                if current_energy >= target_energy:
                    # 构建路线对象
                    route = self._build_route(
                        path, charging_stops, current_energy
                    )
                    
                    if g_score < best_cost:
                        best_route = route
                        best_cost = g_score
                continue
            
            # 扩展邻居
            for neighbor_id in self.adjacency.get(current_id, []):
                edge = self.edges.get((current_id, neighbor_id))
                if edge is None:
                    continue
                
                # 检查能量是否足够
                required_energy = edge.energy_cost
                new_energy = current_energy - required_energy
                
                # 如果能量不足，考虑充电
                if new_energy < min_energy:
                    # 在当前节点充电（如果是充电站）
                    current_node = self.nodes[current_id]
                    if current_node.is_charger:
                        # 充到满电
                        charge_energy = self.usable_capacity - current_energy
                        new_energy = self.usable_capacity - required_energy
                        
                        # 记录充电
                        new_charging_stops = charging_stops + [{
                            'node_id': current_id,
                            'charge_amount': charge_energy,
                            'charge_time': self._estimate_charge_time(
                                charge_energy, current_node.charger_power
                            )
                        }]
                    else:
                        # 无法到达
                        continue
                else:
                    new_charging_stops = charging_stops
                
                # 计算新的成本
                new_g_score = g_score + edge.distance  # 以距离为成本
                
                # 启发式（到终点的直线距离）
                h_score = self._heuristic(neighbor_id, destination)
                new_f_score = new_g_score + h_score
                
                # 加入队列
                new_state = (
                    new_f_score,
                    new_g_score,
                    neighbor_id,
                    new_energy,
                    path + [neighbor_id],
                    new_charging_stops
                )
                heapq.heappush(open_set, new_state)
        
        if best_route:
            logger.info(
                f"找到路线 | 距离: {best_route.total_distance:.1f}km | "
                f"能耗: {best_route.total_energy:.1f}kWh | "
                f"充电次数: {len(best_route.charging_stops)}"
            )
        else:
            logger.warning("未找到可行路线")
        
        return best_route
    
    def _heuristic(self, node1_id: str, node2_id: str) -> float:
        """启发式函数（直线距离）"""
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        # Haversine公式
        lat1, lon1 = np.radians(node1.lat), np.radians(node1.lon)
        lat2, lon2 = np.radians(node2.lat), np.radians(node2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # 地球半径 (km)
        r = 6371
        
        return c * r
    
    def _estimate_charge_time(self, 
                             charge_amount: float,
                             charger_power: float) -> float:
        """估算充电时间（分钟）"""
        # 简化：线性充电
        time_hours = charge_amount / charger_power
        return time_hours * 60
    
    def _build_route(self,
                     path: List[str],
                     charging_stops: List[Dict],
                     final_energy: float) -> Route:
        """构建路线对象"""
        total_distance = 0
        total_time = 0
        total_energy = 0
        segments = []
        
        for i in range(len(path) - 1):
            edge = self.edges.get((path[i], path[i+1]))
            if edge:
                total_distance += edge.distance
                total_time += edge.distance / edge.speed_limit  # 小时
                total_energy += edge.energy_cost
                
                segments.append({
                    'from': path[i],
                    'to': path[i+1],
                    'distance': edge.distance,
                    'time': edge.distance / edge.speed_limit,
                    'energy': edge.energy_cost
                })
        
        # 添加充电时间
        total_charge_time = sum(
            stop['charge_time'] for stop in charging_stops
        )
        total_time += total_charge_time / 60  # 转为小时
        
        return Route(
            nodes=path,
            total_distance=total_distance,
            total_time=total_time,
            total_energy=total_energy,
            charging_stops=charging_stops,
            segments=segments
        )
    
    def find_nearest_charger(self, 
                            node_id: str,
                            max_distance: float = 100) -> Optional[str]:
        """查找最近的充电站"""
        min_dist = float('inf')
        nearest_charger = None
        
        for charger_id, node in self.nodes.items():
            if node.is_charger:
                dist = self._heuristic(node_id, charger_id)
                if dist < min_dist and dist <= max_distance:
                    min_dist = dist
                    nearest_charger = charger_id
        
        return nearest_charger


def test_router():
    """测试路由器"""
    # 创建路由器
    router = EnergyAwareRouter({
        'battery_capacity': 75,
        'usable_capacity': 70,
        'efficiency': 0.15
    })
    
    # 创建测试网络
    # 节点
    router.add_node(Node('A', 37.7749, -122.4194))  # San Francisco
    router.add_node(Node('B', 36.7783, -119.4179))  # Fresno
    router.add_node(Node('C', 35.3733, -119.0187, is_charger=True, charger_power=150))  # Charger
    router.add_node(Node('D', 34.0522, -118.2437))  # Los Angeles
    
    # 边
    router.add_edge(Edge('A', 'B', 300, 120, 500, 0))
    router.add_edge(Edge('B', 'C', 100, 120, 200, 0))
    router.add_edge(Edge('C', 'D', 150, 120, -100, 0))
    router.add_edge(Edge('A', 'D', 600, 120, 1000, 0))  # 直达
    
    # 查找路线
    route = router.find_route(
        origin='A',
        destination='D',
        initial_soc=80,
        min_soc=10,
        target_soc=20
    )
    
    if route:
        logger.info(f"\n路线信息:")
        logger.info(f"  总距离: {route.total_distance:.1f} km")
        logger.info(f"  总时间: {route.total_time:.1f} 小时")
        logger.info(f"  总能耗: {route.total_energy:.1f} kWh")
        logger.info(f"  充电次数: {len(route.charging_stops)}")
        
        for i, segment in enumerate(route.segments):
            logger.info(
                f"\n  路段 {i+1}: {segment['from']} -> {segment['to']}"
            )
            logger.info(f"    距离: {segment['distance']:.1f} km")
            logger.info(f"    能耗: {segment['energy']:.1f} kWh")
    
    logger.success("✅ 路由器测试完成")


if __name__ == "__main__":
    test_router()
