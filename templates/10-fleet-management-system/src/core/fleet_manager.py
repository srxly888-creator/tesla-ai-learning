"""
车队管理器
Fleet manager for coordinating multiple vehicles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq
from collections import defaultdict
from loguru import logger


class VehicleStatus(Enum):
    """车辆状态"""
    AVAILABLE = "available"
    CHARGING = "charging"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Location:
    """位置"""
    latitude: float
    longitude: float
    timestamp: datetime = None
    
    def distance_to(self, other: 'Location') -> float:
        """计算距离（Haversine公式，公里）"""
        lat1, lon1 = np.radians(self.latitude), np.radians(self.longitude)
        lat2, lon2 = np.radians(other.latitude), np.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # 地球半径（公里）
        r = 6371
        
        return c * r


@dataclass
class Vehicle:
    """车辆"""
    id: str
    name: str
    model: str
    status: VehicleStatus
    location: Location
    battery_soc: float  # 百分比
    odometer: float  # 公里
    
    # 容量和约束
    max_range: float = 400  # 公里
    cargo_capacity: float = 500  # kg
    
    # 当前任务
    current_task_id: Optional[str] = None
    estimated_available_time: Optional[datetime] = None
    
    # 历史数据
    total_trips: int = 0
    total_distance: float = 0
    total_energy_used: float = 0
    
    def available_range(self) -> float:
        """可用续航"""
        return self.max_range * (self.battery_soc / 100)
    
    def needs_charging(self, threshold: float = 20) -> bool:
        """是否需要充电"""
        return self.battery_soc < threshold


@dataclass
class Task:
    """任务"""
    id: str
    task_type: str
    priority: TaskPriority
    
    # 位置
    pickup_location: Location
    dropoff_location: Optional[Location] = None
    
    # 时间窗口
    pickup_time_window: Tuple[datetime, datetime]
    dropoff_time_window: Optional[Tuple[datetime, datetime]] = None
    
    # 需求
    cargo_weight: float = 0
    estimated_distance: Optional[float] = None
    estimated_duration: Optional[float] = None  # 分钟
    
    # 分配
    assigned_vehicle: Optional[str] = None
    status: str = "pending"
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class FleetManager:
    """车队管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 车辆和任务
        self.vehicles: Dict[str, Vehicle] = {}
        self.tasks: Dict[str, Task] = {}
        self.pending_tasks: List[Tuple[int, datetime, str]] = []  # 优先级队列
        
        # 充电站
        self.charging_stations: List[Dict] = []
        
        # 统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'total_distance': 0,
            'total_energy': 0
        }
        
        logger.info("车队管理器初始化")
    
    def add_vehicle(self, vehicle: Vehicle):
        """添加车辆"""
        self.vehicles[vehicle.id] = vehicle
        logger.info(f"添加车辆: {vehicle.id} ({vehicle.name})")
    
    def remove_vehicle(self, vehicle_id: str):
        """移除车辆"""
        if vehicle_id in self.vehicles:
            del self.vehicles[vehicle_id]
            logger.info(f"移除车辆: {vehicle_id}")
    
    def update_vehicle_status(self, 
                             vehicle_id: str,
                             status: VehicleStatus,
                             location: Location = None,
                             battery_soc: float = None):
        """更新车辆状态"""
        if vehicle_id not in self.vehicles:
            logger.warning(f"车辆不存在: {vehicle_id}")
            return
        
        vehicle = self.vehicles[vehicle_id]
        vehicle.status = status
        
        if location:
            vehicle.location = location
        
        if battery_soc is not None:
            vehicle.battery_soc = battery_soc
        
        logger.info(
            f"更新车辆状态: {vehicle_id} -> {status.value} | "
            f"SOC: {vehicle.battery_soc:.1f}%"
        )
    
    def add_task(self, task: Task):
        """添加任务"""
        self.tasks[task.id] = task
        self.stats['total_tasks'] += 1
        
        # 加入优先级队列（负优先级用于最大堆）
        heapq.heappush(
            self.pending_tasks,
            (-task.priority.value, task.created_at, task.id)
        )
        
        logger.info(f"添加任务: {task.id} (优先级: {task.priority.name})")
    
    def get_available_vehicles(self) -> List[Vehicle]:
        """获取可用车辆"""
        return [
            v for v in self.vehicles.values()
            if v.status == VehicleStatus.AVAILABLE
        ]
    
    def find_best_vehicle(self, task: Task) -> Optional[Vehicle]:
        """为任务找最佳车辆"""
        available_vehicles = self.get_available_vehicles()
        
        if not available_vehicles:
            return None
        
        # 评分函数
        def score_vehicle(vehicle: Vehicle) -> float:
            score = 0
            
            # 1. 距离因素
            distance = vehicle.location.distance_to(task.pickup_location)
            score -= distance * 2  # 距离越近越好
            
            # 2. 电量因素
            if task.estimated_distance:
                required_soc = (task.estimated_distance / vehicle.max_range) * 100
                if vehicle.battery_soc >= required_soc + 10:  # 预留10%
                    score += 10
                else:
                    score -= 20  # 电量不足惩罚
            
            # 3. 历史表现
            if vehicle.total_trips > 0:
                efficiency = vehicle.total_distance / vehicle.total_trips
                score += min(efficiency / 10, 5)  # 经验加成
            
            return score
        
        # 选择得分最高的车辆
        best_vehicle = max(available_vehicles, key=score_vehicle)
        
        return best_vehicle
    
    def assign_task(self, task_id: str, vehicle_id: str) -> bool:
        """分配任务"""
        if task_id not in self.tasks or vehicle_id not in self.vehicles:
            return False
        
        task = self.tasks[task_id]
        vehicle = self.vehicles[vehicle_id]
        
        # 更新任务
        task.assigned_vehicle = vehicle_id
        task.assigned_at = datetime.now()
        task.status = "assigned"
        
        # 更新车辆
        vehicle.current_task_id = task_id
        vehicle.status = VehicleStatus.IN_USE
        
        # 估算完成时间
        if task.estimated_duration:
            vehicle.estimated_available_time = (
                datetime.now() + timedelta(minutes=task.estimated_duration)
            )
        
        logger.info(f"任务分配: {task_id} -> {vehicle_id}")
        
        return True
    
    def auto_dispatch(self) -> List[Tuple[str, str]]:
        """自动调度"""
        assignments = []
        
        while self.pending_tasks:
            # 获取最高优先级任务
            _, _, task_id = heapq.heappop(self.pending_tasks)
            task = self.tasks[task_id]
            
            if task.status != "pending":
                continue
            
            # 找最佳车辆
            best_vehicle = self.find_best_vehicle(task)
            
            if best_vehicle:
                if self.assign_task(task_id, best_vehicle.id):
                    assignments.append((task_id, best_vehicle.id))
            else:
                # 没有可用车辆，重新加入队列
                heapq.heappush(
                    self.pending_tasks,
                    (-task.priority.value, task.created_at, task_id)
                )
                break
        
        if assignments:
            logger.info(f"自动调度完成 | 分配 {len(assignments)} 个任务")
        
        return assignments
    
    def complete_task(self, task_id: str, 
                     actual_distance: float,
                     actual_energy: float):
        """完成任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        vehicle_id = task.assigned_vehicle
        
        if vehicle_id and vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            
            # 更新车辆状态
            vehicle.status = VehicleStatus.AVAILABLE
            vehicle.current_task_id = None
            vehicle.estimated_available_time = None
            vehicle.total_trips += 1
            vehicle.total_distance += actual_distance
            vehicle.total_energy_used += actual_energy
            vehicle.odometer += actual_distance
            
            # 更新电量
            energy_per_km = actual_energy / actual_distance
            vehicle.battery_soc -= energy_per_km * 10  # 简化估算
        
        # 更新任务状态
        task.status = "completed"
        task.completed_at = datetime.now()
        
        # 更新统计
        self.stats['completed_tasks'] += 1
        self.stats['total_distance'] += actual_distance
        self.stats['total_energy'] += actual_energy
        
        logger.info(f"任务完成: {task_id}")
        
        return True
    
    def optimize_charging_schedule(self,
                                   time_window: Tuple[datetime, datetime],
                                   charger_capacity: int = 5) -> Dict[str, datetime]:
        """
        优化充电调度
        
        Returns:
            车辆ID -> 充电时间
        """
        # 找出需要充电的车辆
        vehicles_needing_charge = [
            v for v in self.vehicles.values()
            if v.needs_charging() and v.status in [VehicleStatus.AVAILABLE]
        ]
        
        if not vehicles_needing_charge:
            return {}
        
        # 按电量排序（电量最低的优先）
        vehicles_needing_charge.sort(key=lambda v: v.battery_soc)
        
        # 时间窗口
        start_time, end_time = time_window
        current_time = start_time
        
        # 充电时长估算（假设150kW充电桩）
        charge_time_per_vehicle = timedelta(minutes=30)  # 简化：30分钟
        
        schedule = {}
        
        for i, vehicle in enumerate(vehicles_needing_charge):
            # 考虑充电桩容量
            if i > 0 and i % charger_capacity == 0:
                current_time += charge_time_per_vehicle
            
            if current_time + charge_time_per_vehicle <= end_time:
                schedule[vehicle.id] = current_time
                current_time += timedelta(minutes=5)  # 时间片
        
        logger.info(f"充电调度完成 | 安排 {len(schedule)} 辆车")
        
        return schedule
    
    def get_fleet_status(self) -> Dict:
        """获取车队状态"""
        status_counts = defaultdict(int)
        total_soc = 0
        total_available_range = 0
        
        for vehicle in self.vehicles.values():
            status_counts[vehicle.status.value] += 1
            total_soc += vehicle.battery_soc
            total_available_range += vehicle.available_range()
        
        return {
            'total_vehicles': len(self.vehicles),
            'status_distribution': dict(status_counts),
            'average_soc': total_soc / len(self.vehicles) if self.vehicles else 0,
            'total_available_range': total_available_range,
            'pending_tasks': len(self.pending_tasks),
            'stats': self.stats
        }
    
    def get_vehicle_recommendations(self, vehicle_id: str) -> List[str]:
        """获取车辆建议"""
        if vehicle_id not in self.vehicles:
            return []
        
        vehicle = self.vehicles[vehicle_id]
        recommendations = []
        
        # 充电建议
        if vehicle.battery_soc < 20:
            recommendations.append("🔋 电量低，建议立即充电")
        elif vehicle.battery_soc < 40:
            recommendations.append("⚡ 建议在下次休息时充电")
        
        # 维护建议
        if vehicle.odometer > 50000:
            recommendations.append("🔧 里程较高，建议检查")
        
        # 效率建议
        if vehicle.total_trips > 10:
            avg_distance = vehicle.total_distance / vehicle.total_trips
            if avg_distance < 10:
                recommendations.append("📈 行程较短，考虑优化路线")
        
        return recommendations


def test_fleet_manager():
    """测试车队管理器"""
    manager = FleetManager()
    
    # 添加车辆
    vehicles = [
        Vehicle(
            id="v1",
            name="Tesla 1",
            model="Model 3",
            status=VehicleStatus.AVAILABLE,
            location=Location(37.7749, -122.4194),
            battery_soc=85,
            odometer=25000,
            max_range=400
        ),
        Vehicle(
            id="v2",
            name="Tesla 2",
            model="Model Y",
            status=VehicleStatus.AVAILABLE,
            location=Location(37.7849, -122.4094),
            battery_soc=45,
            odometer=15000,
            max_range=450
        ),
        Vehicle(
            id="v3",
            name="Tesla 3",
            model="Model S",
            status=VehicleStatus.CHARGING,
            location=Location(37.7649, -122.4294),
            battery_soc=15,
            odometer=60000,
            max_range=500
        ),
    ]
    
    for v in vehicles:
        manager.add_vehicle(v)
    
    # 添加任务
    tasks = [
        Task(
            id="t1",
            task_type="delivery",
            priority=TaskPriority.HIGH,
            pickup_location=Location(37.7849, -122.4094),
            dropoff_location=Location(37.8049, -122.4394),
            pickup_time_window=(
                datetime.now(),
                datetime.now() + timedelta(hours=2)
            ),
            estimated_distance=15,
            estimated_duration=30
        ),
        Task(
            id="t2",
            task_type="pickup",
            priority=TaskPriority.NORMAL,
            pickup_location=Location(37.7749, -122.4194),
            pickup_time_window=(
                datetime.now(),
                datetime.now() + timedelta(hours=3)
            ),
            estimated_distance=10,
            estimated_duration=20
        ),
    ]
    
    for t in tasks:
        manager.add_task(t)
    
    # 自动调度
    assignments = manager.auto_dispatch()
    
    logger.info("\n" + "="*60)
    logger.info("车队状态")
    logger.info("="*60)
    
    fleet_status = manager.get_fleet_status()
    logger.info(f"总车辆数: {fleet_status['total_vehicles']}")
    logger.info(f"状态分布: {fleet_status['status_distribution']}")
    logger.info(f"平均电量: {fleet_status['average_soc']:.1f}%")
    logger.info(f"待处理任务: {fleet_status['pending_tasks']}")
    
    # 充电调度
    logger.info("\n" + "="*60)
    logger.info("充电调度")
    logger.info("="*60)
    
    charge_schedule = manager.optimize_charging_schedule(
        (datetime.now(), datetime.now() + timedelta(hours=8)),
        charger_capacity=2
    )
    
    for vehicle_id, charge_time in charge_schedule.items():
        logger.info(f"  {vehicle_id}: {charge_time.strftime('%H:%M')}")
    
    # 车辆建议
    logger.info("\n" + "="*60)
    logger.info("车辆建议")
    logger.info("="*60)
    
    for vehicle in vehicles:
        recommendations = manager.get_vehicle_recommendations(vehicle.id)
        if recommendations:
            logger.info(f"\n{vehicle.name}:")
            for rec in recommendations:
                logger.info(f"  {rec}")
    
    logger.success("\n✅ 车队管理器测试完成")


if __name__ == "__main__":
    test_fleet_manager()
