"""
充电优化器 - 线性规划方法
Charging optimizer using linear programming
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from pulp import (
        LpProblem, LpMinimize, LpVariable, lpSum,
        LpStatus, value
    )
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    logger.warning("PuLP未安装，优化功能受限")


@dataclass
class ChargingSlot:
    """充电时段"""
    start_time: str
    end_time: str
    power_kw: float
    energy_kwh: float
    cost_usd: float


class ChargingOptimizer:
    """充电优化器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 充电器参数
        self.max_power = config.get('max_power', 11)  # kW
        self.min_power = config.get('min_power', 1.5)  # kW
        self.charging_efficiency = config.get('efficiency', 0.95)
        
        # 电池参数
        self.battery_capacity = config.get('battery_capacity', 75)  # kWh
        
        # 优化参数
        self.time_resolution = config.get('time_resolution', 15)  # 分钟
        self.max_cost_weight = config.get('cost_weight', 1.0)
        self.battery_health_weight = config.get('battery_health_weight', 0.3)
        
        logger.info("充电优化器初始化完成")
    
    def optimize(self,
                electricity_prices: List[float],
                current_soc: float,
                target_soc: float,
                deadline_hours: float,
                current_capacity: float = None) -> List[ChargingSlot]:
        """
        优化充电调度
        
        Args:
            electricity_prices: 未来N个时段的电价列表 ($/kWh)
            current_soc: 当前SOC (%)
            target_soc: 目标SOC (%)
            deadline_hours: 截止时间（小时）
            current_capacity: 当前电池容量 (kWh)，默认等于额定容量
        
        Returns:
            充电时段列表
        """
        if not PULP_AVAILABLE:
            logger.warning("PuLP不可用，使用简化优化")
            return self._simple_optimize(
                electricity_prices, current_soc, target_soc, deadline_hours
            )
        
        # 计算所需能量
        if current_capacity is None:
            current_capacity = self.battery_capacity
        
        current_energy = current_soc / 100 * current_capacity
        target_energy = target_soc / 100 * current_capacity
        required_energy = target_energy - current_energy
        
        if required_energy <= 0:
            logger.info("无需充电")
            return []
        
        # 考虑充电效率
        required_energy_with_loss = required_energy / self.charging_efficiency
        
        # 时间段数量
        n_periods = len(electricity_prices)
        time_step_hours = self.time_resolution / 60
        
        # 创建优化问题
        prob = LpProblem("Charging_Optimization", LpMinimize)
        
        # 决策变量：每个时段的充电功率
        power_vars = [
            LpVariable(f"power_{i}", lowBound=0, upBound=self.max_power)
            for i in range(n_periods)
        ]
        
        # 二进制变量：是否在该时段充电
        charging_vars = [
            LpVariable(f"charging_{i}", cat='Binary')
            for i in range(n_periods)
        ]
        
        # 目标函数：最小化成本
        total_cost = lpSum([
            power_vars[i] * time_step_hours * electricity_prices[i]
            for i in range(n_periods)
        ])
        
        # 添加电池健康成本（避免高功率充电）
        battery_health_cost = lpSum([
            power_vars[i] ** 2 * 0.01  # 二次惩罚
            for i in range(n_periods)
        ])
        
        prob += (
            self.max_cost_weight * total_cost + 
            self.battery_health_weight * battery_health_cost
        )
        
        # 约束1：总充电量满足需求
        total_energy = lpSum([
            power_vars[i] * time_step_hours
            for i in range(n_periods)
        ])
        prob += total_energy >= required_energy_with_loss
        
        # 约束2：在截止时间前完成
        deadline_periods = int(deadline_hours * 60 / self.time_resolution)
        if deadline_periods < n_periods:
            for i in range(deadline_periods, n_periods):
                prob += power_vars[i] == 0
        
        # 约束3：功率限制（如果充电，则功率 >= 最小功率）
        for i in range(n_periods):
            prob += power_vars[i] >= self.min_power * charging_vars[i]
            prob += power_vars[i] <= self.max_power * charging_vars[i]
        
        # 求解
        prob.solve()
        
        # 检查求解状态
        if LpStatus[prob.status] != 'Optimal':
            logger.warning(f"优化失败: {LpStatus[prob.status]}")
            return self._simple_optimize(
                electricity_prices, current_soc, target_soc, deadline_hours
            )
        
        # 提取结果
        schedule = self._extract_schedule(
            power_vars, electricity_prices, time_step_hours
        )
        
        logger.info(f"优化完成 | 总成本: ${value(total_cost):.2f}")
        
        return schedule
    
    def _simple_optimize(self,
                        electricity_prices: List[float],
                        current_soc: float,
                        target_soc: float,
                        deadline_hours: float) -> List[ChargingSlot]:
        """简化优化（贪心算法）"""
        # 计算所需能量
        required_energy = (target_soc - current_soc) / 100 * self.battery_capacity
        required_energy_with_loss = required_energy / self.charging_efficiency
        
        # 找到电价最低的时段
        n_periods = len(electricity_prices)
        time_step_hours = self.time_resolution / 60
        
        # 按电价排序
        price_ranking = sorted(
            range(n_periods),
            key=lambda i: electricity_prices[i]
        )
        
        # 贪心选择
        schedule = []
        remaining_energy = required_energy_with_loss
        deadline_periods = int(deadline_hours * 60 / self.time_resolution)
        
        for period in price_ranking:
            if period >= deadline_periods:
                continue
            
            if remaining_energy <= 0:
                break
            
            # 计算该时段充电量
            energy = min(
                remaining_energy,
                self.max_power * time_step_hours
            )
            
            power = energy / time_step_hours
            cost = energy * electricity_prices[period]
            
            start_time = self._period_to_time(period)
            end_time = self._period_to_time(period + 1)
            
            schedule.append(ChargingSlot(
                start_time=start_time,
                end_time=end_time,
                power_kw=power,
                energy_kwh=energy,
                cost_usd=cost
            ))
            
            remaining_energy -= energy
        
        return schedule
    
    def _extract_schedule(self,
                         power_vars: List,
                         electricity_prices: List[float],
                         time_step_hours: float) -> List[ChargingSlot]:
        """从优化结果提取充电计划"""
        schedule = []
        
        for i, power_var in enumerate(power_vars):
            power = value(power_var)
            
            if power > 0:
                energy = power * time_step_hours
                cost = energy * electricity_prices[i]
                
                start_time = self._period_to_time(i)
                end_time = self._period_to_time(i + 1)
                
                schedule.append(ChargingSlot(
                    start_time=start_time,
                    end_time=end_time,
                    power_kw=power,
                    energy_kwh=energy,
                    cost_usd=cost
                ))
        
        return schedule
    
    def _period_to_time(self, period: int) -> str:
        """时段转时间字符串"""
        hours = (period * self.time_resolution) // 60
        minutes = (period * self.time_resolution) % 60
        return f"{hours:02d}:{minutes:02d}"
    
    def optimize_with_solar(self,
                           electricity_prices: List[float],
                           solar_forecast: List[float],
                           current_soc: float,
                           target_soc: float,
                           deadline_hours: float) -> List[ChargingSlot]:
        """考虑太阳能的优化"""
        # 调整电价：有太阳能时优先使用
        adjusted_prices = []
        for i, (price, solar) in enumerate(zip(electricity_prices, solar_forecast)):
            # 如果太阳能充足，降低有效电价
            if solar > 0:
                # 假设太阳能边际成本为0
                adjusted_price = price * 0.1  # 降低90%
            else:
                adjusted_price = price
            adjusted_prices.append(adjusted_price)
        
        return self.optimize(
            adjusted_prices, current_soc, target_soc, deadline_hours
        )


def test_optimizer():
    """测试优化器"""
    # 创建模拟电价数据（24小时，15分钟分辨率）
    n_periods = 24 * 4
    base_price = 0.12
    
    # 模拟TOU电价（高峰期贵，低谷期便宜）
    electricity_prices = []
    for i in range(n_periods):
        hour = (i * 15) // 60
        if 14 <= hour <= 19:  # 高峰期
            price = base_price * 2.0
        elif 22 <= hour or hour <= 6:  # 低谷期
            price = base_price * 0.5
        else:
            price = base_price
        electricity_prices.append(price)
    
    # 创建优化器
    optimizer = ChargingOptimizer({
        'max_power': 11,
        'battery_capacity': 75,
        'time_resolution': 15
    })
    
    # 优化充电
    schedule = optimizer.optimize(
        electricity_prices=electricity_prices,
        current_soc=20,
        target_soc=90,
        deadline_hours=10
    )
    
    # 打印结果
    logger.info(f"优化结果: {len(schedule)} 个充电时段")
    total_cost = 0
    total_energy = 0
    
    for slot in schedule:
        logger.info(
            f"  {slot.start_time}-{slot.end_time} | "
            f"功率: {slot.power_kw:.1f}kW | "
            f"能量: {slot.energy_kwh:.2f}kWh | "
            f"成本: ${slot.cost_usd:.2f}"
        )
        total_cost += slot.cost_usd
        total_energy += slot.energy_kwh
    
    logger.info(f"总能量: {total_energy:.2f} kWh")
    logger.info(f"总成本: ${total_cost:.2f}")
    
    logger.success("✅ 优化器测试完成")


if __name__ == "__main__":
    test_optimizer()
