"""
故障预测器
Failure predictor for vehicle components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from loguru import logger


@dataclass
class ComponentHealth:
    """组件健康状态"""
    component: str
    health_score: float  # 0-100
    failure_probability: float  # 0-1
    remaining_useful_life: Optional[int]  # 天数
    last_maintenance: Optional[datetime]
    next_maintenance: Optional[datetime]
    recommendation: str


@dataclass
class MaintenanceAlert:
    """维护告警"""
    component: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    recommended_action: str
    urgency: str  # 'immediate', 'soon', 'scheduled'


class FailurePredictor:
    """故障预测器"""
    
    # 组件列表
    COMPONENTS = [
        'battery',
        'motor',
        'brake_pads',
        'brake_rotors',
        'tires',
        'suspension',
        'steering',
        'hvac',
        'cooling_system'
    ]
    
    # 特征列表
    FEATURES = {
        'battery': [
            'capacity', 'internal_resistance', 'temperature_avg',
            'charge_cycles', 'age_days', 'voltage_variance',
            'charging_speed_avg', 'depth_of_discharge_avg'
        ],
        'motor': [
            'temperature_avg', 'vibration_level', 'power_output',
            'efficiency', 'noise_level', 'age_days', 'usage_hours'
        ],
        'brake_pads': [
            'thickness', 'wear_rate', 'temperature_avg',
            'usage_count', 'age_days', 'braking_force_avg'
        ],
        'tires': [
            'tread_depth', 'pressure', 'temperature_avg',
            'mileage', 'age_days', 'wear_pattern_score'
        ]
    }
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 模型
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        
        # 组件健康阈值
        self.health_thresholds = {
            'critical': 20,
            'warning': 40,
            'attention': 60,
            'good': 80
        }
        
        logger.info("故障预测器初始化")
    
    def train(self, 
              data: pd.DataFrame,
              component: str,
              target_column: str = 'failure_within_30_days'):
        """
        训练特定组件的预测模型
        
        Args:
            data: 训练数据
            component: 组件名称
            target_column: 目标列
        """
        logger.info(f"训练 {component} 预测模型...")
        
        # 获取特征
        features = self.FEATURES.get(component, [])
        available_features = [f for f in features if f in data.columns]
        
        if not available_features:
            logger.warning(f"没有找到 {component} 的特征")
            return
        
        X = data[available_features]
        y = data[target_column]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # 评估
        y_pred = model.predict(X_test_scaled)
        logger.info(f"\n{component} 模型性能:")
        logger.info(classification_report(y_test, y_pred))
        
        # 保存模型和scaler
        self.models[component] = model
        self.scalers[component] = scaler
        self.thresholds[component] = {
            'features': available_features
        }
        
        logger.success(f"✅ {component} 模型训练完成")
    
    def predict(self, 
               component_data: Dict[str, float],
               component: str) -> ComponentHealth:
        """
        预测组件健康状态
        
        Args:
            component_data: 组件数据字典
            component: 组件名称
        
        Returns:
            组件健康状态
        """
        if component not in self.models:
            logger.warning(f"没有找到 {component} 的模型")
            return self._get_default_health(component)
        
        # 准备特征
        features = self.thresholds[component]['features']
        X = np.array([[component_data.get(f, 0) for f in features]])
        
        # 标准化
        X_scaled = self.scalers[component].transform(X)
        
        # 预测
        model = self.models[component]
        failure_prob = model.predict_proba(X_scaled)[0, 1]
        
        # 计算健康分数
        health_score = (1 - failure_prob) * 100
        
        # 估算剩余寿命
        rul = self._estimate_rul(component, component_data, failure_prob)
        
        # 生成建议
        recommendation = self._generate_recommendation(
            component, health_score, failure_prob, rul
        )
        
        # 计算下次维护时间
        next_maintenance = self._calculate_next_maintenance(
            component, health_score
        )
        
        return ComponentHealth(
            component=component,
            health_score=health_score,
            failure_probability=failure_prob,
            remaining_useful_life=rul,
            last_maintenance=None,  # 需要从数据库获取
            next_maintenance=next_maintenance,
            recommendation=recommendation
        )
    
    def predict_all(self, 
                   vehicle_data: Dict[str, Dict[str, float]]) -> List[ComponentHealth]:
        """预测所有组件"""
        results = []
        
        for component in self.COMPONENTS:
            if component in vehicle_data:
                health = self.predict(vehicle_data[component], component)
                results.append(health)
        
        return results
    
    def _estimate_rul(self,
                     component: str,
                     data: Dict[str, float],
                     failure_prob: float) -> int:
        """估算剩余使用寿命（天）"""
        # 简化模型：基于故障概率线性估算
        
        # 基础寿命（天）
        base_lifetimes = {
            'battery': 3650,  # 10年
            'motor': 5475,    # 15年
            'brake_pads': 730,  # 2年
            'brake_rotors': 1825,  # 5年
            'tires': 1095,     # 3年
            'suspension': 3650,  # 10年
        }
        
        base_life = base_lifetimes.get(component, 1825)
        
        # 根据故障概率调整
        if failure_prob < 0.1:
            rul = base_life * 0.8
        elif failure_prob < 0.3:
            rul = base_life * 0.5
        elif failure_prob < 0.5:
            rul = base_life * 0.3
        else:
            rul = base_life * 0.1
        
        # 根据使用情况进一步调整
        if 'age_days' in data:
            rul = max(rul - data['age_days'], 0)
        
        return int(rul)
    
    def _generate_recommendation(self,
                                component: str,
                                health_score: float,
                                failure_prob: float,
                                rul: int) -> str:
        """生成维护建议"""
        component_names = {
            'battery': '电池',
            'motor': '电机',
            'brake_pads': '刹车片',
            'brake_rotors': '刹车盘',
            'tires': '轮胎',
            'suspension': '悬挂系统'
        }
        
        name = component_names.get(component, component)
        
        if health_score >= self.health_thresholds['good']:
            return f"{name}状态良好，正常使用即可"
        elif health_score >= self.health_thresholds['attention']:
            return f"{name}状态正常，建议每年检查一次"
        elif health_score >= self.health_thresholds['warning']:
            return f"建议在未来{rul//2}天内检查{name}"
        elif health_score >= self.health_thresholds['critical']:
            return f"建议尽快（{rul//3}天内）更换或维修{name}"
        else:
            return f"⚠️ {name}状态严重，建议立即更换！"
    
    def _calculate_next_maintenance(self,
                                   component: str,
                                   health_score: float) -> datetime:
        """计算下次维护时间"""
        # 基础维护间隔（天）
        maintenance_intervals = {
            'battery': 365,
            'motor': 730,
            'brake_pads': 180,
            'tires': 365,
            'suspension': 730
        }
        
        interval = maintenance_intervals.get(component, 365)
        
        # 根据健康分数调整
        if health_score < self.health_thresholds['critical']:
            interval = 7  # 一周内
        elif health_score < self.health_thresholds['warning']:
            interval = min(interval, 30)  # 一个月内
        elif health_score < self.health_thresholds['attention']:
            interval = min(interval, 90)  # 三个月内
        
        return datetime.now() + timedelta(days=interval)
    
    def _get_default_health(self, component: str) -> ComponentHealth:
        """获取默认健康状态"""
        return ComponentHealth(
            component=component,
            health_score=100,
            failure_probability=0,
            remaining_useful_life=None,
            last_maintenance=None,
            next_maintenance=None,
            recommendation="无法评估，请提供更多数据"
        )
    
    def check_alerts(self, 
                    health_statuses: List[ComponentHealth]) -> List[MaintenanceAlert]:
        """检查是否需要告警"""
        alerts = []
        
        for health in health_statuses:
            if health.health_score < self.health_thresholds['critical']:
                alerts.append(MaintenanceAlert(
                    component=health.component,
                    severity='critical',
                    message=f"{health.component} 状态严重",
                    recommended_action=health.recommendation,
                    urgency='immediate'
                ))
            elif health.health_score < self.health_thresholds['warning']:
                alerts.append(MaintenanceAlert(
                    component=health.component,
                    severity='high',
                    message=f"{health.component} 需要关注",
                    recommended_action=health.recommendation,
                    urgency='soon'
                ))
        
        return alerts
    
    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for component in self.models:
            joblib.dump(
                self.models[component],
                path / f"{component}_model.pkl"
            )
            joblib.dump(
                self.scalers[component],
                path / f"{component}_scaler.pkl"
            )
        
        joblib.dump(self.thresholds, path / "thresholds.pkl")
        
        logger.info(f"模型保存至: {path}")
    
    def load(self, path: str):
        """加载模型"""
        path = Path(path)
        
        for component in self.COMPONENTS:
            model_path = path / f"{component}_model.pkl"
            if model_path.exists():
                self.models[component] = joblib.load(model_path)
                self.scalers[component] = joblib.load(
                    path / f"{component}_scaler.pkl"
                )
        
        self.thresholds = joblib.load(path / "thresholds.pkl")
        
        logger.info(f"模型加载自: {path}")


def test_predictor():
    """测试故障预测器"""
    # 创建模拟训练数据
    np.random.seed(42)
    n_samples = 1000
    
    # 刹车片数据
    brake_data = pd.DataFrame({
        'thickness': np.random.uniform(2, 12, n_samples),
        'wear_rate': np.random.uniform(0.01, 0.1, n_samples),
        'temperature_avg': np.random.uniform(50, 200, n_samples),
        'usage_count': np.random.randint(1000, 50000, n_samples),
        'age_days': np.random.randint(0, 730, n_samples),
        'braking_force_avg': np.random.uniform(100, 500, n_samples),
        'failure_within_30_days': np.random.randint(0, 2, n_samples)
    })
    
    # 训练模型
    predictor = FailurePredictor()
    predictor.train(brake_data, 'brake_pads')
    
    # 预测
    test_data = {
        'thickness': 4.5,
        'wear_rate': 0.08,
        'temperature_avg': 150,
        'usage_count': 35000,
        'age_days': 400,
        'braking_force_avg': 350
    }
    
    health = predictor.predict(test_data, 'brake_pads')
    
    logger.info("\n" + "="*60)
    logger.info("组件健康状态")
    logger.info("="*60)
    logger.info(f"组件: {health.component}")
    logger.info(f"健康分数: {health.health_score:.1f}")
    logger.info(f"故障概率: {health.failure_probability:.2%}")
    logger.info(f"剩余寿命: {health.remaining_useful_life} 天")
    logger.info(f"建议: {health.recommendation}")
    
    # 检查告警
    alerts = predictor.check_alerts([health])
    if alerts:
        logger.info("\n告警:")
        for alert in alerts:
            logger.info(f"  [{alert.severity.upper()}] {alert.message}")
    
    logger.success("\n✅ 故障预测器测试完成")


if __name__ == "__main__":
    test_predictor()
