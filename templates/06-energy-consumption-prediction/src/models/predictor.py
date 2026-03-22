"""
能耗预测器
Energy consumption predictor using ensemble models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import shap

from loguru import logger


@dataclass
class PredictionResult:
    """预测结果"""
    energy_kwh: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    shap_values: Optional[Dict[str, float]] = None


class EnergyPredictor:
    """能耗预测器"""
    
    # 特征列表
    FEATURE_COLUMNS = [
        # 距离和速度
        'distance_km',
        'avg_speed_kmh',
        'max_speed_kmh',
        'speed_variance',
        
        # 海拔
        'elevation_gain_m',
        'elevation_loss_m',
        'net_elevation_m',
        
        # 天气
        'temperature_c',
        'humidity_pct',
        'wind_speed_ms',
        'precipitation_mm',
        
        # 车辆状态
        'initial_soc_pct',
        'battery_capacity_kwh',
        'vehicle_mass_kg',
        
        # 道路
        'highway_ratio',
        'urban_ratio',
        'traffic_density',
        
        # 时间
        'hour_of_day',
        'day_of_week',
        'is_weekend',
        
        # 历史特征
        'avg_energy_last_10_trips',
        'driver_efficiency_score'
    ]
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 模型
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 特征重要性
        self.feature_importance = {}
        
        logger.info("能耗预测器初始化")
    
    def build_models(self):
        """构建集成模型"""
        # XGBoost
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        
        # Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Ridge (作为基线)
        self.models['ridge'] = Ridge(alpha=1.0)
        
        logger.info(f"构建 {len(self.models)} 个模型")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征"""
        df = data.copy()
        
        # 确保所有特征存在
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0  # 默认值
        
        # 特征工程
        # 1. 速度相关特征
        if 'avg_speed_kmh' in df.columns and 'distance_km' in df.columns:
            df['estimated_time_h'] = df['distance_km'] / (df['avg_speed_kmh'] + 1)
        
        # 2. 海拔影响
        if 'elevation_gain_m' in df.columns:
            df['elevation_factor'] = df['elevation_gain_m'] / 1000  # 每1km
        
        # 3. 温度影响
        if 'temperature_c' in df.columns:
            # 最适温度20°C，偏离越大能耗越高
            df['temp_deviation'] = np.abs(df['temperature_c'] - 20)
        
        # 4. 天气综合影响
        weather_cols = ['temperature_c', 'humidity_pct', 'wind_speed_ms', 'precipitation_mm']
        if all(col in df.columns for col in weather_cols):
            df['weather_factor'] = (
                df['temp_deviation'] * 0.3 +
                df['precipitation_mm'] * 0.5 +
                df['wind_speed_ms'] * 0.2
            )
        
        # 5. SOC影响（低SOC时效率略低）
        if 'initial_soc_pct' in df.columns:
            df['soc_factor'] = np.where(
                df['initial_soc_pct'] < 20,
                1.1,  # 低SOC时能耗增加10%
                1.0
            )
        
        return df[self.FEATURE_COLUMNS + ['estimated_time_h', 'elevation_factor', 
                                          'temp_deviation', 'weather_factor', 'soc_factor']]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练模型"""
        logger.info("开始训练...")
        
        # 准备特征
        X_prepared = self.prepare_features(X)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X_prepared)
        
        # 构建模型
        self.build_models()
        
        # 训练每个模型
        for name, model in self.models.items():
            logger.info(f"训练 {name}...")
            model.fit(X_scaled, y)
            
            # 交叉验证
            scores = cross_val_score(model, X_scaled, y, cv=5, 
                                    scoring='neg_mean_absolute_error')
            logger.info(f"  {name} MAE: {-scores.mean():.2f} (±{scores.std():.2f})")
        
        # 计算特征重要性
        self._compute_feature_importance(X_prepared)
        
        self.is_fitted = True
        logger.success("✅ 训练完成")
    
    def predict(self, X: pd.DataFrame, 
               return_confidence: bool = True) -> PredictionResult:
        """预测能耗"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 准备特征
        X_prepared = self.prepare_features(X)
        X_scaled = self.scaler.transform(X_prepared)
        
        # 集成预测
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # 平均集成
        ensemble_pred = np.mean(predictions, axis=0)[0]
        
        # 置信区间
        if return_confidence:
            pred_std = np.std(predictions, axis=0)[0]
            confidence_interval = (
                ensemble_pred - 1.96 * pred_std,
                ensemble_pred + 1.96 * pred_std
            )
        else:
            confidence_interval = (ensemble_pred, ensemble_pred)
        
        # SHAP值（可选）
        shap_values = None
        if self.config.get('compute_shap', False):
            shap_values = self._compute_shap_values(X_scaled[0])
        
        return PredictionResult(
            energy_kwh=ensemble_pred,
            confidence_interval=confidence_interval,
            feature_importance=self.feature_importance,
            shap_values=shap_values
        )
    
    def _compute_feature_importance(self, X: pd.DataFrame):
        """计算特征重要性"""
        # 使用随机森林的特征重要性
        if 'random_forest' in self.models:
            rf_model = self.models['random_forest']
            importances = rf_model.feature_importances_
            
            self.feature_importance = {
                col: imp 
                for col, imp in zip(X.columns, importances)
            }
            
            # 排序
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), 
                      key=lambda x: x[1], reverse=True)
            )
            
            logger.info("特征重要性 (Top 10):")
            for i, (feat, imp) in enumerate(list(self.feature_importance.items())[:10]):
                logger.info(f"  {i+1}. {feat}: {imp:.4f}")
    
    def _compute_shap_values(self, X: np.ndarray) -> Dict[str, float]:
        """计算SHAP值"""
        try:
            if 'xgboost' in self.models:
                explainer = shap.TreeExplainer(self.models['xgboost'])
                shap_vals = explainer.shap_values(X.reshape(1, -1))[0]
                
                return {
                    col: val 
                    for col, val in zip(self.FEATURE_COLUMNS, shap_vals)
                }
        except Exception as e:
            logger.warning(f"SHAP值计算失败: {e}")
        
        return None
    
    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        for name, model in self.models.items():
            joblib.dump(model, path / f"{name}.pkl")
        
        # 保存scaler
        joblib.dump(self.scaler, path / "scaler.pkl")
        
        # 保存特征重要性
        joblib.dump(self.feature_importance, path / "feature_importance.pkl")
        
        logger.info(f"模型保存至: {path}")
    
    def load(self, path: str):
        """加载模型"""
        path = Path(path)
        
        # 构建模型
        self.build_models()
        
        # 加载模型
        for name in self.models.keys():
            model_path = path / f"{name}.pkl"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
        
        # 加载scaler
        self.scaler = joblib.load(path / "scaler.pkl")
        
        # 加载特征重要性
        self.feature_importance = joblib.load(path / "feature_importance.pkl")
        
        self.is_fitted = True
        logger.info(f"模型加载自: {path}")


def test_predictor():
    """测试预测器"""
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'distance_km': np.random.uniform(10, 500, n_samples),
        'avg_speed_kmh': np.random.uniform(40, 120, n_samples),
        'max_speed_kmh': np.random.uniform(60, 150, n_samples),
        'speed_variance': np.random.uniform(5, 30, n_samples),
        'elevation_gain_m': np.random.uniform(0, 1000, n_samples),
        'elevation_loss_m': np.random.uniform(0, 1000, n_samples),
        'net_elevation_m': np.random.uniform(-500, 500, n_samples),
        'temperature_c': np.random.uniform(-10, 40, n_samples),
        'humidity_pct': np.random.uniform(20, 100, n_samples),
        'wind_speed_ms': np.random.uniform(0, 15, n_samples),
        'precipitation_mm': np.random.uniform(0, 20, n_samples),
        'initial_soc_pct': np.random.uniform(10, 100, n_samples),
        'battery_capacity_kwh': 75,
        'vehicle_mass_kg': 2000,
        'highway_ratio': np.random.uniform(0, 1, n_samples),
        'urban_ratio': np.random.uniform(0, 1, n_samples),
        'traffic_density': np.random.uniform(0, 1, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'avg_energy_last_10_trips': np.random.uniform(12, 25, n_samples),
        'driver_efficiency_score': np.random.uniform(60, 100, n_samples)
    })
    
    # 目标变量（能耗）
    # 简化模型：能耗 = 距离 * 基础效率 + 各种修正
    y = (
        data['distance_km'] * 0.15 +  # 基础
        data['elevation_gain_m'] / 100 * 0.5 +  # 海拔
        (data['avg_speed_kmh'] - 80).abs() * 0.02 * data['distance_km'] +  # 速度偏离
        np.abs(data['temperature_c'] - 20) * 0.1 * data['distance_km'] / 100 +  # 温度
        np.random.normal(0, 2, n_samples)  # 噪声
    )
    y = np.clip(y, 0, 150)
    
    # 训练模型
    predictor = EnergyPredictor({'compute_shap': False})
    predictor.fit(data, y)
    
    # 预测
    test_sample = data.iloc[[0]]
    result = predictor.predict(test_sample)
    
    logger.info(f"\n预测结果:")
    logger.info(f"  能耗: {result.energy_kwh:.2f} kWh")
    logger.info(f"  置信区间: ({result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f})")
    
    # 保存和加载测试
    predictor.save('models/energy_predictor')
    
    predictor2 = EnergyPredictor()
    predictor2.load('models/energy_predictor')
    
    result2 = predictor2.predict(test_sample)
    logger.info(f"  加载后预测: {result2.energy_kwh:.2f} kWh")
    
    logger.success("✅ 预测器测试完成")


if __name__ == "__main__":
    test_predictor()
