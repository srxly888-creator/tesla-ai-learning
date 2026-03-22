"""
电池数据预处理器
Battery data preprocessor for cleaning and normalization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger


class BatteryDataPreprocessor:
    """电池数据预处理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scalers = {}
        
        logger.info("电池数据预处理器初始化")
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """完整的数据处理流程"""
        # 1. 清洗数据
        data = self.clean_data(data)
        
        # 2. 处理缺失值
        data = self.handle_missing_values(data)
        
        # 3. 特征工程
        data = self.engineer_features(data)
        
        # 4. 标准化
        data = self.normalize(data)
        
        # 5. 异常检测
        data = self.detect_anomalies(data)
        
        return data
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 删除重复行
        data = data.drop_duplicates()
        
        # 删除明显错误的值
        data = data[
            (data['voltage'] >= 200) & (data['voltage'] <= 450) &  # 电压范围
            (data['current'] >= -500) & (data['current'] <= 500) &  # 电流范围
            (data['temperature'] >= -20) & (data['temperature'] <= 60)  # 温度范围
        ]
        
        # 按时间排序
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"数据清洗完成 | 剩余 {len(data)} 条记录")
        
        return data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 数值列 - 线性插值
        numeric_cols = ['voltage', 'current', 'temperature', 'soc', 'capacity']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = data[col].interpolate(method='linear')
        
        # 如果还有缺失，用前向填充
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        missing_count = data.isnull().sum().sum()
        logger.info(f"缺失值处理完成 | 剩余缺失值: {missing_count}")
        
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        df = data.copy()
        
        # 1. 功率
        df['power'] = df['voltage'] * df['current'] / 1000  # kW
        
        # 2. 能量
        df['energy'] = df['power'] * df.get('duration', 1)  # kWh
        
        # 3. 内阻估算 (V = E - I*R)
        if 'voltage' in df.columns and 'current' in df.columns:
            # 使用简单的线性回归估算内阻
            from sklearn.linear_model import LinearRegression
            
            if len(df) > 10:
                model = LinearRegression()
                X = df['current'].values.reshape(-1, 1)
                y = df['voltage'].values
                model.fit(X, y)
                df['internal_resistance'] = -model.coef_[0]
            else:
                df['internal_resistance'] = 0.05  # 默认值
        
        # 4. 温度特征
        df['temp_rolling_mean'] = df['temperature'].rolling(window=10, min_periods=1).mean()
        df['temp_rolling_std'] = df['temperature'].rolling(window=10, min_periods=1).std()
        
        # 5. 电压特征
        df['voltage_rolling_mean'] = df['voltage'].rolling(window=10, min_periods=1).mean()
        df['voltage_rolling_std'] = df['voltage'].rolling(window=10, min_periods=1).std()
        
        # 6. 充放电状态
        df['is_charging'] = (df['current'] > 5).astype(int)
        df['is_discharging'] = (df['current'] < -5).astype(int)
        
        # 7. SOH (State of Health)
        if 'capacity' in df.columns and 'nominal_capacity' in df.columns:
            df['soh'] = df['capacity'] / df['nominal_capacity'] * 100
        else:
            # 估算SOH
            df['soh'] = 100 - (df['cycle_number'] * 0.02) if 'cycle_number' in df.columns else 100
        
        # 8. 使用强度
        df['usage_intensity'] = abs(df['current']) / 100
        
        # 9. 时间特征
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"特征工程完成 | 特征数量: {len(df.columns)}")
        
        return df
    
    def normalize(self, data: pd.DataFrame, 
                 fit: bool = True) -> pd.DataFrame:
        """数据标准化"""
        df = data.copy()
        
        # 需要标准化的特征
        features_to_scale = [
            'voltage', 'current', 'temperature', 'soc', 'power',
            'temp_rolling_mean', 'temp_rolling_std',
            'voltage_rolling_mean', 'voltage_rolling_std'
        ]
        
        for feature in features_to_scale:
            if feature in df.columns:
                if fit:
                    self.scalers[feature] = StandardScaler()
                    df[feature] = self.scalers[feature].fit_transform(
                        df[[feature]]
                    )
                else:
                    if feature in self.scalers:
                        df[feature] = self.scalers[feature].transform(
                            df[[feature]]
                        )
        
        logger.info("数据标准化完成")
        
        return df
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """异常检测"""
        df = data.copy()
        
        # 使用IQR方法检测异常
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        anomaly_flags = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            anomaly_flags.append(
                (df[col] < lower_bound) | (df[col] > upper_bound)
            )
        
        # 标记异常
        df['is_anomaly'] = np.any(anomaly_flags, axis=0)
        
        anomaly_count = df['is_anomaly'].sum()
        logger.info(f"异常检测完成 | 检测到 {anomaly_count} 个异常点")
        
        return df
    
    def create_sequences(self, data: pd.DataFrame, 
                        sequence_length: int = 50,
                        target_column: str = 'soh') -> Tuple[np.ndarray, np.ndarray]:
        """创建时序数据序列"""
        feature_columns = [
            'voltage', 'current', 'temperature', 'soc', 'power',
            'temp_rolling_mean', 'voltage_rolling_mean', 'usage_intensity'
        ]
        
        # 过滤存在的特征
        feature_columns = [col for col in feature_columns if col in data.columns]
        
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            # 特征序列
            seq = data[feature_columns].iloc[i:i+sequence_length].values
            sequences.append(seq)
            
            # 目标值（下一个时间步的SOH）
            target = data[target_column].iloc[i + sequence_length]
            targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        logger.info(f"创建序列完成 | 形状: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15) -> Tuple:
        """划分数据集"""
        n_samples = len(X)
        
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logger.info(
            f"数据划分完成 | "
            f"训练集: {len(X_train)}, "
            f"验证集: {len(X_val)}, "
            f"测试集: {len(X_test)}"
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def test_preprocessor():
    """测试预处理器"""
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'voltage': np.random.normal(350, 20, n_samples),
        'current': np.random.normal(-50, 30, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'soc': np.random.uniform(20, 100, n_samples),
        'capacity': np.random.normal(75, 2, n_samples),
        'cycle_number': np.arange(n_samples) // 24
    })
    
    # 添加一些缺失值
    data.loc[100:110, 'voltage'] = np.nan
    
    # 预处理
    preprocessor = BatteryDataPreprocessor()
    processed_data = preprocessor.process(data)
    
    logger.info(f"处理后数据形状: {processed_data.shape}")
    logger.info(f"特征列: {processed_data.columns.tolist()}")
    
    # 创建序列
    X, y = preprocessor.create_sequences(processed_data, sequence_length=50)
    
    logger.success("✅ 预处理器测试完成")


if __name__ == "__main__":
    test_preprocessor()
