"""
驾驶风格聚类分析
Driver style clustering using unsupervised learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


class DrivingStyleClusterer:
    """驾驶风格聚类器"""
    
    # 驾驶风格标签
    STYLE_LABELS = {
        0: 'Aggressive',     # 激进型
        1: 'Moderate',       # 温和型
        2: 'Eco',            # 经济型
        3: 'Sporty',         # 运动型
    }
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.n_clusters = config.get('n_clusters', 4)
        self.scaler = StandardScaler()
        self.model = None
        self.pca = None
        
        logger.info(f"驾驶风格聚类器初始化 | 聚类数: {self.n_clusters}")
    
    def extract_features(self, trip_data: pd.DataFrame) -> pd.DataFrame:
        """
        提取驾驶特征
        
        Args:
            trip_data: 行程数据，包含速度、加速度、时间等
        
        Returns:
            特征DataFrame
        """
        features = {}
        
        # 1. 速度特征
        if 'speed' in trip_data.columns:
            features['avg_speed'] = trip_data['speed'].mean()
            features['max_speed'] = trip_data['speed'].max()
            features['speed_std'] = trip_data['speed'].std()
            features['speed_variance'] = trip_data['speed'].var()
        
        # 2. 加速度特征
        if 'acceleration' in trip_data.columns:
            features['avg_acceleration'] = trip_data['acceleration'].mean()
            features['max_acceleration'] = trip_data['acceleration'].max()
            features['min_acceleration'] = trip_data['acceleration'].min()
            
            # 急加速次数 (> 2 m/s^2)
            features['hard_accel_count'] = (trip_data['acceleration'] > 2).sum()
            
            # 急刹车次数 (< -3 m/s^2)
            features['hard_brake_count'] = (trip_data['acceleration'] < -3).sum()
        
        # 3. 能耗特征
        if 'energy_consumption' in trip_data.columns:
            features['avg_energy_rate'] = trip_data['energy_consumption'].mean()
            features['max_energy_rate'] = trip_data['energy_consumption'].max()
            features['total_energy'] = trip_data['energy_consumption'].sum()
        
        # 4. 时间特征
        if 'duration' in trip_data.columns:
            features['trip_duration'] = trip_data['duration'].iloc[0]
        
        if 'distance' in trip_data.columns:
            features['trip_distance'] = trip_data['distance'].iloc[0]
        
        # 5. 驾驶平稳性指标
        if 'speed' in trip_data.columns and 'acceleration' in trip_data.columns:
            # 速度变化率
            speed_changes = np.abs(np.diff(trip_data['speed']))
            features['speed_change_rate'] = speed_changes.mean()
            
            # 加速度平滑度（加速度变化的标准差）
            accel_changes = np.abs(np.diff(trip_data['acceleration']))
            features['accel_smoothness'] = 1 / (1 + accel_changes.std())
        
        # 6. 效率指标
        if 'speed' in trip_data.columns and 'energy_consumption' in trip_data.columns:
            # 能耗效率 (kWh/100km)
            if features.get('trip_distance', 0) > 0:
                features['energy_efficiency'] = (
                    features['total_energy'] / features['trip_distance'] * 100
                )
        
        # 7. 超速比例
        if 'speed' in trip_data.columns and 'speed_limit' in trip_data.columns:
            features['overspeed_ratio'] = (
                trip_data['speed'] > trip_data['speed_limit']
            ).mean()
        
        # 8. 怠速比例
        if 'speed' in trip_data.columns:
            features['idle_ratio'] = (trip_data['speed'] < 1).mean()
        
        return pd.DataFrame([features])
    
    def extract_features_batch(self, 
                               trips: List[pd.DataFrame]) -> pd.DataFrame:
        """批量提取特征"""
        all_features = []
        
        for i, trip in enumerate(trips):
            features = self.extract_features(trip)
            features['trip_id'] = i
            all_features.append(features)
        
        return pd.concat(all_features, ignore_index=True)
    
    def fit(self, features: pd.DataFrame, method: str = 'kmeans'):
        """
        训练聚类模型
        
        Args:
            features: 特征DataFrame
            method: 聚类方法 ('kmeans', 'dbscan', 'hierarchical')
        """
        # 移除非特征列
        feature_cols = [col for col in features.columns if col != 'trip_id']
        X = features[feature_cols].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA降维（可选）
        if self.config.get('use_pca', False):
            n_components = min(self.config.get('pca_components', 10), X.shape[1])
            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)
            logger.info(f"PCA降维: {X.shape[1]} -> {n_components}")
        
        # 聚类
        if method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        elif method == 'dbscan':
            self.model = DBSCAN(
                eps=self.config.get('eps', 0.5),
                min_samples=self.config.get('min_samples', 5)
            )
        elif method == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters
            )
        else:
            raise ValueError(f"不支持的聚类方法: {method}")
        
        # 训练
        self.labels = self.model.fit_predict(X_scaled)
        
        # 评估
        if len(set(self.labels)) > 1:
            silhouette = silhouette_score(X_scaled, self.labels)
            db_score = davies_bouldin_score(X_scaled, self.labels)
            
            logger.info(f"聚类完成 | 轮廓系数: {silhouette:.3f} | DB指数: {db_score:.3f}")
        
        # 统计每个聚类的特征
        self._analyze_clusters(features, self.labels)
        
        return self
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """预测驾驶风格"""
        feature_cols = [col for col in features.columns if col != 'trip_id']
        X = features[feature_cols].values
        X = np.nan_to_num(X, nan=0)
        X_scaled = self.scaler.transform(X)
        
        if self.pca:
            X_scaled = self.pca.transform(X_scaled)
        
        if hasattr(self.model, 'predict'):
            labels = self.model.predict(X_scaled)
        else:
            # DBSCAN等不支持predict
            labels = self.model.fit_predict(X_scaled)
        
        return labels
    
    def _analyze_clusters(self, features: pd.DataFrame, labels: np.ndarray):
        """分析每个聚类的特征"""
        features_with_labels = features.copy()
        features_with_labels['cluster'] = labels
        
        # 计算每个聚类的平均特征
        cluster_stats = features_with_labels.groupby('cluster').mean()
        
        logger.info("\n聚类统计:")
        for cluster_id in range(self.n_clusters):
            if cluster_id in cluster_stats.index:
                stats = cluster_stats.loc[cluster_id]
                logger.info(f"\n聚类 {cluster_id} ({self.STYLE_LABELS.get(cluster_id, 'Unknown')}):")
                logger.info(f"  样本数: {(labels == cluster_id).sum()}")
                logger.info(f"  平均速度: {stats.get('avg_speed', 0):.1f} km/h")
                logger.info(f"  急加速次数: {stats.get('hard_accel_count', 0):.1f}")
                logger.info(f"  急刹车次数: {stats.get('hard_brake_count', 0):.1f}")
                logger.info(f"  能耗效率: {stats.get('energy_efficiency', 0):.1f} kWh/100km")
    
    def visualize_clusters(self, 
                          features: pd.DataFrame,
                          output_path: str = None):
        """可视化聚类结果"""
        feature_cols = [col for col in features.columns if col != 'trip_id']
        X = features[feature_cols].values
        X = np.nan_to_num(X, nan=0)
        X_scaled = self.scaler.transform(X)
        
        # 2D可视化
        if self.pca and self.pca.n_components >= 2:
            X_2d = X_scaled[:, :2]
        else:
            pca_2d = PCA(n_components=2)
            X_2d = pca_2d.fit_transform(X_scaled)
        
        # 绘图
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=self.labels,
            cmap='viridis',
            alpha=0.6,
            s=100
        )
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Driving Style Clusters')
        
        # 添加聚类中心
        if hasattr(self.model, 'cluster_centers_'):
            if self.pca:
                centers_2d = self.pca.transform(
                    self.scaler.inverse_transform(self.model.cluster_centers_)
                )[:, :2]
            else:
                centers_2d = self.model.cluster_centers_[:, :2]
            
            plt.scatter(
                centers_2d[:, 0], centers_2d[:, 1],
                c='red',
                marker='x',
                s=200,
                linewidths=3,
                label='Cluster Centers'
            )
            plt.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化保存至: {output_path}")
        
        plt.show()
    
    def get_style_description(self, cluster_id: int) -> Dict:
        """获取驾驶风格描述"""
        descriptions = {
            0: {
                'style': 'Aggressive',
                'description': '高速、急加速、急刹车频繁',
                'characteristics': [
                    '高平均速度',
                    '频繁急加速和急刹车',
                    '能耗较高',
                    '可能存在安全风险'
                ],
                'recommendations': [
                    '减少急加速可提升15%续航',
                    '保持匀速行驶',
                    '预留更长的刹车距离'
                ]
            },
            1: {
                'style': 'Moderate',
                'description': '正常驾驶风格',
                'characteristics': [
                    '适中的速度',
                    '偶尔急加速/刹车',
                    '能耗正常',
                    '安全性较好'
                ],
                'recommendations': [
                    '继续保持良好驾驶习惯',
                    '可尝试节能驾驶技巧'
                ]
            },
            2: {
                'style': 'Eco',
                'description': '节能驾驶风格',
                'characteristics': [
                    '较低的速度',
                    '平缓的加减速',
                    '能耗最低',
                    '最环保'
                ],
                'recommendations': [
                    '优秀的节能驾驶习惯',
                    '保持当前驾驶方式'
                ]
            },
            3: {
                'style': 'Sporty',
                'description': '运动驾驶风格',
                'characteristics': [
                    '较高速度',
                    '快速加速',
                    '精准控制',
                    '能耗偏高'
                ],
                'recommendations': [
                    '享受驾驶乐趣的同时注意安全',
                    '在赛道或安全环境下发挥性能'
                ]
            }
        }
        
        return descriptions.get(cluster_id, descriptions[1])


def test_clusterer():
    """测试聚类器"""
    # 创建模拟行程数据
    np.random.seed(42)
    n_trips = 200
    trips = []
    
    for i in range(n_trips):
        # 随机生成行程类型
        trip_type = np.random.choice(['aggressive', 'moderate', 'eco', 'sporty'])
        
        if trip_type == 'aggressive':
            speed = np.random.normal(80, 30, 100)
            acceleration = np.random.normal(0.5, 3, 100)
            energy = np.random.normal(20, 8, 100)
        elif trip_type == 'moderate':
            speed = np.random.normal(60, 15, 100)
            acceleration = np.random.normal(0, 1.5, 100)
            energy = np.random.normal(15, 5, 100)
        elif trip_type == 'eco':
            speed = np.random.normal(50, 10, 100)
            acceleration = np.random.normal(0, 0.8, 100)
            energy = np.random.normal(12, 3, 100)
        else:  # sporty
            speed = np.random.normal(90, 25, 100)
            acceleration = np.random.normal(0.3, 2.5, 100)
            energy = np.random.normal(22, 7, 100)
        
        trip_data = pd.DataFrame({
            'speed': np.clip(speed, 0, 150),
            'acceleration': acceleration,
            'energy_consumption': np.clip(energy, 5, 40)
        })
        
        trips.append(trip_data)
    
    # 提取特征
    clusterer = DrivingStyleClusterer({'n_clusters': 4})
    features = clusterer.extract_features_batch(trips)
    
    logger.info(f"提取特征: {features.shape}")
    
    # 聚类
    clusterer.fit(features, method='kmeans')
    
    # 可视化
    clusterer.visualize_clusters(features)
    
    # 获取风格描述
    for i in range(4):
        desc = clusterer.get_style_description(i)
        logger.info(f"\n聚类 {i}: {desc['style']}")
        logger.info(f"描述: {desc['description']}")
    
    logger.success("✅ 聚类器测试完成")


if __name__ == "__main__":
    test_clusterer()
