"""
数据质量检查器
Data quality checker for autonomous driving data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from loguru import logger


@dataclass
class QualityReport:
    """数据质量报告"""
    timestamp: datetime
    total_records: int
    valid_records: int
    invalid_records: int
    issues: List[Dict]
    metrics: Dict[str, float]
    passed: bool


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 质量阈值
        self.thresholds = {
            'completeness': 0.95,  # 完整性
            'accuracy': 0.98,      # 准确性
            'timeliness': 300,     # 时效性（秒）
            'consistency': 0.99,   # 一致性
        }
        
        # 传感器范围
        self.sensor_ranges = {
            'voltage': (200, 450),
            'current': (-500, 500),
            'temperature': (-40, 85),
            'speed': (0, 250),
            'acceleration': (-10, 10),
            'latitude': (-90, 90),
            'longitude': (-180, 180),
            'soc': (0, 100)
        }
        
        logger.info("数据质量检查器初始化")
    
    def check(self, data: pd.DataFrame, 
             schema: Dict = None) -> QualityReport:
        """
        执行数据质量检查
        
        Args:
            data: 待检查的数据
            schema: 数据模式（期望的列和类型）
        
        Returns:
            质量报告
        """
        logger.info(f"开始数据质量检查 | 记录数: {len(data)}")
        
        issues = []
        metrics = {}
        
        # 1. 完整性检查
        completeness_issues, completeness_score = self._check_completeness(data)
        issues.extend(completeness_issues)
        metrics['completeness'] = completeness_score
        
        # 2. 准确性检查
        accuracy_issues, accuracy_score = self._check_accuracy(data)
        issues.extend(accuracy_issues)
        metrics['accuracy'] = accuracy_score
        
        # 3. 时效性检查
        timeliness_issues, timeliness_score = self._check_timeliness(data)
        issues.extend(timeliness_issues)
        metrics['timeliness'] = timeliness_score
        
        # 4. 一致性检查
        consistency_issues, consistency_score = self._check_consistency(data)
        issues.extend(consistency_issues)
        metrics['consistency'] = consistency_score
        
        # 5. 模式检查
        if schema:
            schema_issues = self._check_schema(data, schema)
            issues.extend(schema_issues)
        
        # 6. 异常检测
        anomaly_issues = self._detect_anomalies(data)
        issues.extend(anomaly_issues)
        
        # 计算有效记录数
        valid_records = len(data) - len(set(
            issue.get('index', -1) for issue in issues
        ))
        
        # 判断是否通过
        passed = all([
            metrics['completeness'] >= self.thresholds['completeness'],
            metrics['accuracy'] >= self.thresholds['accuracy'],
            metrics['timeliness'] <= self.thresholds['timeliness'],
            metrics['consistency'] >= self.thresholds['consistency']
        ])
        
        # 生成报告
        report = QualityReport(
            timestamp=datetime.now(),
            total_records=len(data),
            valid_records=valid_records,
            invalid_records=len(data) - valid_records,
            issues=issues,
            metrics=metrics,
            passed=passed
        )
        
        self._log_report(report)
        
        return report
    
    def _check_completeness(self, data: pd.DataFrame) -> Tuple[List[Dict], float]:
        """完整性检查"""
        issues = []
        
        # 检查缺失值
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_ratio = missing_count / len(data)
            
            if missing_ratio > 0.05:  # 超过5%缺失
                issues.append({
                    'type': 'completeness',
                    'column': col,
                    'description': f'高缺失率: {missing_ratio:.2%}',
                    'severity': 'high' if missing_ratio > 0.2 else 'medium',
                    'count': missing_count
                })
        
        # 计算完整性得分
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness_score = 1 - (missing_cells / total_cells)
        
        return issues, completeness_score
    
    def _check_accuracy(self, data: pd.DataFrame) -> Tuple[List[Dict], float]:
        """准确性检查"""
        issues = []
        out_of_range_count = 0
        
        # 检查数值范围
        for col, (min_val, max_val) in self.sensor_ranges.items():
            if col in data.columns:
                # 找出超出范围的值
                mask = (data[col] < min_val) | (data[col] > max_val)
                invalid_count = mask.sum()
                
                if invalid_count > 0:
                    out_of_range_count += invalid_count
                    issues.append({
                        'type': 'accuracy',
                        'column': col,
                        'description': f'超出范围 [{min_val}, {max_val}]',
                        'severity': 'high',
                        'count': invalid_count,
                        'indices': data[mask].index.tolist()[:10]  # 前10个
                    })
        
        # 计算准确性得分
        total_values = data.select_dtypes(include=[np.number]).size
        accuracy_score = 1 - (out_of_range_count / total_values) if total_values > 0 else 1.0
        
        return issues, accuracy_score
    
    def _check_timeliness(self, data: pd.DataFrame) -> Tuple[List[Dict], float]:
        """时效性检查"""
        issues = []
        timeliness_score = 0
        
        if 'timestamp' in data.columns:
            # 转换为datetime
            timestamps = pd.to_datetime(data['timestamp'])
            
            # 计算时间延迟
            now = datetime.now()
            delays = [(now - ts).total_seconds() for ts in timestamps]
            
            avg_delay = np.mean(delays)
            max_delay = np.max(delays)
            
            timeliness_score = avg_delay
            
            if max_delay > 3600:  # 超过1小时
                issues.append({
                    'type': 'timeliness',
                    'description': f'最大延迟: {max_delay:.0f}秒',
                    'severity': 'medium',
                    'avg_delay': avg_delay,
                    'max_delay': max_delay
                })
        
        return issues, timeliness_score
    
    def _check_consistency(self, data: pd.DataFrame) -> Tuple[List[Dict], float]:
        """一致性检查"""
        issues = []
        inconsistencies = 0
        
        # 1. 检查速度和加速度的一致性
        if 'speed' in data.columns and 'acceleration' in data.columns:
            # 速度变化应该与加速度一致
            speed_diff = data['speed'].diff()
            expected_accel = speed_diff / 0.1  # 假设0.1秒采样
            
            inconsistency_mask = np.abs(
                expected_accel - data['acceleration']
            ) > 5  # 允许5 m/s^2误差
            
            inconsistency_count = inconsistency_mask.sum()
            if inconsistency_count > 0:
                inconsistencies += inconsistency_count
                issues.append({
                    'type': 'consistency',
                    'description': '速度和加速度不一致',
                    'severity': 'medium',
                    'count': inconsistency_count
                })
        
        # 2. 检查SOC单调性（充电时应该增加，放电时减少）
        if 'soc' in data.columns and 'current' in data.columns:
            soc_diff = data['soc'].diff()
            
            # 充电时SOC应该增加
            charging_mask = data['current'] > 5
            charging_soc_decrease = (soc_diff[charging_mask] < -0.5).sum()
            
            if charging_soc_decrease > 0:
                inconsistencies += charging_soc_decrease
                issues.append({
                    'type': 'consistency',
                    'description': '充电时SOC减少',
                    'severity': 'high',
                    'count': charging_soc_decrease
                })
        
        # 计算一致性得分
        consistency_score = 1 - (inconsistencies / len(data)) if len(data) > 0 else 1.0
        
        return issues, consistency_score
    
    def _check_schema(self, data: pd.DataFrame, schema: Dict) -> List[Dict]:
        """模式检查"""
        issues = []
        
        expected_columns = schema.get('columns', [])
        expected_types = schema.get('types', {})
        
        # 检查缺失的列
        missing_cols = set(expected_columns) - set(data.columns)
        if missing_cols:
            issues.append({
                'type': 'schema',
                'description': f'缺失列: {missing_cols}',
                'severity': 'high'
            })
        
        # 检查数据类型
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if expected_type not in actual_type:
                    issues.append({
                        'type': 'schema',
                        'column': col,
                        'description': f'类型不匹配: 期望{expected_type}, 实际{actual_type}',
                        'severity': 'medium'
                    })
        
        return issues
    
    def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """异常检测"""
        issues = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = data[col].dropna()
            
            if len(values) < 10:
                continue
            
            # 使用IQR方法检测异常
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            anomalies = ((values < lower_bound) | (values > upper_bound)).sum()
            
            if anomalies > len(values) * 0.01:  # 超过1%异常
                issues.append({
                    'type': 'anomaly',
                    'column': col,
                    'description': f'检测到 {anomalies} 个异常值',
                    'severity': 'medium',
                    'count': anomalies
                })
        
        return issues
    
    def _log_report(self, report: QualityReport):
        """记录质量报告"""
        logger.info("\n" + "="*60)
        logger.info("数据质量报告")
        logger.info("="*60)
        logger.info(f"时间戳: {report.timestamp}")
        logger.info(f"总记录数: {report.total_records}")
        logger.info(f"有效记录: {report.valid_records} ({report.valid_records/report.total_records*100:.1f}%)")
        logger.info(f"无效记录: {report.invalid_records}")
        logger.info("\n质量指标:")
        logger.info(f"  完整性: {report.metrics['completeness']:.2%}")
        logger.info(f"  准确性: {report.metrics['accuracy']:.2%}")
        logger.info(f"  时效性: {report.metrics['timeliness']:.1f}秒")
        logger.info(f"  一致性: {report.metrics['consistency']:.2%}")
        logger.info(f"\n发现问题: {len(report.issues)}")
        
        if report.issues:
            logger.info("\n问题详情:")
            for i, issue in enumerate(report.issues[:5], 1):  # 只显示前5个
                logger.info(f"  {i}. [{issue['severity'].upper()}] {issue['description']}")
        
        logger.info(f"\n质量检查: {'✅ 通过' if report.passed else '❌ 未通过'}")
        logger.info("="*60)
    
    def generate_alert(self, report: QualityReport) -> Optional[str]:
        """生成告警"""
        if not report.passed:
            alert = {
                'timestamp': report.timestamp.isoformat(),
                'severity': 'high',
                'message': '数据质量未达标',
                'details': {
                    'completeness': report.metrics['completeness'],
                    'accuracy': report.metrics['accuracy'],
                    'consistency': report.metrics['consistency'],
                    'issue_count': len(report.issues)
                }
            }
            return json.dumps(alert, indent=2)
        
        return None


def test_quality_checker():
    """测试数据质量检查器"""
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        'voltage': np.random.normal(350, 20, n_samples),
        'current': np.random.normal(-50, 30, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'speed': np.random.uniform(0, 120, n_samples),
        'soc': np.random.uniform(20, 100, n_samples)
    })
    
    # 添加一些问题
    # 1. 缺失值
    data.loc[100:110, 'voltage'] = np.nan
    
    # 2. 超出范围的值
    data.loc[200, 'temperature'] = 100  # 超出范围
    
    # 3. 一致性问题
    data['acceleration'] = 0
    data.loc[50:60, 'acceleration'] = 10  # 与速度变化不一致
    
    # 检查
    checker = DataQualityChecker()
    report = checker.check(data)
    
    # 生成告警
    alert = checker.generate_alert(report)
    if alert:
        logger.info(f"\n告警:\n{alert}")
    
    logger.success("✅ 数据质量检查器测试完成")


if __name__ == "__main__":
    test_quality_checker()
