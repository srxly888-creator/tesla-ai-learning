"""
电池健康预测训练脚本
Training script for battery health prediction
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from data.preprocessor import BatteryDataPreprocessor
from models.lstm import BatteryHealthModel


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    """加载数据"""
    logger.info(f"加载数据: {data_path}")
    
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"不支持的数据格式: {data_path}")
    
    logger.info(f"数据形状: {data.shape}")
    
    return data


def create_data_loaders(X_train, y_train, X_val, y_val, 
                       batch_size: int) -> tuple:
    """创建数据加载器"""
    # 转换为PyTorch张量
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader


def main(args):
    """主函数"""
    logger.info("🔋 电池健康预测模型训练")
    logger.info("="*60)
    
    # 加载配置
    config = load_config(args.config)
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    
    # 加载数据
    data = load_data(data_config['path'])
    
    # 预处理
    logger.info("数据预处理...")
    preprocessor = BatteryDataPreprocessor(data_config)
    processed_data = preprocessor.process(data)
    
    # 创建序列
    X, y = preprocessor.create_sequences(
        processed_data,
        sequence_length=model_config['sequence_length'],
        target_column=data_config['target_column']
    )
    
    # 划分数据集
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        preprocessor.split_data(
            X, y,
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio']
        )
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=training_config['batch_size']
    )
    
    # 更新模型配置
    model_config['input_size'] = X_train.shape[2]
    
    # 创建模型
    logger.info("创建模型...")
    model = BatteryHealthModel(model_config)
    
    # 训练
    logger.info("开始训练...")
    history = model.fit(
        train_loader,
        val_loader,
        epochs=training_config['epochs'],
        early_stopping_patience=training_config['early_stopping_patience']
    )
    
    # 评估
    logger.info("评估模型...")
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    test_metrics = model.validate(test_loader)
    logger.info(
        f"测试集性能 | "
        f"Loss: {test_metrics['loss']:.4f} | "
        f"MAE: {test_metrics['mae']:.4f}"
    )
    
    # 保存模型
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'battery_health_model.pth'
    model.save(str(model_path))
    
    # 保存预处理器
    import joblib
    preprocessor_path = output_dir / 'preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_path)
    
    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    
    logger.success("✅ 训练完成！")
    logger.info(f"模型保存于: {model_path}")
    logger.info(f"预处理器保存于: {preprocessor_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="电池健康预测模型训练")
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/trained',
        help='输出目录'
    )
    
    args = parser.parse_args()
    main(args)
