"""
LSTM电池健康预测模型
LSTM-based battery health prediction model
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger


class LSTMBatteryPredictor(nn.Module):
    """LSTM电池健康预测模型"""
    
    def __init__(self, config: Dict):
        super(LSTMBatteryPredictor, self).__init__()
        
        # 模型参数
        self.input_size = config.get('input_size', 8)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.output_size = config.get('output_size', 1)
        self.dropout = config.get('dropout', 0.2)
        self.bidirectional = config.get('bidirectional', False)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, self.output_size)
        )
        
        # 注意力机制（可选）
        self.attention = AttentionLayer(self.hidden_size) if config.get('use_attention', False) else None
        
        logger.info(f"LSTM模型初始化 | 参数: {self.count_parameters()}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        
        # 使用注意力或最后一个时间步
        if self.attention is not None:
            # 注意力加权
            context, attention_weights = self.attention(lstm_out)
            out = context
        else:
            # 使用最后一个时间步的输出
            out = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        
        return out
    
    def count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionLayer(nn.Module):
    """注意力层"""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # lstm_output: (batch_size, seq_len, hidden_size)
        
        # 计算注意力权重
        attention_weights = self.attention(lstm_output)
        # (batch_size, seq_len, 1)
        
        # 加权求和
        context = torch.sum(lstm_output * attention_weights, dim=1)
        # (batch_size, hidden_size)
        
        return context, attention_weights


class BatteryHealthModel:
    """电池健康预测模型封装"""
    
    def __init__(self, config: Dict, device: str = 'auto'):
        self.config = config
        self.device = self._get_device(device)
        
        # 创建模型
        self.model = LSTMBatteryPredictor(config).to(self.device)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        logger.info(f"模型创建完成 | 设备: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """确定计算设备"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def train_epoch(self, train_loader) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            
            # 计算损失
            loss = self.criterion(predictions.squeeze(), batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            mae = torch.mean(torch.abs(predictions.squeeze() - batch_y))
            total_mae += mae.item()
            num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }
        
        return metrics
    
    def validate(self, val_loader) -> Dict:
        """验证"""
        self.model.eval()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 前向传播
                predictions = self.model(batch_x)
                
                # 计算损失
                loss = self.criterion(predictions.squeeze(), batch_y)
                mae = torch.mean(torch.abs(predictions.squeeze() - batch_y))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }
        
        return metrics
    
    def fit(self, train_loader, val_loader, epochs: int = 100,
           early_stopping_patience: int = 20) -> Dict:
        """完整训练"""
        logger.info(f"开始训练 | Epochs: {epochs}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            
            # 学习率调度
            self.scheduler.step(val_metrics['loss'])
            
            # 日志
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.4f}"
                )
            
            # 早停
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # 保存最佳模型
                self.best_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停于 Epoch {epoch+1}")
                    break
        
        # 恢复最佳模型
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        logger.success("✅ 训练完成")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        
        logger.info(f"模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {})
        
        logger.info(f"模型已加载: {path}")


def test_model():
    """测试模型"""
    # 配置
    config = {
        'input_size': 8,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 1,
        'dropout': 0.2,
        'learning_rate': 0.001
    }
    
    # 创建模型
    model_wrapper = BatteryHealthModel(config)
    
    # 创建模拟数据
    batch_size = 32
    seq_len = 50
    input_size = 8
    
    X = torch.randn(batch_size, seq_len, input_size)
    y = torch.randn(batch_size)
    
    # 前向传播测试
    output = model_wrapper.model(X)
    
    logger.info(f"输入形状: {X.shape}")
    logger.info(f"输出形状: {output.shape}")
    logger.info(f"模型参数: {model_wrapper.model.count_parameters():,}")
    
    logger.success("✅ 模型测试完成")


if __name__ == "__main__":
    test_model()
