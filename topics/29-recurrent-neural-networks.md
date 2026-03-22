# Recurrent Neural Networks for Video and Sequential Data
 ## Introduction
 Recurrent Neural Networks (RNNs) are powerful for processing sequential data like video. This document covers RNN architectures for video understanding and temporal modeling in autonomous driving. ## RNN Architectures
### LSTM (Long Short-Term Memory)
```python
import torch
import torch.nn as nn

import math

class LSTMCell(nn.Module):
    """LSTM cell implementation"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gates
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        
        # Cell state
        self.cell_state = None
        
    def forward(self, x, state=None):
        """Forward pass"""
        if state is None:
            state = self._init_state(x.size(0), self.hidden_dim)
            
            # Initialize gates
            f = torch.sigmoid(self.forget_gate(x))
            i = torch.sigmoid(self.input_gate(x))
            o = torch.sigmoid(self.output_gate(x))
            
            # Cell state
            self.cell_state = (f * state['forget'], i * state['input'], o)
            state['output'] = o * state['cell']
        
        return state['output'], state
```
### GRU (Gated Recurrent Unit)
```python
class GRUCell(nn.Module):
    """GRU cell implementation"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Gates
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.new_gate = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, state=None):
        """Forward pass"""
        if state is None:
            state = self._init_state(x.size(0), self.hidden_dim)
            
            # Gates
            z = torch.sigmoid(self.update_gate(x))
            r = torch.sigmoid(self.reset_gate(x))
            
            # Update
            new_state = torch.tanh(self.new_gate(state))
            state = (1 - z) * state + z * r + z * new_state
            
            return state, state
```
### Bidirectional RNN
```python
class BidirectionalRNN(nn.Module):
    """Bidirectional RNN for temporal modeling"""
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.forward_rnn = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        self.backward_rnn = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        
    def forward(self, x):
        """Bidirectional forward pass"""
        seq_len = x.size(1)
        
        # Forward direction
        h_forward, _ = self.forward_rnn(x)
        
        # Backward direction
        h_backward, _ = self.backward_rnn(x.flip(dims=[0, 1]))
        
        # Concatenate
        h_cat = torch.cat([h_forward, h_backward], dim=2)
        
        return h_cat
```
## Video Understanding Applications
### Action Recognition
```python
class ActionRecognition(nn.Module):
    """Recogn actions in video"""
    
    def __init__(self, num_classes, hidden_dim=256):
        super().__init__()
        # Feature extractor (3D CNN)
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2)
        )
        
        # Temporal modeling
        self.rnn = nn.LSTM(128, hidden_dim, batch_first=True)
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """Forward pass"""
        # x: [B, C, T, H, W]
        batch_size = x.size(0)
        seq_len = x.size(2)
        
        # Extract features for each frame
        frame_features = []
        for t in range(seq_len):
            feat = self.features(x[:, t])
            frame_features.append(feat)
        
        # Stack features
        frame_features = torch.stack(frame_features, dim=1)  # [T, C, H, W]
        
        # Temporal modeling
        rnn_out, _ = self.rnn(frame_features)
        
        # Classify
        logits = self.classifier(rnn_out)
        
        return logits
```
### Video Captioning
```python
class VideoCaptioner(nn.Module):
    """Generate captions for video"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
        )
        
        # CNN encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # RNN decoder
        self.rnn_decoder = nn.LSTM(256, hidden_dim, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        """Forward pass"""
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Encode frames
        encoder_out = self.encoder(x)
        
        # Temporal features
        temporal_features = self.cnn_encoder(encoder_out)
        
        # Decode
        decoder_out, _ = self.rnn_decoder(temporal_features)
        
        # Output
        logits = self.fc(decoder_out)
        
        return logits
```
### Future Prediction
```python
class FuturePredictor(nn.Module):
    """Predict future frames"""
    
    def __init__(self, pred_len=10, hidden_dim=512):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Predictive model
        self.rnn = nn.LSTM(256, hidden_dim, batch_first=True)
        
        # Predictor
        self.predictor = nn.Linear(hidden_dim, pred_len * 3)  # x, y, z per frame
        
    def forward(self, x):
        """Forward pass"""
        # Encode
        encoded = self.encoder(x)
        
        # Predict
        pred, _ = self.rnn(encoded)
        
        # Decode predictions
        predictions = self.predictor(pred)
        
        return predictions
```
## Temporal Modeling in Autonomous Driving
### Trajectory Prediction
```python
class TrajectoryPredictor(nn.Module):
    """Predict future trajectories of vehicles"""
    
    def __init__(self, pred_horizon=30, hidden_dim=512):
        super().__init__()
        # Scene encoder
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Temporal encoder
        self.temporal_encoder = nn.LSTM(256, hidden_dim, batch_first=True)
        
        # Trajectory decoder
        self.trajectory_decoder = nn.Linear(hidden_dim, pred_horizon * 4)  # x, y, z, yaw, vel
        
    def forward(self, x):
        """Forward pass"""
        # Encode scene
        scene_features = self.scene_encoder(x)
        
        # Temporal encoding
        temporal_features, _ = self.temporal_encoder(scene_features)
        
        # Decode trajectories
        trajectories = self.trajectory_decoder(temporal_features)
        
        return trajectories
```
### Behavior Prediction
```python
class BehaviorPredictor(nn.Module):
    """Predict behaviors of other road users"""
    
    def __init__(self, num_behaviors=5, hidden_dim=512):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Temporal model
        self.rnn = nn.LSTM(256, hidden_dim, batch_first=True)
        
        # Behavior classifier
        self.classifier = nn.Linear(hidden_dim, num_behaviors)
        
    def forward(self, x):
        """Forward pass"""
        # Encode
        encoded = self.encoder(x)
        
        # Temporal modeling
        temporal_features, _ = self.rnn(encoded)
        
        # Classify behavior
        behavior = self.classifier(temporal_features)
        
        return behavior
```
## Best Practices
### 1. Handle Variable-Length Sequences
```python
def pad_sequences(sequences, max_len):
    """Pad sequences to max_len"""
    padded = []
    
    for seq in sequences:
        # Pad
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        padded.append(seq)
    
    return torch.stack(padded)
```
### 2. Use Teacher For Testing
```python
class TeacherForcing(nn.Module):
    """Teacher forcing for better convergence"""
    
    def __init__(self, teacher_model):
        super().__init__()
        self.teacher = teacher_model
        
    def forward(self, x):
        # Get teacher outputs
        with torch.no_grad():
            teacher_out = self.teacher(x)
        
        # Get student outputs
        student_out = self.student(x)
        
        return teacher_out, student_out
```
### 3. Gradient Clipping for RNNs
```python
# Clip gradients to prevent exploding
torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
```
## Conclusion
Recurrent Neural Networks are essential for processing sequential data in autonomous driving systems. Their ability to model temporal dependencies makes them particularly valuable for video understanding, trajectory prediction, and behavior prediction. ## References
- "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- "Learning Phrase Representations using RNN Encoder-Decoder" (Cho et al., 2014)
- Tesla AI Day presentations
- "Deep Learning for Video Understanding" papers
---
**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~3500 words  
**Size**: ~20KB
