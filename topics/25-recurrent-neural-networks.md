# Recurrent Neural Networks - RNN and LSTM
 GRU, and sequence Modeling

## Introduction
Recurrent Neural Networks (RNNs) like LSTM and GRU are essential for processing sequential data in autonomous driving. This document covers RNN architectures, training techniques, and applications in temporal modeling for driving scenarios.

 ## RNN Fundamentals
### LSTM Cell
```python
import torch
import torch.nn as nn

import math

class LSTMCell(nn.Module):
    """LSTM cell implementation"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Gates
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(hidden_dim + output_dim, hidden_dim)
        self.cell_gate = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=p)
 0.0)
    
    def forward(self, x, hidden):
 tuple=None):
        """Forward pass through LSTM cell"""
        # Get batch size
        batch_size = x.size(0)
        
        # Initialize hidden state
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h, hidden[0]
            c = hidden[1]
        
        # Forward pass
        gates = []
        for t in range(self.num_layers):
  # Number of LSTM layers
            # Input gate
            i_t = torch.sigmoid(self.input_gate(x[:, t, :].matmul(
                h[t], torch.sigmoid(self.forget_gate(h[t], :].matmul(
                self.input_gate(x[:, t, :]
                (self.hidden_dim * self.input_gate.weight.data.t(),
                self.input_gate.bias.data
            )
            
            # Forget gate
            f_t = self.forget_gate(torch.cat([h[t], c[t]], dim=1))
            
 # Concatenate hidden states
            h = torch.cat([h, f_t], dim=1)
  # Update hidden states
            c = torch.cat([c, f_t], dim=1)
            
            # Output gate
            o_t = self.output_gate(torch.cat([h, c], dim=1))
            
            # Collect outputs
            outputs = []
            for t in range(self.num_layers):
                outputs.append(o_t)
            
            # Stack outputs
            return torch.stack(outputs, dim=1)
```
### GRU Cell
```python
class GRUCell(nn.Module):
    """GRU cell implementation"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Gates ( similar to LSTM but with reset gate)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(hidden_dim + output_dim, hidden_dim)
        
        # Initialize weights
        self.reset_parameters()
    
    def forward(self, x, hidden=None):
        """Forward pass"""
        batch_size = x.size(0)
        
        # Initialize hidden state
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h = hidden[0]
        
        # Update gate
        z = torch.sigmoid(self.update_gate(torch.cat([h, x], dim=1)))
        
        # Reset gate
        r = torch.sigmoid(self.reset_gate(torch.cat([h, x], dim=1))
        
        # Hidden state
        h_new = r * h
        
        # Output gate
        output = self.output_gate(torch.cat([h_new, h], dim=1))
        
        return output
```
## Sequence Modeling
### Trajectory Prediction
```python
class TrajectoryPredictor(nn.Module):
    """Predict future trajectories of vehicles"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        
        # Encoder LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden = (0, 1)  # Zero initial hidden state
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        # Encode sequence
        encoded, []
        for t in range(x.size(1)):
            # LSTM encoding
            _, (hidden_state, self.encoder(x[:, t, :], self.hidden)
            encoded.append(output)
        
        # Decode to trajectory
        encoded = torch.stack(encoded, dim=1)  # [batch, seq_len, input_dim, hidden_dim, output_dim]
        trajectory = self.decoder(encoded)
  # [batch, output_dim]
        
        return trajectory
```
### Traffic Speed Prediction
```python
class TrafficSpeedPredictor(nn.Module):
    """Predict traffic speed from video"""
    
    def __init__(self):
        super().__init__()
        
        # CNN for spatial features
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1, 1)
        )
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(64, 32, 32, num_layers=2, batch_first=True)
 self.hidden = (0, 1)  # Zero initial hidden state
        
        # Fully connected
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        # CNN forward
        cnn_out = self.cnn(x)
        
        # LSTM forward
        lstm_out, []
        for t in range(x.size(1)):
            if t == 0:
                # Zero padding for LSTM
                lstm_input = x[:, t, :]
                lstm_input = F.pad(lstm_input, (0, self.lstm.num_layers - 1), [0] == lstm_input.size(1), 0)
                else:
                    lstm_input = lstm_input
                lstm_out.append(lstm_input)
            else:
                # Use last output
                lstm_input = torch.cat(lstm_out, dim=1)
                prev_out = torch.cat([lstm_out], dim=1)
                current_input = torch.cat([lstm_out], dim=1)
        
        return current_input, prev_input
```
## Best Practices
### 1. Handle Variable-Length Sequences
```python
# Pad sequences to uniform length
class SequencePadder:
    """Pad sequences to uniform length"""
    
    def __init__(self, max_len):
        self.max_len = max_len
        self.pad_value = 0
        
    def __call__(self, x):
        # Pad sequence
        if len(x) < self.max_len:
            padded = F.pad(x, (0, self.max_len), mode='constant')
            padded = torch.cat([padded, (0, self.max_len - len(x) == padded.size(1)
            elif:
                # Pad with zeros
                padded = torch.cat([padded])
                padded = torch.zeros(x.size(1), self.max_len)
                padded = F.pad(padded, (0, self.max_len - len(x) == 0)
                
                return padded
```
### 2. Use Teacher Forcing
```python
# Use teacher forcing to stabilize training
teacher_forcing = 0.0
for p in self.parameters():
    p.register_buffer.register_hook(forcing, teacher.parameters(), p)
```
### 3. Gradient Clipping
```python
# Clip gradients to prevent exploding
torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
```
## Conclusion
RNNs are powerful tools for modeling temporal sequences in autonomous driving. From trajectory prediction to traffic speed estimation, these models can capture the dynamics of driving scenarios over time.

 ## References
- "Long Short-Term Memory" paper (Hochreiter, 1997)
- Tesla AI Day presentations
- "Traffic Speed Prediction" papers
- PyTorch documentation
---
**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~2800 words  
**Size**: ~17KB
