import torch
import torch.nn as nn
#from config import Config
from backend.config import Config

import logging

logger = logging.getLogger(__name__)

class LSTMModule(nn.Module):
    """LSTM for processing sequential malware features"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.config = Config()
        
        # LSTM parameters
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.3
        
        # Create LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0)
            
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout))
            
    def forward(self, x):
        # Reshape input to (batch_size, sequence_length=1, features)
        x = x.unsqueeze(1)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        return out