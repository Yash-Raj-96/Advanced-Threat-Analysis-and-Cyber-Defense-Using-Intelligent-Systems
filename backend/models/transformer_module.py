import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
#from config import Config
from backend.config import Config

import logging

logger = logging.getLogger(__name__)

class TransformerModule(nn.Module):
    """Transformer for processing CVE text and vulnerability data"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.config = Config()
        
        # Transformer parameters
        self.nhead = 4
        self.nhid = 256
        self.nlayers = 2
        self.dropout = 0.3
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, self.nhid)
        
        # Transformer layers
        encoder_layers = TransformerEncoderLayer(
            d_model=self.nhid,
            nhead=self.nhead,
            dim_feedforward=self.nhid,
            dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.nhid, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout))
            
    def forward(self, x):
        # Add sequence dimension (batch_size, seq_len=1, features)
        x = x.unsqueeze(1)
        
        # Embed input
        x = self.embedding(x)
        
        # Transformer expects (seq_len, batch_size, features)
        x = x.transpose(0, 1)
        
        # Forward through transformer
        x = self.transformer_encoder(x)
        
        # Get last sequence output
        x = x[-1]
        
        # Fully connected layer
        x = self.fc(x)
        return x