import torch
import torch.nn as nn
from torch.nn import functional as F
#from .cnn_module import CNNModule
from backend.models.lstm_module import LSTMModule
from backend.models.transformer_module import TransformerModule
#from config import Config
from backend.config import Config
import logging
from backend.models.cnn_module import CNNModule




logger = logging.getLogger(__name__)

class MultiModalThreatDetector(nn.Module):
    """Multi-modal deep learning model for joint threat analysis"""
    def __init__(self, 
                 network_input_dim: int,
                 malware_input_dim: int,
                 cve_input_dim: int,
                 num_classes: int = 2):
        super().__init__()
        self.config = Config()
        
        # Individual modality networks
        self.cnn_network = CNNModule(input_dim=network_input_dim)
        self.lstm_malware = LSTMModule(input_dim=malware_input_dim)
        self.transformer_cve = TransformerModule(input_dim=cve_input_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(768, 256),  # Assuming each module outputs 256-dim features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes))
            
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Linear(256, 3),
            nn.Softmax(dim=1))
            
    def forward(self, x_network, x_malware, x_cve):
        # Process each modality
        network_features = self.cnn_network(x_network)
        malware_features = self.lstm_malware(x_malware)
        cve_features = self.transformer_cve(x_cve)
        
        # Concatenate features
        combined = torch.cat([network_features, malware_features, cve_features], dim=1)
        
        # Attention-weighted fusion
        attention_weights = self.attention(combined)
        weighted_features = torch.stack([
            network_features * attention_weights[:, 0:1],
            malware_features * attention_weights[:, 1:2],
            cve_features * attention_weights[:, 2:3]
        ], dim=1).sum(dim=1)
        
        # Final classification
        output = self.fusion(weighted_features)
        
        return output, attention_weights