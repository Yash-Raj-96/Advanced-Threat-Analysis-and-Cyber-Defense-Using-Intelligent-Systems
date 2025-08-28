import torch
import torch.nn as nn
#from config import Config
from backend.config import Config

import logging

logger = logging.getLogger(__name__)

class CNNModule(nn.Module):
    """CNN for processing network traffic data"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.config = Config()
        
        # Calculate input channels based on input dimension
        self.input_channels = 1  # Treat as 1 channel (like grayscale image)
        self.input_height = int(input_dim ** 0.5)
        self.input_width = input_dim // self.input_height
        self.actual_input_dim = self.input_height * self.input_width
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate flattened size after convolutions
        self.flattened_size = self._get_conv_output_size()
        
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def _get_conv_output_size(self):
        """Calculate the output size after convolutional layers"""
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, 
                                self.input_height, self.input_width)
            dummy = self.conv_layers(dummy)
            return dummy.view(1, -1).shape[1]
            
    def forward(self, x):
        batch_size = x.size(0)
        x = x[:, :self.actual_input_dim]
        x = x.view(batch_size, 1, self.input_height, self.input_width)
        
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
