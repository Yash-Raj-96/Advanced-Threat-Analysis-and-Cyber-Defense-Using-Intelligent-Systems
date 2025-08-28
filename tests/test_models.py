import pytest
import torch
import numpy as np
from backend.models.cnn_module import CNNModule
from backend.models.lstm_module import LSTMModule
from backend.models.transformer_module import TransformerModule
from backend.models.multi_modal_model import MultiModalThreatDetector
from backend.config import Config

@pytest.fixture
def sample_inputs():
    """Generate sample inputs for each modality"""
    return {
        'network': torch.randn(2, 100),  # 2 samples, 100 features
        'malware': torch.randn(2, 50),    # 2 samples, 50 features
        'cve': torch.randn(2, 10)         # 2 samples, 10 features
    }

def test_cnn_module():
    """Test CNN module initialization and forward pass"""
    model = CNNModule(input_dim=100)
    x = torch.randn(2, 100)  # batch_size=2, features=100
    output = model(x)
    assert output.shape == (2, 256)  # Should match fc layer output

def test_lstm_module():
    """Test LSTM module initialization and forward pass"""
    model = LSTMModule(input_dim=50)
    x = torch.randn(2, 50)  # batch_size=2, features=50
    output = model(x)
    assert output.shape == (2, 256)  # Should match fc layer output

def test_transformer_module():
    """Test Transformer module initialization and forward pass"""
    model = TransformerModule(input_dim=10)
    x = torch.randn(2, 10)  # batch_size=2, features=10
    output = model(x)
    assert output.shape == (2, 256)  # Should match fc layer output

def test_multi_modal_model(sample_inputs):
    """Test multi-modal model integration"""
    model = MultiModalThreatDetector(
        network_input_dim=100,
        malware_input_dim=50,
        cve_input_dim=10
    )
    
    outputs, attention = model(
        sample_inputs['network'],
        sample_inputs['malware'],
        sample_inputs['cve']
    )
    
    # Check output shapes
    assert outputs.shape == (2, 2)  # 2 samples, 2 classes
    assert attention.shape == (2, 3)  # 2 samples, 3 modalities
    
    # Attention weights should sum to 1
    assert torch.allclose(attention.sum(dim=1), torch.ones(2))

def test_model_device_placement():
    """Test models can move to GPU if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiModalThreatDetector(100, 50, 10).to(device)
    assert next(model.parameters()).device == device