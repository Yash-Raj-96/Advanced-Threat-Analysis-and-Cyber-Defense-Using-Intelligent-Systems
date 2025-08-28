import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from backend.data_processing.data_loader import DataLoader
from backend.data_processing.preprocessor import DataPreprocessor
from backend.config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def sample_network_data(config):
    # Create a small sample of network data
    data = {
        'Flow Duration': [10, 20, 30],
        'Total Fwd Packets': [5, 10, 15],
        'Total Bwd Packets': [3, 6, 9],
        'Label': ['Benign', 'Malicious', 'Benign']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_malware_data(config):
    # Create a small sample of malware data
    np.random.seed(42)
    data = {
        'feature_1': np.random.rand(3),
        'feature_2': np.random.rand(3),
        'label': [0, 1, 0]
    }
    return pd.DataFrame(data)

def test_data_loader(config):
    """Test that data loader can initialize properly"""
    loader = DataLoader()
    assert loader.config == config

def test_load_network_data(sample_network_data, tmp_path):
    """Test network data loading and preprocessing"""
    # Save sample data to temp directory
    test_file = tmp_path / "test_network.csv"
    sample_network_data.to_csv(test_file, index=False)
    
    # Mock the config path
    class TestConfig:
        NETWORK_LOGS = tmp_path
    
    loader = DataLoader()
    loader.config = TestConfig()
    
    # Test loading
    df = loader.load_network_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

def test_preprocess_network_data(sample_network_data):
    """Test network data preprocessing"""
    preprocessor = DataPreprocessor()
    X, y, _ = preprocessor.preprocess_network_data(sample_network_data)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == len(y)

def test_preprocess_malware_data(sample_malware_data):
    """Test malware data preprocessing"""
    preprocessor = DataPreprocessor()
    X, y, _ = preprocessor.preprocess_malware_data(sample_malware_data)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == len(y)

def test_preprocessor_saving(config, sample_network_data, tmp_path):
    """Test that preprocessor saves correctly"""
    # Mock the config path
    class TestConfig:
        PROCESSED_DATA_DIR = tmp_path
    
    preprocessor = DataPreprocessor()
    preprocessor.config = TestConfig()
    
    X, y, processor = preprocessor.preprocess_network_data(sample_network_data)
    
    # Check files were saved
    assert (tmp_path / 'preprocessor.joblib').exists()