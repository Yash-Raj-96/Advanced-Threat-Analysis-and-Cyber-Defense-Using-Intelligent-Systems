import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from backend.threat_analysis.pipeline import ThreatAnalysisPipeline
from backend.config import Config
import torch

@pytest.fixture
def pipeline():
    """Fixture for threat analysis pipeline with mocked model"""
    with patch('backend.threat_analysis.pipeline.MultiModalThreatDetector') as mock_model:
        # Configure mock model
        mock_model.return_value.return_value = (
            torch.tensor([[0.8, 0.2]]),  # predictions
            torch.tensor([[0.3, 0.4, 0.3]])  # attention weights
        )
        
        pipeline = ThreatAnalysisPipeline()
        pipeline.model = mock_model()
        yield pipeline

def test_pipeline_initialization(pipeline):
    """Test pipeline initializes correctly"""
    assert pipeline is not None
    assert pipeline.model is not None

def test_network_analysis(pipeline):
    """Test network traffic analysis"""
    sample_data = pd.DataFrame({
        'Flow Duration': [10, 20],
        'Total Fwd Packets': [5, 10],
        'Label': ['Benign', 'Malicious']
    })
    
    result = pipeline.analyze_network_traffic(sample_data)
    
    assert 'predictions' in result
    assert 'probabilities' in result
    assert 'attention_weights' in result
    assert 'explanations' in result
    assert result['predictions'].shape == (2,)  # 2 samples

def test_malware_analysis(pipeline):
    """Test malware analysis"""
    sample_data = pd.DataFrame({
        'feature_1': [0.1, 0.2],
        'feature_2': [0.3, 0.4],
        'label': [0, 1]
    })
    
    result = pipeline.analyze_malware(sample_data)
    
    assert 'predictions' in result
    assert 'probabilities' in result
    assert 'attention_weights' in result
    assert 'explanations' in result
    assert result['predictions'].shape == (2,)  # 2 samples

def test_vulnerability_analysis(pipeline):
    """Test vulnerability assessment"""
    sample_data = {
        'CVE_Items': [{
            'cve': {
                'description': {
                    'description_data': [{
                        'value': 'Sample vulnerability',
                        'lang': 'en'
                    }]
                }
            },
            'impact': {
                'baseMetricV3': {
                    'cvssV3': {
                        'baseScore': 7.5
                    }
                }
            }
        }]
    }
    
    result = pipeline.analyze_vulnerability(sample_data)
    
    assert 'predictions' in result
    assert 'probabilities' in result
    assert 'attention_weights' in result
    assert 'explanations' in result
    assert result['predictions'].shape == (1,)  # 1 CVE

@patch('backend.threat_analysis.pipeline.ThreatExplainer')
def test_explanations_included(mock_explainer, pipeline):
    """Test that explanations are included in results"""
    # Configure mock explainer
    mock_explainer.return_value.explain_network.return_value = {
        'shap_values': [0.1, 0.2],
        'lime_explanation': [('feature1', 0.5)]
    }
    
    sample_data = pd.DataFrame({
        'Flow Duration': [10],
        'Total Fwd Packets': [5],
        'Label': ['Benign']
    })
    
    result = pipeline.analyze_network_traffic(sample_data)
    assert 'explanations' in result
    assert 'shap_values' in result['explanations']
    assert 'lime_explanation' in result['explanations']