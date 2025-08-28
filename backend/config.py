import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

class Config:
    API_KEY = os.getenv("API_KEY", "default_fallback_key") 
    
    """Centralized configuration for the Cyber Defense System"""

    def __init__(self):
        # =====================
        # Directory Structure
        # =====================
        self.ROOT_DIR = Path(__file__).resolve().parent.parent
        self.DATA_DIR = self.ROOT_DIR / os.getenv('DATA_DIR', 'data')
        self.RAW_DATA_DIR = self.DATA_DIR / 'raw'
        self.PROCESSED_DATA_DIR = self.DATA_DIR / 'processed'
        self.MODEL_DIR = self.ROOT_DIR / 'models'
        self.REPORTS_DIR = self.ROOT_DIR / 'reports'
        self.OUTPUTS_DIR = self.ROOT_DIR / 'outputs'
        
        # =====================
        # Data File Paths (For API Access)
        # =====================
        #self.INTRUSION_DATA = self.ROOT_DIR / os.getenv('INTRUSION_DATA', 'data/processed/cic_ids2017.csv')
        #self.MALWARE_FEATURES = self.ROOT_DIR / os.getenv('MALWARE_FEATURES', 'data/raw/malware/train_ember_2018_v2_features.parquet')
        #self.MALWARE_FEATURES = self.ROOT_DIR / os.getenv('MALWARE_FEATURES', 'data/processed/ember_scaler_stats.csv')
        #self.VULNERABILITY_DATA = self.ROOT_DIR / os.getenv('VULNERABILITY_DATA', 'data/processed/nvd_processed.csv')


        # Raw data paths
        self.NETWORK_LOGS_DIR = self.RAW_DATA_DIR / 'network_logs'
        self.MALWARE_DATA = self.RAW_DATA_DIR / 'malware' / 'train_ember_2018_v2_features.parquet'
        self.CVE_DATA = self.RAW_DATA_DIR / 'cve' / 'nvdcve-2.0-2025.json'

        # Outputs
        self.INTERPRETABILITY_DIR = self.OUTPUTS_DIR / 'interpretability'
        self.ANALYSIS_RESULTS_DIR = self.OUTPUTS_DIR / 'analysis_results'
        self.THREAT_MODEL_PATH = self.MODEL_DIR / 'threat_detector.pkl'

        # =====================
        # Model Configuration
        # =====================
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 1e-4
        self.EARLY_STOPPING_PATIENCE = 5
        self.LR_PATIENCE = 3
        self.SEED = 42
        self.MODEL_LAST_UPDATED = "2025-06-15"

        # =====================
        # MLflow Configuration
        # =====================
        self.MLFLOW_TRACKING = True
        self.MLFLOW_URI = os.getenv('MLFLOW_URI', 'http://localhost:5000')
        self.MLFLOW_EXPERIMENT = "CyberDefenseSystem"

        # =====================
        # API Configuration
        # =====================
        self.API_HOST = os.getenv('API_HOST', '127.0.0.1')
        self.API_PORT = int(os.getenv('API_PORT', 8000))
        self.USE_SSL = os.getenv('USE_SSL', 'False').lower() == 'true'
        self.SSL_KEY_PATH = self.ROOT_DIR / 'ssl' / 'key.pem' if self.USE_SSL else None
        self.SSL_CERT_PATH = self.ROOT_DIR / 'ssl' / 'cert.pem' if self.USE_SSL else None
        self.ALLOWED_ORIGINS = json.loads(os.getenv('ALLOWED_ORIGINS', '["*"]'))

        # =====================
        # Frontend Configuration
        # =====================
        self.FRONTEND_DIR = self.ROOT_DIR / 'frontend'
        self.STATIC_DIR = self.FRONTEND_DIR / 'static'
        self.TEMPLATES_DIR = self.FRONTEND_DIR / 'templates'
        self.VISUALIZATION_CACHE = self.STATIC_DIR / 'visualizations'

        # =====================
        # Testing Configuration
        # =====================
        self.TEST_DATA_DIR = self.ROOT_DIR / 'tests' / 'test_data'
        self.TEST_NETWORK_SAMPLES = self.TEST_DATA_DIR / 'network_samples.parquet'
        self.TEST_MALWARE_SAMPLES = self.TEST_DATA_DIR / 'malware_samples.parquet'
        self.TEST_CVE_SAMPLES = self.TEST_DATA_DIR / 'cve_samples.json'

        # =====================
        # Threat Analysis
        # =====================
        self.THREAT_THRESHOLD = 0.7
        self.REAL_TIME_POLL_INTERVAL = 5  # seconds
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

        # Create folders
        self._create_directories()

    def _create_directories(self):
        """Create required directories if they don't exist"""
        dirs_to_create = [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODEL_DIR,
            self.REPORTS_DIR,
            self.NETWORK_LOGS_DIR,
            self.OUTPUTS_DIR,
            self.INTERPRETABILITY_DIR,
            self.ANALYSIS_RESULTS_DIR,
            self.VISUALIZATION_CACHE,
            self.TEST_DATA_DIR,
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def frontend_settings(self) -> Dict[str, Any]:
        """Get frontend-specific settings"""
        return {
            'api_base_url': f"{'https' if self.USE_SSL else 'http'}://{self.API_HOST}:{self.API_PORT}",
            'visualization_refresh': 30,  # seconds
            'max_threats_displayed': 100
        }
