import logging
import sys
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import torch
from backend.config import Config
from backend.data_processing.data_loader import DataLoader
from backend.data_processing.preprocessor import DataPreprocessor
from backend.models.model_trainer import ModelTrainer
from backend.threat_analysis.pipeline import ThreatAnalysisPipeline

from datetime import datetime, timedelta

app = FastAPI(title="Cyber Defense API")

# ✅ Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ✅ Return recent threats (now wrapped in a dict)
@app.get("/threats/recent")
def recent_threats():
    now = datetime.now()
    sample_threats = [
        {"id": 1, "type": "Intrusion", "confidence": 0.91, "timestamp": (now - timedelta(minutes=5)).isoformat()},
        {"id": 2, "type": "Malware", "confidence": 0.87, "timestamp": (now - timedelta(minutes=2)).isoformat()},
    ]
    return {"threats": sample_threats}

# ✅ Set up logging
def setup_logging():
    logging.basicConfig(
        level=Config().LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cyber_defense.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ])

# ✅ Main pipeline runner
def run_main_pipeline():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = Config()

    try:
        logger.info("🚀 Starting Advanced Threat Analysis System...")

        # 1. Load data
        data_loader = DataLoader()
        network_data, malware_data, cve_data = data_loader.load_all_data()

        # 2. Preprocess data
        preprocessor = DataPreprocessor()
        X_network, y_network, _ = preprocessor.preprocess_network_data(network_data)
        if y_network is None or len(y_network) == 0:
            raise ValueError("Missing network labels. Check the output of preprocess_network_data.")

        X_malware, y_malware, _ = preprocessor.preprocess_malware_data(malware_data)
        X_cve = preprocessor.extract_cve_features(cve_data)

        if not isinstance(X_cve, np.ndarray) or X_cve.size == 0:
            logger.warning("⚠️ CVE features are empty, using dummy fallback.")
            X_cve = np.zeros((X_network.shape[0], 5), dtype=np.float32)

        # 3. Train model
        trainer = ModelTrainer()
        input_dims = (X_network.shape[1], X_malware.shape[1], X_cve.shape[1])
        train_dataset, val_dataset, test_dataset = trainer.create_datasets(
            X_network, X_malware, X_cve, y_network
        )

        model = trainer.train_model(train_dataset, val_dataset, input_dims)

        # 4. Evaluate model
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
        test_loss, test_acc = trainer.evaluate(test_loader)
        logger.info(f"✅ Test Performance — Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

        # 5. Initialize threat pipeline and run demo
        pipeline = ThreatAnalysisPipeline(model=model)
        logger.info("🎯 Running demo analysis...")
        network_result = pipeline.analyze_network_traffic(network_data.sample(10))
        malware_result = pipeline.analyze_malware(malware_data.sample(10))
        cve_result = pipeline.analyze_vulnerability(cve_data)

        logger.info("📊 Network result:")
        logger.info(network_result)
        logger.info("🦠 Malware result:")
        logger.info(malware_result)
        logger.info("🔓 Vulnerability result:")
        logger.info(cve_result)

        logger.info("✅ Advanced Threat Analysis System completed successfully.")

    except Exception as e:
        logger.error(f"❌ System failed: {str(e)}", exc_info=True)
        raise
