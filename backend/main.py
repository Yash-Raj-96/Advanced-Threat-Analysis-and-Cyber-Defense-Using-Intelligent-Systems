import logging
import pandas as pd
import mlflow
from pathlib import Path
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
import sys
from typing import Tuple, Dict, Optional, Union, Any
import json
from datetime import datetime
from backend.utils.performance_monitor import PerformanceMonitor

# Local imports
from backend.config import Config
from backend.data_processing.data_loader import DataLoader
from backend.data_processing.preprocessor import DataPreprocessor
from backend.models.model_trainer import ModelTrainer
from backend.threat_analysis.pipeline import ThreatAnalysisPipeline
from backend.utils.performance_monitor import PerformanceMonitor
from backend.utils.data_validator import DataValidator

from fastapi import FastAPI
from backend.api.routes import dashboard  # adjust based on your path

app = FastAPI()
app.include_router(dashboard.router)


class AdvancedThreatAnalyzer:
    """Main class for the Advanced Threat Analysis System"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor()
        self.data_validator = DataValidator()
        
    def setup_logging(self) -> None:
        """Configure logging system"""
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cyber_defense.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        # Suppress noisy library logs
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all required datasets with validation"""
        with self.performance_monitor.track("Data Loading"):
            data_loader = DataLoader()
            
            try:
                network_data = data_loader.load_network_data()
                malware_data = data_loader.load_malware_data()
                cve_data = data_loader.load_cve_data()
                
                # Validate data integrity
                self.data_validator.validate_network_data(network_data)
                self.data_validator.validate_malware_data(malware_data)
                self.data_validator.validate_cve_data(cve_data)
                
                return network_data, malware_data, cve_data
                
            except Exception as e:
                self.logger.error(f"Data loading failed: {str(e)}")
                raise

    def preprocess_data(self, 
                       network_data: pd.DataFrame, 
                       malware_data: pd.DataFrame, 
                       cve_data: pd.DataFrame
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess all data types with fallback mechanisms"""
        with self.performance_monitor.track("Data Preprocessing"):
            preprocessor = DataPreprocessor(self.config)
            
            try:
                # Network data processing
                X_network, y_network, network_features = preprocessor.preprocess_network_data(network_data)
                if y_network is None or len(y_network) == 0:
                    raise ValueError("Network labels are missing or empty")
                
                # Malware data processing
                X_malware, y_malware, malware_features = preprocessor.preprocess_malware_data(malware_data)
                
                # CVE data processing with fallback
                try:
                    X_cve = preprocessor.extract_cve_features(cve_data)
                    if not isinstance(X_cve, np.ndarray) or X_cve.size == 0:
                        raise ValueError("Empty CVE features")
                except Exception as e:
                    self.logger.warning(f"CVE processing failed, using fallback: {str(e)}")
                    X_cve = np.zeros((X_network.shape[0], 5), dtype=np.float32)
                
                return X_network, y_network, X_malware, y_malware, X_cve
                
            except Exception as e:
                self.logger.error(f"Data preprocessing failed: {str(e)}")
                raise

    def train_model(self, 
                   X_network: np.ndarray, 
                   X_malware: np.ndarray, 
                   X_cve: np.ndarray, 
                   y_network: np.ndarray
                  ) -> Tuple[object, TorchDataLoader]:
        """Train the multi-modal threat detection model"""
        with self.performance_monitor.track("Model Training"):
            try:
                trainer = ModelTrainer()
                input_dims = (X_network.shape[1], X_malware.shape[1], X_cve.shape[1])
                
                # Create datasets with stratified sampling
                train_dataset, val_dataset, test_dataset = trainer.create_datasets(
                    X_network, X_malware, X_cve, y_network
                )
                
                # Initialize MLflow tracking
                with mlflow.start_run():
                    mlflow.log_params({
                        "network_features": X_network.shape[1],
                        "malware_features": X_malware.shape[1],
                        "cve_features": X_cve.shape[1],
                        "batch_size": self.config.BATCH_SIZE
                    })
                    
                    # Train model with early stopping
                    model = trainer.train_model(
                        train_dataset, 
                        val_dataset, 
                        input_dims,
                        patience=self.config.EARLY_STOPPING_PATIENCE
                    )
                    
                    # Evaluate on test set
                    test_loader = TorchDataLoader(
                        test_dataset, 
                        batch_size=self.config.BATCH_SIZE
                    )
                    test_loss, test_acc = trainer.evaluate(test_loader)
                    
                    mlflow.log_metrics({
                        "test_loss": test_loss,
                        "test_accuracy": test_acc
                    })
                    
                    self.logger.info(f"✅ Test Performance — Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
                    
                    return model, test_loader
                    
            except Exception as e:
                self.logger.error(f"Model training failed: {str(e)}")
                raise

    def analyze_threats(self, 
                        model: object, 
                        network_data: pd.DataFrame, 
                        malware_data: pd.DataFrame, 
                        cve_data: pd.DataFrame
                       ) -> Dict[str, pd.DataFrame]:
        """Run comprehensive threat analysis"""
        with self.performance_monitor.track("Threat Analysis"):
            try:
                pipeline = ThreatAnalysisPipeline(model=model)
                
                # Sample data for demonstration (stratified if possible)
                network_sample = self._get_balanced_sample(network_data, 'Label', 10)
                malware_sample = malware_data.sample(10, random_state=self.config.RANDOM_SEED)
                
                results = {
                    "network": pipeline.analyze_network_traffic(network_sample),
                    "malware": pipeline.analyze_malware(malware_sample),
                    "vulnerability": pipeline.analyze_vulnerability(cve_data)
                }
                
                # Log results summary
                self._log_analysis_results(results)
                
                # Save results to files
                self._save_analysis_results(results)
                
                return results
                
            except Exception as e:
                self.logger.error(f"Threat analysis failed: {str(e)}")
                raise

    def _get_balanced_sample(self, 
                           data: pd.DataFrame, 
                           label_col: str, 
                           n_samples: int
                          ) -> pd.DataFrame:
        """Get stratified sample when possible"""
        if label_col in data.columns:
            return data.groupby(label_col).apply(
                lambda x: x.sample(min(len(x), n_samples // 2), random_state=self.config.RANDOM_SEED)
            ).reset_index(drop=True)
        return data.sample(n_samples, random_state=self.config.RANDOM_SEED)

    def _log_analysis_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """Log analysis results in a structured way"""
        for analysis_type, result in results.items():
            self.logger.info(f"📊 {analysis_type.capitalize()} Analysis Results:")
            if isinstance(result, pd.DataFrame):
                self.logger.info(result.describe().to_string())
                threats = result.get('threat_score', result.get('prediction', pd.Series()))
                if not threats.empty:
                    threat_stats = threats.value_counts(normalize=True)
                    self.logger.info(f"Threat Distribution:\n{threat_stats.to_string()}")
            else:
                self.logger.info(str(result))              


    def _save_analysis_results(self, results: Dict[str, Any]) -> None:
        """
        Save analysis results to the configured output directory.

        - If the result is a Pandas DataFrame, save it as a CSV.
        - If the result is a dictionary or other JSON-serializable object, save it as JSON.

        Args:
            results (Dict[str, Any]): A dictionary where keys are analysis types
                                    (e.g., 'network', 'malware') and values are
                                    either DataFrames or JSON-serializable objects.
        """
        output_dir = Path(self.config.ANALYSIS_RESULTS_DIR)
        output_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for analysis_type, result in results.items():
            if isinstance(result, pd.DataFrame):
                # Save as CSV
                file_path = output_dir / f"{analysis_type}_results_{timestamp}.csv"
                result.to_csv(file_path, index=False)
            else:
                # Save as JSON
                file_path = output_dir / f"{analysis_type}_results_{timestamp}.json"
                with open(file_path, 'w') as f:
                    json.dump(result, f, indent=4, default=str)

            self.logger.info(f"💾 Saved {analysis_type} results to {file_path}")


    def run(self) -> None:
        """Main execution flow for the threat analysis system"""
        try:
            self.setup_logging()
            self.logger.info("🚀 Starting Advanced Threat Analysis System")
            
            # Load and preprocess data
            network_data, malware_data, cve_data = self.load_data()
            X_network, y_network, X_malware, y_malware, X_cve = self.preprocess_data(
                network_data, malware_data, cve_data
            )
            
            # Train and evaluate model
            model, test_loader = self.train_model(X_network, X_malware, X_cve, y_network)
            
            # Perform threat analysis
            analysis_results = self.analyze_threats(model, network_data, malware_data, cve_data)
            
            # Generate performance report
            perf_report = self.performance_monitor.generate_report()
            self.logger.info("⏱️ Performance Metrics:\n" + json.dumps(perf_report, indent=2))
            
            self.logger.info("✅ Advanced Threat Analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Critical system failure: {str(e)}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    analyzer = AdvancedThreatAnalyzer()
    analyzer.run()