import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path
import json
import mlflow
from collections import defaultdict
from backend.config import Config
import torch
import joblib


# Correct relative imports using absolute path
from backend.config import Config
from backend.models.multi_modal_model import MultiModalThreatDetector
from backend.data_processing.preprocessor import DataPreprocessor
from backend.utils.interpretability import ThreatExplainer
from backend.threat_analysis.validation import ThreatValidator

logger = logging.getLogger(__name__)

class ThreatAnalysisPipeline:
    """Enhanced real-time threat analysis pipeline with multi-modal support"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the threat analysis pipeline
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocessor = DataPreprocessor(self.config)
        self.explainer = ThreatExplainer()
        self.validator = None
        self._setup_directories()
        self.load_model()
        
    def _setup_directories(self):
        """Ensure required directories exist"""
        self.config.ANALYSIS_RESULTS_DIR.mkdir(exist_ok=True, parents=True)
        

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load trained threat detection model
        """
        try:
            model_path = model_path or self.config.MODEL_DIR / 'threat_detector.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            self.model = joblib.load(model_path)
            logger.info(f"Threat detection model loaded from {model_path}")
        except Exception as e:
            logger.exception("Failed to load threat detection model")
            raise

            
    def analyze_multi_modal_threat(
        self,
        network_data: Optional[pd.DataFrame] = None,
        malware_data: Optional[pd.DataFrame] = None,
        cve_data: Optional[List[Dict]] = None,
        explain: bool = True
    ) -> Dict[str, Any]:

        """
        Comprehensive multi-modal threat analysis
        
        Args:
            network_data: DataFrame containing network features
            malware_data: DataFrame containing malware features
            cve_data: List of CVE dictionaries
            explain: Whether to generate explanations
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Initialize results dictionary
            results = {
                'timestamp': datetime.now().isoformat(),
                'modalities_processed': [],
                'predictions': None,
                'explanations': None
            }
            
            # Preprocess available data
            processed = {}
            if network_data is not None:
                X_net, _, _ = self.preprocessor.preprocess_network_data(network_data)
                processed['network'] = torch.FloatTensor(X_net).to(self.device)
                results['modalities_processed'].append('network')
                
            if malware_data is not None:
                X_mal, _, _ = self.preprocessor.preprocess_malware_data(malware_data)
                processed['malware'] = torch.FloatTensor(X_mal).to(self.device)
                results['modalities_processed'].append('malware')
                
            if cve_data is not None:
                X_cve = self._extract_cve_features(cve_data)
                processed['cve'] = torch.FloatTensor(X_cve).to(self.device)
                results['modalities_processed'].append('cve')
                
            if not processed:
                raise ValueError("No input data provided for analysis")
                
            # Create dummy tensors for missing modalities
            input_tensors = {}
            for modality in ['network', 'malware', 'cve']:
                if modality in processed:
                    input_tensors[modality] = processed[modality]
                else:
                    # Use zeros with correct dimension
                    dim = self.model.input_dims[['network', 'malware', 'cve'].index(modality)]
                    input_tensors[modality] = torch.zeros(
                        (len(next(iter(processed.values()))), dim)
                    ).to(self.device)
            
            # Run model inference
            with torch.no_grad():
                outputs, interpret = self.model(
                    input_tensors['network'],
                    input_tensors['malware'],
                    input_tensors['cve']
                )
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)
                
            # Store predictions
            results.update({
                'predictions': predictions.cpu().numpy(),
                'probabilities': probs.cpu().numpy(),
                'attention_weights': interpret['attention_weights'].cpu().numpy()
            })
            
            # Generate explanations if requested
            if explain:
                results['explanations'] = self._generate_explanations(
                    processed, 
                    predictions
                )
                
            # Log analysis results
            self._log_analysis(results)
            
            return results
            
        except Exception as e:
            logger.exception("Error during multi-modal threat analysis")
            raise
            
    def _generate_explanations(
        self,
        processed_data: Dict[str, torch.Tensor],
        predictions: torch.Tensor
    ) -> Dict[str, Dict]:
        """
        Generate explanations for model predictions
        
        Args:
            processed_data: Dictionary of processed input tensors
            predictions: Model predictions
            
        Returns:
            Dictionary of explanations per modality
        """
        explanations = {}
        
        for modality, tensor in processed_data.items():
            try:
                # Convert tensor to numpy for explainer
                data = tensor.cpu().numpy()
                
                # Get sample indices where threat was detected
                threat_indices = np.where(predictions.cpu().numpy() == 1)[0]
                if len(threat_indices) == 0:
                    threat_indices = [0]  # Explain first sample if no threats found
                    
                # Explain first detected threat (or first sample)
                sample_idx = threat_indices[0]
                sample = data[sample_idx:sample_idx+1]
                
                if modality == 'network':
                    explanations[modality] = self.explainer.explain_network(sample)
                elif modality == 'malware':
                    explanations[modality] = self.explainer.explain_malware(sample)
                elif modality == 'cve':
                    explanations[modality] = self.explainer.explain_vulnerability(sample)
                    
                # Add feature importance if available
                if hasattr(self.model, 'get_feature_importance'):
                    importance = self.model.get_feature_importance(
                        *[tensor[sample_idx:sample_idx+1] for tensor in [
                            processed_data.get('network', torch.zeros(1, self.model.input_dims[0])),
                            processed_data.get('malware', torch.zeros(1, self.model.input_dims[1])),
                            processed_data.get('cve', torch.zeros(1, self.model.input_dims[2]))
                        ]]
                    )
                    explanations[modality].update(importance)
                    
            except Exception as e:
                logger.warning(f"Failed to generate {modality} explanation: {str(e)}")
                explanations[modality] = {'error': str(e)}
                
        return explanations
        
    def _extract_cve_features(self, cve_data: List[Dict]) -> np.ndarray:
        """
        Extract features from CVE JSON data
        
        Args:
            cve_data: List of CVE items in NVD JSON format
            
        Returns:
            Numpy array of extracted features
        """
        features = []
        
        for item in cve_data:
            try:
                # Extract basic metadata
                cve_id = item.get('cve', {}).get('CVE_data_meta', {}).get('ID', '')
                published_date = item.get('publishedDate', '')
                
                # Calculate days since publication
                days_published = 0
                if published_date:
                    pub_date = datetime.strptime(published_date[:10], "%Y-%m-%d")
                    days_published = (datetime.now() - pub_date).days
                
                # Extract impact metrics
                impact = item.get('impact', {})
                base_metrics = impact.get('baseMetricV3', impact.get('baseMetricV2', {}))
                cvss_data = base_metrics.get('cvssV3', base_metrics.get('cvssV2', {}))
                
                # Description features
                descriptions = item.get('cve', {}).get('description', {}).get('description_data', [])
                desc_text = descriptions[0]['value'] if descriptions else ''
                desc_length = len(desc_text)
                has_exploit = any(
                    'exploit' in ref.get('url', '').lower() or 
                    'exploit' in ref.get('name', '').lower()
                    for ref in item.get('cve', {}).get('references', {}).get('reference_data', [])
                )
                
                # Reference features
                ref_count = len(item.get('cve', {}).get('references', {}).get('reference_data', []))
                
                features.append([
                    days_published,
                    cvss_data.get('baseScore', 0),
                    cvss_data.get('attackVector', 0),
                    cvss_data.get('attackComplexity', 0),
                    cvss_data.get('privilegesRequired', 0),
                    cvss_data.get('userInteraction', 0),
                    cvss_data.get('scope', 0),
                    cvss_data.get('confidentialityImpact', 0),
                    cvss_data.get('integrityImpact', 0),
                    cvss_data.get('availabilityImpact', 0),
                    desc_length,
                    int(has_exploit),
                    ref_count
                ])
                
            except Exception as e:
                logger.warning(f"Error processing CVE item: {str(e)}")
                continue
                
        return np.array(features) if features else np.zeros((0, 13))
        
            
    def _log_analysis(self, results: Dict) -> None:
        """
        Log analysis results to MLflow and local CSV storage

        Args:
            results: Analysis results dictionary
        """
        try:
            # Save JSON report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = self.config.ANALYSIS_RESULTS_DIR
            base_path.mkdir(parents=True, exist_ok=True)

            json_path = base_path / f"threat_analysis_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)

            # Save selected parts to CSV
            csv_path = base_path / f"threat_analysis_{timestamp}.csv"
            df = pd.DataFrame({
                'timestamp': [results['timestamp']],
                'modalities_processed': [", ".join(results['modalities_processed'])],
                'prediction': [results['predictions'][0] if results['predictions'] is not None else None],
                'threat_confidence': [results['probabilities'][0][1] if results['probabilities'] is not None else None]
            })
            df.to_csv(csv_path, index=False)

            # Optional: Log to MLflow
            if self.config.MLFLOW_TRACKING:
                with mlflow.start_run(run_name=f"ThreatAnalysis_{timestamp}"):
                    mlflow.log_dict(results, "threat_analysis.json")
                    mlflow.log_artifact(str(csv_path))  # log .csv too
                    if 'probabilities' in results:
                        mlflow.log_metrics({
                            'threat_detection_rate': float((results['predictions'] == 1).mean()),
                            'avg_threat_confidence': float(results['probabilities'][:, 1].mean())
                        })

        except Exception as e:
            logger.warning(f"Failed to log analysis results: {str(e)}")

            
    def validate_model(self, test_dataset) -> Dict:
        """
        Run comprehensive model validation
        
        Args:
            test_dataset: PyTorch Dataset for validation
            
        Returns:
            Dictionary containing validation results
        """
        try:
            self.validator = ThreatValidator(self.model, test_dataset)
            results = self.validator.evaluate_performance()
            report = self.validator.generate_threat_report()
            
            # Save validation report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.config.ANALYSIS_RESULTS_DIR / f"model_validation_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            return report
            
        except Exception as e:
            logger.exception("Model validation failed")
            raise
        
    def save_analysis_results(self, results: pd.DataFrame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_csv = self.config.ANALYSIS_RESULTS_DIR / f"threat_analysis_{timestamp}.csv"
        output_path_json = self.config.ANALYSIS_RESULTS_DIR / f"threat_analysis_{timestamp}.json"

        try:
            results.to_csv(output_path_csv, index=False)
            results.to_json(output_path_json, orient="records", indent=2)
            self.logger.info(f"✅ Saved analysis results to {output_path_csv} and {output_path_json}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save analysis results: {str(e)}")

    def load_intrusion_data(self) -> pd.DataFrame:
        """Load CIC-IDS2017 intrusion detection dataset"""
        path = self.config.INTRUSION_DATA
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to load intrusion data from {path}: {e}")
            return pd.DataFrame()

    def load_malware_data(self) -> pd.DataFrame:
        """Load EMBER malware dataset"""
        path = self.config.MALWARE_FEATURES
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to load malware data from {path}: {e}")
            return pd.DataFrame()

    def load_vulnerability_data(self) -> pd.DataFrame:
        """Load NVD CVE processed vulnerability dataset"""
        path = self.config.VULNERABILITY_DATA
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to load vulnerability data from {path}: {e}")
            return pd.DataFrame()


'''
    def load_intrusion_data(self):
        """Load CIC-IDS2017 intrusion detection dataset"""
        path = "backend/data/cic_ids2017.csv"
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Failed to load intrusion data: {e}")
            return pd.DataFrame()

    def load_malware_data(self):
        """Load EMBER malware dataset"""
        path = "backend/data/train_ember_2018_v2_features.parquet"
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to load malware data: {e}")
            return pd.DataFrame()

    def load_vulnerability_data(self):
        """Load NVD CVE processed vulnerability dataset"""
        path = "backend/data/nvd_processed.csv"
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Failed to load vulnerability data: {e}")
            return pd.DataFrame()
'''