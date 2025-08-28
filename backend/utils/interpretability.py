import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Tuple, Optional
from backend.config import Config
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow

logger = logging.getLogger(__name__)

class ThreatExplainer:
    """Provides model-agnostic interpretability for threat detection models"""
    
    def __init__(self, model=None, config=None):
        self.config = config or Config()
        self.model = model
        self._setup_directories()
        
    def _setup_directories(self):
        """Ensure required directories exist"""
        self.config.INTERPRETABILITY_DIR.mkdir(exist_ok=True, parents=True)
        
    def explain_prediction(
        self,
        X_network: np.ndarray = None,
        X_malware: np.ndarray = None,
        X_cve: np.ndarray = None,
        sample_idx: int = 0,
        num_samples: int = 100,
        num_features: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Explain model predictions using SHAP and LIME
        
        Args:
            X_*: Input features for each modality
            sample_idx: Index of sample to explain
            num_samples: Background samples for SHAP
            num_features: Top features to show
            
        Returns:
            Dictionary containing SHAP and LIME explanations
        """
        explanations = {}
        
        try:
            if X_network is not None:
                explanations['network'] = self._explain_modality(
                    X_network,
                    modality_name='network',
                    sample_idx=sample_idx,
                    num_samples=num_samples,
                    num_features=num_features
                )
                
            if X_malware is not None:
                explanations['malware'] = self._explain_modality(
                    X_malware,
                    modality_name='malware',
                    sample_idx=sample_idx,
                    num_samples=num_samples,
                    num_features=num_features
                )
                
            if X_cve is not None:
                explanations['cve'] = self._explain_modality(
                    X_cve,
                    modality_name='cve',
                    sample_idx=sample_idx,
                    num_samples=num_samples,
                    num_features=4  # Fewer features for CVE
                )
                
            # Log explanations to MLflow
            self._log_explanations(explanations)
            
            return explanations
            
        except Exception as e:
            logger.exception("Error generating explanations")
            raise

    def _explain_modality(
        self,
        X: np.ndarray,
        modality_name: str,
        sample_idx: int,
        num_samples: int,
        num_features: int
    ) -> Dict[str, Any]:
        """Generate explanations for a single modality"""
        if len(X) <= sample_idx:
            raise ValueError(f"Sample index {sample_idx} out of range for {modality_name} data")
            
        sample = X[sample_idx:sample_idx+1]
        background = X[:num_samples]
        
        # SHAP Explanation
        shap_explainer = shap.KernelExplainer(
            self._predict_wrapper(modality_name),
            background
        )
        shap_values = shap_explainer.shap_values(sample)
        
        # LIME Explanation
        feature_names = self._get_feature_names(modality_name, X.shape[1])
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            background,
            feature_names=feature_names,
            mode="classification",
            verbose=True
        )
        lime_exp = lime_explainer.explain_instance(
            sample[0],
            self._predict_wrapper(modality_name),
            num_features=num_features
        )
        
        # Generate visualization
        fig_path = self._plot_feature_importance(
            modality_name,
            shap_values,
            feature_names,
            sample_idx
        )
        
        return {
            'shap_values': shap_values,
            'lime_explanation': lime_exp.as_list(),
            'visualization': fig_path
        }

    def _predict_wrapper(self, modality_name: str):
        """Create prediction function for explainers"""
        def predict_fn(X):
            if self.model is None:
                # Fallback dummy prediction
                return np.random.rand(len(X), 2)
                
            # Convert to model input format
            inputs = {
                'network': np.zeros((len(X), self.model.input_dims[0])) if modality_name != 'network' else X,
                'malware': np.zeros((len(X), self.model.input_dims[1])) if modality_name != 'malware' else X,
                'cve': np.zeros((len(X), self.model.input_dims[2])) if modality_name != 'cve' else X
            }
            
            with torch.no_grad():
                outputs = self.model(
                    torch.FloatTensor(inputs['network']),
                    torch.FloatTensor(inputs['malware']),
                    torch.FloatTensor(inputs['cve'])
                )
                return torch.softmax(outputs, dim=1).numpy()
                
        return predict_fn

    def _get_feature_names(self, modality_name: str, num_features: int) -> list:
        """Get meaningful feature names for each modality"""
        if modality_name == 'network':
            return [f'net_feat_{i}' for i in range(num_features)]
        elif modality_name == 'malware':
            return [f'mal_feat_{i}' for i in range(num_features)]
        else:  # CVE
            return ['base_score', 'exploitability', 'impact', 'desc_complexity']

    def _plot_feature_importance(
        self,
        modality_name: str,
        shap_values: np.ndarray,
        feature_names: list,
        sample_idx: int
    ) -> Optional[str]:
        """Generate and save feature importance plot"""
        try:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                feature_names=feature_names,
                plot_type="bar",
                show=False
            )
            plt.title(f"{modality_name.upper()} Feature Importance")
            
            # Save plot
            fig_path = self.config.INTERPRETABILITY_DIR / \
                      f"{modality_name}_feature_importance_{sample_idx}.png"
            plt.savefig(fig_path)
            plt.close()
            
            return str(fig_path)
            
        except Exception as e:
            logger.warning(f"Could not generate feature plot: {str(e)}")
            return None

    def _log_explanations(self, explanations: Dict):
        """Log explanations to MLflow"""
        try:
            for modality, explanation in explanations.items():
                # Log SHAP values
                if 'shap_values' in explanation:
                    mlflow.log_dict(
                        {'shap_values': explanation['shap_values'].tolist()},
                        f"explanations/{modality}_shap.json"
                    )
                
                # Log LIME explanation
                if 'lime_explanation' in explanation:
                    mlflow.log_dict(
                        {'lime_explanation': explanation['lime_explanation']},
                        f"explanations/{modality}_lime.json"
                    )
                
                # Log visualization
                if 'visualization' in explanation and explanation['visualization']:
                    mlflow.log_artifact(explanation['visualization'])
                    
        except Exception as e:
            logger.warning(f"Failed to log explanations: {str(e)}")

    def compare_threats(
        self,
        X_benign: Dict[str, np.ndarray],
        X_malicious: Dict[str, np.ndarray],
        num_samples: int = 5
    ) -> Dict[str, Any]:
        """Compare feature patterns between benign and malicious samples"""
        comparison = {}
        
        for modality in ['network', 'malware', 'cve']:
            if modality in X_benign and modality in X_malicious:
                benign_exp = self.explain_prediction(
                    **{modality: X_benign[modality][:num_samples]}
                )
                malicious_exp = self.explain_prediction(
                    **{modality: X_malicious[modality][:num_samples]}
                )
                
                comparison[modality] = {
                    'benign': benign_exp.get(modality, {}),
                    'malicious': malicious_exp.get(modality, {})
                }
        
        return comparison