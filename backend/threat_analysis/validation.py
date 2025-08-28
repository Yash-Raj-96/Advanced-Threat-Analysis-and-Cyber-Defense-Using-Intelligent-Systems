# backend/threat_analysis/validation.py
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc)
import mlflow
from typing import Dict, Tuple, Optional
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import torch
import matplotlib
matplotlib.use('Agg') 

logger = logging.getLogger(__name__)

class ThreatValidator:
    """Comprehensive threat model validation and analysis"""
    
    def __init__(self, model, test_dataset, config=None):
        """
        Args:
            model: Trained threat detection model
            test_dataset: PyTorch Dataset containing test data
            config: Configuration object
        """
        self.model = model
        self.test_dataset = test_dataset
        self.config = config or self._default_config()
        self._setup_directories()
        
        self.metrics = {}
        self.classification_reports = {}
        self.threat_coverage = {}
        self.error_analysis = {}

    def _default_config(self):
        """Default configuration if none provided"""
        class Config:
            VALIDATION_DIR = Path("reports/validation")
            SEED = 42
            THRESHOLDS = {
                'network': 0.7,
                'malware': 0.8,
                'cve': 0.6
            }
        return Config()

    def _setup_directories(self):
        """Ensure required directories exist"""
        self.config.VALIDATION_DIR.mkdir(exist_ok=True, parents=True)

    def evaluate_performance(self) -> Dict:
        """
        Comprehensive model evaluation with:
        - Standard classification metrics
        - Threat-specific analysis
        - Error analysis
        - Threshold optimization
        """
        try:
            # Get predictions and probabilities
            y_true, y_pred, y_probs = self._get_predictions()
            
            # Calculate metrics
            self.metrics = self._calculate_metrics(y_true, y_pred, y_probs)
            self.classification_reports = classification_report(
                y_true, y_pred, output_dict=True, target_names=['benign', 'malicious']
            )
            
            # Threat-specific analysis
            self.threat_coverage = self._analyze_threat_coverage(y_true, y_pred)
            
            # Error analysis
            self.error_analysis = self._analyze_errors(y_true, y_pred, y_probs)
            
            # Log everything
            self._log_results()
            
            return {
                'metrics': self.metrics,
                'classification_report': self.classification_reports,
                'threat_coverage': self.threat_coverage,
                'error_analysis': self.error_analysis
            }
            
        except Exception as e:
            logger.exception("Error during model evaluation")
            raise

    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions and probabilities on test set"""
        y_true = []
        y_pred = []
        y_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_dataset:
                x_net, x_mal, x_cve, y = batch
                outputs = self.model(
                    x_net.to(self.model.device),
                    x_mal.to(self.model.device),
                    x_cve.to(self.model.device)
                )
                
                probs = torch.softmax(outputs, dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(probs.argmax(1).cpu().numpy())
                y_probs.extend(probs[:, 1].cpu().numpy())  # Malicious class probability
                
        return np.array(y_true), np.array(y_pred), np.array(y_probs)

    def _calculate_metrics(self, y_true, y_pred, y_probs) -> Dict:
        """Calculate comprehensive performance metrics"""
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp),
            'recall': tp / (tp + fn),
            'f1_score': 2 * tp / (2 * tp + fp + fn),
            'false_positive_rate': fp / (fp + tn),
            'true_positive_rate': tp / (tp + fn),
            'balanced_accuracy': (tp/(tp+fn) + tn/(tn+fp)) / 2
        }
        
        # Probability-based metrics
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        metrics['roc_auc'] = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        metrics['pr_auc'] = auc(recall, precision)
        
        # Optimal threshold finding
        metrics['optimal_threshold'] = self._find_optimal_threshold(y_true, y_probs)
        
        return metrics

    def _find_optimal_threshold(self, y_true, y_probs) -> float:
        """Find optimal decision threshold using Youden's J statistic"""
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]

    def _analyze_threat_coverage(self, y_true, y_pred) -> Dict:
        """Analyze detection rates by threat type"""
        coverage = defaultdict(dict)
        
        # Get threat type metadata if available
        has_metadata = hasattr(self.test_dataset, 'threat_metadata')
        
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if has_metadata:
                meta = self.test_dataset.threat_metadata[i]
                threat_type = meta.get('threat_type', 'unknown')
                subtype = meta.get('subtype', 'unknown')
            else:
                threat_type = 'unknown'
                subtype = 'unknown'
                
            key = f"{threat_type}_{subtype}"
            
            if key not in coverage:
                coverage[key] = {
                    'detected': 0,
                    'missed': 0,
                    'total': 0,
                    'false_alarms': 0
                }
                
            if true == 1:  # Actual threat
                coverage[key]['total'] += 1
                if pred == 1:
                    coverage[key]['detected'] += 1
                else:
                    coverage[key]['missed'] += 1
            else:  # Benign
                if pred == 1:
                    coverage[key]['false_alarms'] += 1
                    
        # Convert counts to rates
        for key in coverage:
            total = coverage[key]['total']
            if total > 0:
                coverage[key]['detection_rate'] = coverage[key]['detected'] / total
                coverage[key]['miss_rate'] = coverage[key]['missed'] / total
            else:
                coverage[key]['detection_rate'] = 0
                coverage[key]['miss_rate'] = 0
                
            coverage[key]['false_alarm_rate'] = coverage[key]['false_alarms'] / len(y_true)
            
        return dict(coverage)

    def _analyze_errors(self, y_true, y_pred, y_probs) -> Dict:
        """Analyze false positives and false negatives"""
        errors = {
            'false_positives': [],
            'false_negatives': []
        }
        
        for i, (true, pred, prob) in enumerate(zip(y_true, y_pred, y_probs)):
            if true != pred:
                error = {
                    'index': i,
                    'confidence': prob if pred == 1 else 1 - prob,
                    'true_label': 'malicious' if true == 1 else 'benign',
                    'predicted_label': 'malicious' if pred == 1 else 'benign'
                }
                
                if hasattr(self.test_dataset, 'get_sample_metadata'):
                    error.update(self.test_dataset.get_sample_metadata(i))
                
                if true == 0 and pred == 1:
                    errors['false_positives'].append(error)
                elif true == 1 and pred == 0:
                    errors['false_negatives'].append(error)
                    
        # Sort by confidence
        errors['false_positives'].sort(key=lambda x: x['confidence'], reverse=True)
        errors['false_negatives'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return errors

    def _log_results(self):
        """Log all validation results to MLflow"""
        try:
            # Log metrics
            mlflow.log_metrics(self.metrics)
            
            # Log reports
            mlflow.log_dict(
                self.classification_reports,
                "classification_report.json"
            )
            
            mlflow.log_dict(
                self.threat_coverage,
                "threat_coverage.json"
            )
            
            mlflow.log_dict(
                self.error_analysis,
                "error_analysis.json"
            )
            
            # Log visualizations
            self._plot_roc_curve()
            self._plot_threat_coverage()
            
        except Exception as e:
            logger.warning(f"Failed to log validation results: {str(e)}")

    def _plot_roc_curve(self):
        """Generate and save ROC curve plot"""
        y_true, _, y_probs = self._get_predictions()
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        plot_path = self.config.VALIDATION_DIR / f"roc_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        plt.close()
        
        mlflow.log_artifact(str(plot_path))

    def _plot_threat_coverage(self):
        """Generate threat coverage visualization"""
        coverage = pd.DataFrame.from_dict(self.threat_coverage, orient='index')
        
        if not coverage.empty:
            plt.figure(figsize=(12, 6))
            coverage['detection_rate'].sort_values().plot(
                kind='barh',
                title='Threat Detection Rates by Type'
            )
            plt.xlabel('Detection Rate')
            plt.ylabel('Threat Type')
            plt.tight_layout()
            
            plot_path = self.config.VALIDATION_DIR / f"threat_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path)
            plt.close()
            
            mlflow.log_artifact(str(plot_path))

    def generate_threat_report(self, save_path: str = None) -> Dict:
        """Generate comprehensive PDF/HTML report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': self.metrics,
            'classification_report': self.classification_reports,
            'threat_coverage': self.threat_coverage,
            'error_analysis_summary': {
                'false_positives': len(self.error_analysis['false_positives']),
                'false_negatives': len(self.error_analysis['false_negatives']),
                'high_confidence_errors': len([
                    e for e in self.error_analysis['false_positives'] + 
                    self.error_analysis['false_negatives'] 
                    if e['confidence'] > 0.9
                ])
            },
            'recommendations': self._generate_recommendations()
        }
        
        if save_path:
            save_path = Path(save_path)
            if save_path.suffix == '.json':
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=4)
            # Could add HTML/PDF generation here
        
        return report

    def _generate_recommendations(self) -> Dict:
        """Generate actionable recommendations based on validation"""
        recs = {}
        
        # Threshold tuning
        if self.metrics['false_positive_rate'] > 0.1:
            recs['threshold_tuning'] = {
                'current_fpr': self.metrics['false_positive_rate'],
                'suggested_threshold': min(1.0, self.metrics['optimal_threshold'] + 0.1),
                'expected_impact': "Reduce false positives by ~15-20%"
            }
        
        # Coverage improvements
        for threat_type, stats in self.threat_coverage.items():
            if stats['detection_rate'] < 0.7:
                recs[f'improve_{threat_type}_detection'] = {
                    'current_rate': stats['detection_rate'],
                    'action': f"Add more {threat_type} samples to training",
                    'priority': "high" if stats['miss_rate'] > 0.5 else "medium"
                }
        
        return recs