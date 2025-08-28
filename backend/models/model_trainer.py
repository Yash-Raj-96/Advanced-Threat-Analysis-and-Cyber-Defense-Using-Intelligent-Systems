import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from .multi_modal_model import MultiModalThreatDetector
from backend.config import Config
from scipy import sparse
import numpy as np
import logging
import mlflow

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = nn.CrossEntropyLoss()

    def _convert_to_tensor(self, data):
        """Convert input data to PyTorch tensor, handling sparse matrices and pandas DataFrames"""
        if sparse.issparse(data):
            data = data.toarray()
        elif hasattr(data, 'values'):
            data = data.values
        return torch.FloatTensor(data)

    def create_datasets(self, X_network, X_malware, X_cve, y):
        """Create PyTorch datasets from processed data with consistent splitting"""
        try:
            # Convert all inputs to tensors
            X_network_t = self._convert_to_tensor(X_network)
            X_malware_t = self._convert_to_tensor(X_malware)
            X_cve_t = self._convert_to_tensor(X_cve)

            if y is None:
                raise ValueError("Labels (y) are None. Please ensure they are generated during preprocessing.")
            
            y_array = y.values if hasattr(y, 'values') else y
            if isinstance(y_array, (list, np.ndarray)) and len(y_array) == 0:
                raise ValueError("Labels (y) are empty. Cannot create dataset.")
            
            y_t = torch.LongTensor(y_array)

            # Align dataset lengths to the smallest one
            min_len = min(len(X_network_t), len(X_malware_t), len(X_cve_t), len(y_t))
            X_network_t = X_network_t[:min_len]
            X_malware_t = X_malware_t[:min_len]
            X_cve_t = X_cve_t[:min_len]
            y_t = y_t[:min_len]

            logger.info(
                f"Input sizes aligned to minimum length: {min_len} | "
                f"Network: {X_network_t.shape}, Malware: {X_malware_t.shape}, "
                f"CVE: {X_cve_t.shape}, Labels: {y_t.shape}"
            )

            # Split into train/val/test (60/20/20)
            indices = np.arange(min_len)
            train_idx, test_idx = train_test_split(
                indices, 
                test_size=0.2, 
                random_state=self.config.SEED,
                stratify=y_t.numpy() if len(np.unique(y_t)) > 1 else None
            )
            train_idx, val_idx = train_test_split(
                train_idx, 
                test_size=0.25,  # 0.25 x 0.8 = 0.2
                random_state=self.config.SEED,
                stratify=y_t[train_idx].numpy() if len(np.unique(y_t[train_idx])) > 1 else None
            )

            # Create subsets
            train_data = (
                X_network_t[train_idx],
                X_malware_t[train_idx],
                X_cve_t[train_idx],
                y_t[train_idx]
            )
            
            val_data = (
                X_network_t[val_idx],
                X_malware_t[val_idx],
                X_cve_t[val_idx],
                y_t[val_idx]
            )
            
            test_data = (
                X_network_t[test_idx],
                X_malware_t[test_idx],
                X_cve_t[test_idx],
                y_t[test_idx]
            )

            # Create datasets
            train_dataset = TensorDataset(*train_data)
            val_dataset = TensorDataset(*val_data)
            test_dataset = TensorDataset(*test_data)

            logger.info(
                f"✅ Created datasets - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
            )
            
            return train_dataset, val_dataset, test_dataset

        except Exception as e:
            logger.exception("❌ Error creating datasets")
            raise


    def train_model(self, train_dataset, val_dataset, input_dims):
        """Train the multi-modal model with MLflow logging and early stopping"""
        try:
            self.model = MultiModalThreatDetector(
                network_input_dim=input_dims[0],
                malware_input_dim=input_dims[1],
                cve_input_dim=input_dims[2]
            ).to(self.device)

            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.config.LR_PATIENCE,
                factor=0.5
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.BATCH_SIZE * 2,
                num_workers=4,
                pin_memory=True
            )

            mlflow.set_tracking_uri(self.config.MLFLOW_URI)

            with mlflow.start_run(run_name="Multimodal-Threat-Detector"):
                # 🔹 Log static parameters and tags
                mlflow.set_tags({
                    "stage": "training",
                    "model_type": "MultiModalThreatDetector"
                })
                mlflow.log_params({
                    "batch_size": self.config.BATCH_SIZE,
                    "learning_rate": self.config.LEARNING_RATE,
                    "epochs": self.config.EPOCHS,
                    "early_stopping_patience": self.config.EARLY_STOPPING_PATIENCE
                })

                best_val_loss = float('inf')
                best_epoch = 0
                patience_counter = 0

                for epoch in range(self.config.EPOCHS):
                    self.model.train()
                    train_loss, correct, total = 0.0, 0, 0

                    for batch in train_loader:
                        x_net, x_mal, x_cve, y = [t.to(self.device, non_blocking=True) for t in batch]

                        optimizer.zero_grad(set_to_none=True)
                        outputs, _ = self.model(x_net, x_mal, x_cve)
                        loss = self.criterion(outputs, y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()

                        train_loss += loss.item() * x_net.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()

                    train_loss /= len(train_loader.dataset)
                    train_acc = correct / total
                    val_loss, val_acc = self.evaluate(val_loader)
                    scheduler.step(val_loss)

                    # 🔹 Log metrics
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    }, step=epoch)

                    logger.info(
                        f"Epoch {epoch+1}/{self.config.EPOCHS} - "
                        f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                    )

                    # 🔹 Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        patience_counter = 0
                        self.save_model()
                        # 🔹 Log the model as an MLflow artifact
                        mlflow.pytorch.log_model(self.model, artifact_path="best_model")
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                            logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                            break

                logger.info(f"✅ Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                return self.model


        except Exception as e:
            mlflow.end_run()
            logger.exception("Error during training")
            raise

    def evaluate(self, data_loader):
        """Evaluate model performance with reduced memory usage"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                x_net, x_mal, x_cve, y = [t.to(self.device, non_blocking=True) for t in batch]
                outputs, _ = self.model(x_net, x_mal, x_cve)
                loss = self.criterion(outputs, y)

                total_loss += loss.item() * x_net.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def save_model(self):
        """Save and register the trained model with MLflow"""
        import os
        model_dir = self.config.MODEL_DIR
        model_dir.mkdir(exist_ok=True)

        save_path = model_dir / 'threat_detector_latest.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': vars(self.config)
        }, save_path)

        logger.info(f"✅ Model saved to {save_path}")

        # Log artifact file to MLflow
        mlflow.log_artifact(str(save_path), artifact_path="model")

        # Register model (shows up under 'Models' tab in MLflow UI)
        mlflow.pytorch.log_model(
            self.model,
            artifact_path="pytorch_model",
            registered_model_name="CyberDefenseThreatDetector"
        )
