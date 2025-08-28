import pandas as pd 
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict
import os
import json
from datetime import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from backend.config import Config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config): 
    #def __init__(self):
        self.config = Config()
        processed_dir = self.config.PROCESSED_DATA_DIR
        processed_dir.mkdir(parents=True, exist_ok=True)

        self.network_preprocessor_path = processed_dir / 'network_preprocessor.joblib'
        self.malware_scaler_path = processed_dir / 'malware_scaler.joblib'
        self.malware_imputer_path = processed_dir / 'malware_imputer.joblib'
        self.label_encoder_path = processed_dir / 'label_encoder.joblib'
        self.cve_preprocessor_path = processed_dir / 'cve_preprocessor.joblib'
        self.temp_dir = processed_dir / 'temp'
        self.temp_dir.mkdir(exist_ok=True)

    def _cleanup_temp_files(self):
        """Remove temporary files used during processing"""
        for file in self.temp_dir.glob('*.npy'):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp file {file}: {e}")

    def extract_cve_features(self, cve_data: Union[List[Dict], List[str]]) -> np.ndarray:
        """
        Extract features from CVE data which can be either:
        - List of dictionaries (already parsed JSON)
        - List of JSON strings (needs parsing)
        
        Args:
            cve_data: List of CVE items (either as dicts or JSON strings)
        Returns:
            numpy array of extracted features
        """
        try:
            if not cve_data:
                logger.warning("No CVE data provided")
                return np.array([])

            features = []
            for item in cve_data:
                # Handle both string and dict input
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse CVE item as JSON: {item[:100]}...")
                        continue
                elif not isinstance(item, dict):
                    logger.warning(f"Unexpected CVE item type {type(item)}")
                    continue

                # Extract basic CVE metadata
                cve_info = item.get('cve', {})
                if not cve_info:  # Handle case where 'cve' key might be missing
                    cve_info = item
                    
                cve_meta = cve_info.get('CVE_data_meta', {})
                cve_id = cve_meta.get('ID', '')
                
                # Get dates (try multiple possible fields)
                published_date = item.get('publishedDate') or item.get('published_date') or ''
                last_modified_date = item.get('lastModifiedDate') or item.get('last_modified_date') or ''
                
                # Calculate days since published
                try:
                    pub_date = datetime.strptime(published_date, "%Y-%m-%dT%H:%MZ") if published_date else None
                    days_since_published = (datetime.now() - pub_date).days if pub_date else 0
                except (ValueError, TypeError) as e:
                    logger.debug(f"Couldn't parse date {published_date}: {e}")
                    days_since_published = 0
                
                # Extract impact metrics (try both V3 and V2)
                impact = item.get('impact', {})
                base_metrics = impact.get('baseMetricV3', impact.get('baseMetricV2', {}))
                cvss_data = base_metrics.get('cvssV3', base_metrics.get('cvssV2', {}))
                
                # Safely calculate reference and description counts
                references_count = 0
                if isinstance(cve_info, dict):
                    references = cve_info.get('references', [])
                    if isinstance(references, list):
                        references_count = sum(                                
                            len(ref.get('reference_data', [])) if isinstance(ref, dict) else 0
                            for ref in references
                        )

                description_count = 0
                if isinstance(cve_info.get('description'), dict):
                    description_data = cve_info['description'].get('description_data', [])
                    if isinstance(description_data, list):
                        description_count = len(description_data)
                    # Create feature vector with fallback values
                feature_vec = [
                    days_since_published,
                    cvss_data.get('baseScore', 0),
                    self._cvss_vector_to_num(cvss_data.get('attackVector', '')),
                    self._cvss_vector_to_num(cvss_data.get('attackComplexity', '')),
                    self._cvss_vector_to_num(cvss_data.get('privilegesRequired', '')),
                    self._cvss_vector_to_num(cvss_data.get('userInteraction', '')),
                    self._cvss_vector_to_num(cvss_data.get('scope', '')),
                    self._cvss_vector_to_num(cvss_data.get('confidentialityImpact', '')),
                    self._cvss_vector_to_num(cvss_data.get('integrityImpact', '')),
                    self._cvss_vector_to_num(cvss_data.get('availabilityImpact', '')),
                    references_count,
                    description_count
                ]

                features.append(feature_vec)

                # Final check and array conversion
                if not features:
                    return np.array([])
                features_array = np.array(features, dtype=np.float32)

            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            features_array = imputer.fit_transform(features_array)
            
            # Scale features
            scaler = StandardScaler()
            features_array = scaler.fit_transform(features_array)
            
            joblib.dump(imputer, self.cve_preprocessor_path.with_name('cve_imputer.joblib'))
            joblib.dump(scaler, self.cve_preprocessor_path.with_name('cve_scaler.joblib'))
            
            logger.info(f"✅ Extracted {features_array.shape[0]} CVE features")
            return features_array

        except Exception as e:
            logger.exception("❌ Error extracting CVE features")
            raise

    @staticmethod
    def _cvss_vector_to_num(vector_value: str) -> float:
        """Convert CVSS vector strings to numerical values"""
        vector_map = {
            '': 0,
            'NONE': 0,
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 0.8,
            'CRITICAL': 1.0,
            'NETWORK': 0.8,
            'ADJACENT_NETWORK': 0.6,
            'LOCAL': 0.4,
            'PHYSICAL': 0.2,
            'UNCHANGED': 0.5,
            'CHANGED': 0.8
        }
        return vector_map.get(vector_value.upper(), 0)

    def preprocess_network_data(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], ColumnTransformer]:
        try:
            if df.empty:
                raise ValueError("❌ Input network DataFrame is empty.")

            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(axis=1, thresh=0.8 * len(df))

            # Attempt to find a suitable label column
            label_candidates = ['Label', 'label', 'Attack_Label', 'attack_cat', 'Class', 'Target']
            label_col = next((col for col in df.columns if col.strip() in label_candidates), None)

            if not label_col:
                raise ValueError(f"❌ Label column not found in DataFrame. Checked: {label_candidates}")

            y = df[label_col]
            X = df.drop(columns=[label_col])

            if y.isnull().all() or y.empty:
                raise ValueError(f"❌ Label column '{label_col}' is empty or contains only null values.")

            if X.empty:
                raise ValueError("❌ No usable features found after dropping label column.")

            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                #('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=10))  # or limit categories

            ])

            preprocessor = ColumnTransformer([
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ], sparse_threshold=1.0)  # Keep sparse output

            if fit:
                X_processed = preprocessor.fit_transform(X)
                joblib.dump(preprocessor, self.network_preprocessor_path)
            else:
                preprocessor = joblib.load(self.network_preprocessor_path)
                X_processed = preprocessor.transform(X)

            # Encode labels if they are categorical
            if y is not None and y.dtype == 'object':
                if fit:
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
                    joblib.dump(label_encoder, self.label_encoder_path)
                else:
                    label_encoder = joblib.load(self.label_encoder_path)
                    y = label_encoder.transform(y)

            logger.info("✅ Network data preprocessing completed successfully.")
            return X_processed, y, preprocessor

        except Exception as e:
            logger.exception(f"❌ Error preprocessing network data: {str(e)}")
            raise

    def preprocess_malware_data(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        chunk_size: int = 10000
    ) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
        try:
            if df.empty:
                raise ValueError("Input malware DataFrame is empty.")

            X = df.drop(columns=['label'], errors='ignore')
            y = df['label'] if 'label' in df.columns else None

            # Optimize memory usage by downcasting
            for col in X.select_dtypes(include='float64').columns:
                X[col] = pd.to_numeric(X[col], downcast='float32')
            for col in X.select_dtypes(include='int64').columns:
                X[col] = pd.to_numeric(X[col], downcast='integer')

            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()

            # Create memory-mapped array for the final output
            output_shape = (X.shape[0], X.shape[1])
            output_path = self.temp_dir / 'malware_processed.npy'
            X_scaled = np.memmap(output_path, dtype='float32', mode='w+', shape=output_shape)

            if fit:
                # First pass: fit the imputer and scaler
                for start in range(0, X.shape[0], chunk_size):
                    chunk = X.iloc[start:start + chunk_size]
                    imputed_chunk = imputer.fit_transform(chunk)
                    scaler.partial_fit(imputed_chunk)
                
                joblib.dump(imputer, self.malware_imputer_path)
                joblib.dump(scaler, self.malware_scaler_path)
            else:
                imputer = joblib.load(self.malware_imputer_path)
                scaler = joblib.load(self.malware_scaler_path)

            # Second pass: transform the data in chunks
            for start in range(0, X.shape[0], chunk_size):
                end = min(start + chunk_size, X.shape[0])
                chunk = X.iloc[start:end]
                
                # Impute and scale
                imputed_chunk = imputer.transform(chunk)
                scaled_chunk = scaler.transform(imputed_chunk)
                
                # Store in memory-mapped array
                X_scaled[start:end] = scaled_chunk.astype('float32')
            
            # Flush changes to disk
            X_scaled.flush()
            
            logger.info(f"✅ Malware data preprocessing completed. Processed shape: {X_scaled.shape}")
            return X_scaled, y, scaler

        except Exception as e:
            self._cleanup_temp_files()
            logger.exception("❌ Error preprocessing malware data.")
            raise
        finally:
            # Clean up the memmap file if we're not returning it
            if 'X_scaled' in locals() and isinstance(X_scaled, np.memmap):
                X_scaled._mmap.close()

    @staticmethod
    def load_preprocessor(path: Union[str, Path]):
        try:
            return joblib.load(path)
        except Exception:
            logger.exception(f"❌ Failed to load preprocessor from {path}.")
            raise