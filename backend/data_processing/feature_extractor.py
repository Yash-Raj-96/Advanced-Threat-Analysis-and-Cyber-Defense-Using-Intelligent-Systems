# backend/data_processing/feature_extractor.py

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)

    def extract_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and scale numeric features from network data."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"Extracting numeric network features: {numeric_cols}")
            features = df[numeric_cols].fillna(0)
            scaled = self.scaler.fit_transform(features)
            return pd.DataFrame(scaled, columns=numeric_cols)
        except Exception as e:
            logger.error(f"Error in extract_network_features: {str(e)}")
            raise

    def extract_malware_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from malware metadata."""
        try:
            # Example: file size, entropy, api calls count (adjust to your data)
            selected_cols = ['file_size', 'entropy', 'api_call_count']
            available_cols = [col for col in selected_cols if col in df.columns]
            logger.info(f"Extracting malware features: {available_cols}")
            features = df[available_cols].fillna(0)
            scaled = self.scaler.fit_transform(features)
            return pd.DataFrame(scaled, columns=available_cols)
        except Exception as e:
            logger.error(f"Error in extract_malware_features: {str(e)}")
            raise

    def extract_cve_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract textual features from CVE vulnerability descriptions."""
        try:
            if 'description' not in df.columns:
                raise ValueError("Missing 'description' column in CVE data")

            logger.info("Vectorizing CVE descriptions with TF-IDF")
            descriptions = df['description'].fillna("").astype(str).apply(self._clean_text)
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)
            return pd.DataFrame(tfidf_matrix.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
        except Exception as e:
            logger.error(f"Error in extract_cve_features: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning for CVE descriptions."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
