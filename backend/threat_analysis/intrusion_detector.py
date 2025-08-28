# backend/threat_analysis/intrusion_detector.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class IntrusionDetector:
    def __init__(self, model_path="backend/models/intrusion_model.pkl", scaler_path="backend/models/intrusion_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load_components()

    def _load_components(self):
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except Exception:
            # Fall back to fresh model if no existing components
            self.model = RandomForestClassifier(n_estimators=100)
            self.scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def predict(self, X: pd.DataFrame):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled), self.model.predict_proba(X_scaled)

# ✅ Add this function for frontend integration
def detect_intrusion(input_df: pd.DataFrame):
    detector = IntrusionDetector()
    predictions, probabilities = detector.predict(input_df)
    return predictions, probabilities
