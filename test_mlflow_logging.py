# test_mlflow_logging.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from mlflow_logger import log_metrics_and_shap

BASE_PATH = r"C:\Users\yashu\Downloads\cyber_defense_system-1 - Copy - Copy - Copy\data"

# === Intrusion Detection ===
print("[INFO] Loading intrusion data...")
df_intr = pd.read_csv(f"{BASE_PATH}\\cic_ids2017.csv").dropna()

# Drop non-numeric
non_numeric_cols = df_intr.select_dtypes(include=["object"]).columns.tolist()
print(f"[INFO] Dropping non-numeric columns from intrusion data: {non_numeric_cols}")
df_intr_numeric = df_intr.drop(columns=non_numeric_cols)

X_intr = df_intr_numeric.drop(columns=["Label"], errors="ignore")
y_intr = LabelEncoder().fit_transform(df_intr["Label"]) if "Label" in df_intr.columns else None

X_train, X_test, y_train, y_test = train_test_split(X_intr, y_intr, test_size=0.2)
model_intr = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
log_metrics_and_shap(model_intr, pd.DataFrame(X_test, columns=X_intr.columns), y_test, "Intrusion Detection")

# === Malware Classification ===
print("[INFO] Loading malware data...")
df_mal = pd.read_csv(f"{BASE_PATH}\\ember_scaler_stats.csv").dropna()

# Drop non-numeric features
non_numeric_mal = df_mal.select_dtypes(include=["object"]).columns.tolist()
non_numeric_mal.remove("target") if "target" in non_numeric_mal else None  # keep target
print(f"[INFO] Dropping non-numeric malware columns: {non_numeric_mal}")
df_mal_clean = df_mal.drop(columns=non_numeric_mal)

# Target setup
target_col = "target" if "target" in df_mal_clean.columns else df_mal_clean.columns[-1]
X_mal = df_mal_clean.drop(columns=[target_col])
y_mal = LabelEncoder().fit_transform(df_mal_clean[target_col])

X_train, X_test, y_train, y_test = train_test_split(X_mal, y_mal, test_size=0.2)
model_mal = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
log_metrics_and_shap(model_mal, pd.DataFrame(X_test, columns=X_mal.columns), y_test, "Malware Classification")

# === Vulnerability Scoring ===
print("[INFO] Loading vulnerability data...")
df_vuln = pd.read_csv(f"{BASE_PATH}\\nvd_processed.csv").dropna()

X_vuln = df_vuln.drop(columns=["severity_score", "cve_id"], errors="ignore")
y_vuln = df_vuln["severity_score"]

X_train, X_test, y_train, y_test = train_test_split(X_vuln, y_vuln, test_size=0.2)
model_vuln = GradientBoostingRegressor().fit(X_train, y_train)
log_metrics_and_shap(model_vuln, pd.DataFrame(X_test, columns=X_vuln.columns), y_test, "Vulnerability Scoring")
