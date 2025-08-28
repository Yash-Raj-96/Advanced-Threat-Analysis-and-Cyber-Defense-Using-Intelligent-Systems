import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def process_ember_dataset(input_path, output_dir="data"):
    """
    Process the EMBER malware dataset from parquet to training-ready format.
    Saves StandardScaler object and its stats (mean, std, var, etc.) to CSV.
    """
    print(f"📂 Loading EMBER dataset from: {input_path}")
    
    # Load parquet file
    df = pd.read_parquet(input_path)
    print(f"📊 Dataset shape: {df.shape}")
    
    # Basic preprocessing
    print("🧹 Cleaning data...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Identify label column
    label_candidates = ['Label', 'label', 'Attack_Label', 'attack_cat', 'Class', 'Target']
    label_col = next((col for col in df.columns if col.strip() in label_candidates), None)

    if not label_col:
        print("❌ Label column not found. Available columns:")
        print(df.columns.tolist())
        raise ValueError(f"Label column not found in DataFrame. Checked: {label_candidates}")

    y = df[label_col]
    X = df.drop(columns=[label_col])

    if y.isnull().all() or y.empty:
        raise ValueError(f"❌ Label column '{label_col}' is empty or contains only null values.")
    if X.empty:
        raise ValueError("❌ No usable features found after dropping label column.")

    # Select numeric columns only
    X = X.select_dtypes(include=np.number)

    print("⚖️ Scaling numeric features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train).to_parquet(os.path.join(output_dir, "X_train.parquet"))
    pd.DataFrame(X_test).to_parquet(os.path.join(output_dir, "X_test.parquet"))
    pd.DataFrame(y_train).to_parquet(os.path.join(output_dir, "y_train.parquet"))
    pd.DataFrame(y_test).to_parquet(os.path.join(output_dir, "y_test.parquet"))

    # Save StandardScaler object
    joblib.dump(scaler, os.path.join(output_dir, "ember_scaler.pkl"))

    # Save scaler stats as CSV
    feature_names = X.columns if hasattr(scaler, "feature_names_in_") else [f"feature_{i}" for i in range(X.shape[1])]
    scaler_stats_df = pd.DataFrame({
        "feature_name": feature_names,
        "feature_index": range(len(scaler.mean_)),
        "mean": scaler.mean_,
        "std": scaler.scale_,
        "var": scaler.var_,
        "n_features_in_": [scaler.n_features_in_] * len(scaler.mean_)
    })
    scaler_stats_df.to_csv(os.path.join(output_dir, "ember_scaler_stats.csv"), index=False)

    print(f"✅ EMBER processing complete. Saved to: {output_dir}")
    print(f"📊 Dataset shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    print("📁 Scaler saved as: ember_scaler.pkl")
    print("📄 Scaler stats saved as: ember_scaler_stats.csv ✅ CSV with: mean, std, var, feature index, name, count")

if __name__ == "__main__":
    input_file = "data/raw/network_logs/train_ember_2018_v2_features.parquet"
    process_ember_dataset(input_file)
