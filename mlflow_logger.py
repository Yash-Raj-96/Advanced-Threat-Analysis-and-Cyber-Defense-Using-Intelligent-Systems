# mlflow_logger.py

import os
import mlflow
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, mean_absolute_error
)

mlflow.set_tracking_uri("file:///mlruns")

def log_metrics_and_shap(model, X, y_true, task_name):
    mlflow.set_experiment(f"{task_name} Evaluation")
    
    with mlflow.start_run(run_name=f"{task_name} Run"):
        y_pred = model.predict(X)
        y_proba = getattr(model, "predict_proba", lambda x: None)(X)

        # === Metrics ===
        if task_name == "Intrusion Detection":
            mlflow.log_metric("Precision", precision_score(y_true, y_pred, average="weighted"))
            mlflow.log_metric("Recall", recall_score(y_true, y_pred, average="weighted"))
            mlflow.log_metric("F1 Score", f1_score(y_true, y_pred, average="weighted"))

        elif task_name == "Malware Classification":
            auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else 0.0
            mlflow.log_metric("AUC-ROC", auc)

        elif task_name == "Vulnerability Scoring":
            mae = mean_absolute_error(y_true, y_pred)
            mlflow.log_metric("MAE", mae)

        # === Visualization ===
        fig, ax = plt.subplots()

        if task_name == "Malware Classification":
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            ax.plot(fpr, tpr, label='ROC curve')
            ax.set_title("ROC Curve")
            ax.legend()

        elif task_name == "Intrusion Detection":
            cm = confusion_matrix(y_true, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
            ax.set_title("Confusion Matrix")

        elif task_name == "Vulnerability Scoring":
            mae = mean_absolute_error(y_true, y_pred)
            ax.text(0.5, 0.5, f"MAE: {mae:.4f}", ha='center', va='center', fontsize=14)
            ax.set_title("Regression Evaluation")
            ax.axis("off")

        img_path = f"{task_name.lower().replace(' ', '_')}_eval.png"
        plt.savefig(img_path)
        mlflow.log_artifact(img_path)
        plt.close()

        # === SHAP
        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            shap.summary_plot(shap_values, X, show=False)
            shap_path = f"{task_name.lower().replace(' ', '_')}_shap.png"
            plt.savefig(shap_path)
            mlflow.log_artifact(shap_path)
            plt.close()
        except Exception as e:
            print(f"[!] SHAP failed: {e}")

        model_path = f"{task_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"[✓] Logged to MLflow: {task_name}")
