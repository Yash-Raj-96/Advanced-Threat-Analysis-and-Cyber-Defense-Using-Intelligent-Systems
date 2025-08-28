import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)
import shap
import pandas as pd
import os
import numpy as np

# === Classification Reports ===
def generate_classification_report(y_true, y_pred, output_path):
    """Generate and save classification report to a text file."""
    report = classification_report(y_true, y_pred)
    with open(output_path, 'w') as f:
        f.write(report)

def calculate_accuracy(y_true, y_pred):
    """Calculate and return accuracy."""
    return accuracy_score(y_true, y_pred)

# === Confusion Matrix ===
def generate_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

# === Charts for Attack Analysis ===
def generate_attack_bar_chart(df, save_path):
    """Generate a bar chart of attack categories."""
    if 'Label' in df.columns:
        attack_counts = df['Label'].value_counts()
        attack_counts.plot(kind='bar', color='steelblue')
        plt.title('Number of Attacks per Category')
        plt.xlabel('Attack Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def generate_attack_pie_chart(df, save_path):
    """Generate a pie chart of top attack types."""
    if 'Label' in df.columns:
        attack_counts = df['Label'].value_counts().head(5)
        attack_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
        plt.ylabel('')
        plt.title('Top 5 Attack Types')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# === Pairplot Visualization ===
def generate_pairplot(df, save_path):
    """Generate a seaborn pairplot of numeric features and labels."""
    if 'Label' in df.columns:
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.shape[1] >= 2:
            sample_df = numeric_df.copy()
            sample_df['Label'] = df['Label']
            sns.pairplot(sample_df.sample(n=min(500, len(sample_df))), hue='Label', diag_kind='kde')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

# === Interpretability using SHAP ===
def generate_interpretability_visual(input_features, save_path):
    """
    Generate a SHAP summary plot for a simple model on a single instance (interpretability).
    This is a mock/demo SHAP explanation to simulate explainability.
    """
    try:
        from sklearn.linear_model import LogisticRegression

        # Dummy model & data for SHAP explanation
        X_sample = np.random.rand(100, len(input_features))
        y_sample = np.random.randint(0, 2, 100)

        model = LogisticRegression()
        model.fit(X_sample, y_sample)

        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(np.array([input_features]))

        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"SHAP interpretability plot generation failed: {e}")
