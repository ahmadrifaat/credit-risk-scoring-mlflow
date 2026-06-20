import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import dagshub

# Dagshub 
REPO_OWNER = "ahmadrifaat" 
REPO_NAME = "Workflow-CI"

if "GITHUB_ACTIONS" not in os.environ:
    print("Berjalan di lokal: Menginisialisasi DagsHub...")
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
else:
    print("Berjalan di GitHub Actions: Melewati dagshub.init() karena menggunakan Environment Variables.")

if "GITHUB_ACTIONS" in os.environ:
    mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

mlflow.set_experiment("Latihan Credit Scoring")

def train_baseline_model():
    data_path = os.path.join('..', 'namadataset_preprocessing', 'credit_data_processed.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('namadataset_preprocessing', 'credit_data_processed.csv')

    # Load Data 
    df = pd.read_csv(data_path)
    X = df.drop(columns=['risk_rating'])
    y = df['risk_rating']
    
    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Autologging 
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    print("Memulai pelatihan Baseline Model (Direkam langsung oleh MLflow Project)...")
    
    # Eksekusi langsung tanpa dibungkus 'with mlflow.start_run()'
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train, y_train)
    
    # Prediksi data pengujian
    y_pred = baseline_model.predict(X_test)

    # Evaluasi Baseline
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # Simpan artifak
    os.makedirs('outputs', exist_ok=True)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Baseline Confusion Matrix')
    cm_path = 'outputs/baseline_confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path) 

    # JSON info kustom
    metric_info = {
        "status_evaluasi": "Sukses Baseline Via CI",
        "jumlah_data_testing": len(X_test),
        "baseline_accuracy": acc,
        "baseline_precision": prec,
        "baseline_recall": rec
    }
    json_path = 'outputs/baseline_metric_info.json'
    with open(json_path, 'w') as f:
        json.dump(metric_info, f, indent=4)
    mlflow.log_artifact(json_path) 
    
    print("\n[SUKSES] Seluruh metrik dan artifak baseline berhasil dicatat!")

if __name__ == "__main__":
    train_baseline_model()