import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import dagshub

# inisialisasi
REPO_OWNER = "ahmadrifaat" 
REPO_NAME = "Workflow-CI"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_experiment("Latihan Credit Scoring")

def train_and_tune_model():
    # Path data
    data_path = os.path.join('..', 'namadataset_preprocessing', 'credit_data_processed.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('namadataset_preprocessing', 'credit_data_processed.csv')

    # Load Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['risk_rating'])
    y = df['risk_rating']
    
    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # autolog
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    # GridSearchCV
    with mlflow.start_run() as run: 
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'random_state': [42]
        }
        
        grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # best mdel
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Evaluasi 
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # artifak custom
        os.makedirs('outputs', exist_ok=True)
        
        # Plot Confusion Matrix kustom
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Training Confusion Matrix')
        cm_path = 'outputs/training_confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path) 

        # JSON info kustom
        metric_info = {
            "status_evaluasi": "Sukses",
            "jumlah_data_testing": len(X_test),
            "cv_best_score": grid_search.best_score_,
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec
        }
        json_path = 'outputs/metric_info.json'
        with open(json_path, 'w') as f:
            json.dump(metric_info, f, indent=4)
        mlflow.log_artifact(json_path) 
        
        print("\nAutolog berhasil, cek DagsHub")

if __name__ == "__main__":
    train_and_tune_model()