# ============================================================
# Model Training Script for Wellness Tourism Package Prediction
# ============================================================
# This script loads preprocessed data from Hugging Face, trains
# an XGBoost model with GridSearchCV, logs all experiments with
# MLflow, and uploads the best model to the Hugging Face model hub.

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# ── MLflow Configuration ───────────────────────────────────
# In production (GitHub Actions), MLflow server runs locally on port 5000
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

# ── Configuration ──────────────────────────────────────────
# Replace <your-hf-username> with your Hugging Face username
HF_DATASET_REPO = "panda1391/tourism-package-prediction"
HF_MODEL_REPO = "panda1391/tourism-package-prediction"

api = HfApi(token=os.getenv("HF_TOKEN"))

# ── Step 1: Load Preprocessed Data from Hugging Face ──────
print("Loading train/test splits from Hugging Face...")
Xtrain = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytrain.csv").squeeze()
ytest = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytest.csv").squeeze()
print(f"Loaded: Xtrain {Xtrain.shape}, Xtest {Xtest.shape}")

# ── Step 2: Define Features ────────────────────────────────
# All features are numeric after preprocessing — apply StandardScaler
numeric_features = list(Xtrain.columns)

# ── Step 3: Handle Class Imbalance ────────────────────────
# Target is imbalanced (~4.2:1 No:Yes). Use scale_pos_weight in XGBoost
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight (scale_pos_weight): {class_weight:.2f}")

# ── Step 4: Define Pipeline ────────────────────────────────
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric='logloss'
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# ── Step 5: Define Hyperparameter Grid ────────────────────
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 4, 5],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.6, 0.8],
    'xgbclassifier__reg_lambda': [0.5, 1.0],
}

# ── Step 6: Train with MLflow Tracking ────────────────────
with mlflow.start_run(run_name="wellness_tourism_gridsearch"):

    # GridSearchCV with 5-fold cross-validation, optimizing recall
    grid_search = GridSearchCV(
        model_pipeline, param_grid,
        cv=5, scoring='recall',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations as nested MLflow runs
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_recall_cv", mean_score)
            mlflow.log_metric("std_recall_cv", std_score)

    # Log best parameters in the parent run
    mlflow.log_params(grid_search.best_params_)
    print("Best Parameters:", grid_search.best_params_)

    # Evaluate best model with classification threshold
    best_model = grid_search.best_estimator_
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log evaluation metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score'],
    })

    print("\nTraining Classification Report:")
    print(classification_report(ytrain, y_pred_train))
    print("\nTest Classification Report:")
    print(classification_report(ytest, y_pred_test))

    # Save best model locally
    model_path = "tourism-package-prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally: {model_path}")

# ── Step 7: Upload Best Model to Hugging Face ──────────────
print("\nUploading best model to Hugging Face model hub...")

# Check if model repo exists; create if not
try:
    api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
    print(f"Model repo '{HF_MODEL_REPO}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating model repo '{HF_MODEL_REPO}'...")
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False)
    print(f"Model repo '{HF_MODEL_REPO}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)
print(f"Model uploaded to: https://huggingface.co/{HF_MODEL_REPO}")
