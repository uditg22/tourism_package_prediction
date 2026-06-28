# ============================================================
# Data Preparation Script for Wellness Tourism Package Prediction
# ============================================================
# This script loads the raw dataset from Hugging Face, performs
# comprehensive data cleaning, feature engineering, encoding,
# and uploads the train/test splits back to Hugging Face.

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# ── Configuration ──────────────────────────────────────────
# Replace <your-hf-username> with your Hugging Face username
HF_REPO_ID = "panda1391/tourism-package-prediction"
DATASET_PATH = f"hf://datasets/{HF_REPO_ID}/tourism.csv"

# Initialize API client using the HF_TOKEN environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# ── Step 1: Load Dataset ───────────────────────────────────
print("Loading dataset from Hugging Face...")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded. Shape: {df.shape}")

# ── Step 2: Drop Unnecessary Columns ─────────────────────
# Drop the index column (Unnamed: 0) and CustomerID (unique identifier)
# CustomerID is not predictive — it's just an identifier
cols_to_drop = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {cols_to_drop}")

# ── Step 3: Clean Gender Data ─────────────────────────────
# Fix known data quality issue: 'Fe Male' should be 'Female'
print(f"Gender values before cleaning: {df['Gender'].unique()}")
df['Gender'] = df['Gender'].str.strip()  # Remove whitespace
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
print(f"Gender values after cleaning: {df['Gender'].unique()}")

# ── Step 4: Normalize Marital Status ──────────────────────
# Combine 'Unmarried' into 'Single' as they mean the same
print(f"MaritalStatus values before: {df['MaritalStatus'].unique()}")
df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})
print(f"MaritalStatus values after: {df['MaritalStatus'].unique()}")

# ── Step 5: Create Age Groups ─────────────────────────────
# Bin continuous age into 5 meaningful groups for better generalization
age_bins = [0, 25, 35, 45, 55, 100]
age_labels = ['Young Adult (18-25)', 'Early Career (26-35)',
              'Mid Career (36-45)', 'Senior Professional (46-55)', 'Elder (56+)']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
df.drop(columns=['Age'], inplace=True)  # Drop raw Age; AgeGroup replaces it
print(f"AgeGroup distribution:\n{df['AgeGroup'].value_counts()}")

# ── Step 6: Detect and Handle Anomalies (IQR Method) ──────
# Cap outliers using IQR fencing (Winsorization) for numeric columns
numeric_cols = ['DurationOfPitch', 'NumberOfTrips', 'MonthlyIncome',
                'NumberOfFollowups', 'NumberOfChildrenVisiting',
                'NumberOfPersonVisiting', 'PreferredPropertyStar']

print("\nAnomaly detection and capping using IQR method:")
for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        # Cap outliers (Winsorize) rather than dropping rows
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"  {col}: {outliers} outlier(s) capped | range [{lower_bound:.2f}, {upper_bound:.2f}]")

# ── Step 7: Handle Missing Values ─────────────────────────
# Impute missing values: median for numeric, mode for categorical
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object', 'category']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"\nMissing values after imputation: {df.isnull().sum().sum()}")

# ── Step 8: Encode Categorical Features ───────────────────
# Label encode ordinal categorical features
# Ordinal: Designation (Executive < Manager < Senior Manager < AVP < VP)
# AgeGroup is already ordered via pd.cut labels

designation_order = {'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4}
df['Designation'] = df['Designation'].map(designation_order)

# Label encode AgeGroup (ordinal — order matters)
age_group_order = {
    'Young Adult (18-25)': 0,
    'Early Career (26-35)': 1,
    'Mid Career (36-45)': 2,
    'Senior Professional (46-55)': 3,
    'Elder (56+)': 4
}
df['AgeGroup'] = df['AgeGroup'].astype(str).map(age_group_order)

# One-hot encode nominal categorical features
nominal_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
                'ProductPitched']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# Ensure boolean columns are int (for compatibility with XGBoost)
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

print(f"\nFinal dataset shape after encoding: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── Step 9: Define Target and Features ────────────────────
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nTarget distribution:\n{y.value_counts()}")
print(f"Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}")

# ── Step 10: Train/Test Split ──────────────────────────────
# Stratify on target to maintain class proportions in both splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {Xtrain.shape}, Test shape: {Xtest.shape}")

# ── Step 11: Save Splits Locally ──────────────────────────
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("\nTrain/test splits saved locally.")

# ── Step 12: Upload Splits to Hugging Face ─────────────────
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print(f"Uploaded: {file_path}")

print("\nData preparation complete. All splits uploaded to Hugging Face.")
