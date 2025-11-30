"""
Train and save the MPG prediction model.
This script is run during deployment to generate the pickle files.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from category_encoders import HashingEncoder
import gdown

print("[v0] Starting model training...")

# Download dataset from Google Drive
file_id = "1Brb-2ij5S5Ndt-P0da1DdWlmR1wwyfIn"
file_name = "vehicles_dataset.csv"

print("[v0] Downloading dataset...")
gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=True)

# Load dataset
df = pd.read_csv(file_name)
print(f"[v0] Dataset loaded: {df.shape}")

# Fill missing values
df['Engine_Cylinders'] = df['Engine_Cylinders'].fillna(df['Engine_Cylinders'].mean())
df['Engine_Size'] = df['Engine_Size'].fillna(df['Engine_Size'].mean())
df['Drive_Type'] = df['Drive_Type'].fillna(df['Drive_Type'].mode()[0])

# One-Hot Encoding
one_hot_cols = ['Drive_Type', 'Fuel_Type', 'Vehicle Class/Type']
df_encoded = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

# Hashing Encoding
hash_cols = ['Car_Brand']
hash_enc = HashingEncoder(cols=hash_cols, n_components=16)
df_encoded = pd.concat([
    df_encoded.drop(columns=hash_cols),
    hash_enc.fit_transform(df_encoded[hash_cols])
], axis=1)

# Outlier Detection
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(df_encoded)
df_encoded['outlier'] = outliers
df_filtered = df_encoded[df_encoded['outlier'] == 1].copy()
cleaned_df = df_filtered.drop(columns=['outlier'])

print("[v0] Data preprocessing complete")

# Separate features and target
X = cleaned_df.drop(columns=['Combined_MPG'])
y = cleaned_df['Combined_MPG']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Train model
print("[v0] Training RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=300, 
    max_depth=None, 
    min_samples_split=10, 
    min_samples_leaf=1, 
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"[v0] Model R² Score: {score:.4f}")

# Save model files
print("[v0] Saving model files...")
joblib.dump(model, "mpg_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train.columns.tolist(), "columns.pkl")

print("[v0] Model training complete!")
