import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

os.makedirs("models", exist_ok=True)

# Load raw data (use yield_df.csv so preprocessing is explicit here)
df = pd.read_csv("data/yield_df.csv")

# Drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)

# Drop rows with missing target
df = df.dropna(subset=['hg/ha_yield'])

# Separate features and target
X = df.drop(columns=['hg/ha_yield']).copy()
y = df['hg/ha_yield']

# Detect column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Simple explicit preprocessing (easy to explain in viva)
# 1) Numeric: fill missing values with median
medians = {}
if len(numeric_cols) > 0:
    medians = X[numeric_cols].median()
    X[numeric_cols] = X[numeric_cols].fillna(medians)

# 2) Categorical: fill missing with string 'missing' and label-encode
label_encoders = {}
for col in categorical_cols:
    X[col] = X[col].fillna('missing').astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Save label encoders so UI / other scripts can reuse them
if len(label_encoders) > 0:
    joblib.dump(label_encoders, "models/label_encoders_pipeline.pkl")

# Save medians and feature order so UI can use exactly the same preprocessing
feature_order = numeric_cols + categorical_cols
with open('models/feature_order.json', 'w') as f:
    json.dump(feature_order, f)
if len(medians) > 0:
    # medians may be a pandas Series; convert to dict
    med_dict = medians.to_dict() if hasattr(medians, 'to_dict') else dict(medians)
    with open('models/medians.json', 'w') as f:
        json.dump(med_dict, f)

# Helper to train and save a model with simple preprocessing applied
def train_and_save_simple(model, name_prefix):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{name_prefix} Evaluation:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    # Save model
    model_path = f"models/{name_prefix}.pkl"
    joblib.dump(model, model_path)

    # Save metrics
    metrics = {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}
    with open(f"models/metrics_{name_prefix}.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save feature importances / coefficients
    feature_names = numeric_cols + categorical_cols
    try:
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            feat_imp = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
            feat_imp = feat_imp.reindex(feat_imp.coefficient.abs().sort_values(ascending=False).index)
            feat_imp.to_csv(f"models/{name_prefix}_feature_importances.csv", index=False)
            print(f"Saved linear feature importances to models/{name_prefix}_feature_importances.csv")
        elif hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            feat_imp = pd.DataFrame({'feature': feature_names, 'importance': fi})
            feat_imp = feat_imp.sort_values('importance', ascending=False)
            feat_imp.to_csv(f"models/{name_prefix}_feature_importances.csv", index=False)
            print(f"Saved tree feature importances to models/{name_prefix}_feature_importances.csv")
    except Exception as e:
        print('Could not extract feature importances:', e)

# Train linear model (with simple preprocessing applied)
linear_model = LinearRegression()
train_and_save_simple(linear_model, 'crop_yield_model')

# Train decision tree model (with same preprocessing)
tree_model = DecisionTreeRegressor()
train_and_save_simple(tree_model, 'crop_yield_tree')

print('Simple training complete.')
