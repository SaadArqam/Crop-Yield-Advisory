import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import os
import json

os.makedirs("models", exist_ok=True)

df=pd.read_csv("data/processed_data.csv")

# drop unnamed if present
if 'Unnamed: 0' in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

# print(df.head())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)


print("Model Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# save model
joblib.dump(model, "models/crop_yield_model.pkl")

# save metrics
metrics = {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}
with open("models/metrics_linear.json", "w") as f:
    json.dump(metrics, f, indent=2)

# save feature importances / coefficients
try:
    coefs = model.coef_
    feature_names = list(X.columns)
    feat_imp = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    feat_imp = feat_imp.reindex(feat_imp.coefficient.abs().sort_values(ascending=False).index)
    feat_imp.to_csv("models/linear_feature_importances.csv", index=False)
    print("Saved linear feature importances to models/linear_feature_importances.csv")
except Exception as e:
    print("Could not extract coefficients:", e)

print("Model saved.")
