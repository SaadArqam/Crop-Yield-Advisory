import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import os
import json

os.makedirs("models", exist_ok=True)

df=pd.read_csv("data/processed_data.csv")

X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=DecisionTreeRegressor()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print("Decision Tree Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# save metrics
metrics = {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}
with open("models/metrics_tree.json", "w") as f:
    json.dump(metrics, f, indent=2)

# save model
joblib.dump(model,"models/crop_yield_tree.pkl")
print("Decision Tree model saved.")

# save feature importances if available
try:
    fi = model.feature_importances_
    feature_names = list(X.columns)
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": fi})
    feat_imp = feat_imp.sort_values("importance", ascending=False)
    feat_imp.to_csv("models/tree_feature_importances.csv", index=False)
    print("Saved tree feature importances to models/tree_feature_importances.csv")
except Exception as e:
    print("Could not extract feature importances:", e)