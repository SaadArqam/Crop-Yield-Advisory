import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import os

os.makedirs("models", exist_ok=True)

df=pd.read_csv("data/processed_data.csv")

X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=DecisionTreeRegressor()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Decision Tree Evaluation:")
print("MAE:", mean_absolute_error(y_test,y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2:", r2_score(y_test,y_pred))

joblib.dump(model,"models/crop_yield_tree.pkl")
print("Decision Tree model saved.")