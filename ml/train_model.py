import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import os

df=pd.read_csv("data/processed_data.csv")


df.drop("Unnamed: 0", axis=1,inplace=True)

X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

# print(df.head())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_absolute_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)


print("Model Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)




os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/crop_yield_model.pkl")

print("Model saved.")
