import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("models", exist_ok=True)

df=pd.read_csv("data/yield_df.csv")

# drop unnamed column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)

# drop missing values (persist the change)
df = df.dropna()

categorical_cols = df.select_dtypes(include=["object"]).columns

label_encoders = {}
for i in categorical_cols:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].astype(str))
    label_encoders[i] = le

# save label encoders for later use in the app
joblib.dump(label_encoders, "models/label_encoders.pkl")

# write processed data
df.to_csv("data/processed_data.csv", index=False)
print("Data preprocessing done")
print("Processed shape:", df.shape)