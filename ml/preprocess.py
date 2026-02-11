import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("data/yield_df.csv")

df.dropna()

categorical_cols=df.select_dtypes(include=["object"]).columns

label_encoders={}
for i in categorical_cols:
    le=LabelEncoder()
    df[i]=le.fit_transform(df[i])
    label_encoders[i]=le



df.to_csv("data/processed_data.csv",index=False)
print("Data preprocessing done")
print("Processed shape:", df.shape)