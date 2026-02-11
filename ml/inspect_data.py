import pandas as pd

df=pd.read_csv("data/yield_df.csv")

print("Shape:",df.shape)
print("Size:",df.size)
print("\nColumns:")
print(df.columns)
# print(df.head())

df = df.drop(['Unnamed: 0'], axis=1).reset_index(drop=True)
print("\nFirst 5 rows:")
print(df.head())

# X = df.drop(columns=['hg/ha_yield'])
# y = df['hg/ha_yield']

