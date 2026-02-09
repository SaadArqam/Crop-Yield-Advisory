import pandas as pd

df=pd.read_csv("data/yield_df.csv")

print("Shape:",df.shape)
print("Size:",df.size)
print("\nColumns:")
print(df.columns)

df = df.drop(columns=['Unnamed: 0'],inplace=True)

# X = df.drop(columns=['hg/ha_yield'])
# y = df['hg/ha_yield']

