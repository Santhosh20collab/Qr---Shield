import pandas as pd

df = pd.read_csv("dataset/metadata.csv")
print(df.head())
print("\nTotal images:", len(df))
print("\nLabel counts:\n", df["label"].value_counts())
print("\nPayload type counts:\n", df["payload_type"].value_counts())