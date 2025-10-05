import pandas as pd
df = pd.read_csv("data/metadata.csv")
df['period'] = -1.0
df.to_csv("data/metadata.csv", index=False)
print("period set to -1 for all rows")
