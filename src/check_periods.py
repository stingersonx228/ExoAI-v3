import pandas as pd
df = pd.read_csv("data/metadata.csv")
print("Всего записей:", len(df))
print("period <= 0 или NaN:", ((df['period']<=0) | (df['period'].isna())).sum())
print(df[['id','label','period']].head(30))
print(df['period'].describe())
