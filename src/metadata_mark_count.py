import pandas as pd
meta = pd.read_csv("data/metadata.csv")
print(meta['label'].value_counts())
