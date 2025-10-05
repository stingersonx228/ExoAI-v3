import pandas as pd
import numpy as np
df = pd.read_csv("data/metadata.csv")
np.random.seed(42)
df.loc[df['label']=='confirmed', 'period'] = np.random.uniform(3.0, 8.0, size=len(df[df['label']=='confirmed']))
df.loc[df['label']=='candidate', 'period'] = np.random.uniform(1.0, 2.8, size=len(df[df['label']=='candidate']))
df.loc[df['label']=='false', 'period'] = np.random.uniform(12.0, 25.0, size=len(df[df['label']=='false']))
df.to_csv("data/metadata.csv", index=False)
print("period заполнены по label диапазонам")
