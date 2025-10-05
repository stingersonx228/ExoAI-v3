import pandas as pd
import matplotlib.pyplot as plt
from src.train import preprocess_lightcurve
import os

os.makedirs("debug_fold_plots", exist_ok=True)

df = pd.read_csv("data/metadata.csv")
samples = []
for lab in ['confirmed','candidate','false']:
    sub = df[df['label']==lab]
    if len(sub)==0:
        continue
    row = sub.iloc[0]
    samples.append((lab, row['filepath'], float(row['period'])))

for lab, path, p in samples:
    seq = preprocess_lightcurve(path, p)
    fn = f"debug_fold_plots/fold_{lab}.png"
    plt.figure(figsize=(10,3))
    plt.plot(seq, '-o', markersize=2)
    plt.title(f"{lab}  period={p:.3f}")
    plt.xlabel("index")
    plt.ylabel("flux (normalized)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fn, dpi=150)
    plt.close()
    print("saved", fn)
