import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUT_CSV = os.path.join(DATA_DIR, "metadata.csv")

def classify_flux_stats(flux, q1, q2):
    std = np.std(flux)
    if std <= q1:
        return "confirmed"
    elif std <= q2:
        return "candidate"
    else:
        return "false"

def main():
    rows = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f != "metadata.csv"]
    if not files:
        print("❌ В папке data нет CSV-файлов!")
        return

    std_values = []
    for fname in files:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, fname))
            if "flux" in df.columns:
                std_values.append(np.std(df["flux"].dropna()))
        except Exception:
            continue

    if not std_values:
        print("❌ Нет корректных данных для анализа.")
        return

    q1, q2 = np.quantile(std_values, [0.33, 0.66])

    for i, fname in enumerate(sorted(files), start=1):
        fpath = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            if "flux" not in df.columns:
                continue
            flux = df["flux"].dropna().to_numpy()
            if len(flux) == 0:
                continue
            label = classify_flux_stats(flux, q1, q2)
            period = float(np.random.uniform(1.0, 5.0))
            rows.append([i, period, fpath, label])
        except Exception:
            continue

    df_meta = pd.DataFrame(rows, columns=["id", "period", "filepath", "label"])
    df_meta.to_csv(OUTPUT_CSV, index=False)
    counts = df_meta["label"].value_counts().to_dict()

    print(f"✅ metadata.csv создан: {len(df_meta)} файлов")
    for k, v in counts.items():
        print(f"  • {k}: {v}")

if __name__ == "__main__":
    main()


