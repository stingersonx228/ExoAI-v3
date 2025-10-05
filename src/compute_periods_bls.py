import pandas as pd
import numpy as np
from src.train import load_lightcurve_csv, detrend_flux
import os

try:
    from astropy.timeseries import BoxLeastSquares
except Exception:
    raise SystemExit("Install astropy: pip install astropy")

MIN_P = 0.2
MAX_P = 30.0
N_PERIODS = 2000
DURATION_FRAC = 0.02

df = pd.read_csv("data/metadata.csv")
for i, row in df.iterrows():
    path = row['filepath']
    if not os.path.exists(path):
        continue
    t, f = load_lightcurve_csv(path)
    if len(t) < 10:
        df.at[i, 'period'] = -1.0
        continue
    f = detrend_flux(t, f, window=101, polyorder=3)
    f = (f - np.mean(f)) / (np.std(f) + 1e-9)
    try:
        model = BoxLeastSquares(t, f)
        periods = np.linspace(MIN_P, MAX_P, N_PERIODS)
        durations = DURATION_FRAC * periods
        power = model.power(periods, durations).power
        best_idx = np.nanargmax(power)
        best_period = float(periods[best_idx])
        df.at[i, 'period'] = best_period
    except Exception:
        df.at[i, 'period'] = -1.0

df.to_csv("data/metadata.csv", index=False)
print("BLS periods computed and saved to data/metadata.csv")
