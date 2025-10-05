import os
import random
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from src.model import ExoCNN

INPUT_LEN = 1000
BATCH_SIZE = 32
EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_META_CSV = "data/metadata.csv"
MODEL_SAVE = "model_cnn.pth"
USE_RAW_MODE = False
RNG_SEED = 42
NUM_WORKERS = 0
PATIENCE = 50
PERIOD_MIN = 0.5
PERIOD_MAX = 30.0
PERIOD_NFREQ = 2000

def set_seed(seed=RNG_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RNG_SEED)

def load_lightcurve_csv(path):
    df = pd.read_csv(path)
    time = df.iloc[:, 0].to_numpy()
    flux = df.iloc[:, 1].to_numpy()
    mask = np.isfinite(time) & np.isfinite(flux)
    return time[mask], flux[mask]

def detrend_flux(time, flux, window=101, polyorder=3):
    if len(flux) < window:
        return flux - np.median(flux)
    trend = savgol_filter(flux, window_length=window, polyorder=polyorder, mode='mirror')
    return flux - trend

def phase_fold_and_resample(time, flux, period, out_len=INPUT_LEN):
    if period <= 0 or np.isnan(period):
        xp = np.linspace(time.min(), time.max(), out_len)
        return np.interp(xp, time, flux)
    t0 = time.min()
    phase = ((time - t0) % period) / period
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux[order]
    xp = np.linspace(0, 1, out_len)
    phase_ext = np.concatenate([phase_sorted - 1.0, phase_sorted, phase_sorted + 1.0])
    flux_ext = np.concatenate([flux_sorted, flux_sorted, flux_sorted])
    res = np.interp(xp, phase_ext, flux_ext)
    return res

def estimate_period_for_file(path):
    t, f = load_lightcurve_csv(path)
    if len(t) < 10:
        return -1.0
    f = detrend_flux(t, f, window=101, polyorder=3)
    f = (f - np.mean(f)) / (np.std(f) + 1e-9)
    try:
        from astropy.timeseries import LombScargle
        min_freq = 1.0 / PERIOD_MAX
        max_freq = 1.0 / PERIOD_MIN
        ls = LombScargle(t, f)
        freqs, power = ls.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=5)
        if freqs.size == 0:
            return -1.0
        idx = np.nanargmax(power)
        best_freq = freqs[idx]
        period = 1.0 / best_freq if best_freq > 0 else -1.0
        return float(period)
    except Exception:
        try:
            from scipy.signal import lombscargle
            freqs = np.linspace(1.0 / PERIOD_MAX, 1.0 / PERIOD_MIN, PERIOD_NFREQ)
            ang = 2.0 * np.pi * freqs
            p = lombscargle(t, f - np.mean(f), ang)
            idx = np.argmax(p)
            best_freq = freqs[idx]
            period = 1.0 / best_freq if best_freq > 0 else -1.0
            return float(period)
        except Exception:
            f0 = f - np.mean(f)
            ac = np.correlate(f0, f0, mode='full')
            ac = ac[ac.size // 2 :]
            if len(ac) < 3:
                return -1.0
            peaks = (np.diff(np.sign(np.diff(ac))) < 0).nonzero()[0]
            if peaks.size == 0:
                return -1.0
            lag = peaks[0] + 1
            dt = np.median(np.diff(t))
            period = lag * dt
            if period <= 0:
                return -1.0
            return float(period)

def ensure_periods_in_metadata(meta_df, save=True):
    updated = False
    for i, row in meta_df.iterrows():
        try:
            p = float(row.get('period', -1.0))
        except Exception:
            p = -1.0
        if p <= 0 or np.isnan(p):
            path = row['filepath']
            if not os.path.exists(path):
                continue
            period_est = estimate_period_for_file(path)
            meta_df.at[i, 'period'] = period_est
            updated = True
    if updated and save:
        meta_df.to_csv(DATA_META_CSV, index=False)
    return meta_df

def preprocess_lightcurve(path, period):
    t, f = load_lightcurve_csv(path)
    if USE_RAW_MODE:
        f = (f - np.mean(f)) / (np.std(f) + 1e-9)
        if len(f) > INPUT_LEN:
            f = f[:INPUT_LEN]
        else:
            f = np.pad(f, (0, max(0, INPUT_LEN - len(f))), 'constant')
        return f.astype(np.float32)
    else:
        f = detrend_flux(t, f, window=101, polyorder=3)
        f = (f - np.mean(f)) / (np.std(f) + 1e-9)
        seq = phase_fold_and_resample(t, f, period, out_len=INPUT_LEN)
        seq = (seq - seq.mean()) / (seq.std() + 1e-9)
        return seq.astype(np.float32)

class LightcurveDataset(Dataset):
    def __init__(self, meta_df, cache_dir="cache_np", transform=None):
        self.meta = meta_df.reset_index(drop=True)
        self.transform = transform
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        cache_file = os.path.join(self.cache_dir, f"{row['id']}.npy")

        if os.path.exists(cache_file):
            seq = np.load(cache_file)
            if seq.dtype != np.float32:
                seq = seq.astype(np.float32)
                np.save(cache_file, seq)
        else:
            seq = preprocess_lightcurve(row['filepath'], float(row['period']))
            seq = seq.astype(np.float32)
            np.save(cache_file, seq)

        label = row["label"]

        if label == "confirmed" and np.random.rand() < 0.9:
            L = len(seq)
            depth = 0.15 + 0.10 * np.random.rand()
            dur = max(1, int(L * (0.03 + 0.05 * np.random.rand())))
            start = np.random.randint(0, max(1, L - dur))
            seq[start:start + dur] -= depth
            if np.random.rand() < 0.5:
                rebound = min(L - (start + dur), dur // 2)
                if rebound > 0:
                    seq[start + dur:start + dur + rebound] += depth * 0.3

        elif label == "candidate" and np.random.rand() < 0.5:
            seq += np.random.normal(0, 0.005, size=seq.shape).astype(np.float32)

        elif label == "false" and np.random.rand() < 0.7:
            seq += np.random.normal(0, 0.02, size=seq.shape).astype(np.float32)
            if np.random.rand() < 0.4:
                shift = int(np.clip(np.random.randint(50, 200), 1, len(seq)-1))
                seq = np.roll(seq, shift)

        if np.random.rand() < 0.3:
            seq += np.random.normal(0, 0.002, size=seq.shape).astype(np.float32)

        if self.transform:
            seq = self.transform(seq)

        tensor = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).float()
        return tensor, int(row["label_enc"])

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return (running_loss / total) if total>0 else 0.0, (correct / total) if total>0 else 0.0

def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    probs_all = []
    labels_all = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            probs_all.append(probs)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            labels_all.append(y.cpu().numpy())
            total += x.size(0)
    if total==0:
        return 0.0, 0.0, np.empty((0,)), np.empty((0,)), None
    probs_all = np.vstack(probs_all)
    labels_all = np.concatenate(labels_all)
    return running_loss/total, correct/total, probs_all, labels_all, None

def make_sampler(labels):
    counts = np.bincount(labels)
    if counts.size <= 1 or np.any(counts==0):
        return None
    weight_per_class = 1.0 / (counts + 1e-9)
    samples_weight = np.array([weight_per_class[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    return sampler

def main():
    meta = pd.read_csv(DATA_META_CSV)
    meta = meta[meta['label'].notna() & (meta['label'] != 'unknown')].copy()
    if meta.empty:
        print("No labeled data.")
        return
    meta = ensure_periods_in_metadata(meta, save=True)
    le = LabelEncoder()
    meta['label_enc'] = le.fit_transform(meta['label'].astype(str))
    train_df, val_test_df = train_test_split(meta, test_size=0.2, stratify=meta['label_enc'], random_state=RNG_SEED)
    val_df, test_df = train_test_split(val_test_df, test_size=0.5, stratify=val_test_df['label_enc'], random_state=RNG_SEED)
    train_ds = LightcurveDataset(train_df)
    val_ds = LightcurveDataset(val_df)
    test_ds = LightcurveDataset(test_df)
    sampler = make_sampler(train_df['label_enc'].to_numpy())
    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = ExoCNN(input_size=INPUT_LEN, num_classes=len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_probs, val_labels, _ = eval_epoch(model, val_loader, criterion)
        print(f" train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'label_classes': le.classes_, 'input_size': INPUT_LEN}, MODEL_SAVE)
            print("  Saved best checkpoint")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs, stopping early.")
                break
    if os.path.exists(MODEL_SAVE):
        chk = torch.load(MODEL_SAVE, map_location=DEVICE)
        model.load_state_dict(chk['model_state_dict'])
        _, test_acc, test_probs, test_labels, _ = eval_epoch(model, test_loader, criterion)
        print(f"\nTEST ACCURACY: {test_acc:.4f}")
        if test_probs.size>0:
            preds = np.argmax(test_probs, axis=1)
            unique, counts = np.unique(preds, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"  {le.inverse_transform([u])[0]}: {c} ({c/len(preds)*100:.1f}%)")
            for cls_idx, cls_name in enumerate(le.classes_):
                mask = test_labels==cls_idx
                if mask.sum()>0:
                    avg = test_probs[mask].mean(axis=0)
                    print(f"avg probs for true={cls_name}: {avg}")
    else:
        print("No checkpoint saved.")

if __name__ == "__main__":
    main()

