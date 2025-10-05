import torch, numpy as np, pandas as pd
from src.train import preprocess_lightcurve, INPUT_LEN
from src.model import ExoCNN
chk = torch.load("model_cnn.pth", map_location="cpu")
classes = list(chk['label_classes'])
model = ExoCNN(input_size=chk.get('input_size', INPUT_LEN), num_classes=len(classes))
model.load_state_dict(chk['model_state_dict'])
model.eval()
meta = pd.read_csv("data/metadata.csv")
meta = meta[meta['label'].notna() & (meta['label']!='unknown')].reset_index(drop=True)
from tqdm import tqdm
probs_per_true = {c: [] for c in classes}
for _, r in tqdm(meta.iterrows(), total=len(meta)):
    seq = preprocess_lightcurve(r['filepath'], float(r['period']))
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    probs_per_true[r['label']].append(p)
for k,v in probs_per_true.items():
    if len(v)>0:
        avg = np.mean(np.vstack(v), axis=0)
        print(k, "avg probs:", avg)
