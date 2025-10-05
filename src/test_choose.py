import torch, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from src.train import preprocess_lightcurve, INPUT_LEN, DATA_META_CSV
from src.model import ExoCNN

chk = torch.load("model_cnn.pth", map_location="cpu")
classes = list(chk['label_classes'])
model = ExoCNN(input_size=chk.get('input_size', INPUT_LEN), num_classes=len(classes))
model.load_state_dict(chk['model_state_dict'])
model.eval()

meta = pd.read_csv(DATA_META_CSV)
meta = meta[meta['label'].notna() & (meta['label'] != 'unknown')].reset_index(drop=True)
from tqdm import tqdm
y_true, y_pred = [], []
for _, r in tqdm(meta.iterrows(), total=len(meta)):
    seq = preprocess_lightcurve(r['filepath'], float(r['period']))
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        p = torch.softmax(out, dim=1).cpu().numpy()[0]
    pred = int(p.argmax())
    y_true.append(r['label'])
    y_pred.append(classes[pred])

print(classification_report(y_true, y_pred, labels=classes))
cm = confusion_matrix(y_true, y_pred, labels=classes)
print("classes order:", classes)
print("confusion matrix:\n", cm)
