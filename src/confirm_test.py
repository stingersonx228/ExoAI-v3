import pandas as pd, torch, numpy as np
from src.train import preprocess_lightcurve, INPUT_LEN
from src.model import ExoCNN
chk = torch.load("model_cnn.pth", map_location="cpu")
classes = list(chk['label_classes'])
model = ExoCNN(input_size=chk.get('input_size', INPUT_LEN), num_classes=len(classes))
model.load_state_dict(chk['model_state_dict'])
model.eval()
meta = pd.read_csv("data/metadata.csv")
errors = []
for _, r in meta.iterrows():
    if r['label']!='confirmed': continue
    seq = preprocess_lightcurve(r['filepath'], float(r['period']))
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    pred = classes[int(p.argmax())]
    if pred != 'confirmed':
        errors.append((r['id'], r['filepath'], pred, p.tolist()))
print("Misclassified confirmed (id, path, pred, probs):")
for e in errors[:20]:
    print(e)
