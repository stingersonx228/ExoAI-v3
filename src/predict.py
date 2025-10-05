import torch
import torch.nn.functional as F
import numpy as np
from src.model import ExoCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model_cnn.pth"

def predict(flux_array):
    chk = torch.load(MODEL_PATH, map_location=DEVICE)
    label_classes = chk.get('label_classes', ['confirmed', 'candidate', 'false'])
    input_len = chk.get('input_size', 1000)
    flux_array = np.array(flux_array, dtype=np.float32)
    if len(flux_array) != input_len:
        flux_array = np.interp(
            np.linspace(0, len(flux_array) - 1, input_len),
            np.arange(len(flux_array)), flux_array
        )
    flux_array = (flux_array - np.mean(flux_array)) / (np.std(flux_array) + 1e-9)
    model = ExoCNN(input_size=input_len, num_classes=len(label_classes)).to(DEVICE)
    model.load_state_dict(chk['model_state_dict'])
    model.eval()
    x = torch.tensor(flux_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs, label_classes

if __name__ == "__main__":
    fake_flux = np.random.randn(200).astype(np.float32)
    probs, classes = predict(fake_flux)
    for cls, p in zip(classes, probs):
        print(f"{cls}: {p:.3f}")
    print("Predicted class:", classes[np.argmax(probs)])


