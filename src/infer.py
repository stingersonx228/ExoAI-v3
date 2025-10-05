import torch
import pandas as pd
import numpy as np
import glob
import os

from src.model import ExoCNN
from src.train import preprocess_lightcurve, INPUT_LEN, DEVICE, MODEL_SAVE


def infer(filepath, period):
    chk = torch.load(MODEL_SAVE, map_location=DEVICE)
    model = ExoCNN(input_size=INPUT_LEN, num_classes=len(chk['label_classes'])).to(DEVICE)
    model.load_state_dict(chk['model_state_dict'])
    model.eval()

    seq = preprocess_lightcurve(filepath, period)
    x = torch.from_numpy(seq).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1).item()
    label = chk['label_classes'][pred]
    return label


if __name__ == "__main__":
    csv_files = glob.glob("data/*.csv")

    for fpath in csv_files:
        if "metadata" in os.path.basename(fpath).lower():
            continue  
        
        test_period = 5.0
        label = infer(fpath, test_period)
        print(f"{os.path.basename(fpath)} -> {label}")

