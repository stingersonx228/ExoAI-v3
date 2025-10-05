import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.predict import predict

st.title("🌌 Exoplanet AI — Классификатор экзопланет")

uploaded_file = st.file_uploader("Загрузите CSV с колонками 'time' и 'flux'", type="csv")

def plot_lightcurve(time, flux):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, flux, '-o', markersize=2, alpha=0.7)
    ax.set_title('Lightcurve')
    ax.set_xlabel('Time')
    ax.set_ylabel('Flux')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'flux' not in df.columns:
        st.error("❌ В CSV должен быть столбец 'flux'")
    else:
        st.success(f"✅ Загружено {len(df)} измерений")
        flux = df['flux'].values
        time = df[df.columns[0]].values
        plot_lightcurve(time, flux)

        
        target_len = 1000
        flux_array = np.array(flux, dtype=np.float32)
        flux_array = (flux_array - np.mean(flux_array)) / (np.std(flux_array) + 1e-9)
        if len(flux_array) > target_len:
            flux_array = flux_array[:target_len]
        else:
            flux_array = np.pad(flux_array, (0, target_len - len(flux_array)), 'constant')

        probs, classes = predict(flux_array)
        st.subheader("🎯 Результаты классификации:")
        for cls, p in zip(classes, probs):
            st.write(f"- **{cls}:** {p:.3f}")
        pred_class = classes[np.argmax(probs)]
        st.success(f"🪐 Итоговое предсказание: **{pred_class}**")


