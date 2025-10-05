# 🌌 ExoAI v3 — Классификатор экзопланет по кривым блеска

ExoAI — это модель на PyTorch и Streamlit-веб-интерфейс, определяющий, является ли наблюдаемая кривая блеска звезды **экзопланетой (confirmed)**, **кандидатом (candidate)** или **ложным сигналом (false)**.

---

## 🚀 Возможности

- Автоматическая предобработка данных (нормализация, фолдирование, детрендинг)
- Обучение сверточной нейросети (ExoCNN)
- Веб-интерфейс на Streamlit для интерактивного тестирования
- Поддержка загрузки CSV-файлов с `time` и `flux`
- Сохранение и загрузка модели (`model_cnn.pth`)

---

## 🧩 Структура проекта
ExoAI-v3/
├── src/
│ ├── model.py # Определение модели ExoCNN
│ ├── train.py # Тренировка модели
│ ├── predict.py # Предсказания по CSV
│ ├── inspect_probs.py # Анализ вероятностей классов
│ └── ...
├── data/
│ └── metadata.csv # Метаданные световых кривых
├── app.py # Streamlit интерфейс
├── requirements.txt # Зависимости проекта
├── .gitignore
└── README.md


---

## ⚙️ Установка и запуск

```bash
git clone https://github.com/stingersonx228/ExoAI-v3.git
cd ExoAI-v3
pip install -r requirements.txt




▶️ Запуск Streamlit-приложения
streamlit run app.py



#После этого открой http://localhost:8501 в браузере и загрузи CSV с колонками:

time, flux


🧠 Обучение модели
python -m src.train





Автор: @stingersonx228


---

### 📄 **LICENSE (MIT)**

```text
MIT License

Copyright (c) 2025 stingersonx228

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.