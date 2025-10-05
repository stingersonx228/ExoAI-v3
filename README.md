# 🪐 ExoAI v3 (final build)

Нейросеть для классификации транзитных кривых света на **confirmed**, **candidate** и **false** экзопланеты.

## 🚀 Возможности
- Автоматическое чтение и препроцессинг кривых света  
- Фазовое фолдингование по периоду  
- Обучение CNN для 3-классовой классификации  
- Визуализация вероятностей и проверка на новых данных

## 🧠 Структура проекта
ExoAI v3/
│
├── src/
│   ├── train.py
│   ├── model.py
│   ├── generate_metadata.py
│   ├── inspect_probs.py
│   ├── check_periods.py
│   └── ...
│
├── data/
│   ├── metadata.csv
│   └── lightcurves/...
│            
│
├── model_cnn.pth          
│
├── requirements.txt
├── README.md
└── .gitignore
