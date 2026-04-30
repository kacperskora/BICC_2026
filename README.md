# BICC 2026
**Projekt badawczy na konkurs Bussiness Intelligence Case Challenge (edycja 2026)**

---

## 📈 O projekcie
Celem projektu jest budowa zaawansowanego systemu do prognozowania rynków finansowych, łączącego klasyczne podejście ekonometryczne z nowoczesnymi technikami uczenia głębokiego (Deep Learning). System skupia się na trzech kluczowych aspektach:
1. **Prognozowanie zmienności** (Volatility Modeling)
2. **Predykcja kierunku zmian cen** (Classification)
3. **Estymacja stóp zwrotu** (Regression)

---

## 🛠 Struktura Projektu

Projekt jest podzielony na moduły, aby zapewnić czystość kodu i umożliwić równoległą pracę zespołu:
```text
BICC_2026/
├── data/               # Dane (raw - nieedytowalne, processed - po transformacjach)
├── notebooks/          # Analiza EDA, eksperymenty, wizualizacje SHAP
├── src/                # Kod źródłowy (serce systemu)
│   ├── data_loader.py  # Pobieranie danych, czyszczenie, Triple Barrier Labeling
│   ├── features.py     # Inżynieria cech, wskaźniki techniczne, Fractional Diff
│   ├── models/         # Implementacje modeli (XGBoost, LSTM, TFT)
│   └── backtest.py     # Silnik symulacji handlu (Transaction Costs, Sharpe Ratio)
├── tests/              # Testy jednostkowe logiki finansowej
├── main.py             # Główny skrypt uruchamiający cały pipeline
├── requirements.txt    # Zależności projektu
└── README.md           # Dokumentacja projektu
