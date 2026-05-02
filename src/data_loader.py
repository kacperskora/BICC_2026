import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Uniwersalny loader dla plików CSV/Parquet.
    Automatycznie wykrywa kolumnę czasu i ustawia ją jako indeks.
    """
    path = Path(file_path)
    
    # Obsługa różnych formatów
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError("Nieobsługiwany format pliku. Użyj CSV lub Parquet.")

    # Szukanie kolumny z datą (szukamy fraz 'date', 'time', 'timestamp')
    date_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'timestamp'])]
    
    if date_cols:
        col = date_cols[0]
        df[col] = pd.to_datetime(df[col])
        df = df.sort_values(col).set_index(col)
        print(f"✅ Ustawiono indeks czasowy na kolumnie: {col}")
    else:
        print("⚠️ Nie znaleziono kolumny daty. Indeks pozostaje domyślny.")

    return df

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Naprawia typy danych: usuwa przecinki, konwertuje na float, 
    czyści procenty i usuwa puste kolumny.
    """
    df = df.copy()
    
    # 1. Usuń kolumny, które są całkowicie puste (jak Twój Vol.)
    df = df.dropna(axis=1, how='all')
    
    # 2. Napraw kolumny numeryczne (usuń przecinki i zmień na float)
    cols_to_fix = ['Price', 'Open', 'High', 'Low']
    for col in cols_to_fix:
        if col in df.columns and df[col].dtype == 'object':
            # Usuwamy przecinki (separatory tysięcy) i zmieniamy na float
            df[col] = df[col].str.replace(',', '').astype(float)
            
    # 3. Napraw kolumnę Change % (usuń % i zamień na ułamek)
    if 'Change %' in df.columns and df['Change %'].dtype == 'object':
        df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100
        
    return df

def analyze_data_content(df: pd.DataFrame):
    """
    Generuje szybki raport o zawartości danych: braki, typy, statystyki.
    """
    print("\n" + "="*30)
    print("📊 RAPORT ZAWARTOŚCI DANYCH")
    print("="*30)
    
    print(f"Liczba wierszy: {df.shape[0]}")
    print(f"Liczba kolumn: {df.shape[1]}")
    print("\n--- BRAKI W DANYCH ---")
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        print(nulls[nulls > 0])
    else:
        print("Brak brakujących danych.")

    print("\n--- TYPY I ZAKRESY ---")
    report = pd.DataFrame({
        'Typ': df.dtypes,
        'Min': df.min(),
        'Max': df.max(),
        'Unique': df.nunique()
    })
    print(report)
    
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"\nZakres czasowy: {df.index.min()} do {df.index.max()}")
        print(f"Częstotliwość danych: {pd.infer_freq(df.index) if len(df) > 3 else 'Nieznana'}")
    
    print("="*30 + "\n")

if __name__ == "__main__":
    # Test lokalny
    # df = load_data("data/raw/twoj_plik.csv")
    # analyze_data_content(df)
    pass