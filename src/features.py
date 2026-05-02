import pandas as pd
import numpy as np

def add_momentum_features(df, window=14):
    """
    RODZINA: MOMENTUM (Pęd)
    Sprawdza siłę i szybkość trendu.
    """
    df = df.copy()
    
    # RSI - klasyk siły trendu
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['mom_RSI'] = 100 - (100 / (1 + rs))
    
    # ROC (Rate of Change) - procentowa zmiana ceny w czasie
    df['mom_ROC'] = df['Price'].pct_change(periods=window)
    
    return df

def add_mean_reversion_features(df, window=20):
    """
    RODZINA: MEAN REVERSION (Powrót do średniej)
    Szuka statystycznych odchyleń od normy.
    """
    df = df.copy()
    
    # Rolling Z-score (Standard Score)
    sma = df['Price'].rolling(window=window).mean()
    std = df['Price'].rolling(window=window).std()
    df['mr_Zscore'] = (df['Price'] - sma) / std
    
    # Distance to SMA - odległość procentowa od średniej 200-dniowej (klasyk)
    sma_long = df['Price'].rolling(window=200).mean()
    df['mr_DistSMA200'] = (df['Price'] - sma_long) / sma_long
    
    return df

def add_volatility_features(df, window=14):
    """
    RODZINA: VOLATILITY (Zmienność)
    Mierzy niepewność i ryzyko na rynku.
    """
    df = df.copy()
    
    # Log-return Volatility (Kroczące odchylenie standardowe zwrotów)
    returns = np.log(df['Price'] / df['Price'].shift(1))
    df['vol_RollingStd'] = returns.rolling(window=window).std() * np.sqrt(252) # Anualizowana
    
    # ATR (Average True Range) - uproszczony jeśli brak OHLC
    # Jeśli masz High/Low, warto tu wstawić pełny wzór ATR
    df['vol_Range'] = df['Price'].rolling(window=window).max() - df['Price'].rolling(window=window).min()
    
    return df

def add_regime_features(df):
    """
    RODZINA: MARKET REGIME (Stan rynku)
    Filtry określające w jakiej fazie rynku jesteśmy.
    """
    df = df.copy()
    
    # Binary Regime: 1 jeśli powyżej średniej 200d (Hossa), 0 jeśli poniżej (Bessa)
    sma_200 = df['Price'].rolling(window=200).mean()
    df['reg_IsBull'] = (df['Price'] > sma_200).astype(int)
    
    return df

def build_feature_matrix(df):
    """
    Główna funkcja budująca całą macierz cech.
    """
    df = add_momentum_features(df)
    df = add_mean_reversion_features(df)
    df = add_volatility_features(df)
    df = add_regime_features(df)
    
    # Dodajemy target (np. jutrzejszy log-return)
    df['target_next_return'] = np.log(df['Price'].shift(-1) / df['Price'])
    
    return df.dropna()