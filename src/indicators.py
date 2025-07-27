import pandas as pd
import numpy as np

def detect_candlestick_patterns(df):
    """
    Detect candlestick patterns using custom logic and return as binary features.
    """
    patterns = pd.DataFrame(index=df.index)
    
    # Basic patterns (retained from previous implementation)
    is_hammer = (df['Close'] > df['Open']) & \
                (df['Low'] <= df['Open'] - (df['Open'] - df['Close']) * 2) & \
                (df['High'] - df['Close'] < (df['Open'] - df['Low']) * 0.1)
    patterns['Hammer'] = is_hammer.astype(int)
    
    is_bullish_engulfing = (df['Close'].shift(1) < df['Open'].shift(1)) & \
                           (df['Close'] > df['Open']) & \
                           (df['Open'] < df['Close'].shift(1)) & \
                           (df['Close'] > df['Open'].shift(1))
    patterns['Bullish_Engulfing'] = is_bullish_engulfing.astype(int)
    
    is_bearish_engulfing = (df['Close'].shift(1) > df['Open'].shift(1)) & \
                           (df['Close'] < df['Open']) & \
                           (df['Open'] > df['Close'].shift(1)) & \
                           (df['Close'] < df['Open'].shift(1))
    patterns['Bearish_Engulfing'] = is_bearish_engulfing.astype(int)
    
    is_doji = abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1
    patterns['Doji'] = is_doji.astype(int)
    
    # New patterns with custom logic
    # Morning Star (Bullish reversal over 3 days)
    is_morning_star = (df['Close'].shift(2) < df['Open'].shift(2)) & \
                      (abs(df['Close'].shift(1) - df['Open'].shift(1)) <= (df['High'].shift(1) - df['Low'].shift(1)) * 0.1) & \
                      (df['Close'] > df['Open']) & \
                      (df['Close'] > df['Open'].shift(2)) & \
                      (df['Open'] > df['Close'].shift(1))
    patterns['Morning_Star'] = is_morning_star.astype(int)
    
    # Evening Star (Bearish reversal over 3 days)
    is_evening_star = (df['Close'].shift(2) > df['Open'].shift(2)) & \
                      (abs(df['Close'].shift(1) - df['Open'].shift(1)) <= (df['High'].shift(1) - df['Low'].shift(1)) * 0.1) & \
                      (df['Close'] < df['Open']) & \
                      (df['Close'] < df['Open'].shift(2)) & \
                      (df['Open'] < df['Close'].shift(1))
    patterns['Evening_Star'] = is_evening_star.astype(int)
    
    # Shooting Star (Bearish reversal)
    is_shooting_star = (df['Close'] < df['Open']) & \
                       (df['High'] >= df['Open'] + (df['Open'] - df['Close']) * 2) & \
                       (df['Low'] - df['Close'] < (df['High'] - df['Open']) * 0.1)
    patterns['Shooting_Star'] = is_shooting_star.astype(int)
    
    # Piercing Line (Bullish reversal over 2 days)
    is_piercing_line = (df['Close'].shift(1) < df['Open'].shift(1)) & \
                       (df['Close'] > df['Open']) & \
                       (df['Open'] < df['Close'].shift(1)) & \
                       (df['Close'] > (df['Open'].shift(1) + df['Close'].shift(1)) / 2)
    patterns['Piercing_Line'] = is_piercing_line.astype(int)
    
    # Dark Cloud Cover (Bearish reversal over 2 days)
    is_dark_cloud_cover = (df['Close'].shift(1) > df['Open'].shift(1)) & \
                          (df['Close'] < df['Open']) & \
                          (df['Open'] > df['Close'].shift(1)) & \
                          (df['Close'] < (df['Open'].shift(1) + df['Close'].shift(1)) / 2)
    patterns['Dark_Cloud_Cover'] = is_dark_cloud_cover.astype(int)
    
    return patterns

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    """
    if not all(col in df.columns for col in ['Close', 'High', 'Low', 'Volume', 'Open']):
        raise ValueError(f"DataFrame must contain ['Close', 'High', 'Low', 'Volume', 'Open'], got {list(df.columns)}")
    
    # Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    df['MACD'] = macd
    df['MACD_Signal'] = macd.ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['BB_Mid'] + 2 * std
    df['BB_Low'] = df['BB_Mid'] - 2 * std
    
    # VWAP
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum().replace(0, np.finfo(float).eps)
    
    # Additional Features
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
    
    # New Features from Previous Update
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Stochastic Oscillator (%K, %D)
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, np.finfo(float).eps)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    
    # Add Candlestick Patterns
    patterns = detect_candlestick_patterns(df)
    df = pd.concat([df, patterns], axis=1)
    
    # Drop NaN values
    df = df.dropna()
    return df