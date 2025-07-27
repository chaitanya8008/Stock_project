import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

def prepare_data(df, features, output_file=None):
    """
    Prepare the DataFrame by selecting features and handling missing values.
    """
    if not all(feature in df.columns for feature in features):
        missing = [f for f in features if f not in df.columns]
        raise ValueError(f"Missing features in DataFrame: {missing}")
    
    df_prepared = df[features].copy()
    df_prepared = df_prepared.dropna()
    
    if output_file:
        df_prepared.to_csv(output_file)
    return df_prepared

def prepare_lstm_data(df, target_column, sequence_length, return_all=False):
    """
    Prepare data for LSTM by creating sequences and adding a target.
    """
    # Select features and target
    features = [col for col in df.columns if col != target_column]
    df_features = df[features].values
    df_target = df[target_column].values.reshape(-1, 1)  # Reshape for scaling
    
    # Standardize features and target
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)
    scaled_target = scaler.fit_transform(df_target).flatten()  # Scale target separately
    
    # Create sequences
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(scaled_features[i:(i + sequence_length)])
        y.append(scaled_target[i + sequence_length])
    X, y = np.array(X), np.array(y)
    
    if return_all:
        logging.info(f"Returning full dataset for {len(X)} sequences")
        return X, y, scaler
    else:
        # Return last sequence for prediction
        return X[-1:], y[-1:], scaler