from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

def build_lstm_model(input_shape):
    """
    Build an LSTM model for stock price prediction.
    """
    model = Sequential([
        LSTM(units=100, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    logging.info(f"Model built with input_shape={input_shape}, units=100, layers=1, dropout=0.2")
    return model

def train_lstm_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
    """
    Train the LSTM model and return the trained model and history.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                       validation_split=validation_split, verbose=1)
    best_val_loss = min(history.history['val_loss'])
    logging.info(f"Training completed. Best val_loss: {best_val_loss}")
    return model, history