import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid tkinter issues
from data_fetcher import fetch_stock_data
from indicators import add_technical_indicators
from data_preprocessor import prepare_data, prepare_lstm_data
from lstm_model import build_lstm_model, train_lstm_model
from visualizer import plot_predictions
import logging
import warnings
from tensorflow.keras.models import save_model
from math import sqrt

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of NIFTY 50 tickers (partial list; update with full list)
NIFTY50_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    # Add remaining tickers to complete 50
]

def predict_future_prices(model, last_sequence, scaler, final_data, future_steps=5):
    """
    Predict future prices using the trained model.
    """
    future_predictions = []
    current_sequence = last_sequence.copy()  # Shape: (1, sequence_length, n_features)
    
    n_features = current_sequence.shape[2]
    close_idx = final_data.columns.get_loc('Close')
    
    for _ in range(future_steps):
        current_sequence_reshaped = current_sequence  # Already has batch dimension
        next_prediction = model.predict(current_sequence_reshaped, verbose=0)
        next_price = next_prediction[0, 0]  # Scaled prediction
        future_predictions.append(next_price)
        
        # Update the last timestep with the predicted scaled Close
        new_sequence = np.zeros_like(current_sequence)
        new_sequence[0, :-1] = current_sequence[0, 1:]  # Shift sequence, access first sample
        new_sequence[0, -1, close_idx] = next_price  # Update Close at the last timestep
        current_sequence = new_sequence
    
    # Inverse transform predictions to get raw Close prices
    future_predictions_scaled = np.array(future_predictions).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_predictions_scaled).flatten()
    
    return future_prices

def process_nifty50_stocks(tickers, start_date, end_date, sequence_length, features, output_dir, plot_dir):
    """
    Process all NIFTY 50 stocks with k-fold cross-validation and save models.
    """
    results = []
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        raw_data_file = os.path.join(output_dir, 'raw', f"{ticker}_raw_data.csv")
        stock_data = fetch_stock_data(ticker, start_date, end_date, output_dir='data/raw')
        if stock_data is None or stock_data.empty:
            logging.warning(f"No data fetched for {ticker}")
            continue
        
        logging.info(f"{ticker} DataFrame:\n{stock_data.head()}\nColumns: {list(stock_data.columns)}\nShape: {stock_data.shape}")
        if isinstance(stock_data.columns, pd.MultiIndex):
            logging.info(f"{ticker} MultiIndex columns detected, flattening to first level")
            stock_data.columns = stock_data.columns.get_level_values(0)
            logging.info(f"{ticker} Flattened columns: {list(stock_data.columns)}")
        
        stock_data = add_technical_indicators(stock_data)
        if stock_data is None or stock_data.empty:
            logging.warning(f"No data after cleaning for {ticker}")
            continue
        
        final_data_file = os.path.join(output_dir, 'processed', f"{ticker}_final_data.csv")
        final_data = prepare_data(stock_data, features, final_data_file)
        if final_data is None or final_data.empty:
            logging.warning(f"No data after preprocessing for {ticker}")
            continue
        
        logging.info(f"Final data columns for {ticker}: {list(final_data.columns)}")
        
        X, y, scaler = prepare_lstm_data(final_data, 'Close', sequence_length, return_all=True)
        if X is None or len(X) < 2:
            logging.warning(f"Insufficient data for LSTM training for {ticker}")
            continue
        
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Ensure last_sequence has batch dimension
        last_sequence = X[-1:,:,:]  # Reshape to (1, sequence_length, n_features)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores, rmse_scores, mape_scores = [], [], []
        best_model = None
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            logging.info(f"Processing fold {fold + 1} for {ticker}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = build_lstm_model((sequence_length, X_train.shape[2]))
            model, history = train_lstm_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
            
            fold_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            h5_path = os.path.join(output_dir, 'models', f"{ticker}_fold_{fold + 1}_{fold_timestamp}.keras")
            model.save(h5_path)  # Use native Keras format
            
            best_model = model
            
            predictions = model.predict(X_test, verbose=0)
            
            # Inverse transform predictions and actual values
            y_test_scaled = np.zeros((len(y_test), 1))
            y_test_scaled[:, 0] = y_test
            predictions_scaled = np.zeros((len(predictions), 1))
            predictions_scaled[:, 0] = predictions.flatten()
            y_test_inv = scaler.inverse_transform(y_test_scaled).flatten()
            predictions_inv = scaler.inverse_transform(predictions_scaled).flatten()
            
            mae = mean_absolute_error(y_test_inv, predictions_inv)
            rmse = sqrt(mean_squared_error(y_test_inv, predictions_inv))
            mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
            
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            mape_scores.append(mape)
        
        best_h5_path = os.path.join(output_dir, 'models', f"{ticker}_best_{timestamp}.keras")
        best_model.save(best_h5_path)  # Use native Keras format
        
        avg_mae = np.mean(mae_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_mape = np.mean(mape_scores)
        
        future_predictions = predict_future_prices(best_model, last_sequence, scaler, final_data, future_steps=5)
        
        future_dates = pd.date_range(start=final_data.index[-1], periods=6, freq='B')[1:]
        print(f"\nFuture Price Predictions for {ticker}:")
        for date, pred in zip(future_dates, future_predictions):
            print(f"{date.date()}: {pred:.2f} INR")
        
        # Prepare dates for plotting (use test set dates from final_data index)
        test_dates = final_data.index[test_idx]
        
        plot_file = os.path.join(plot_dir, f"{ticker}_prediction_{timestamp}.png")
        try:
            plot_predictions(test_dates, y_test_inv, predictions_inv, f"{ticker} - Actual vs Predicted Prices", plot_file)
        except Exception as e:
            logging.error(f"Error generating plot for {ticker}: {e}")
            continue
        
        results.append({
            'Ticker': ticker,
            'MAE': avg_mae,
            'RMSE': avg_rmse,
            'MAPE': avg_mape,
            'Pred_Day1': future_predictions[0],
            'Pred_Day2': future_predictions[1],
            'Pred_Day3': future_predictions[2],
            'Pred_Day4': future_predictions[3],
            'Pred_Day5': future_predictions[4]
        })
        print(f"{ticker} - MAE: {avg_mae:.2f}, RMSE: {avg_rmse:.2f}, MAPE: {avg_mape:.2f}%")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'nifty50_results.csv'), index=False)
    return results_df

if __name__ == "__main__":
    # Parameters
    start_date = "2020-01-01"
    end_date = "2025-07-25"
    sequence_length = 60
    features = ['Close', 'MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'BB_Mid', 'VWAP', 'Return', 'Volatility', 'ATR', 'Stoch_K', 'Stoch_D', 'OBV', 'Hammer', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Doji', 'Morning_Star', 'Evening_Star', 'Shooting_Star', 'Piercing_Line', 'Dark_Cloud_Cover']
    output_dir = "data"
    plot_dir = "data/results"
    
    results = process_nifty50_stocks(NIFTY50_TICKERS, start_date, end_date, sequence_length, features, output_dir, plot_dir)
    print("\nSummary of Results:")
    print(results)