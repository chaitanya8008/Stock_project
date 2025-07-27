# NIFTY 50 Stock Price Prediction

This project predicts stock prices for NIFTY 50 companies using an LSTM neural network. It fetches historical data, computes technical indicators, trains a model with 5-fold cross-validation, and visualizes results with date-based plots.

## Features
- Fetches NIFTY 50 stock data via `yfinance`.
- Calculates indicators like MA, RSI, and MACD.
- Trains LSTM with cross-validation and predicts 5-day futures.
- Plots actual vs. predicted prices with date labels.

# Technical Indicators
This project leverages a comprehensive set of technical indicators to enhance stock price predictions. The following indicators are calculated and utilized:
- **Close**: The closing price of the stock.
- **MA50**: 50-day Moving Average for trend analysis.
- **MA200**: 200-day Moving Average for long-term trends.
- **RSI**: Relative Strength Index to measure overbought/oversold conditions.
- **MACD**: Moving Average Convergence Divergence for momentum trading.
- **MACD_Signal**: Signal line for MACD crossovers.
- **BB_High, BB_Low, BB_Mid**: Bollinger Bands (Upper, Lower, and Middle) for volatility and trend boundaries.
- **VWAP**: Volume Weighted Average Price for trade execution insights.
- **Return**: Price return for performance evaluation.
- **Volatility**: Price volatility to assess risk.
- **ATR**: Average True Range for volatility measurement.
- **Stoch_K, Stoch_D**: Stochastic Oscillator (%K and %D) for momentum.
- **OBV**: On-Balance Volume for volume trend analysis.
- **Hammer, Doji, Bullish_Engulfing, Bearish_Engulfing, Morning_Star, Evening_Star, Shooting_Star, Piercing_Line,          Dark_Cloud_Cover**: Candlestick patterns for reversal and continuation signals.

## Requirements
- Python 3.8+
- Dependencies in `requirements.txt`

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/Bhavesh216/IITR-CLOUDXLAB-STOCK-PRICE-PREDICTION.git
Set up a virtual environment:
Windows: python -m venv venv && venv\Scripts\activate
macOS/Linux: python3 -m venv venv && source venv/bin/activate
# Install dependencies:
pip install -r requirements.txt
# Run the main script:
python src/main.py
# Output
Data: data/raw/, data/processed/
Models: data/models/{ticker}_best_*.keras
Plots: data/results/{ticker}_prediction_*.png
Results: data/nifty50_results.csv

# Project Structure
nifty50_stock_prediction/
├── data/           # Output files
├── src/            # Source code
│   ├── data_fetcher.py
│   ├── indicators.py
│   ├── data_preprocessor.py
│   ├── lstm_model.py
│   ├── visualizer.py
│   ├── main.py
│   └── test_visualizer.py
├── requirements.txt
├── README.md
└── venv/           # Optional virtual env

# Configuration
Edit src/main.py to adjust:

start_date and end_date (e.g., "2020-01-01", "2025-07-25")
sequence_length (default: 60)
features list
output_dir and plot_dir
Troubleshooting
No Data: Check internet and date range.
Plot Issues: Ensure matplotlib is installed and data/results/ is writable.
Straight Lines: May indicate underfitting; adjust lstm_model.py.
Errors: Share the log for help.


# Contact
For support, open an issue or email bhaveshkhullar1093@gmail.com
