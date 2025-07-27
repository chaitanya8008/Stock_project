import yfinance as yf
import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_stock_data(ticker, start_date, end_date, output_dir):
    """
    Fetch historical stock data from yfinance and save to CSV.
    Returns None if data fetch fails or data is empty.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            logging.warning(f"No data for {ticker}")
            return None
        # Handle multi-level columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            logging.info(f"MultiIndex columns detected for {ticker}, flattening to first level")
            stock_data.columns = stock_data.columns.get_level_values(0)
        logging.info(f"{ticker} fetched columns: {list(stock_data.columns)}")
        output_file = os.path.join(output_dir, f"{ticker}_raw_data.csv")
        stock_data.to_csv(output_file)
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None