import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_predictions(dates, actual, predicted, title, output_file):
    """
    Plot actual vs predicted prices with proper date formatting on x-axis.
    
    Args:
        dates: List of date strings or datetime objects
        actual: List of actual prices
        predicted: List of predicted prices
        title: Plot title
        output_file: Output file path
    """
    plt.figure(figsize=(14, 6))
    
    # Convert string dates to datetime objects if needed
    if isinstance(dates[0], str):
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
    
    plt.plot(dates, actual, label='Actual Price')
    plt.plot(dates, predicted, label='Predicted Price')
    
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Price')
    
    # Format x-axis to show years
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()