import pandas as pd
import datetime
import requests
from io import StringIO

def load_data_marketwatch(ticker='SPY', start_years=3):
    """
    Load historical data from MarketWatch
    :param ticker: Stock symbol (e.g., 'SPY')
    :param start_years: Number of years from today to retrieve historical data
    :return: DataFrame with columns Date, Open, High, Low, Close, Volume
    """
    # Define the time range
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=start_years * 365)
    
    # Format dates for MarketWatch (assumes 'MM/DD/YYYY' format)
    start_date_str = start_date.strftime('%m/%d/%Y')
    end_date_str = end_date.strftime('%m/%d/%Y')
    
    # Construct the URL for MarketWatch historical data
    # NOTE: Update this URL if MarketWatch changes its download endpoint
    url = f"https://www.marketwatch.com/investing/fund/{ticker}/download-data?startDate={start_date_str}&endDate={end_date_str}"
    
    # Fetch data
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from MarketWatch (status code: {response.status_code})")
    
    # Convert CSV response to pandas DataFrame
    data = StringIO(response.text)
    df = pd.read_csv(data)
    
    # Convert 'Date' to datetime and ensure proper sorting
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    
    # Rename columns if necessary (MarketWatch may have different headers)
    df.rename(columns={
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }, inplace=True)
    
    return df
