import yfinance as yf
import pandas as pd
import datetime

def load_data(ticker='SPY', start_years=3, interval='1d'):
    """
    获取历史数据
    :param ticker: 股票代码，默认 'SPY'
    :param start_years: 数据开始的年份数（距离当前）
    :param interval: 数据时间间隔
    :return: DataFrame 包含 Open, High, Low, Close, Volume 等
    """
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=start_years * 365)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return df
