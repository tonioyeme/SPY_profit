import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    数据清洗与特征工程
    :param df: 原始数据 DataFrame
    :return: 处理后的 DataFrame
    """
    # 清洗数据
    df = df.dropna()
    df = df.asfreq('B')  # 对齐交易日
    df.ffill(inplace=True)  # 前值填充

    # 生成目标值（未来5天的最高价和最低价）
    forecast_horizon = 5
    df['Highest_Next_Week'] = df['High'].shift(-1).rolling(forecast_horizon).max()
    df['Lowest_Next_Week'] = df['Low'].shift(-1).rolling(forecast_horizon).min()

    # 删除预测无法计算的行
    df.dropna(inplace=True)

    # 特征工程
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    window_rsi = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window_rsi).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss))
    
    # 布林带宽度
    df['std_20'] = df['Close'].rolling(20).std()
    df['Boll_Width'] = (df['std_20'] * 2).mean() / df['Close'].rolling(20).mean()
    
    # 收益率特征
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    df.dropna(inplace=True)
    return df
