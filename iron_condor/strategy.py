import numpy as np
import pandas as pd

# 从同级目录导入 train_model (示例)，假设 model.py 中有 train_model 函数可用
from .model import train_model

##############################
# 计算Iron Condor行权价
##############################
def calculate_iron_condor(predicted_high, predicted_low, safety_gap=2.0):
    """
    根据预测得到的下一周高价、低价，为 Iron Condor 设置执行价
    :param predicted_high: 模型预测的未来最高价
    :param predicted_low:  模型预测的未来最低价
    :param safety_gap:     安全边际(固定加减值)，可根据策略经验值设定
    :return: (sell_call_strike, buy_call_strike, sell_put_strike, buy_put_strike)
    """
    sell_call_strike = predicted_high + safety_gap
    buy_call_strike  = sell_call_strike + 2  # 保护腿再远2美元，示例值
    sell_put_strike  = predicted_low - safety_gap
    buy_put_strike   = sell_put_strike  - 2

    return sell_call_strike, buy_call_strike, sell_put_strike, buy_put_strike


##############################
# 简单策略函数 (不使用模型)
##############################
def simple_iron_condor_strategy(df, idx, safety_gap=2.0):
    """
    一个示例: 仅根据当前Close, 取上下2%区间，并叠加 safety_gap
    :param df:   DataFrame（包含至少 'Close' 列）
    :param idx:  当前所在行的索引（用于取当日价格）
    :param safety_gap: 安全边际
    :return: (sell_call_strike, buy_call_strike, sell_put_strike, buy_put_strike)
    """
    current_price = df['Close'].iloc[idx]

    # 如果是 Series，就取它的第 0 个元素,如此一来，current_price 一定是 float
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0]
    else:
        current_price = current_price

    # 假设预测后未来一周的高点可能 +2%，低点 -2%（仅示例）
    predicted_high = current_price * 1.02
    predicted_low  = current_price * 0.98

    return calculate_iron_condor(predicted_high, predicted_low, safety_gap)


##############################
# 基于模型的策略函数
##############################
def model_based_iron_condor_strategy(df, idx, model_high, model_low=None, features=None, safety_gap=2.0):
    """
    使用训练好的模型预测下一周最高价(及最低价)来设置 Iron Condor
    :param df:          DataFrame（预处理后，包含特征列）
    :param idx:         当前行索引
    :param model_high:  预测最高价的模型 (RandomForestRegressor)
    :param model_low:   预测最低价的模型 (可选，如果没有，就做一个简单假设)
    :param features:    需要提取的特征列，用于模型输入
    :param safety_gap:  安全边际
    :return: (sell_call_strike, buy_call_strike, sell_put_strike, buy_put_strike)
    """
    if features is None:
        # 若未指定特征列，默认用下面几项 (需与训练时一致)
        features = ['SMA_5','SMA_20','RSI_14','Boll_Width','Return_1d','Return_5d']

    # 取当前这行的特征
    X_current = df[features].iloc[idx:idx+1]  # DataFrame

    # 预测未来最高价
    predicted_high = model_high.predict(X_current)[0]

    # 如果也有 model_low，就用来预测最低价；否则用简单估计
    if model_low:
        predicted_low = model_low.predict(X_current)[0]
    else:
        # 简单处理: 用 predicted_high * 0.98 做示例
        predicted_low = predicted_high * 0.98

    # 调用计算 Iron Condor 行权价的函数
    return calculate_iron_condor(predicted_high, predicted_low, safety_gap)


##############################
# （可选）示例: 训练模型 & 返回策略
##############################
def load_or_train_model_high(df, features=None):
    """
    演示：在 strategy 中直接训练模型并返回，用于预测 Highest_Next_Week。
    实际环境中通常把训练放到单独流程，然后在策略中直接 load 模型文件。
    :param df:  预处理后的数据
    :param features:  特征列
    :return: (model_high, metrics)
    """
    if features is None:
        features = ['SMA_5','SMA_20','RSI_14','Boll_Width','Return_1d','Return_5d']

    # 目标：最高价
    X = df[features].copy()
    y_high = df['Highest_Next_Week'].copy()  # 在 preprocessing 中已生成
    
    # 调用 model.py 中的 train_model
    model_high, metrics_high = train_model(X, y_high, test_ratio=0.2, random_state=42)
    return model_high, metrics_high


def load_or_train_model_low(df, features=None):
    """
    类似地，训练最低价模型
    """
    if features is None:
        features = ['SMA_5','SMA_20','RSI_14','Boll_Width','Return_1d','Return_5d']

    X = df[features].copy()
    y_low = df['Lowest_Next_Week'].copy()
    
    model_low, metrics_low = train_model(X, y_low, test_ratio=0.2, random_state=42)
    return model_low, metrics_low


##############################
# 组合示例：结合两模型做策略
##############################
def two_model_iron_condor_strategy(df, idx, model_high, model_low, features=None, safety_gap=2.0):
    """
    同时用高价模型 & 低价模型来预测下一周价格区间
    :param df:         DataFrame
    :param idx:        当前行索引
    :param model_high: 预测最高价模型
    :param model_low:  预测最低价模型
    :param features:   特征列
    :param safety_gap: 安全边际
    :return: 行权价组合 (sell_call, buy_call, sell_put, buy_put)
    """
    if features is None:
        features = ['SMA_5','SMA_20','RSI_14','Boll_Width','Return_1d','Return_5d']
    
    X_current = df[features].iloc[idx:idx+1]

    predicted_high = model_high.predict(X_current)[0]
    predicted_low  = model_low.predict(X_current)[0]

    return calculate_iron_condor(predicted_high, predicted_low, safety_gap)
