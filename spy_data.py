# 如果没有安装 yfinance, 需要先安装:
# pip install yfinance

### Part1 准备环境与数据

import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# 1.机器学习相关
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
plt.show()

end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=3*365)  # 3年


## 2.使用 yfinance 获取 SPY 的历史 K线数据
df = yf.download('SPY', start=start_date, end=end_date, interval='1d')
df.head()

### Part 2 数据预处理与特征工程
## 1.数据清理：去除缺失值或异常值；简单处理，若有缺失值则用前值填充。
df = df.dropna()
df = df.asfreq('B')  # 将数据对齐到交易日频率（Business day）
df.fillna(method='ffill', inplace=True)  # 前值填充

## 2.生成目标值
# 我们假设一周约5个交易日
forecast_horizon = 5

# Shift - 获取“未来5天”区间的最高价、最低价
df['Highest_Next_Week'] = df['High'].shift(-1).rolling(forecast_horizon).max()
df['Lowest_Next_Week'] = df['Low'].shift(-1).rolling(forecast_horizon).min()

# 删除后面 forecast_horizon 行（因无法计算未来）
df.dropna(inplace=True)

df.head(10)

# 结果中我们会看到每一行新增了 Highest_Next_Week 和 Lowest_Next_Week，表示未来 5 个交易日的最高/最低。
# 实际策略可能更复杂，比如用“加权收盘价”或者再加点波动率、考虑成交量等。


## 3.生成特征 此处只演示 2 个简化指标，实际应用中可自行扩展。
# 简单移动平均线 (SMA)
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()

# 相对强弱指数 RSI(14) -- 简化版本
window_rsi = 14
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window_rsi).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window_rsi).mean()
df['RSI_14'] = 100 - (100 / (1 + gain / loss))

# 布林带(可选) - 这里只演示布林带宽度
df['MA_20'] = df['Close'].rolling(20).mean()
df['std_20'] = df['Close'].rolling(20).std()
df['Boll_Width'] = (df['std_20']*2) / df['MA_20']

# 去除因 rolling 导致的 NaN
df.dropna(inplace=True)

#合并到特征表，上面新增的列即是特征，我们再补充几个简单的价格变化率 (Price Change Rate)
df['Return_1d'] = df['Close'].pct_change(1)
df['Return_5d'] = df['Close'].pct_change(5)
df.dropna(inplace=True)

# 现在，我们的特征大致包含：SMA_5, SMA_20, RSI_14, Boll_Width, Return_1d, Return_5d
# 还可以加上交易量、隐含波动率数据(VIX) 等。此处仅作示范。


### Part3 构建机器学习模型
##1.准备训练集与测试集
# 目标：预测 Highest_Next_Week 和 Lowest_Next_Week。
# 以多目标回归方式分别构建，或者先做单一模型分别训练。如果为了简单，就演示一个模型先预测 Highest_Next_Week；再用同样逻辑再训练一个预测 Lowest_Next_Week 的模型。
features = [
    'SMA_5', 'SMA_20', 'RSI_14', 'Boll_Width',
    'Return_1d', 'Return_5d'
]

# --- 预测 最高价 ---
X = df[features].copy()
y_high = df['Highest_Next_Week'].copy()

# 时间序列切分 - 例如留最后 6 个月做测试
split_date = df.index[-120]  # 120交易日作为测试，大约半年
X_train = X[X.index <= split_date]
X_test  = X[X.index >  split_date]
y_train = y_high[y_high.index <= split_date]
y_test  = y_high[y_high.index >  split_date]

#或者使用 TimeSeriesSplit 做交叉验证（Walk-forward）：
#tscv = TimeSeriesSplit(n_splits=5)
#for train_idx, val_idx in tscv.split(X):
    # X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    # y_train, y_val = y_high.iloc[train_idx], y_high.iloc[val_idx]
    # 做模型训练和评估

##2.训练随机森林模型
#RandomForestRegressor 为例做示范，可根据情况换用 XGBoost、LightGBM、NN 等
model_high = RandomForestRegressor(
    n_estimators=100, 
    max_depth=5, 
    random_state=42
)
model_high.fit(X_train, y_train)

# 预测
y_pred_high = model_high.predict(X_test)

# 评估
mae = mean_absolute_error(y_test, y_pred_high)
mse = mean_squared_error(y_test, y_pred_high)
rmse = np.sqrt(mse)

print(f"Predict 'Highest_Next_Week': MAE={mae:.2f}, RMSE={rmse:.2f}")

#如果要同时预测 Highest_Next_Week & Lowest_Next_Week，可以：
#再训练一个 model_low = RandomForestRegressor(...) 目标是 y_low = df['Lowest_Next_Week']；
#或者用多输出回归 MultiOutputRegressor(...)。

### Part4 使用模型预测下一周价格区间
##1.预测区间
#假设我们已经训练好：
# model_high 用来预测 Highest_Next_Week
# model_low 用来预测 Lowest_Next_Week
# 假设取测试集最后一条数据（真实中应取当日收盘后来做预测）
# 在实际策略中，每个交易日收盘后，我们拿当天的特征输入模型，得到对未来一周最高价、最低价的预测，然后再加一点安全边际设定 Iron Condor 的卖方行权价。
# 示例：我们在 测试集最后一日（也可以是最新实盘数据）拿到特征 X_today，用模型预测下一周最高 & 最低。


X_latest = X_test.iloc[-1:].copy()  # dataframe with 1 row
predicted_high = model_high.predict(X_latest)[0]

# 如果有model_low:
# predicted_low = model_low.predict(X_latest)[0]
# 此处为演示，假设我们也训练了
predicted_low = predicted_high * 0.98  # 仅示例，真实应用要训练model_low

current_price = df['Close'].iloc[-1]  # 当前收盘价
print(f"今日收盘价: {current_price:.2f}")
print(f"未来一周预测最高价: {predicted_high:.2f}")
print(f"未来一周预测最低价: {predicted_low:.2f}")

#此时得到一个区间 [predicted_low, predicted_high]。真实策略中，你需要更精细的风控来决定多大的安全边际。

##2.设置iron condor的上下行权价
#设想要做一周到期的 Iron Condor（周五卖出，下周四或周五到期）：
# 从预测区间上下沿稍微“再远一点”(比如多留 2~5 美元的安全垫)；
# 在这个基础上分别卖出看涨期权 + 卖出看跌期权，同时买入更远虚值的对冲腿。
# 简单示例逻辑：
safety_gap = 2.0  # 给自己留2美元的缓冲
sell_call_strike = predicted_high + safety_gap
sell_put_strike = predicted_low - safety_gap

print(f"Iron Condor 卖方执行价 (示例):")
print(f"  - 卖出看涨期权行权价 ~ {sell_call_strike:.2f}")
print(f"  - 卖出看跌期权行权价 ~ {sell_put_strike:.2f}")

# 至于买入对冲腿(保护腿)，可再在这两个执行价外各加 1~2美元
buy_call_strike = sell_call_strike + 2
buy_put_strike  = sell_put_strike  - 2
print(f"保护腿行权价(示例):")
print(f"  - 买入看涨期权行权价 ~ {buy_call_strike:.2f}")
print(f"  - 买入看跌期权行权价 ~ {buy_put_strike:.2f}")

#在实际交易平台，还需要去查询对应执行价的期权合约代码、权利金报价、隐含波动率等，然后下单。


###Part 5 进一步的强化学习思路 (简要)

# 如果想做得更加自动化、让模型“自学”何时开仓 / 何时平仓 / 何时滚动，可以考虑强化学习(RL)。简单流程：

# 创建交易环境 (Environment)
# 在 Gym 或者某种自定义环境中，模拟每日收盘后做决策：卖哪些执行价的期权？是否平仓？
# 奖励(Reward) = 策略的盈亏 - 交易成本 - 风险敞口惩罚；
# 定义状态 (State)
# 当前标的价格、波动率、时间到期剩余天数、Greeks等；
# 定义动作 (Action)
# 卖(买)哪档看涨/看跌期权；平仓、滚动等操作；
# 训练 Agent
# 采用 Deep Q-Network(DQN)、PPO、SAC 等常见 RL 算法在此环境中反复训练。
# 需要较大的计算量，并且要注意不确定度和模拟的逼真程度。
# 难点：

# 期权定价本身是个动态多维问题(隐含波动率曲面、时间衰减、Gamma、Vega 等)，模拟环境可能要引入比较复杂的期权定价模型或真实历史 order book 数据。
# 对于短期周度 Iron Condor，爆发性风险(黑天鹅)随时存在，RL 需要足够多的极端场景训练，避免过拟合到正常行情。