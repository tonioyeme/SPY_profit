"""
main.py - 项目主入口
"""

from iron_condor import load_data, train_model, backtest_iron_condor
from iron_condor.preprocessing import preprocess_data
from iron_condor.strategy import (
    calculate_iron_condor, 
    simple_iron_condor_strategy  # 引入策略函数
)
from iron_condor.visualization_old import plot_dynamic_iron_condor
from iron_condor.export_and_plot import export_to_csv

try:
    import plotly.graph_objects as go
except ImportError:
    print("Plotly is not installed. Please install it using 'pip install plotly'.")

from iron_condor.visualization import plot_iron_condor

# 1. 加载数据
data = load_data()

# 2. 数据预处理
processed_data = preprocess_data(data)

# 3. 选择特征和目标
features = ['SMA_5', 'SMA_20', 'RSI_14', 'Boll_Width', 'Return_1d', 'Return_5d']
X = processed_data[features]
y_high = processed_data['Highest_Next_Week']
y_low = processed_data['Lowest_Next_Week']  # 目前暂不使用，可扩展

# 4. 训练模型 (这里只演示预测最高价)
model_high, metrics_high = train_model(X, y_high)
print(f"Model High Prediction - MAE: {metrics_high['mae']:.2f}, RMSE: {metrics_high['rmse']:.2f}")

# 5. 使用训练好的模型做一次简单预测
latest_features = X.iloc[-1:].copy()
predicted_high = model_high.predict(latest_features)[0]
predicted_low = predicted_high * 0.98  # 示例：假设低点为最高价的98%

# 6. 计算 Iron Condor 卖方执行价
iron_condor = calculate_iron_condor(predicted_high, predicted_low)
print(f"Iron Condor Strategy (Sample): {iron_condor}")

# 7. 回测演示
#    这里仅示例：在回测中并没有真正使用训练好的 'model_high'；
#    可在 strategy.py 中写一个基于 model_high 的策略函数，然后在此传给 backtest_iron_condor
backtest_results = backtest_iron_condor(
    processed_data,
    strategy_fn = lambda df, i: simple_iron_condor_strategy(df, i),
    stop_loss_ratio=0.08  # 8% 止损
)

# Calculate win_rate if not present
if 'cumulative_win_rate' not in backtest_results.columns:
    backtest_results['cumulative_win_rate'] = backtest_results['pnl'].gt(0).cumsum() / (backtest_results.index + 1)

print(backtest_results.head(10))
print(f"Win Rate: {backtest_results['cumulative_win_rate'].iloc[-1]:.2%}")
print("回测完成！")

# 8. 动态绘制 Iron Condor 执行价和警告线
# 定义绘制区间
start_idx = 0
end_idx = len(processed_data) - 1
plot_slice = processed_data.iloc[start_idx:end_idx+1]

# 动态计算 Iron Condor 价格范围
short_call_prices = []
short_put_prices = []
warn_call_prices = []
warn_put_prices = []

for _, row in plot_slice.iterrows():
    current_price = row['Close']
    predicted_high = current_price * 1.02  # 示例：预测未来一周高点上涨 2%
    predicted_low = current_price * 0.98  # 示例：预测未来一周低点下跌 2%

    # 计算 Iron Condor 的上下执行价
    sell_call, _, sell_put, _ = calculate_iron_condor(predicted_high, predicted_low)
    short_call_prices.append(sell_call)
    short_put_prices.append(sell_put)

    # 计算警告线
    warn_call_prices.append(sell_call - 1)  # 示例：警告线比卖出看涨执行价低 1
    warn_put_prices.append(sell_put + 1)   # 示例：警告线比卖出看跌执行价高 1

# 调用绘图函数
# plot_dynamic_iron_condor(
#     df=processed_data,
#     start_idx=start_idx,
#     end_idx=end_idx,
#     short_call_prices=short_call_prices,
#     short_put_prices=short_put_prices,
#     warn_call_prices=warn_call_prices,
#     warn_put_prices=warn_put_prices,
#     color_call='blue',
#     color_put='blue',
#     color_warn_call='red',
#     color_warn_put='red',
#     label_call='Sell Call (Blue)',
#     label_put='Sell Put (Blue)',
#     label_warn_call='Warn Call (Red)',
#     label_warn_put='Warn Put (Red)'
# )

# 定义 df_price 和 results_df
df_price = processed_data[['Close']]
results_df = backtest_results

#export csv
export_to_csv(results_df, win_rate=None, filename='iron_condor_results.csv')

# plotly可视化
plot_iron_condor(df_price, results_df, title="Iron Condor Strategy Visualization")