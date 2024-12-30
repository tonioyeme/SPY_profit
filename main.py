from iron_condor import load_data, train_model, calculate_iron_condor
from iron_condor.preprocessing import preprocess_data

# 加载数据
data = load_data()

# 数据预处理
processed_data = preprocess_data(data)

# 选择特征和目标值
features = ['SMA_5', 'SMA_20', 'RSI_14', 'Boll_Width', 'Return_1d', 'Return_5d']
X = processed_data[features]
y_high = processed_data['Highest_Next_Week']
y_low = processed_data['Lowest_Next_Week']

# 训练模型 (这里只预测最高价，可扩展预测最低价)
model_high, metrics_high = train_model(X, y_high)

# 显示模型评估结果
print(f"Model High Prediction - MAE: {metrics_high['mae']:.2f}, RMSE: {metrics_high['rmse']:.2f}")

# 预测
latest_features = X.iloc[-1:].copy()
predicted_high = model_high.predict(latest_features)[0]
predicted_low = predicted_high * 0.98  # 假设，实际应训练低价模型

# 计算 Iron Condor 策略
iron_condor = calculate_iron_condor(predicted_high, predicted_low)
print(f"Iron Condor Strategy: {iron_condor}")
