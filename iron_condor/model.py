from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_model(X, y, test_ratio=0.2, random_state=42):
    """
    训练随机森林模型
    :param X: 特征矩阵
    :param y: 目标值
    :param test_ratio: 测试集占比
    :param random_state: 随机种子
    :return: 模型, 测试集预测结果, 评价指标
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, {'mae': mae, 'rmse': rmse}
