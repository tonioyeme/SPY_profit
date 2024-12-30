def calculate_iron_condor(predicted_high, predicted_low, safety_gap=2.0):
    """
    根据预测值计算 Iron Condor 的执行价
    :param predicted_high: 预测的最高价
    :param predicted_low: 预测的最低价
    :param safety_gap: 安全边界
    :return: 执行价配置
    """
    sell_call_strike = predicted_high + safety_gap
    sell_put_strike = predicted_low - safety_gap
    buy_call_strike = sell_call_strike + 2
    buy_put_strike = sell_put_strike - 2

    return {
        "sell_call": sell_call_strike,
        "sell_put": sell_put_strike,
        "buy_call": buy_call_strike,
        "buy_put": buy_put_strike
    }
