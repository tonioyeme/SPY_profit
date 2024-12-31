def check_stop_loss(current_loss, capital, stop_loss_ratio=0.08):
    """
    检查是否触发止损
    :param current_loss: 当前浮亏(>0 表示亏损额度)
    :param capital: 该策略/组合的初始资金或总资金
    :param stop_loss_ratio: 止损比例
    :return: 是否触发止损 (bool)
    """
    if current_loss >= stop_loss_ratio * capital:
        return True
    return False
