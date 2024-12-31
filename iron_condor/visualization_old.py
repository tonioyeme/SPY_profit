import matplotlib.pyplot as plt

def plot_dynamic_iron_condor(
    df,
    start_idx,
    end_idx,
    short_call_prices,
    short_put_prices,
    warn_call_prices=None,
    warn_put_prices=None,
    color_call='blue',
    color_put='blue',
    color_warn_call='red',
    color_warn_put='red',
    label_call='Short Call',
    label_put='Short Put',
    label_warn_call='Warning Call',
    label_warn_put='Warning Put'
):
    """
    绘制动态 Iron Condor 价格区间和 SPY 收盘价走势。
    
    :param df:                DataFrame，至少包含 'Close' 列
    :param start_idx:         绘制开始的索引
    :param end_idx:           绘制结束的索引
    :param short_call_prices: 卖出看涨执行价序列
    :param short_put_prices:  卖出看跌执行价序列
    :param warn_call_prices:  警告线(看涨)序列 (可选)
    :param warn_put_prices:   警告线(看跌)序列 (可选)
    """
    # 1. 取绘图区间的数据
    plot_slice = df.iloc[start_idx : end_idx+1]
    dates = plot_slice.index
    close_prices = plot_slice['Close']

    plt.figure(figsize=(12, 6))

    # 2. 绘制收盘价走势
    plt.plot(dates, close_prices, label='SPY Close', color='black')

    # 3. 绘制动态 Iron Condor 卖出看涨/看跌线
    plt.plot(dates, short_call_prices, color=color_call, linestyle='--', label=label_call)
    plt.plot(dates, short_put_prices, color=color_put, linestyle='--', label=label_put)

    # 4. 如果提供了警告线，绘制它们
    if warn_call_prices is not None:
        plt.plot(dates, warn_call_prices, color=color_warn_call, linestyle=':', label=label_warn_call)
    if warn_put_prices is not None:
        plt.plot(dates, warn_put_prices, color=color_warn_put, linestyle=':', label=label_warn_put)

    # 5. 设置图表样式
    plt.title("SPY Price with Dynamic Iron Condor Lines")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
