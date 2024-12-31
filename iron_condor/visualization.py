import pandas as pd
try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("Plotly is not installed. Please install it using 'pip install plotly'.")

def plot_iron_condor(df_price, df_backtest, title="Iron Condor Backtest Visualization"):
    """
    使用 Plotly 可视化标的价格 & Iron Condor 卖方执行价等。
    :param df_price:     DataFrame，包含至少 'Close' 列，index 为日期
    :param df_backtest:  回测结果 DataFrame，需包含:
                         ['trade_date', 'sell_call_strike', 'sell_put_strike', ...]
    :param title:        图表标题
    """

    print("debug visualization")
    print(df_price.head(10))
    print(df_price.index)
    print(df_price.info())
    print("Columns in df_price:", df_price.columns)

    print("\nBacktest Results:")
    print(df_backtest.head(10))

    # 如果 'Close' 是多级索引，提取正确列
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price = df_price[('Close', 'SPY')]  # 正确引用多级索引列
    else:
        if 'Close' in df_price.columns:
            df_price = df_price['Close']  # 如果是单级索引
        else:
            raise KeyError("The 'Close' column is not found in df_price. Available columns are: {}".format(df_price.columns))

    fig = go.Figure()

    # 1) 标的价格线
    fig.add_trace(go.Scatter(
        x = df_price.index,
        y = df_price,
        mode='lines',
        name='SPY Price',
        line=dict(color='blue')
    ))

    # Add scale bar
    fig.add_trace(go.Scatter(
        x=[df_price.index[0], df_price.index[0]],
        y=[df_price.min(), df_price.min() + (df_price.max() - df_price.min()) * 0.1],
        mode='lines',
        line=dict(color='black', width=2),
        name='Scale Bar',
        showlegend=False
    ))

    # 2) 在回测结果中，每笔交易对应的 trade_date ~ (到期日) 之间画水平线
    for idx, row in df_backtest.iterrows():
        trade_date = row['trade_date']  # 当日
        expiration_date = row['expiration_date']
        sc_strike  = row['sell_call_strike']
        sp_strike  = row['sell_put_strike']
        warning_call = row.get('warning_call', None)
        warning_put = row.get('warning_put', None)

        # 卖方执行价 Iron Condor 上边界（Sell Call）
        fig.add_trace(go.Scatter(
            x = [trade_date, expiration_date],
            y = [sc_strike, sc_strike],
            mode='lines+markers',
            line=dict(color='red', dash='dash'),
            marker=dict(symbol='circle-dot', color='red', size=3),
            name='Sell Call Strike',
            showlegend=False
        ))

        # 卖方执行价 Iron Condor 下边界（Sell Put）
        fig.add_trace(go.Scatter(
            x = [trade_date, expiration_date],
            y = [sp_strike, sp_strike],
            mode='lines+markers',
            marker=dict(symbol='circle-dot', color='red', size=3),
            line=dict(color='green', dash='dash'),
            name='Sell Put Strike',
            showlegend=False
        ))

        # 警告区间（可选）
        if warning_call is not None:
            fig.add_trace(go.Scatter(
                x=[trade_date, expiration_date],
                y=[warning_call, warning_call],
                mode='lines+markers',
                name='Warning Call',
                line=dict(color='orange', dash='dot'),
                showlegend=False
            ))

        if warning_put is not None:
            fig.add_trace(go.Scatter(
                x=[trade_date, expiration_date],
                y=[warning_put, warning_put],
                mode='lines+markers',
                name='Warning Put',
                line=dict(color='orange', dash='dot'),
                showlegend=False
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )

    fig.show()
