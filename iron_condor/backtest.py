import pandas as pd
import numpy as np
from .option_pricing import iron_condor_price

def backtest_iron_condor(
    df,
    strategy_fn,
    stop_loss_ratio=0.08,  # 8% 止损
    r=0.02,                # 假设无风险利率 2%
    sigma=0.20,            # 假设年化隐含波动率 20%
    days_to_expiry=5
):
    """
    简易回测：根据 strategy_fn 给出的 Iron Condor 执行价，在 df 上滚动模拟。

    :param df: 包含真实收盘价等数据的DataFrame(每日记录), 需包含 df['Close']
    :param strategy_fn: 函数, 输入 (df, idx) => (sell_call, buy_call, sell_put, buy_put)
    :param stop_loss_ratio: 设置最大亏损比(如0.08=8%), 简化示例
    :param r: 无风险利率
    :param sigma: 年化隐含波动率(简化为同一个值)
    :param days_to_expiry: 到期日(默认为5交易日)
    :return: 回测记录 DataFrame
    """

    records = []
    total_length = len(df)

    for i in range(total_length - days_to_expiry):
        # 当日价格
        today_price = df['Close'].iloc[i]
        if isinstance(today_price, pd.Series):
            today_price = today_price.iloc[0]

        # 使用策略函数得到行权价
        sell_call_strike, buy_call_strike, sell_put_strike, buy_put_strike = strategy_fn(df, i)
        sell_call_strike = float(sell_call_strike)
        buy_call_strike  = float(buy_call_strike)
        sell_put_strike  = float(sell_put_strike)
        buy_put_strike   = float(buy_put_strike)

        # 计算 Iron Condor 建仓时的净权利金、最大损失、最大收益
        T = days_to_expiry / 252.0  # 假设一年252个交易日
        net_premium, max_loss, max_profit = iron_condor_price(
            S=today_price,
            sell_call=sell_call_strike,
            buy_call=buy_call_strike,
            sell_put=sell_put_strike,
            buy_put=buy_put_strike,
            T=T, r=r, sigma=sigma
        )

        # 取未来 days_to_expiry 天的行情
        future_slice = df['Close'].iloc[i+1 : i+1+days_to_expiry]
        if len(future_slice) < days_to_expiry:
            # 数据不够，不再做回测
            #break
            continue

        final_price = future_slice.iloc[-1]
        if isinstance(final_price, pd.Series):
            final_price = final_price.iloc[0]

        max_price_in_period = future_slice.max()
        min_price_in_period = future_slice.min()
        if isinstance(max_price_in_period, pd.Series):
            max_price_in_period = max_price_in_period.iloc[0]
        if isinstance(min_price_in_period, pd.Series):
            min_price_in_period = min_price_in_period.iloc[0]

        # 根据到期时的价位, 估算盈亏(简化示例)
        call_spread_width = buy_call_strike - sell_call_strike
        put_spread_width  = sell_put_strike - buy_put_strike

        # 1) 若 final_price > buy_call_strike => 亏损 = (call_spread_width - net_premium)
        # 2) 若 final_price < buy_put_strike  => 亏损 = (put_spread_width - net_premium)
        # 3) 否则盈利 = net_premium
        if final_price > buy_call_strike:
            pl_at_expiry = -(call_spread_width - net_premium)
        elif final_price < buy_put_strike:
            pl_at_expiry = -(put_spread_width - net_premium)
        else:
            pl_at_expiry = net_premium

        # 中途止损检查(简化示例)
        stop_loss_triggered = False
        potential_loss = 0.0

        # 上侧止损：如果 max_price_in_period > sell_call_strike => 浮亏估计
        if max_price_in_period > sell_call_strike:
            potential_loss = (max_price_in_period - sell_call_strike) - net_premium
            if potential_loss > stop_loss_ratio * today_price:
                stop_loss_triggered = True

        # 下侧止损：如果 min_price_in_period < sell_put_strike => 浮亏估计
        if (not stop_loss_triggered) and (min_price_in_period < sell_put_strike):
            potential_loss = (sell_put_strike - min_price_in_period) - net_premium
            if potential_loss > stop_loss_ratio * today_price:
                stop_loss_triggered = True

        # 若止损 => 亏损 = potential_loss(若 potential_loss < 0，需做容错)
        if stop_loss_triggered:
            final_pl = -max(potential_loss, 0)
        else:
            final_pl = pl_at_expiry

        # 计算到期日期
        trade_date = df.index[i]
        expiration_date = df.index[i + days_to_expiry]

        # 保存记录
        record = {
            'trade_date': trade_date,
            'expiration_date': expiration_date,
            'sell_call_strike': sell_call_strike,
            'buy_call_strike': buy_call_strike,
            'sell_put_strike': sell_put_strike,
            'buy_put_strike': buy_put_strike,
            'net_premium': net_premium,
            'max_loss': max_loss,
            'max_profit': max_profit,
            'final_price': final_price,
            'stop_loss_triggered': stop_loss_triggered,
            'pnl': final_pl
        }
        records.append(record)

    result_df = pd.DataFrame(records)

    # Debug print to check pnl values
    print(result_df[['trade_date', 'pnl']])

    # Calculate cumulative win rate
    result_df['cumulative_wins'] = result_df['pnl'].gt(0).cumsum()
    result_df['cumulative_trades'] = result_df.index + 1
    result_df['cumulative_win_rate'] = result_df['cumulative_wins'] / result_df['cumulative_trades']

    return result_df
