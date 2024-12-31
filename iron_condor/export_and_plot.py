# export_and_plot.py
import pandas as pd
import os

def export_to_csv(results_df, win_rate, filename='iron_condor_results.csv'):
    """
    将回测/策略计算结果输出为 CSV:
    需要包含：
      - Iron Condor 上下边缘
      - 警告区间上下边缘
      - 减仓日期
      - 到期日期
      - 天数
      - 最大亏损
      - 最大盈利
      - 止损
      - 平仓日期
      - 持仓天数
      - 收益(PnL)
      - 胜率
    :param results_df: 包含上述字段的 DataFrame
    :param win_rate: 胜率
    :param filename: 输出的 CSV 文件名
    """
    # 确保 DataFrame 至少包含必须的列（可以自行完善检查）
    required_cols = [
        'sell_call_strike', 'sell_put_strike',
        'warning_call', 'warning_put',
        'reduce_date', 'expiration_date',
        'days_to_expiry', 'max_loss', 'max_profit',
        'stop_loss_triggered', 'close_date',
        'holding_days', 'pnl', 'cumulative win_rate'
    ]
    # 简单检查
    missing_cols = [c for c in required_cols if c not in results_df.columns]
    if missing_cols:
        print(f"[Warning] Missing columns in results_df: {missing_cols}. Please ensure all columns exist.")

    results_df.to_csv(filename, index=False)
    print(f"CSV Exported: {os.path.abspath(filename)}")