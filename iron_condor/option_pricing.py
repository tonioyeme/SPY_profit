import math
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm


# 根据卖出看涨、买入看涨、卖出看跌、买入看跌的行权价，计算 Iron Condor 建仓时的净权利金与风险敞口
# 注意：
# 真实行情需使用期权市场实际报价(Bid/Ask) 或者 期权希腊值 + IV 来定价；
# 下面仅示例公式，可根据需要替换为更准确的市场数据。

def black_scholes_call(S, K, T, r, sigma):
    """
    简化版Black-Scholes公式 - 看涨期权定价
    :param S: 标的现价
    :param K: 执行价
    :param T: 距到期时间(按年计)
    :param r: 无风险利率
    :param sigma: 波动率(年化)
    :return: call期权理论价格
    """
    S = float(S)
    K = float(K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    简化版Black-Scholes公式 - 看跌期权定价
    """
    S = float(S)
    K = float(K)
    d1 = (math.log(S / K) + (r + 0.5*sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    put_price = K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return put_price

def iron_condor_price(
    S, # 当前标的价格
    sell_call, buy_call,
    sell_put, buy_put,
    T, r, sigma
):
    """
    计算Iron Condor策略建仓时的净权利金(理论价值)
    :param S: 当前标的现价
    :param sell_call: 卖出看涨期权的执行价
    :param buy_call:  买入看涨期权的执行价
    :param sell_put:  卖出看跌期权的执行价
    :param buy_put:   买入看跌期权的执行价
    :param T: 距到期时间(按年计)
    :param r: 无风险利率
    :param sigma: 年化隐含波动率(假设同IV)
    :return: (净权利金, 最大风险, 最大收益)
    """
    # 计算各腿期权价格
    call_sell_price = black_scholes_call(S, sell_call, T, r, sigma)
    call_buy_price  = black_scholes_call(S, buy_call, T, r, sigma)
    put_sell_price  = black_scholes_put(S, sell_put, T, r, sigma)
    put_buy_price   = black_scholes_put(S, buy_put, T, r, sigma)

    # Iron Condor: 卖出看涨 + 卖出看跌 - (买入看涨 + 买入看跌)
    net_premium_received = (call_sell_price + put_sell_price) - (call_buy_price + put_buy_price)
    net_premium_received = float(net_premium_received)

    # 最大收益：即净收取的权利金
    max_profit = net_premium_received

    # 风险敞口:
    # Iron Condor最大损失 = 差距中较大的腿 - 净权利金 (简化场景：卖Put到买Put行权差 or 卖Call到买Call行权差)
    # call spread 宽度 = buy_call - sell_call
    # put spread 宽度  = sell_put - buy_put
    call_spread_width = float(buy_call) - float(sell_call)
    put_spread_width  = float(sell_put) - float(buy_put)

    max_loss = min(call_spread_width, put_spread_width) - net_premium_received

    return net_premium_received, max_loss, max_profit
