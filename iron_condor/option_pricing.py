import math
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum

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

    max_loss = max(call_spread_width, put_spread_width) - net_premium_received

    return net_premium_received, max_loss, max_profit



class OptionType(Enum):
    CALL = "call"
    PUT = "put"

@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class VolatilityModel:
    """波动率模型基类"""
    def __init__(self):
        self.parameters = {}
        
    def fit(self, strikes: np.ndarray, market_prices: np.ndarray, 
            spot: float, rate: float, time: float):
        raise NotImplementedError
        
    def get_vol(self, strike: float, time: float) -> float:
        raise NotImplementedError

class SVIModel(VolatilityModel):
    """
    随机波动率模型 (Stochastic Volatility Inspired)
    用于拟合波动率曲面
    """
    def __init__(self):
        super().__init__()
        self.parameters = {
            'a': 0,  # 整体水平
            'b': 0,  # 斜率
            'rho': 0,  # 相关性
            'm': 0,  # 最小点位置
            'sigma': 0  # 曲率
        }
    
    def _w(self, k: float) -> float:
        """SVI方差函数"""
        a, b, rho, m, sigma = self.parameters.values()
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
    
    def fit(self, strikes: np.ndarray, market_prices: np.ndarray, 
            spot: float, rate: float, time: float):
        """拟合SVI参数"""
        log_strikes = np.log(strikes / spot)
        implied_vars = np.square(market_prices) * time
        
        def objective(params):
            self.parameters = {
                'a': params[0], 'b': params[1], 'rho': params[2],
                'm': params[3], 'sigma': params[4]
            }
            model_vars = np.array([self._w(k) for k in log_strikes])
            return np.sum(np.square(model_vars - implied_vars))
        
        # 添加约束确保模型合理性
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[1] - 0},  # b > 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[2]**2},  # |ρ| < 1
            {'type': 'ineq', 'fun': lambda x: x[4] - 0}  # σ > 0
        ]
        
        result = minimize(
            objective,
            x0=[0, 0.1, 0, 0, 0.1],
            constraints=constraints,
            method='SLSQP'
        )
        
        if not result.success:
            raise ValueError("SVI拟合失败")
            
        self.parameters = {
            'a': result.x[0], 'b': result.x[1], 'rho': result.x[2],
            'm': result.x[3], 'sigma': result.x[4]
        }
    
    def get_vol(self, strike: float, time: float) -> float:
        """获取特定执行价和到期时间的波动率"""
        w = self._w(np.log(strike))
        return np.sqrt(w / time)

class HestonModel:
    """
    Heston随机波动率模型
    dS = rSdt + √vS dW₁
    dv = κ(θ-v)dt + σ√v dW₂
    dW₁dW₂ = ρdt
    """
    def __init__(self, kappa: float, theta: float, sigma: float, 
                 rho: float, v0: float):
        self.kappa = kappa  # 均值回归速度
        self.theta = theta  # 长期波动率
        self.sigma = sigma  # 波动率的波动率
        self.rho = rho     # 价格-波动率相关性
        self.v0 = v0       # 初始波动率
        
    def characteristic_function(self, u: complex, T: float, S: float, 
                             r: float) -> complex:
        """特征函数"""
        kappa, theta, sigma, rho, v0 = self.kappa, self.theta, self.sigma, self.rho, self.v0
        
        d = np.sqrt((rho * sigma * u * 1j - kappa)**2 + sigma**2 * (u * 1j + u**2))
        g = (kappa - rho * sigma * u * 1j - d) / (kappa - rho * sigma * u * 1j + d)
        
        C = (r * u * 1j * T + kappa * theta / sigma**2 * 
             ((kappa - rho * sigma * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T))/(1 - g))))
        
        D = ((kappa - rho * sigma * u * 1j - d) * (1 - np.exp(-d * T))) / (sigma**2 * (1 - g * np.exp(-d * T)))
        
        return np.exp(C + D * v0 + 1j * u * np.log(S))
    
    def price_european(self, S: float, K: float, T: float, r: float, 
                      option_type: OptionType, N: int = 100) -> float:
        """使用特征函数方法定价欧式期权"""
        def integrand(u: float, x: float) -> complex:
            if option_type == OptionType.CALL:
                return (np.exp(-1j * u * np.log(K)) * 
                       self.characteristic_function(u - 1j/2, T, S, r) / 
                       (1j * u * self.characteristic_function(-1j/2, T, S, r)))
            else:
                return (np.exp(-1j * u * np.log(K)) * 
                       self.characteristic_function(u - 1j/2, T, S, r) / 
                       (1j * u * self.characteristic_function(-1j/2, T, S, r)))
        
        # 数值积分
        du = 0.1
        u = np.arange(0, N) * du
        sum_real = np.sum(np.real(integrand(u, np.log(S))))
        
        price = S * np.exp(-r * T) * (0.5 + sum_real * du / np.pi)
        
        if option_type == OptionType.PUT:
            # 使用put-call平价
            price = price - S + K * np.exp(-r * T)
            
        return max(0, price)

