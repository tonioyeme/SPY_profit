from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

def check_stop_loss(
    current_price: float,
    position: Dict[str, float],
    stop_loss_ratio: float = 0.08,  # Default 8% max loss
    warning_threshold: float = 0.05  # Default 5% warning threshold
) -> Tuple[bool, Optional[str]]:
    """
    Check if stop loss should be triggered for an Iron Condor position
    
    Args:
        current_price: Current price of the underlying
        position: Dictionary containing strike prices:
            - sell_call_strike
            - buy_call_strike 
            - sell_put_strike
            - buy_put_strike
            - entry_credit (initial credit received)
        stop_loss_ratio: Maximum loss tolerance as ratio of max possible loss
        warning_threshold: Threshold for warning before stop loss
        
    Returns:
        Tuple of (should_stop_loss, warning_message)
    """
    # Extract position details
    sell_call = position['sell_call_strike']
    buy_call = position['buy_call_strike']
    sell_put = position['sell_put_strike'] 
    buy_put = position['buy_put_strike']
    entry_credit = position.get('entry_credit', 0)
    
    # Calculate max possible loss (width of the widest spread - credit received)
    call_spread_width = buy_call - sell_call
    put_spread_width = sell_put - buy_put
    max_loss = max(call_spread_width, put_spread_width) - entry_credit
    
    # Calculate current theoretical loss
    current_loss = 0
    
    if current_price >= sell_call:
        # Price above short call - losing money on call spread
        loss = min(current_price - sell_call, call_spread_width)
        current_loss = loss - entry_credit
    elif current_price <= sell_put:
        # Price below short put - losing money on put spread
        loss = min(sell_put - current_price, put_spread_width)
        current_loss = loss - entry_credit
        
    # Check stop loss ratio
    loss_ratio = abs(current_loss) / max_loss if max_loss else 0
    
    # Generate warning message if approaching stop loss
    warning_msg = None
    if loss_ratio >= warning_threshold and loss_ratio < stop_loss_ratio:
        warning_msg = f"Warning: Position loss ratio {loss_ratio:.1%} approaching stop loss threshold {stop_loss_ratio:.1%}"
        
    return loss_ratio >= stop_loss_ratio, warning_msg

def calculate_position_greeks(
    current_price: float,
    position: Dict[str, float],
    days_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate aggregate Greeks for the Iron Condor position
    
    Args:
        current_price: Current price of underlying
        position: Dictionary with strike prices
        days_to_expiry: Days until expiration
        volatility: Current implied volatility 
        risk_free_rate: Risk-free interest rate
        
    Returns:
        Dictionary with aggregate Greek values:
        - delta
        - gamma 
        - theta
        - vega
    """
    # Implementation of Greeks calculation
    # This would use Black-Scholes or similar model to calculate option Greeks
    # Then combine them for all legs of the Iron Condor
    # For now returning dummy values
    return {
        'delta': 0.0,
        'gamma': 0.0,
        'theta': 0.0,
        'vega': 0.0
    }

def check_position_limits(
    portfolio: pd.DataFrame,
    new_position: Dict[str, float],
    max_positions: int = 10,
    max_gamma: float = 100,
    max_vega: float = 1000
) -> Tuple[bool, Optional[str]]:
    """
    Check if adding new position would exceed portfolio risk limits
    
    Args:
        portfolio: DataFrame of existing positions
        new_position: Dictionary with new position details
        max_positions: Maximum number of concurrent positions
        max_gamma: Maximum absolute gamma exposure
        max_vega: Maximum absolute vega exposure
        
    Returns:
        Tuple of (position_allowed, reason_if_not_allowed)
    """
    # Check number of positions
    if len(portfolio) >= max_positions:
        return False, f"Maximum positions ({max_positions}) reached"
        
    # Check gamma risk
    # total_gamma = portfolio['gamma'].sum() + new_position_gamma
    # if abs(total_gamma) > max_gamma:
    #     return False, f"Total gamma exposure would exceed limit"
        
    # Check vega risk
    # total_vega = portfolio['vega'].sum() + new_position_vega
    # if abs(total_vega) > max_vega:
    #     return False, f"Total vega exposure would exceed limit"
        
    return True, None

def adjust_position(
    current_price: float,
    position: Dict[str, float],
    days_to_expiry: float,
    adjustment_threshold: float = 0.2  # 20% ITM
) -> Optional[Dict[str, float]]:
    """
    Determine if and how position should be adjusted
    
    Args:
        current_price: Current price of underlying
        position: Dictionary with current position details
        days_to_expiry: Days until expiration
        adjustment_threshold: How far ITM before adjusting
        
    Returns:
        Dictionary with adjustment instructions or None if no adjustment needed
    """
    sell_call = position['sell_call_strike']
    sell_put = position['sell_put_strike']
    
    # Check if short strikes are threatened
    call_itm_pct = (current_price - sell_call) / sell_call if current_price > sell_call else 0
    put_itm_pct = (sell_put - current_price) / sell_put if current_price < sell_put else 0
    
    if max(call_itm_pct, put_itm_pct) >= adjustment_threshold:
        return {
            'action': 'roll',
            'side': 'call' if call_itm_pct > put_itm_pct else 'put',
            'suggested_adjustment': 'Roll threatened side up/down by 2-3 strikes'
        }
        
    return None
