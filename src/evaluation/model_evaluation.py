import numpy as np
from scipy.stats import norm

def calculate_implied_volatility(market_price, S, K, T, r, option_type='call'):
    def option_price(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def objective(sigma):
        return abs(option_price(sigma) - market_price)

    result = minimize_scalar(objective, bounds=(0.01, 5), method='bounded')
    return result.x

def calculate_mse(predicted_prices, actual_prices):
    return np.mean((predicted_prices - actual_prices) ** 2)

def calculate_mae(predicted_prices, actual_prices):
    return np.mean(np.abs(predicted_prices - actual_prices))

def calculate_mape(predicted_prices, actual_prices):
    return np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100