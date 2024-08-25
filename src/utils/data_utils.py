import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def calculate_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def calculate_historical_volatility(returns, window=30):
    return returns.rolling(window=window).std() * np.sqrt(252)

def calibrate_heston_model(returns, initial_params):
    def objective(params):
        kappa, theta, sigma, rho, v0 = params
        model_var = v0 * np.exp(-kappa * np.arange(len(returns))) + theta * (1 - np.exp(-kappa * np.arange(len(returns))))
        return np.sum((returns**2 - model_var)**2)

    result = minimize(objective, initial_params, method='L-BFGS-B', 
                      bounds=((0, 10), (0, 1), (0, 2), (-1, 1), (0, 1)))
    return result.x

def get_option_data(ticker, date):
    stock = yf.Ticker(ticker)
    options = stock.option_chain(date)
    return options.calls, options.puts