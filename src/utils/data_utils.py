import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
from scipy.optimize import minimize
from src.models.heston_model import HestonModel
from src.models.black_scholes import BlackScholes
from src.evaluation.model_evaluation import calculate_implied_volatility 
from scipy.optimize import differential_evolution

class LocalVolatilityBS:
    def __init__(self, S0, T, r, strikes, local_vols):
        self.S0 = S0
        self.T = T
        self.r = r
        self.strikes = strikes
        self.local_vols = local_vols

    def __call__(self, K):
        vol = np.interp(K, self.strikes, self.local_vols)
        return BlackScholes(self.S0, K, self.T, self.r, vol)

def local_volatility_bs(market_data, S0, r, T):
    def objective(sigma, K, market_price):
        bs = BlackScholes(S0, K, T, r, sigma)
        model_price = bs.call_price()
        return (model_price - market_price)**2

    local_vols = []
    for _, row in market_data.iterrows():
        K = row['Strike']
        market_price = row['Market Call']
        result = minimize(objective, x0=[0.3], args=(K, market_price), bounds=[(0.01, 2)])
        local_vols.append(result.x[0])
    
    return LocalVolatilityBS(S0, T, r, market_data['Strike'].values, local_vols)

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def calculate_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def calculate_historical_volatility(returns, window=30):
    return returns.rolling(window=window).std() * np.sqrt(252)

def calibrate_heston_model(returns, S0, V0, r, T):
    def objective(params):
        kappa, theta, sigma, rho = params
        model = HestonModel(S0, V0, kappa, theta, sigma, rho, r, T)
        simulated_paths = model.generate_paths(len(returns), 1)[0]
        simulated_returns = np.diff(np.log(simulated_paths))
        return np.sum((returns.values - simulated_returns.flatten())**2)

    bounds = [(0.1, 10), (0.01, 1), (0.01, 1), (-0.99, 0.99)]
    result = minimize(objective, [1.5, 0.04, 0.3, -0.7], bounds=bounds, method='L-BFGS-B')
    return result.x

def get_option_data(ticker, date):
    stock = yf.Ticker(ticker)
    
    expirations = stock.options
    target_date = datetime.strptime(date, '%Y-%m-%d')
    nearest_expiration = min(expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))

    print(f"Using option expiration date: {nearest_expiration}")

    options = stock.option_chain(nearest_expiration)
    
    calls = options.calls
    puts = options.puts
    
    expiration_date = datetime.strptime(nearest_expiration, '%Y-%m-%d')
    T = (expiration_date - target_date).days / 365.0
    
    return calls, puts, T

def check_put_call_parity(S, K, r, T, call_price, put_price):
    left_side = call_price - put_price
    right_side = S - K * np.exp(-r * T)
    return np.abs(left_side - right_side)

def heston_price_vector(params, S0, K, r, T, option_type):
    kappa, theta, sigma, rho, v0 = params
    heston = HestonModel(S0, v0, kappa, theta, sigma, rho, r, T)
    return np.array([heston.price_european_option(k, option_type, 5000, 100) for k in K])

def recalibrate_models(market_data, S0, r, T):
    print("Calibrating Local Volatility Black-Scholes model...")
    bs_model = local_volatility_bs(market_data, S0, r, T)
    
    print("Calibrating Heston model...")
    def heston_objective(params):
        model_prices = heston_price_vector(params, S0, market_data['Strike'].values, r, T, 'call')
        weights = np.exp(-(market_data['Strike'] - S0)**2 / (2 * S0**2))  # Give more weight to ATM options
        return np.sum(weights * (model_prices - market_data['Market Call'])**2)
    
    bounds = [(0.1, 10), (0.01, 1), (0.01, 1), (-0.99, 0.99), (0.01, 1)]
    
    pbar = tqdm(total=100, desc="Heston Calibration")
    last_update = [0]
    start_time = time.time()
    max_time = 300 

    def update_progress(xk, convergence):
        nonlocal start_time
        current = int(100 * (1 - convergence))
        if current > last_update[0]:
            pbar.update(current - last_update[0])
            last_update[0] = current
        
        if time.time() - start_time > max_time:
            return True 

    result = differential_evolution(heston_objective, bounds, maxiter=50, popsize=15, 
                                    callback=update_progress, polish=False, 
                                    updating='immediate', workers=-1)
    pbar.close()
    
    if not result.success:
        print("Warning: Heston calibration did not converge. Using best found solution.")
    
    heston_model = HestonModel(S0, result.x[4], result.x[0], result.x[1], result.x[2], result.x[3], r, T)
    
    print(f"Heston calibration completed. Final error: {result.fun:.6f}")
    print(f"Optimal parameters: kappa={result.x[0]:.4f}, theta={result.x[1]:.4f}, sigma={result.x[2]:.4f}, rho={result.x[3]:.4f}, v0={result.x[4]:.4f}")
    
    return bs_model, heston_model