import numpy as np
import pandas as pd
from src.models.black_scholes import BlackScholes
from src.models.heston_model import HestonModel
from src.simulation.monte_carlo import MonteCarloSimulation
from src.utils.data_utils import fetch_stock_data, calculate_returns, calculate_historical_volatility, calibrate_heston_model, get_option_data
from src.evaluation.model_evaluation import calculate_implied_volatility, calculate_mse, calculate_mae, calculate_mape

def main():
    # Fetch Tesla stock data
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    tesla_data = fetch_stock_data('TSLA', start_date, end_date)
    
    # Calculate returns and historical volatility
    returns = calculate_returns(tesla_data['Close'])
    hist_volatility = calculate_historical_volatility(returns)
    
    # Set parameters
    S0 = tesla_data['Close'].iloc[-1]  # Current stock price (TSLA)
    K = S0  # At-the-money strike price
    T = 1  # Time to maturity (in years)
    r = 0.05  # Risk-free rate (you may want to use actual treasury yields)
    sigma = hist_volatility.iloc[-1]  # Use the most recent historical volatility
    t_choose = 0.5  # Time to choose for the chooser option

    # Calibrate Heston model
    initial_params = [2, 0.04, 0.3, -0.7, hist_volatility.iloc[-1]**2]
    kappa, theta, sigma_v, rho, V0 = calibrate_heston_model(returns, initial_params)

    # Simulation parameters
    num_simulations = 100000
    num_steps = 252  # Daily steps for a year

    # Black-Scholes model
    bs_model = BlackScholes(S0, K, T, r, sigma)
    bs_chooser_price = bs_model.chooser_option_price(t_choose)
    print(f"Black-Scholes Chooser Option Price: {bs_chooser_price:.4f}")

    # Monte Carlo simulation with Black-Scholes model
    mc_bs = MonteCarloSimulation(bs_model, num_simulations, num_steps)
    mc_bs_price = mc_bs.price_chooser_option(K, t_choose)
    print(f"Monte Carlo Black-Scholes Chooser Option Price: {mc_bs_price:.4f}")

    # Heston model
    heston_model = HestonModel(S0, V0, kappa, theta, sigma_v, rho, r, T)
    
    # Monte Carlo simulation with Heston model
    mc_heston = MonteCarloSimulation(heston_model, num_simulations, num_steps)
    mc_heston_price = mc_heston.price_chooser_option(K, t_choose)
    print(f"Monte Carlo Heston Chooser Option Price: {mc_heston_price:.4f}")

    # Fetch actual option data (note: chooser options are not commonly traded, so we'll use regular options for comparison)
    calls, puts = get_option_data('TSLA', tesla_data.index[-1].strftime('%Y-%m-%d'))
    
    # Find the closest strike price to our K
    closest_call = calls.iloc[(calls['strike'] - K).abs().argsort()[:1]]
    closest_put = puts.iloc[(puts['strike'] - K).abs().argsort()[:1]]

    # Calculate implied volatilities
    call_iv = calculate_implied_volatility(closest_call['lastPrice'].values[0], S0, closest_call['strike'].values[0], T, r, 'call')
    put_iv = calculate_implied_volatility(closest_put['lastPrice'].values[0], S0, closest_put['strike'].values[0], T, r, 'put')

    print(f"Implied Volatility (Call): {call_iv:.4f}")
    print(f"Implied Volatility (Put): {put_iv:.4f}")

    # Compare model prices with market prices
    market_call_price = closest_call['lastPrice'].values[0]
    market_put_price = closest_put['lastPrice'].values[0]

    bs_call_price = bs_model.call_price()
    bs_put_price = bs_model.put_price()

    heston_call_price = heston_model.price_european_option(K, 'call', num_simulations, num_steps)
    heston_put_price = heston_model.price_european_option(K, 'put', num_simulations, num_steps)

    # Calculate error metrics
    models = ['Black-Scholes', 'Heston']
    call_prices = [bs_call_price, heston_call_price]
    put_prices = [bs_put_price, heston_put_price]

    for model, call_price, put_price in zip(models, call_prices, put_prices):
        print(f"\n{model} Model Evaluation:")
        print(f"Call Price - Model: {call_price:.4f}, Market: {market_call_price:.4f}")
        print(f"Put Price - Model: {put_price:.4f}, Market: {market_put_price:.4f}")
        
        mse = calculate_mse(np.array([call_price, put_price]), np.array([market_call_price, market_put_price]))
        mae = calculate_mae(np.array([call_price, put_price]), np.array([market_call_price, market_put_price]))
        mape = calculate_mape(np.array([call_price, put_price]), np.array([market_call_price, market_put_price]))
        
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main()