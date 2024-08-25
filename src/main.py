import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.models.black_scholes import BlackScholes, calibrate_local_volatility
from src.models.heston_model import calibrate_heston
from src.utils.visualization_utils import plot_stock_price_and_volatility, plot_option_prices_comparison, plot_chooser_option_comparison
from src.evaluation.model_evaluation import calculate_implied_volatility, calculate_mse, calculate_mae, calculate_mape
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def calculate_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def calculate_historical_volatility(returns, window=30):
    return returns.rolling(window=window).std() * np.sqrt(252)

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

def price_option(args):
    K, market_call, market_put, S0, r, T, bs_model, heston_model = args
    
    bs = bs_model(K)
    bs_call_price = bs.call_price()
    bs_put_price = bs.put_price()
    
    heston_call_price = heston_model.price_european_option(K, 'call')
    heston_put_price = heston_model.price_european_option(K, 'put')
    
    call_iv = calculate_implied_volatility(market_call, S0, K, T, r, 'call')
    put_iv = calculate_implied_volatility(market_put, S0, K, T, r, 'put')
    
    parity_violation = check_put_call_parity(S0, K, r, T, market_call, market_put)
    
    return {
        'Strike': K,
        'Market Call': market_call,
        'Market Put': market_put,
        'BS Call': bs_call_price,
        'BS Put': bs_put_price,
        'Heston Call': heston_call_price,
        'Heston Put': heston_put_price,
        'Call IV': call_iv,
        'Put IV': put_iv,
        'Parity Violation': parity_violation
    }

def simulate_real_world_chooser(S0, K, r, T, t_choose, call_iv, put_iv):
    bs_call = BlackScholes(S0, K, T, r, call_iv)
    bs_put = BlackScholes(S0, K, T, r, put_iv)
    
    call_value = bs_call.call_price()
    put_value = bs_put.put_price()
    
    chooser_value = np.exp(-r * t_choose) * np.maximum(call_value, put_value)
    return chooser_value

def get_risk_free_rate():
    try:
        tnx = yf.Ticker('^TNX')
        history = tnx.history(period="1d")
        if not history.empty:
            return history['Close'].iloc[-1] / 100
        else:
            print("Warning: Could not fetch current Treasury yield. Using default value.")
            return 0.05
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}. Using default value.")
        return 0.05

def main():
    print("Fetching stock data...")
    ticker = 'TSLA'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    tesla_data = fetch_stock_data(ticker, start_date, end_date)
    
    print("Calculating returns and volatility...")
    returns = calculate_returns(tesla_data['Close'])
    hist_volatility = calculate_historical_volatility(returns)
    
    plot_stock_price_and_volatility(tesla_data, hist_volatility)
    
    S0 = tesla_data['Close'].iloc[-1]
    r = get_risk_free_rate()  # Use the new function to get the risk-free rate
    print(f"Current stock price: {S0:.2f}")
    print(f"Using risk-free rate: {r:.4f}")
    print("Fetching option data...")
    current_date = tesla_data.index[-1].strftime('%Y-%m-%d')
    calls, puts, T = get_option_data(ticker, current_date)
    
    strike_range = calls[(calls['strike'] >= 0.8*S0) & (calls['strike'] <= 1.2*S0)]
    
    results = []
    for _, option in strike_range.iterrows():
        K = option['strike']
        market_call_price = option['lastPrice']
        market_put_price = puts[puts['strike'] == K]['lastPrice'].values[0]
        
        results.append({
            'Strike': K,
            'Market Call': market_call_price,
            'Market Put': market_put_price,
        })
    
    results_df = pd.DataFrame(results)
    
    logger.info("Calibrating models...")
    logger.info("Calibrating Black-Scholes model...")
    bs_model = calibrate_local_volatility(results_df, S0, r, T)
    
    logger.info("Calibrating Heston model...")
    heston_model = calibrate_heston(results_df, S0, r, T, max_time=300)  # 5 minutes max

    args_list = [(row['Strike'], row['Market Call'], row['Market Put'], S0, r, T, bs_model, heston_model) 
                 for _, row in results_df.iterrows()]
    
    print("Pricing options...")
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(tqdm(executor.map(price_option, args_list), total=len(args_list), desc="Pricing Options"))
    
    results_df = pd.DataFrame(results)
    
    print("\nResults:")
    print(results_df)
    
    print("Plotting option prices comparison...")
    plot_option_prices_comparison(results_df)
    
    print("Pricing chooser options...")
    t_choose = T / 2  # Assume choice time is halfway to expiration
    chooser_results = []
    
    for _, row in results_df.iterrows():
        K = row['Strike']
        
        bs_chooser = bs_model(K).chooser_option_price(t_choose)
        heston_chooser = heston_model.chooser_option_price(t_choose, K)
        
        real_world_chooser = simulate_real_world_chooser(S0, K, r, T, t_choose, row['Call IV'], row['Put IV'])
        
        chooser_results.append({
            'Strike': K,
            'BS Chooser': bs_chooser,
            'Heston Chooser': heston_chooser,
            'Real-World Chooser': real_world_chooser
        })
    
    chooser_df = pd.DataFrame(chooser_results)
    
    print("\nChooser Option Results:")
    print(chooser_df)
    
    print("Plotting chooser option comparison...")
    plot_chooser_option_comparison(chooser_df)

    
    print("Calculating error metrics...")
    for model in ['BS', 'Heston']:
        call_mse = calculate_mse(results_df[f'{model} Call'], results_df['Market Call'])
        put_mse = calculate_mse(results_df[f'{model} Put'], results_df['Market Put'])
        call_mae = calculate_mae(results_df[f'{model} Call'], results_df['Market Call'])
        put_mae = calculate_mae(results_df[f'{model} Put'], results_df['Market Put'])
        
        print(f"\n{model} Model Evaluation:")
        print(f"Call MSE: {call_mse:.6f}")
        print(f"Put MSE: {put_mse:.6f}")
        print(f"Call MAE: {call_mae:.6f}")
        print(f"Put MAE: {put_mae:.6f}")

    print("\nProgram completed successfully!")

if __name__ == "__main__":
    main()