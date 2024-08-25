import numpy as np
import pandas as pd
from src.utils.data_utils import fetch_stock_data, calculate_returns, calculate_historical_volatility, get_option_data, check_put_call_parity, recalibrate_models, LocalVolatilityBS
from src.evaluation.model_evaluation import calculate_implied_volatility, calculate_mse, calculate_mae, calculate_mape
from src.utils.visualization_utils import plot_stock_price_and_volatility, plot_option_prices_comparison
from src.models.black_scholes import BlackScholes
from src.models.heston_model import HestonModel
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def price_option(args):
    K, market_call, market_put, S0, r, T, bs_model, heston_model = args
    
    bs = bs_model(K)  # Get Black-Scholes model with local volatility
    bs_call_price = bs.call_price()
    bs_put_price = bs.put_price()
    
    heston_call_price = heston_model.price_european_option(K, 'call', 5000, 100)
    heston_put_price = heston_model.price_european_option(K, 'put', 5000, 100)
    
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

def main():
    print("Fetching stock data...")
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    tesla_data = fetch_stock_data('TSLA', start_date, end_date)
    
    print("Calculating returns and volatility...")
    returns = calculate_returns(tesla_data['Close'])
    hist_volatility = calculate_historical_volatility(returns)
    
    plot_stock_price_and_volatility(tesla_data, hist_volatility)
    
    S0 = tesla_data['Close'].iloc[-1]
    r = 0.05  # Risk-free rate (you may want to fetch this dynamically)
    print(f"Using risk-free rate: {r:.4f}")

    print("Fetching option data...")
    current_date = tesla_data.index[-1].strftime('%Y-%m-%d')
    print(f"\nFetching option data for date: {current_date}")
    calls, puts, T = get_option_data('TSLA', current_date)
    
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
    
    print("Recalibrating models...")
    try:
        bs_model, heston_model = recalibrate_models(results_df, S0, r, T)
    except Exception as e:
        print(f"Error during model recalibration: {e}")
        print("Falling back to default model parameters.")
        default_vol = 0.3
        bs_model = LocalVolatilityBS(S0, T, r, results_df['Strike'].values, [default_vol] * len(results_df))
        heston_model = HestonModel(S0, 0.04, 2, 0.04, 0.3, -0.7, r, T)  # Using default Heston parameters

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