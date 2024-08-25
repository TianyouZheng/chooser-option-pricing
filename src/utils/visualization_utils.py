import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

def plot_stock_price_and_volatility(data, volatility):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot stock price
    ax1.plot(data.index, data['Close'], label='Close Price')
    ax1.set_title('TSLA Stock Price')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # Plot historical volatility
    ax2.plot(volatility.index, volatility, label='Historical Volatility')
    ax2.set_title('TSLA Historical Volatility')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('stock_price_and_volatility.png')
    plt.close()

def plot_option_prices_comparison(models, call_prices, put_prices, market_call_price, market_put_price):
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, call_prices, width, label='Model Call Price')
    rects2 = ax.bar(x + width/2, put_prices, width, label='Model Put Price')

    ax.axhline(y=market_call_price, color='r', linestyle='--', label='Market Call Price')
    ax.axhline(y=market_put_price, color='g', linestyle='--', label='Market Put Price')

    ax.set_ylabel('Option Price')
    ax.set_title('Comparison of Model and Market Option Prices')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.savefig('option_prices_comparison.png')
    plt.close()

def plot_monte_carlo_paths(paths, title):
    plt.figure(figsize=(10, 6))
    plt.plot(paths[:, :100].T)  # Plot first 100 paths for clarity
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_chooser_option_payoff(S, K, t_choose, T, r, sigma):
    S_range = np.linspace(0.5 * K, 1.5 * K, 100)
    tau = T - t_choose
    d1 = (np.log(S_range / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    call_values = S_range * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    put_values = K * np.exp(-r * tau) * norm.cdf(-d2) - S_range * norm.cdf(-d1)
    chooser_values = np.maximum(call_values, put_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, call_values, label='Call Option')
    plt.plot(S_range, put_values, label='Put Option')
    plt.plot(S_range, chooser_values, label='Chooser Option')
    plt.axvline(x=K, color='r', linestyle='--', label='Strike Price')
    plt.title('Chooser Option Payoff at Choice Time')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Value')
    plt.legend()
    plt.savefig('chooser_option_payoff.png')
    plt.close()

def plot_option_prices_comparison(results_df):
    plt.figure(figsize=(12, 8))
    
    plt.plot(results_df['Strike'], results_df['Market Call'], 'ro-', label='Market Call')
    plt.plot(results_df['Strike'], results_df['Market Put'], 'go-', label='Market Put')
    plt.plot(results_df['Strike'], results_df['BS Call'], 'b--', label='BS Call')
    plt.plot(results_df['Strike'], results_df['BS Put'], 'c--', label='BS Put')
    plt.plot(results_df['Strike'], results_df['Heston Call'], 'm:', label='Heston Call')
    plt.plot(results_df['Strike'], results_df['Heston Put'], 'y:', label='Heston Put')
    
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title('Comparison of Model and Market Option Prices')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('option_prices_comparison.png')
    plt.close()

    # Plot implied volatilities
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Strike'], results_df['Call IV'], 'ro-', label='Call IV')
    plt.plot(results_df['Strike'], results_df['Put IV'], 'bo-', label='Put IV')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility Smile')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('implied_volatility_smile.png')
    plt.close()

def plot_chooser_option_comparison(chooser_df):
    plt.figure(figsize=(12, 8))
    plt.plot(chooser_df['Strike'], chooser_df['Real-World Chooser'], 'ro-', label='Simulated Real-World')
    plt.plot(chooser_df['Strike'], chooser_df['BS Chooser'], 'b--', label='Black-Scholes')
    plt.plot(chooser_df['Strike'], chooser_df['Heston Chooser'], 'g:', label='Heston')
    
    plt.xlabel('Strike Price')
    plt.ylabel('Chooser Option Price')
    plt.title('Comparison of Chooser Option Prices')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('chooser_option_comparison.png')
    plt.close()