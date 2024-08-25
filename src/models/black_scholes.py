import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class LocalVolatilityBS:
    def __init__(self, S0, T, r, strikes, local_vols):
        self.S0 = S0
        self.T = T
        self.r = r
        self.vol_interpolator = interp1d(strikes, local_vols, kind='cubic', fill_value='extrapolate')

    def __call__(self, K):
        return BlackScholes(self.S0, K, self.T, self.r, self.vol_interpolator(K))

class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        return self.S * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_price(self):
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * norm.cdf(-self.d1())

    def chooser_option_price(self, t_choose):
        tau = self.T - t_choose
        d1_tau = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * tau) / (self.sigma * np.sqrt(tau))
        d2_tau = d1_tau - self.sigma * np.sqrt(tau)
        call_value = self.S * norm.cdf(d1_tau) - self.K * np.exp(-self.r * tau) * norm.cdf(d2_tau)
        put_value = self.K * np.exp(-self.r * tau) * norm.cdf(-d2_tau) - self.S * norm.cdf(-d1_tau)
        return np.maximum(call_value, put_value)

def calibrate_single_option(args):
    S0, K, T, r, market_price = args
    def objective(sigma):
        bs = BlackScholes(S0, K, T, r, sigma)
        model_price = bs.call_price()
        return (model_price - market_price)**2
    
    result = minimize(objective, x0=0.3, method='Nelder-Mead', options={'maxiter': 50})
    return result.x[0]

def calibrate_local_volatility(market_data, S0, r, T):
    strikes = market_data['Strike'].values
    market_prices = market_data['Market Call'].values
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        args = [(S0, K, T, r, price) for K, price in zip(strikes, market_prices)]
        local_vols = list(executor.map(calibrate_single_option, args))
    
    return LocalVolatilityBS(S0, T, r, strikes, local_vols)