import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        return self.S * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_price(self):
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * norm.cdf(-self.d1())

    def chooser_option_price(self, t_choose):
        # Time to choose for the chooser option
        tau = self.T - t_choose
        
        d1_tau = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * tau) / (self.sigma * np.sqrt(tau))
        d2_tau = d1_tau - self.sigma * np.sqrt(tau)
        
        call_value = self.S * norm.cdf(d1_tau) - self.K * np.exp(-self.r * tau) * norm.cdf(d2_tau)
        put_value = self.K * np.exp(-self.r * tau) * norm.cdf(-d2_tau) - self.S * norm.cdf(-d1_tau)
        
        return np.maximum(call_value, put_value)