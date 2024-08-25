import numpy as np

class HestonModel:
    def __init__(self, S0, V0, kappa, theta, sigma, rho, r, T):
        self.S0 = S0  # Initial stock price
        self.V0 = V0  # Initial variance
        self.kappa = kappa  # Rate of mean reversion
        self.theta = theta  # Long-term variance
        self.sigma = sigma  # Volatility of volatility
        self.rho = rho  # Correlation between stock price and variance
        self.r = r  # Risk-free rate
        self.T = T  # Time to maturity

    def generate_paths(self, num_paths, num_steps):
        dt = self.T / num_steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.zeros((num_paths, num_steps + 1))
        V = np.zeros((num_paths, num_steps + 1))
        
        S[:, 0] = self.S0
        V[:, 0] = self.V0
        
        for i in range(1, num_steps + 1):
            Z1 = np.random.normal(0, 1, num_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, num_paths)
            
            S[:, i] = S[:, i-1] * np.exp((self.r - 0.5 * V[:, i-1]) * dt + np.sqrt(V[:, i-1]) * sqrt_dt * Z1)
            V[:, i] = np.maximum(V[:, i-1] + self.kappa * (self.theta - V[:, i-1]) * dt + self.sigma * np.sqrt(V[:, i-1]) * sqrt_dt * Z2, 0)
        
        return S, V

    def price_european_option(self, K, option_type, num_paths, num_steps):
        S, _ = self.generate_paths(num_paths, num_steps)
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(S[:, -1] - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S[:, -1], 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return option_price

    def price_chooser_option(self, K, t_choose, num_paths, num_steps):
        S, _ = self.generate_paths(num_paths, num_steps)
        
        # Find the index corresponding to the choice time
        choose_index = int(t_choose / self.T * num_steps)
        
        # Calculate the value of call and put options at the choice time
        remaining_time = self.T - t_choose
        discount_factor = np.exp(-self.r * remaining_time)
        
        call_values = discount_factor * np.maximum(S[:, -1] - K, 0)
        put_values = discount_factor * np.maximum(K - S[:, -1], 0)
        
        # The chooser option value is the maximum of call and put values
        chooser_values = np.maximum(call_values, put_values)
        
        # Discount the chooser option value back to t=0
        option_price = np.exp(-self.r * t_choose) * np.mean(chooser_values)
        return option_price