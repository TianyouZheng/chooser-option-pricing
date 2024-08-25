import numpy as np
from numba import jit

class HestonModel:
    def __init__(self, S0, V0, kappa, theta, sigma, rho, r, T):
        self.S0 = S0
        self.V0 = V0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.T = T

    @staticmethod
    @jit(nopython=True)
    def _generate_paths(S0, V0, kappa, theta, sigma, rho, r, T, num_paths, num_steps):
        dt = T / num_steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.zeros((num_paths, num_steps + 1))
        V = np.zeros((num_paths, num_steps + 1))
        
        S[:, 0] = S0
        V[:, 0] = V0
        
        for i in range(1, num_steps + 1):
            Z1 = np.random.normal(0, 1, num_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, num_paths)
            
            S[:, i] = S[:, i-1] * np.exp((r - 0.5 * V[:, i-1]) * dt + np.sqrt(V[:, i-1]) * sqrt_dt * Z1)
            V[:, i] = np.maximum(V[:, i-1] + kappa * (theta - V[:, i-1]) * dt + sigma * np.sqrt(V[:, i-1]) * sqrt_dt * Z2, 0)
        
        return S, V

    def generate_paths(self, num_paths, num_steps):
        return self._generate_paths(self.S0, self.V0, self.kappa, self.theta, self.sigma, self.rho, self.r, self.T, num_paths, num_steps)

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