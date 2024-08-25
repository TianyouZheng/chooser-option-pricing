import numpy as np
from src.models.black_scholes import BlackScholes
from src.models.heston_model import HestonModel

class MonteCarloSimulation:
    def __init__(self, model, num_simulations, num_steps):
        self.model = model
        self.num_simulations = num_simulations
        self.num_steps = num_steps

    def run_simulation(self):
        if isinstance(self.model, BlackScholes):
            return self._simulate_black_scholes()
        elif isinstance(self.model, HestonModel):
            return self._simulate_heston()
        else:
            raise ValueError("Unsupported model type")

    def _simulate_black_scholes(self):
        dt = self.model.T / self.num_steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.zeros((self.num_simulations, self.num_steps + 1))
        S[:, 0] = self.model.S
        
        for i in range(1, self.num_steps + 1):
            Z = np.random.normal(0, 1, self.num_simulations)
            S[:, i] = S[:, i-1] * np.exp((self.model.r - 0.5 * self.model.sigma**2) * dt + self.model.sigma * sqrt_dt * Z)
        
        return S

    def _simulate_heston(self):
        return self.model.generate_paths(self.num_simulations, self.num_steps)

    def price_chooser_option(self, K, t_choose):
        S = self.run_simulation()
        
        choose_index = int(t_choose / self.model.T * self.num_steps)
        remaining_time = self.model.T - t_choose
        discount_factor = np.exp(-self.model.r * remaining_time)
        
        call_values = discount_factor * np.maximum(S[:, -1] - K, 0)
        put_values = discount_factor * np.maximum(K - S[:, -1], 0)
        
        chooser_values = np.maximum(call_values, put_values)
        option_price = np.exp(-self.model.r * t_choose) * np.mean(chooser_values)
        
        return option_price

    def compute_confidence_interval(self, prices, confidence_level=0.95):
        mean_price = np.mean(prices)
        std_error = np.std(prices, ddof=1) / np.sqrt(len(prices))
        z_score = norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_error
        return mean_price - margin_of_error, mean_price + margin_of_error