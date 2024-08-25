import numpy as np
from scipy.stats import norm

class MonteCarloSimulation:
    def __init__(self, model, num_simulations, num_steps):
        self.model = model
        self.num_simulations = num_simulations
        self.num_steps = num_steps

    def run_simulation(self):
        if hasattr(self.model, 'generate_paths'):
            return self.model.generate_paths(self.num_simulations, self.num_steps)
        else:
            raise NotImplementedError("The model doesn't have a generate_paths method")

    def price_european_option(self, K, option_type):
        S = self.run_simulation()
        
        # If S is a tuple (as returned by HestonModel), take only the first element
        if isinstance(S, tuple):
            S = S[0]
        
        # Ensure S is a numpy array
        S = np.array(S)
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(S[:, -1] - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S[:, -1], 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
        option_price = np.exp(-self.model.r * self.model.T) * np.mean(payoffs)
        return option_price

    def price_chooser_option(self, K, t_choose):
        S = self.run_simulation()
        
        # If S is a tuple (as returned by HestonModel), take only the first element
        if isinstance(S, tuple):
            S = S[0]
        
        # Ensure S is a numpy array
        S = np.array(S)
        
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