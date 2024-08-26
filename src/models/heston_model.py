import numpy as np
from scipy.optimize import minimize
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def price_european_option(self, K, option_type, num_simulations=10000, num_steps=100):
        dt = self.T / num_steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.full(num_simulations, self.S0)
        V = np.full(num_simulations, self.V0)
        
        for _ in range(num_steps):
            Z1 = np.random.normal(0, 1, num_simulations)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, num_simulations)
            
            S *= np.exp((self.r - 0.5 * V) * dt + np.sqrt(V) * sqrt_dt * Z1)
            V = np.maximum(V + self.kappa * (self.theta - V) * dt + self.sigma * np.sqrt(V) * sqrt_dt * Z2, 0)
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(S - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return option_price
    
    def chooser_option_price(self, t_choose, K, num_simulations=10000, num_steps=100):
        dt = self.T / num_steps
        steps_to_choose = int(t_choose / dt)
        steps_after_choose = num_steps - steps_to_choose
        
        S = np.full(num_simulations, self.S0)
        V = np.full(num_simulations, self.V0)
        
        for _ in range(steps_to_choose):
            Z1 = np.random.normal(0, 1, num_simulations)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, num_simulations)
            
            S *= np.exp((self.r - 0.5 * V) * dt + np.sqrt(V) * dt * Z1)
            V = np.maximum(V + self.kappa * (self.theta - V) * dt + self.sigma * np.sqrt(V) * dt * Z2, 0)
        
        call_values = np.zeros(num_simulations)
        put_values = np.zeros(num_simulations)
        
        for _ in range(steps_after_choose):
            Z1 = np.random.normal(0, 1, num_simulations)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, num_simulations)
            
            S *= np.exp((self.r - 0.5 * V) * dt + np.sqrt(V) * dt * Z1)
            V = np.maximum(V + self.kappa * (self.theta - V) * dt + self.sigma * np.sqrt(V) * dt * Z2, 0)
        
        call_values = np.maximum(S - K, 0)
        put_values = np.maximum(K - S, 0)
        
        chooser_payoffs = np.maximum(call_values, put_values)
        
        chooser_price = np.exp(-self.r * self.T) * np.mean(chooser_payoffs)
        return chooser_price
    
def calibrate_heston(market_data, S0, r, T, max_time=60):
    logger.info("Starting Heston model calibration")
    
    def objective(params):
        kappa, theta, sigma, rho, V0 = params
        logger.info(f"Trying parameters: kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}, rho={rho:.4f}, V0={V0:.4f}")
        
        model = HestonModel(S0, V0, kappa, theta, sigma, rho, r, T)
        
        model_prices = []
        for K in market_data['Strike']:
            price = model.price_european_option(K, 'call')
            model_prices.append(price)
        
        market_prices = market_data['Market Call'].values
        mse = np.mean((np.array(model_prices) - market_prices)**2)
        logger.info(f"Mean Squared Error: {mse:.6f}")
        return mse

    initial_guess = [1.5, 0.04, 0.3, -0.7, 0.04]  # kappa, theta, sigma, rho, V0
    bounds = [(0.1, 10), (0.01, 1), (0.01, 1), (-0.99, 0.99), (0.01, 1)]

    start_time = time.time()

    def callback(xk):
        elapsed = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed:.2f} seconds")
        if elapsed > max_time:
            logger.info("Max time reached. Stopping optimization.")
            return True

    logger.info("Starting optimization")
    result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=bounds, 
                      options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3}, callback=callback)

    logger.info("Optimization completed")
    logger.info(f"Optimization success: {result.success}")
    logger.info(f"Number of iterations: {result.nit}")
    logger.info(f"Final MSE: {result.fun:.6f}")

    if not result.success:
        logger.warning("Heston calibration did not converge. Using best found solution.")

    kappa, theta, sigma, rho, V0 = result.x
    logger.info(f"Final parameters: kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}, rho={rho:.4f}, V0={V0:.4f}")
    
    return HestonModel(S0, V0, kappa, theta, sigma, rho, r, T)