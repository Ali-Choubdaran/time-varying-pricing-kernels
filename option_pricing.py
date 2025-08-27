"""
Inferring Subjective Probabilities from Option Prices via Time-Varying Pricing Kernels

This module implements a novel methodology for estimating time-varying stochastic discount 
functions (SDFs) where the SDF coefficients depend on risk-neutral moments. This approach 
allows for non-local relationships between risk-neutral and subjective probabilities, 
resolving the monotonicity puzzle in asset pricing.

Key Innovation:
Instead of the traditional approach where p(r) = q(r)/m(r) depends only locally on the 
pricing kernel at return r, my method makes the entire SDF depend on risk-neutral moments,
creating a non-local relationship where the entire risk-neutral distribution affects 
subjective probabilities at each point.

Mathematical Framework:
- SDF(r) = exp(a₀ + a₁·r + a₂·r² + ... + aₖ·rᵏ)
- Each coefficient aᵢ = polynomial(risk_neutral_moments) 
- a₀ determined by normalization constraint ∑p(r) = 1
- Estimation via maximum likelihood using realized market returns

Author: Ali Choubdaran
Institution: London School of Economics
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings
import os

@dataclass
class Config:
    """
    Configuration parameters for the estimation
    
    This class centralizes all hyperparameters for the model specification and optimization.
    Users can easily modify the model complexity and data range through these parameters.
    """
    # Return grid parameters - defines the range of possible returns
    min_return: float = 0.8    # Minimum gross return (1+r = 0.8, so r = -20%)
    max_return: float = 1.2    # Maximum gross return (1+r = 1.2, so r = +20%)  
    n_grid_points: int = 401   # Number of grid points between min and max
    
    # SDF polynomial specification
    sdf_degree: int = 3        # Degree of SDF polynomial: a₀ + a₁·r + a₂·r² + a₃·r³
    coef_degree: int = 2       # Degree of coefficient polynomials in moments
    
    # Risk-neutral moments to include in the model
    moment_orders: List[int] = None  # Orders of moments to compute [1,2,3,4] = mean,var,skew,kurt
    
    # Optimization parameters
    max_iterations: int = 10000
    tolerance: float = 1e-6
    
    def __post_init__(self):
        """Set default moment orders if not specified"""
        if self.moment_orders is None:
            self.moment_orders = [1, 2, 3, 4]


class DataProcessor:
    """
    Handles loading and preprocessing of Arrow-Debreu price data
    
    This class manages the interface between raw option-derived data and the estimation
    pipeline. It loads Arrow-Debreu prices computed from option price data and applies
    data quality filters.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.return_grid = self._create_return_grid()
        
    def _create_return_grid(self) -> np.ndarray:
        """Create evenly spaced return grid from min to max return"""
        return np.linspace(
            self.config.min_return, 
            self.config.max_return, 
            self.config.n_grid_points
        )
    
    def load_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the Arrow-Debreu price data
        
        The input data should be a Stata file containing:
        - 'date': Trading dates
        - 'mr': Market returns (simple returns)
        - 'ar800', 'ar801', ..., 'ar1200': Arrow-Debreu prices for different return levels
        - 'first_ar', 'last_ar': Coverage indicators for data quality filtering
        
        Args:
            file_path: Path to the Stata file containing preprocessed option data
            
        Returns:
            dates: Array of trading dates
            returns: Array of realized market returns (simple returns)
            arrow_debreu: Array of Arrow-Debreu prices (n_dates × n_grid_points)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        try:
            df = pd.read_stata(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load Stata file {file_path}: {e}")
        
        # Extract basic time series data
        dates = df['date'].values
        returns = df['mr'].values  # Market returns (simple returns)
        
        # Extract Arrow-Debreu prices across the return grid
        # Column names follow pattern: ar800, ar801, ..., ar1200
        # where ar800 corresponds to gross return 0.8 (-20% simple return)
        ar_columns = [f'ar{int(self.config.min_return * 1000 + i)}' 
                     for i in range(self.config.n_grid_points)]
        
        try:
            arrow_debreu = df[ar_columns].values
        except KeyError as e:
            available_cols = [col for col in df.columns if col.startswith('ar')]
            raise ValueError(
                f"Missing Arrow-Debreu columns. Expected: {ar_columns[:3]}...{ar_columns[-3:]}. "
                f"Found: {available_cols[:5]}..."
            )
        
        # Apply data quality filters
        dates_filtered, returns_filtered, arrow_debreu_filtered = self._apply_filters(
            dates, returns, arrow_debreu, df
        )
        
        return dates_filtered, returns_filtered, arrow_debreu_filtered
    
    def _apply_filters(self, dates: np.ndarray, returns: np.ndarray, 
                      arrow_debreu: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply data quality filters to ensure reliable estimation
        
        Filters applied:
        1. Returns within reasonable range (to avoid outliers)
        2. Adequate Arrow-Debreu price coverage across the return grid
        """
        # Filter 1: Remove observations with extreme returns
        return_filter = (
            (returns >= self.config.min_return - 1) & 
            (returns <= self.config.max_return - 1)
        )
        
        # Filter 2: Ensure adequate coverage of Arrow-Debreu prices
        # first_ar and last_ar indicate the range of available prices
        coverage_filter = (
            (df["first_ar"] <= 900) & 
            (df["last_ar"] >= 1100)
        )
        
        # Combine all filters
        combined_filter = return_filter & coverage_filter
        
        return (
            dates[combined_filter],
            returns[combined_filter], 
            arrow_debreu[combined_filter]
        )


class MomentCalculator:
    """
    Computes various types of risk-neutral moments from Arrow-Debreu prices
    
    This class converts Arrow-Debreu prices into risk-neutral probability distributions
    and computes different types of moments that will be used as explanatory variables
    in the time-varying SDF specification.
    """
    
    def __init__(self, config: Config, return_grid: np.ndarray):
        self.config = config
        self.return_grid = return_grid
        self.simple_returns = return_grid - 1  # Convert gross to simple returns
        
    def calculate_moments(self, arrow_debreu: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate different types of risk-neutral moments for each time period
        
        Computes three types of moments:
        1. Raw moments: E[rᵏ] for k in moment_orders
        2. Central moments: E[(r - E[r])ᵏ] for k in moment_orders  
        3. Standardized moments: Normalized central moments with mean=0, std=1
        
        Args:
            arrow_debreu: Array of Arrow-Debreu prices (n_dates × n_grid_points)
            
        Returns:
            Dictionary containing:
            - 'raw': Raw moments (n_dates × n_moments)
            - 'central': Central moments (n_dates × n_moments)
            - 'standardized': Standardized central moments (n_dates × n_moments)
        """
        # Convert Arrow-Debreu prices to risk-neutral probabilities
        risk_neutral_probs = self._normalize_probabilities(arrow_debreu)
        
        moments = {}
        
        # Compute different types of moments
        moments['raw'] = self._calculate_raw_moments(risk_neutral_probs)
        moments['central'] = self._calculate_central_moments(risk_neutral_probs)
        moments['standardized'] = self._standardize_moments(moments['central'])
        
        return moments
    
    def _normalize_probabilities(self, arrow_debreu: np.ndarray) -> np.ndarray:
        """Convert Arrow-Debreu prices to probabilities by normalizing to sum to 1"""
        sums = np.sum(arrow_debreu, axis=1, keepdims=True)
        return arrow_debreu / sums
    
    def _calculate_raw_moments(self, probs: np.ndarray) -> np.ndarray:
        """Calculate raw moments E[rᵏ] for each specified order"""
        n_dates = probs.shape[0]
        n_moments = len(self.config.moment_orders)
        moments = np.zeros((n_dates, n_moments))
        
        for i, order in enumerate(self.config.moment_orders):
            powered_returns = np.power(self.simple_returns, order)
            moments[:, i] = np.sum(probs * powered_returns, axis=1)
            
        return moments
    
    def _calculate_central_moments(self, probs: np.ndarray) -> np.ndarray:
        """Calculate central moments E[(r - E[r])ᵏ] for each specified order"""
        # First compute the mean for each time period
        mean_returns = np.sum(probs * self.simple_returns, axis=1, keepdims=True)
        
        n_dates = probs.shape[0]
        n_moments = len(self.config.moment_orders)
        central_moments = np.zeros((n_dates, n_moments))
        
        for i, order in enumerate(self.config.moment_orders):
            if order == 1:
                central_moments[:, i] = 0  # Central moment of order 1 is always 0
            else:
                deviations = (self.simple_returns - mean_returns.flatten()[:, np.newaxis])
                powered_deviations = np.power(deviations, order)
                central_moments[:, i] = np.sum(probs * powered_deviations, axis=1)
                
        return central_moments
    
    def _standardize_moments(self, central_moments: np.ndarray) -> np.ndarray:
        """
        Standardize moments to have zero mean and unit variance across time
        
        This normalization helps with numerical stability in optimization and makes
        coefficients more interpretable across different moment orders.
        """
        standardized = np.zeros_like(central_moments)
        
        for i in range(central_moments.shape[1]):
            moment_series = central_moments[:, i]
            mean_moment = np.mean(moment_series)
            std_moment = np.std(moment_series)
            
            if std_moment > 1e-10:  # Avoid division by zero
                standardized[:, i] = (moment_series - mean_moment) / std_moment
            else:
                standardized[:, i] = moment_series - mean_moment
                
        return standardized


class SDFSpecification:
    """
    Specifies the time-varying Stochastic Discount Function as a polynomial
    
    This class implements the core innovation: a polynomial SDF where coefficients
    depend on risk-neutral moments, creating non-local relationships between
    risk-neutral and subjective probabilities.
    
    Mathematical Structure:
    SDF(r,t) = exp(a₀ₜ + a₁ₜ·r + a₂ₜ·r² + ... + aₖₜ·rᵏ)
    
    where each coefficient aᵢₜ is a polynomial in risk-neutral moments:
    aᵢₜ = bᵢ₀ + bᵢ₁·m₁ₜ + bᵢ₂·m₂ₜ + ... + polynomial terms
    
    Key constraint: a₀ₜ is not estimated but determined by normalization ∑p(r,t) = 1
    """
    
    def __init__(self, config: Config, return_grid: np.ndarray):
        self.config = config
        self.return_grid = return_grid
        self.simple_returns = return_grid - 1  # Convert to simple returns for polynomial
        self.n_params = self._calculate_param_count()
        
    def _calculate_param_count(self) -> int:
        """
        Calculate total number of parameters to estimate
        
        Parameter structure:
        - SDF polynomial: a₀ + a₁·r + a₂·r² + ... + aₖ·rᵏ
        - a₀ determined by normalization → only estimate a₁, ..., aₖ (k coefficients)
        - Each coefficient aᵢ = polynomial in moments of degree coef_degree
        - Total: k × n_moments × (coef_degree + 1) parameters
        
        Returns:
            Total number of parameters to estimate
        """
        n_free_coefficients = self.config.sdf_degree  # Exclude a₀
        n_moments = len(self.config.moment_orders)
        n_coef_params = self.config.coef_degree + 1  # Constant + linear + quadratic + ...
        
        return n_free_coefficients * n_moments * n_coef_params
    
    def compute_sdf_linear_part(self, parameters: np.ndarray, moments: np.ndarray, 
                               time_index: int) -> np.ndarray:
        """
        Compute the linear part of log SDF (excluding the intercept a₀)
        
        This function computes: a₁ₜ·r + a₂ₜ·r² + ... + aₖₜ·rᵏ
        where each coefficient aᵢₜ depends on moments at time t
        
        Args:
            parameters: Model parameters (flattened array)
            moments: Risk-neutral moments array (n_dates × n_moments)
            time_index: Which time period to compute for
            
        Returns:
            Linear part of log SDF evaluated over the return grid
        """
        n_free_coef = self.config.sdf_degree
        n_moments = len(self.config.moment_orders)
        n_coef_params = self.config.coef_degree + 1
        
        # Reshape flattened parameters into 3D structure
        # Dimensions: [coefficient_index, moment_index, polynomial_power]
        params_reshaped = parameters.reshape(n_free_coef, n_moments, n_coef_params)
        
        # Compute time-varying coefficients a₁ₜ, a₂ₜ, ..., aₖₜ
        sdf_coefficients = np.zeros(n_free_coef)
        
        for i in range(n_free_coef):  # For each SDF coefficient aᵢ₊₁
            for j, moment_val in enumerate(moments[time_index]):  # For each moment
                # Each coefficient is a polynomial in moments
                poly_contribution = np.sum([
                    params_reshaped[i, j, k] * (moment_val ** k) 
                    for k in range(n_coef_params)
                ])
                sdf_coefficients[i] += poly_contribution
        
        # Evaluate polynomial: a₁·r + a₂·r² + ... + aₖ·rᵏ
        linear_part = np.zeros_like(self.simple_returns)
        for i, coef in enumerate(sdf_coefficients):
            linear_part += coef * (self.simple_returns ** (i + 1))  # Powers start from 1
        
        return linear_part
    
    def compute_log_sdf(self, parameters: np.ndarray, moments: np.ndarray, 
                       risk_neutral_probs: np.ndarray, time_index: int) -> np.ndarray:
        """
        Compute complete log(SDF) with proper normalization using numerical stability tricks
        
        This function:
        1. Computes the linear part of log SDF
        2. Determines a₀ₜ from the constraint that subjective probabilities sum to 1
        3. Uses log-space arithmetic to prevent numerical overflow
        
        Mathematical details:
        - p(r,t) = q(r,t) / SDF(r,t) where q are risk-neutral probabilities
        - Constraint: ∑ᵣ p(r,t) = 1
        - This determines: a₀ₜ = log(∑ᵣ q(r,t) · exp(-linear_part))
        
        Args:
            parameters: Model parameters
            moments: Risk-neutral moments array
            risk_neutral_probs: Risk-neutral probabilities for time t
            time_index: Which time period to compute for
            
        Returns:
            Complete log(SDF) = a₀ₜ + linear_part over the return grid
        """
        # Get the "free" part of log SDF (without intercept)
        linear_part = self.compute_sdf_linear_part(parameters, moments, time_index)
        
        # Compute a₀ₜ using log-space tricks for numerical stability
        # We need to compute: exp(a₀) = ∑ q(r) · exp(-linear_part)
        # To prevent overflow in exp(-linear_part), we use the identity:
        # exp(-linear_part) = exp(-linear_part - max_shift + max_shift)
        #                   = exp(max_shift) · exp(-linear_part - max_shift)
        # where max_shift = max(-linear_part) ensures exp(-linear_part - max_shift) ≤ 1
        
        max_shift = np.max(-linear_part)  # Maximum of values we'll exponentiate
        exp_stable = np.exp(-linear_part - max_shift)  # All values ≤ 1, numerically safe
        sum_weighted = np.sum(risk_neutral_probs * exp_stable)
        
        if sum_weighted > 1e-10:
            a0 = np.log(sum_weighted) + max_shift  # Add back the shift
        else:
            # Fallback for edge cases
            a0 = np.log(np.sum(risk_neutral_probs))
        
        # Return complete log(SDF) = a₀ + linear_part
        return a0 + linear_part


class LogLikelihood:
    """
    Implements the negative log-likelihood function for maximum likelihood estimation
    
    This class evaluates the likelihood of observed market returns under the model.
    The likelihood at time t is the subjective probability assigned to the realized return.
    
    Mathematical framework:
    L = ∏ₜ p(rₜ | parameters) where rₜ is the realized return at time t
    log L = ∑ₜ log p(rₜ | parameters) = ∑ₜ [log q(rₜ) - log SDF(rₜ)]
    
    Since log q(rₜ) is constant across parameter values, we minimize:
    -log L ∝ ∑ₜ log SDF(rₜ)
    """
    
    def __init__(self, returns: np.ndarray, moments: np.ndarray, 
                 arrow_debreu: np.ndarray, sdf_spec: SDFSpecification):
        """
        Initialize the likelihood function
        
        Args:
            returns: Realized market returns (simple returns)
            moments: Risk-neutral moments for each time period
            arrow_debreu: Arrow-Debreu prices for each time period
            sdf_spec: SDF specification object
        """
        self.returns = returns
        self.moments = moments  
        self.arrow_debreu = arrow_debreu
        self.sdf_spec = sdf_spec
        self.return_grid = sdf_spec.return_grid
        
    def __call__(self, parameters: np.ndarray) -> float:
        """
        Evaluate the negative log-likelihood function
        
        For each time period:
        1. Compute log(SDF) over the return grid
        2. Find the grid point closest to the realized return
        3. Add log(SDF) at that point to the objective function
        
        Args:
            parameters: Model parameters to evaluate
            
        Returns:
            Negative log-likelihood value (to be minimized)
        """
        try:
            total_log_likelihood = 0.0
            
            for t in range(len(self.returns)):
                # Get risk-neutral probabilities for this time period
                risk_neutral_probs = self.arrow_debreu[t] / np.sum(self.arrow_debreu[t])
                
                # Compute log(SDF) over the entire return grid
                log_sdf = self.sdf_spec.compute_log_sdf(
                    parameters, self.moments, risk_neutral_probs, t
                )
                
                # Find grid point closest to realized return
                realized_return_gross = self.returns[t] + 1  # Convert to gross return
                closest_index = np.argmin(np.abs(self.return_grid - realized_return_gross))
                
                # Add log(SDF) at realized return to objective
                total_log_likelihood += log_sdf[closest_index]
                
            return total_log_likelihood
            
        except Exception:
            # Return large penalty for any numerical issues
            return 1e10


class SimpleOptimizer:
    """
    Multi-method optimization with robust handling of numerical issues
    
    This optimizer implements a multi-start, multi-method approach to handle the
    challenging optimization landscape typical in nonlinear maximum likelihood problems.
    
    Strategy:
    1. Try multiple random starting points to avoid local optima
    2. Use multiple optimization algorithms in sequence
    3. Chain results: use best solution from one method as starting point for next
    4. Robust error handling for numerical issues
    """
    
    def __init__(self, objective_function, n_params: int, logger: Optional[logging.Logger] = None):
        self.objective = objective_function
        self.n_params = n_params
        self.logger = logger or logging.getLogger('optimizer')
        self.best_params = None
        self.best_value = np.inf
        self.function_evals = 0
        
    def optimize(self, n_random_starts: int = 3, param_bounds: float = 2.0) -> Dict:
        """
        Run multi-start, multi-method optimization
        
        Args:
            n_random_starts: Number of random initial points to try
            param_bounds: Range for random initialization [-param_bounds, param_bounds]
            
        Returns:
            Dictionary with optimization results including best parameters and diagnostics
        """
        self.logger.info(f"Starting optimization with {n_random_starts} random starts")
        self.logger.info(f"Parameter bounds: [-{param_bounds}, {param_bounds}]")
        
        results = []
        
        for start_idx in range(n_random_starts):
            self.logger.info(f"\n--- Random Start {start_idx + 1}/{n_random_starts} ---")
            
            # Generate random starting point
            x0 = np.random.uniform(-param_bounds, param_bounds, self.n_params)
            
            # Test initial point validity
            initial_value = self._safe_eval(x0)
            self.logger.info(f"Initial point value: {initial_value:.6f}")
            
            if initial_value >= 1e9:
                self.logger.warning("Initial point invalid, skipping...")
                continue
                
            # Try different optimization methods in sequence
            methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']
            
            for method in methods:
                self.logger.info(f"Trying method: {method}")
                
                try:
                    result = self._single_optimization(x0, method, param_bounds)
                    
                    # Record result
                    results.append({
                        'start_idx': start_idx,
                        'method': method,
                        'success': getattr(result, 'success', True),
                        'fun': result.fun,
                        'x': result.x.copy(),
                        'nfev': getattr(result, 'nfev', 0),
                        'message': getattr(result, 'message', 'Completed')
                    })
                    
                    self.logger.info(f"Method {method} completed: value={result.fun:.6f}")
                    
                    # Update best result
                    if result.fun < self.best_value:
                        self.best_value = result.fun
                        self.best_params = result.x.copy()
                        self.logger.info(f"New best result: {self.best_value:.6f}")
                        
                except Exception as e:
                    self.logger.warning(f"Method {method} failed: {str(e)}")
                    
                # Chain methods: use best result as starting point for next method
                if self.best_params is not None:
                    x0 = self.best_params.copy()
        
        # Log final results
        self.logger.info(f"\n--- Optimization Complete ---")
        self.logger.info(f"Total function evaluations: {self.function_evals}")
        self.logger.info(f"Best value found: {self.best_value:.6f}")
        if self.best_params is not None:
            self.logger.info(f"Best parameters (first 5): {self.best_params[:5]}")
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'function_evals': self.function_evals,
            'results': results,
            'success': self.best_value < 1e9
        }
    
    def _single_optimization(self, x0: np.ndarray, method: str, param_bounds: float):
        """Run a single optimization with the specified method"""
        
        if method == 'L-BFGS-B':
            # Bounded optimization
            bounds = [(-param_bounds, param_bounds)] * self.n_params
            result = minimize(
                self._safe_eval, x0, 
                method=method, 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
        else:
            # Unbounded methods
            result = minimize(
                self._safe_eval, x0,
                method=method,
                options={'maxiter': 1000}
            )
            
        return result
    
    def _safe_eval(self, params: np.ndarray) -> float:
        """Safely evaluate objective function with error handling and logging"""
        self.function_evals += 1
        
        try:
            value = self.objective(params)
            
            # Periodic progress logging
            if self.function_evals % 500 == 0:
                self.logger.info(f"Evaluation {self.function_evals}: {value:.6f}")
                
            return float(value)
            
        except Exception as e:
            self.logger.warning(f"Function evaluation failed: {str(e)}")
            return 1e10


def setup_logging(log_file: str = 'estimation.log') -> logging.Logger:
    """
    Setup comprehensive logging for the estimation process
    
    Creates both file and console handlers to track optimization progress
    and diagnose any issues that arise during estimation.
    """
    logger = logging.getLogger('option_estimation')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for persistent logging
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def run_estimation(file_path: str, config: Optional[Config] = None) -> Dict:
    """
    Complete estimation pipeline from data loading to objective function setup
    
    This function orchestrates the entire estimation process:
    1. Load and validate Arrow-Debreu price data
    2. Compute risk-neutral moments
    3. Set up the SDF specification
    4. Create the log-likelihood objective function
    5. Perform basic diagnostics
    
    Args:
        file_path: Path to the Stata data file containing Arrow-Debreu prices
        config: Model configuration (uses defaults if None)
        
    Returns:
        Dictionary containing all components needed for optimization
    """
    # Use default configuration if none provided
    if config is None:
        config = Config()
    
    # Set up logging
    logger = setup_logging('estimation.log')
    logger.info("="*60)
    logger.info("STARTING ESTIMATION PIPELINE")
    logger.info("="*60)
    logger.info(f"Configuration: {config}")
    
    # Initialize core components
    data_processor = DataProcessor(config)
    moment_calc = MomentCalculator(config, data_processor.return_grid)
    sdf_spec = SDFSpecification(config, data_processor.return_grid)
    
    logger.info(f"Model specification:")
    logger.info(f"  - SDF degree: {config.sdf_degree}")
    logger.info(f"  - Coefficient degree: {config.coef_degree}")  
    logger.info(f"  - Moment orders: {config.moment_orders}")
    logger.info(f"  - Total parameters: {sdf_spec.n_params}")
    
    # Load and validate data
    logger.info(f"Loading data from: {file_path}")
    try:
        dates, returns, arrow_debreu = data_processor.load_data(file_path)
        logger.info(f"Successfully loaded {len(dates)} observations")
        logger.info(f"Date range: {dates[0]} to {dates[-1]}")
        logger.info(f"Return range: {np.min(returns):.4f} to {np.max(returns):.4f}")
        logger.info(f"Arrow-Debreu shape: {arrow_debreu.shape}")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise
    
    # Calculate risk-neutral moments
    logger.info("Computing risk-neutral moments...")
    moments_dict = moment_calc.calculate_moments(arrow_debreu)
    
    # Use standardized moments for estimation (can be changed to 'raw' or 'central')
    moments = moments_dict['standardized']
    logger.info(f"Moments computed successfully:")
    logger.info(f"  - Shape: {moments.shape}")
    logger.info(f"  - Moment statistics:\n{pd.DataFrame(moments).describe()}")
    
    # Set up the objective function
    logger.info("Setting up log-likelihood objective function...")
    objective = LogLikelihood(returns, moments, arrow_debreu, sdf_spec)
    
    # Test objective function with random parameters
    logger.info("Testing objective function with random parameters...")
    test_params = np.random.normal(0, 0.1, sdf_spec.n_params)
    test_likelihood = objective(test_params)
    logger.info(f"Test likelihood value: {test_likelihood:.6f}")
    
    if test_likelihood >= 1e9:
        logger.warning("Test evaluation returned large penalty - check data quality")
    else:
        logger.info("Objective function test passed - ready for optimization")
    
    return {
        'dates': dates,
        'returns': returns,
        'arrow_debreu': arrow_debreu,
        'moments': moments,
        'sdf_spec': sdf_spec,
        'objective': objective,
        'config': config,
        'logger': logger
    }


def run_optimization(results_dict: Dict, n_starts: int = 3, param_bounds: float = 2.0) -> Dict:
    """
    Execute the optimization process to find maximum likelihood estimates
    
    This function takes the estimation setup and runs the optimization to find
    the parameter values that maximize the likelihood of observed returns.
    
    Args:
        results_dict: Output dictionary from run_estimation()
        n_starts: Number of random starting points to try
        param_bounds: Bounds for random parameter initialization
        
    Returns:
        Updated results dictionary including optimization results
    """
    logger = results_dict['logger']
    objective = results_dict['objective']
    sdf_spec = results_dict['sdf_spec']
    
    logger.info("\n" + "="*60)
    logger.info("STARTING OPTIMIZATION")
    logger.info("="*60)
    logger.info(f"Parameters to estimate: {sdf_spec.n_params}")
    logger.info(f"Random starts: {n_starts}")
    logger.info(f"Parameter bounds: [{-param_bounds:.1f}, {param_bounds:.1f}]")
    
    # Initialize and run optimizer
    optimizer = SimpleOptimizer(objective, sdf_spec.n_params, logger)
    opt_results = optimizer.optimize(n_starts, param_bounds)
    
    # Add optimization results to the results dictionary
    results_dict['optimization'] = opt_results
    results_dict['optimizer'] = optimizer
    
    # Log final summary
    if opt_results['success']:
        logger.info(f"\n🎉 OPTIMIZATION SUCCESSFUL!")
        logger.info(f"Best objective value: {opt_results['best_value']:.6f}")
        logger.info(f"Total function evaluations: {opt_results['function_evals']}")
        logger.info(f"Methods attempted: {len(opt_results['results'])}")
    else:
        logger.warning(f"\n⚠️  OPTIMIZATION HAD ISSUES")
        logger.warning(f"Check log for details. Consider adjusting parameters.")
    
    return results_dict


def print_results_summary(results_dict: Dict) -> None:
    """
    Print a formatted summary of estimation results
    
    Args:
        results_dict: Complete results from estimation and optimization
    """
    config = results_dict['config']
    opt_results = results_dict.get('optimization', {})
    
    print("\n" + "="*60)
    print("ESTIMATION RESULTS SUMMARY")
    print("="*60)
    
    # Model specification
    print(f"Model Specification:")
    print(f"  SDF Polynomial Degree: {config.sdf_degree}")
    print(f"  Coefficient Polynomial Degree: {config.coef_degree}")
    print(f"  Risk-Neutral Moments: {config.moment_orders}")
    print(f"  Total Parameters: {results_dict['sdf_spec'].n_params}")
    
    # Data summary
    print(f"\nData Summary:")
    print(f"  Observations: {len(results_dict['dates'])}")
    print(f"  Date Range: {results_dict['dates'][0]} to {results_dict['dates'][-1]}")
    print(f"  Return Range: {np.min(results_dict['returns']):.3f} to {np.max(results_dict['returns']):.3f}")
    
    # Optimization results
    if opt_results:
        print(f"\nOptimization Results:")
        if opt_results['success']:
            print(f"  Status: ✅ Successful")
            print(f"  Best Objective: {opt_results['best_value']:.6f}")
            print(f"  Function Evaluations: {opt_results['function_evals']}")
            print(f"  Methods Tried: {len(opt_results['results'])}")
        else:
            print(f"  Status: ❌ Issues encountered")
            print(f"  Check 'estimation.log' for details")
    
    print(f"\nNext Steps:")
    print(f"  • Check 'estimation.log' for detailed progress")
    print(f"  • Examine parameter estimates in results_dict['optimization']['best_params']")
    print(f"  • Consider running additional diagnostics or robustness checks")


# Example usage and demonstration
if __name__ == "__main__":
    """
    Example usage of the estimation framework
    
    This section demonstrates how to use the code with the provided data file.
    Users should modify the file path and configuration as needed.
    """
    
    # Configuration - modify as needed
    FILE_PATH = "Q_ready_401_detailed.dta"  
    
    try:
        print("Inferring Subjective Probabilities from Option Prices")
        print("=" * 60)
        print("Initializing estimation with time-varying pricing kernels...")
        
        # Model configuration
        config = Config(
            min_return=0.8,       # -20% to +20% return range
            max_return=1.2,
            n_grid_points=401,    # Match data file structure
            sdf_degree=3,         # Cubic polynomial in returns
            coef_degree=2,        # Quadratic dependence on moments
            moment_orders=[2, 3]  # Use variance and skewness (reduced for faster estimation)
        )
        
        print(f"Model Configuration:")
        print(f"  • Parameter count: {config.sdf_degree * len(config.moment_orders) * (config.coef_degree + 1)}")
        print(f"  • SDF specification: exp(a₀ + a₁r + a₂r² + a₃r³)")
        print(f"  • Coefficient variation: Quadratic polynomials in risk-neutral moments")
        print(f"  • Normalization: a₀ determined by ∑p(r) = 1 constraint")
        
        # Run estimation pipeline
        print(f"\nRunning estimation pipeline...")
        results = run_estimation(FILE_PATH, config)
        
        print(f"✅ Data loaded successfully!")
        print(f"   {len(results['dates'])} observations from {results['dates'][0]} to {results['dates'][-1]}")
        
        # Run optimization
        print(f"\nRunning optimization (this may take several minutes)...")
        final_results = run_optimization(
            results,
            n_starts=2,          # Reduce for faster testing
            param_bounds=1.5     # Conservative bounds
        )
        
        # Display results
        print_results_summary(final_results)
        
        print(f"\n📋 Complete results saved in 'final_results' dictionary")
        print(f"📄 Detailed logs saved in 'estimation.log'")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data file not found at '{FILE_PATH}'")
        print(f"\nPlease ensure you have:")
        print(f"  1. Downloaded Q_ready_401_detailed.dta")
        print(f"  2. Updated FILE_PATH to the correct location")
        print(f"  3. File contains the required columns (date, mr, ar800-ar1200, first_ar, last_ar)")
        
    except Exception as e:
        print(f"\n❌ Error during estimation: {e}")
        print(f"Check 'estimation.log' for detailed error information")
        print(f"\nTroubleshooting tips:")
        print(f"  • Verify data file format and column names")
        print(f"  • Try reducing model complexity (fewer moments or lower polynomial degrees)")
        print(f"  • Check parameter bounds and optimization settings")

