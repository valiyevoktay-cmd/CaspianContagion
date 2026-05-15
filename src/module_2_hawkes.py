import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger("CaspianContagion.HawkesMLE")

@dataclass
class HawkesParams:
    """Dataclass to strictly type the calibrated Hawkes parameters."""
    mu: float
    alpha: float
    beta: float
    is_success: bool
    log_likelihood: float

class HawkesMLECalibrator:
    """
    Module 2: Hawkes Engine & MLE Calibration.
    
    Calibrates a univariate Hawkes Process with an exponential kernel using 
    Maximum Likelihood Estimation (MLE). Optimized for short rolling windows 
    typical in microstructure research.
    """

    def __init__(self, initial_guess: Tuple[float, float, float] = (0.1, 0.5, 1.0)):
        """
        Initializes the calibrator.

        Args:
            initial_guess (Tuple[float, float, float]): Starting values for optimization (mu, alpha, beta).
        """
        self.initial_guess = np.array(initial_guess, dtype=np.float64)
        
        # Bounds: mu > 0, alpha > 0, beta > 0.
        # Note: We DO NOT enforce alpha < beta (stationarity), 
        # as we actively want to observe explosive critical states (alpha >= beta).
        self.bounds = ((1e-5, None), (1e-5, None), (1e-5, None))
        
    @staticmethod
    def _compute_recursive_intensity(t: np.ndarray, beta: float) -> np.ndarray:
        """
        Computes the recursive sum of the exponential kernel R(i) in O(N) time.
        R(i) = exp(-beta * (t_i - t_{i-1})) * (1 + R(i-1))

        Args:
            t (np.ndarray): Array of normalized event timestamps.
            beta (float): Decay rate parameter.

        Returns:
            np.ndarray: Array of R_i values.
        """
        n = len(t)
        R = np.zeros(n, dtype=np.float64)
        if n == 0:
            return R
            
        # O(N) loop is fast enough in standard Python for short windows (<1000 events)
        # For ultra-low latency C++ production, this block is cythonized.
        for i in range(1, n):
            dt = t[i] - t[i-1]
            R[i] = np.exp(-beta * dt) * (1.0 + R[i-1])
            
        return R

    def _negative_log_likelihood(self, params: np.ndarray, t: np.ndarray, t_max: float) -> float:
        """
        Calculates the Negative Log-Likelihood (NLL) for a given set of parameters.

        Args:
            params (np.ndarray): Current guess for [mu, alpha, beta].
            t (np.ndarray): Event timestamps.
            t_max (float): Total duration of the observation window.

        Returns:
            float: NLL value (to be minimized). Returns infinity if constraints are violated.
        """
        mu, alpha, beta = params
        
        # Numerical safety guardrails
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return np.inf

        n = len(t)
        if n == 0:
            return mu * t_max  # If no events, penalize high mu

        # 1. Sum of log-intensities at event times
        R = self._compute_recursive_intensity(t, beta)
        intensities = mu + alpha * R
        
        if np.any(intensities <= 0):
            return np.inf
            
        log_term = np.sum(np.log(intensities))
        
        # 2. Integral of the intensity function over [0, t_max]
        integral_term = mu * t_max + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (t_max - t)))
        
        # We want to maximize LL, hence minimize -LL
        nll = -(log_term - integral_term)
        return nll

    def calibrate(self, timestamps: np.ndarray, window_duration: Optional[float] = None) -> Dict[str, float]:
        """
        Runs the L-BFGS-B optimization to find the optimal Hawkes parameters.

        Args:
            timestamps (np.ndarray): 1D array of event timestamps (must be sorted).
            window_duration (Optional[float]): Length of the observation window. 
                                               If None, uses the last timestamp.

        Returns:
            Dict[str, float]: Calibrated parameters {mu, alpha, beta} and diagnostic metrics.
        """
        try:
            if len(timestamps) < 3:
                logger.warning("Not enough events to calibrate Hawkes process. Returning NaNs.")
                return {'mu': np.nan, 'alpha': np.nan, 'beta': np.nan, 'is_success': 0.0, 'log_likelihood': np.nan}

            # Normalize timestamps to start at 0 for numerical stability
            t_normalized = timestamps - timestamps[0]
            
            # Duration in the same units as normalized timestamps
            t_max = window_duration if window_duration else t_normalized[-1]
            
            if t_max <= 0:
                raise ValueError("Observation window duration must be strictly positive.")

            # Run optimization
            result = minimize(
                fun=self._negative_log_likelihood,
                x0=self.initial_guess,
                args=(t_normalized, t_max),
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': 500, 'ftol': 1e-6}
            )

            hawkes_res = HawkesParams(
                mu=result.x[0],
                alpha=result.x[1],
                beta=result.x[2],
                is_success=result.success,
                log_likelihood=-result.fun
            )

            if not hawkes_res.is_success:
                logger.debug(f"MLE convergence failed: {result.message}. Using best found parameters.")

            return {
                'mu': hawkes_res.mu,
                'alpha': hawkes_res.alpha,
                'beta': hawkes_res.beta,
                'is_success': 1.0 if hawkes_res.is_success else 0.0,
                'log_likelihood': hawkes_res.log_likelihood
            }

        except Exception as e:
            logger.error(f"Error during Hawkes MLE calibration: {e}")
            return {'mu': np.nan, 'alpha': np.nan, 'beta': np.nan, 'is_success': 0.0, 'log_likelihood': np.nan}


if __name__ == '__main__':
    # ---------------------------------------------------------
    # DUMMY TESTING: HAWKES ENGINE CALIBRATION
    # ---------------------------------------------------------
    logger.info("Initializing dummy Hawkes calibration test...")
    
    # Generate a synthetic clustering event sequence (mimicking microstructural panic)
    # Background events (Poisson)
    np.random.seed(42)
    base_events = np.cumsum(np.random.exponential(scale=2.0, size=20))
    
    # Adding a localized "Flash Crash" cluster (burst of events)
    burst_start = base_events[-1] + 1.0
    burst_events = burst_start + np.cumsum(np.random.exponential(scale=0.1, size=50))
    
    # Combine and sort timestamps
    all_timestamps = np.sort(np.concatenate([base_events, burst_events]))
    
    logger.info(f"Generated {len(all_timestamps)} synthetic aggressive orders.")

    # Initialize and calibrate
    calibrator = HawkesMLECalibrator(initial_guess=(0.5, 1.0, 2.0))
    results = calibrator.calibrate(timestamps=all_timestamps)
    
    print("\n--- MLE CALIBRATION RESULTS ---")
    print(f"Base Intensity (mu):    {results['mu']:.4f}")
    print(f"Jump Size (alpha):      {results['alpha']:.4f}")
    print(f"Decay Rate (beta):      {results['beta']:.4f}")
    print(f"Contagion Index (a/b):  {results['alpha']/results['beta']:.4f}  <-- If >= 1.0, critical regime!")
    print(f"Convergence Success:    {bool(results['is_success'])}")
    print(f"Log-Likelihood:         {results['log_likelihood']:.2f}")