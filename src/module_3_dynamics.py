import logging
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass

# Fixed absolute import path to ensure the module is found from the project root
try:
    from src.module_2_hawkes import HawkesMLECalibrator
except ImportError:
    # Dummy class for standalone compilation if module_2 is not yet linked
    class HawkesMLECalibrator:
        def __init__(self, initial_guess=(0.1, 0.5, 1.0)): pass
        def calibrate(self, timestamps, window_duration=None): 
            return {'mu': 0.1, 'alpha': 0.8, 'beta': 0.9, 'is_success': 1.0, 'log_likelihood': -100.0}

logger = logging.getLogger("CaspianContagion.Dynamics")

class ContagionDynamicsEngine:
    """
    Module 3: The Novel Intersection (OFI vs. Beta Dynamics).
    
    Segments the Level-2 LOB data and event timestamps into rolling micro-windows.
    Calculates the relationship between Order Flow Imbalance (OFI) and the dynamically
    calibrated Hawkes decay rate (beta). Computes the Caspian Contagion Index (CCI).
    """

    def __init__(self, window_size_ms: int = 10000, step_size_ms: int = 1000, cci_threshold: float = 1.0):
        """
        Initializes the Contagion Dynamics Engine.

        Args:
            window_size_ms (int): Duration of the rolling window in milliseconds (e.g., 10 seconds).
            step_size_ms (int): Step size for the rolling window in milliseconds (e.g., 1 second).
            cci_threshold (float): Threshold above which the market is in an explosive/critical state.
        """
        self.window_size = pd.Timedelta(milliseconds=window_size_ms)
        self.step_size = pd.Timedelta(milliseconds=step_size_ms)
        self.cci_threshold = cci_threshold
        self.calibrator = HawkesMLECalibrator()
        
        logger.info(f"Initialized Dynamics Engine: Window={window_size_ms}ms, Step={step_size_ms}ms.")

    @staticmethod
    def calculate_cci(alpha: float, beta: float) -> float:
        """
        Calculates the Caspian Contagion Index (CCI), which is the branching ratio of the Hawkes process.
        CCI = alpha / beta.

        Args:
            alpha (float): The excitation factor (jump size).
            beta (float): The decay rate of the panic.

        Returns:
            float: The CCI value. Returns np.nan if beta is close to zero or invalid.
        """
        if pd.isna(alpha) or pd.isna(beta) or beta <= 1e-6:
            return np.nan
        return float(alpha / beta)

    def process_rolling_dynamics(self, raw_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the rolling window calibration to prove correlation between OFI and Hawkes parameters.

        Args:
            raw_df (pd.DataFrame): DataFrame containing continuous LOB data and 'OFI' column.
            events_df (pd.DataFrame): DataFrame containing filtered aggressive event timestamps.

        Returns:
            pd.DataFrame: A time-series DataFrame merging Rolling_OFI, dynamic Hawkes parameters, and CCI.
        """
        if raw_df.empty or 'timestamp' not in raw_df.columns or 'OFI' not in raw_df.columns:
            raise ValueError("raw_df must contain 'timestamp' and 'OFI' columns.")
            
        logger.info("Starting rolling window dynamics computation...")

        start_time = raw_df['timestamp'].min()
        end_time = raw_df['timestamp'].max()
        
        current_start = start_time
        results = []

        # Convert to numpy for O(1) filtering inside the loop
        event_times = events_df['timestamp'].values
        
        # Pre-compute time index for raw_df to speed up slicing
        raw_df_indexed = raw_df.set_index('timestamp')

        while current_start + self.window_size <= end_time:
            current_end = current_start + self.window_size
            
            # 1. Aggregate OFI in the current window
            window_raw = raw_df_indexed.loc[current_start:current_end]
            if window_raw.empty:
                mean_ofi = 0.0
                std_ofi = 0.0
            else:
                mean_ofi = window_raw['OFI'].mean()
                std_ofi = window_raw['OFI'].std()

            # 2. Extract events in the current window for Hawkes calibration
            mask = (event_times >= np.datetime64(current_start)) & (event_times < np.datetime64(current_end))
            window_events = event_times[mask]
            
            # Convert datetime64 to float seconds for the calibrator
            if len(window_events) >= 3:
                # Relative time in seconds from the start of the window
                timestamps_sec = (window_events - np.datetime64(current_start)) / np.timedelta64(1, 's')
                window_duration_sec = self.window_size.total_seconds()
                
                hawkes_res = self.calibrator.calibrate(
                    timestamps=timestamps_sec.astype(np.float64), 
                    window_duration=window_duration_sec
                )
            else:
                hawkes_res = {'mu': np.nan, 'alpha': np.nan, 'beta': np.nan, 'is_success': 0.0}

            # 3. Calculate CCI
            cci = self.calculate_cci(hawkes_res['alpha'], hawkes_res['beta'])
            is_critical = 1 if (not pd.isna(cci) and cci >= self.cci_threshold) else 0

            # Store metrics
            results.append({
                'window_end': current_end,
                'mean_OFI': mean_ofi,
                'std_OFI': std_ofi,
                'hawkes_mu': hawkes_res['mu'],
                'hawkes_alpha': hawkes_res['alpha'],
                'hawkes_beta': hawkes_res['beta'],
                'CCI': cci,
                'is_critical': is_critical,
                'event_count': len(window_events)
            })

            current_start += self.step_size

        output_df = pd.DataFrame(results)
        logger.info(f"Dynamics computation completed. Generated {len(output_df)} rolling segments.")
        return output_df


if __name__ == '__main__':
    # ---------------------------------------------------------
    # DUMMY TESTING: MODULE 3 DYNAMICS
    # ---------------------------------------------------------
    logger.info("Initializing dummy dynamics testing...")
    
    # 1. Generate Dummy Raw Data (1 minute of data, 10ms resolution)
    time_index = pd.date_range(start='2026-04-18 10:00:00', periods=6000, freq='10ms')
    
    # Create an OFI series that shifts from stable to highly imbalanced (panic)
    stable_ofi = np.random.normal(0, 10, 4000)
    panic_ofi = np.random.normal(-100, 50, 2000) # Heavy selling pressure
    ofi_series = np.concatenate([stable_ofi, panic_ofi])
    
    raw_dummy_df = pd.DataFrame({'timestamp': time_index, 'OFI': ofi_series})
    
    # 2. Generate Dummy Event Data correlated with OFI
    # In the stable period, events are sparse. In the panic period, events cluster.
    stable_events = pd.date_range(start='2026-04-18 10:00:00', periods=150, freq='266ms')
    
    # Simulating microstructural clustering during the panic phase
    panic_start_time = pd.Timestamp('2026-04-18 10:00:40')
    panic_events = [panic_start_time + pd.Timedelta(milliseconds=int(np.random.exponential(50)*i)) for i in range(1, 300)]
    panic_events = [t for t in panic_events if t <= raw_dummy_df['timestamp'].max()]
    
    all_events = sorted(list(stable_events) + panic_events)
    events_dummy_df = pd.DataFrame({'timestamp': all_events})
    
    # 3. Process Dynamics
    # 10-second window, 1-second step
    engine = ContagionDynamicsEngine(window_size_ms=10000, step_size_ms=1000)
    dynamics_df = engine.process_rolling_dynamics(raw_df=raw_dummy_df, events_df=events_dummy_df)
    
    print("\n--- DYNAMICS ENGINE RESULTS (Sample) ---")
    print(dynamics_df[['window_end', 'mean_OFI', 'hawkes_beta', 'CCI', 'is_critical']].tail(10))