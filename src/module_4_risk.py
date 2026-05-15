import logging
import pandas as pd
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger("CaspianContagion.RiskManager")

class MarketState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HALTED = "HALTED"
    COOLDOWN = "COOLDOWN"

class KillswitchEngine:
    """
    Module 4: HFT Risk Manager & Killswitch.
    Monitors the Caspian Contagion Index (CCI) and manages the trading system's state.
    """

    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 1.0, cooldown_periods: int = 5):
        """
        Args:
            warning_threshold: CCI level to trigger spread widening.
            critical_threshold: CCI level to trigger total liquidity pull.
            cooldown_periods: Number of safe windows required before trading resumes.
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.cooldown_periods = cooldown_periods
        
        # Initial State
        self.current_state = MarketState.NORMAL
        self.cooldown_counter = 0

    def process_tick(self, timestamp: pd.Timestamp, cci: float, ofi: float) -> Dict[str, Any]:
        """
        Evaluates the current microstructural risk and returns the required trading action.
        """
        action = "CONTINUE_QUOTING"
        
        # Handle missing or calculating values
        if pd.isna(cci):
            return {"timestamp": timestamp, "state": self.current_state.value, "action": "WAITING_DATA"}

        # 1. HALT LOGIC (Absolute Priority)
        if cci >= self.critical_threshold:
            if self.current_state != MarketState.HALTED:
                logger.warning(f"[{timestamp}] KILLSWITCH TRIGGERED! CCI: {cci:.3f}")
            
            self.current_state = MarketState.HALTED
            self.cooldown_counter = self.cooldown_periods # Reset cooldown timer
            action = "PULL_ALL_LIQUIDITY"

        # 2. COOLDOWN LOGIC (Recovery phase)
        elif self.current_state == MarketState.HALTED or self.current_state == MarketState.COOLDOWN:
            if cci < self.warning_threshold:
                self.current_state = MarketState.COOLDOWN
                self.cooldown_counter -= 1
                action = f"COOLDOWN_ACTIVE ({self.cooldown_counter} left)"
                
                if self.cooldown_counter <= 0:
                    logger.info(f"[{timestamp}] Cooldown complete. Resuming NORMAL operations.")
                    self.current_state = MarketState.NORMAL
                    action = "RESUME_QUOTING"
            else:
                # If CCI spikes back up during cooldown, reset the timer
                self.cooldown_counter = self.cooldown_periods
                action = "COOLDOWN_RESET_DUE_TO_INSTABILITY"

        # 3. WARNING LOGIC
        elif cci >= self.warning_threshold:
            self.current_state = MarketState.WARNING
            # Determine market direction based on OFI to skew quotes
            direction = "ASK" if ofi < 0 else "BID" 
            action = f"WIDEN_SPREADS_SKEW_{direction}"

        # 4. NORMAL LOGIC
        else:
            self.current_state = MarketState.NORMAL
            action = "MAINTAIN_TIGHT_SPREADS"

        return {
            "timestamp": timestamp,
            "cci": round(cci, 3),
            "state": self.current_state.value,
            "action": action
        }

    def batch_evaluate(self, dynamics_df: pd.DataFrame, tick_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the risk engine across a historical DataFrame for backtesting.
        """
        # Surgical Fix: Enforce identical datetime precision (ns) before asof merge
        dyn_prep = dynamics_df.copy()
        dyn_prep['window_end'] = pd.to_datetime(dyn_prep['window_end']).astype('datetime64[ns]')
        
        tick_prep = tick_df[['timestamp', 'OFI']].copy()
        tick_prep['timestamp'] = pd.to_datetime(tick_prep['timestamp']).astype('datetime64[ns]')

        # Map nearest OFI to dynamics timeframe for backtesting
        merged_df = pd.merge_asof(
            dyn_prep.sort_values('window_end'),
            tick_prep.sort_values('timestamp'),
            left_on='window_end',
            right_on='timestamp',
            direction='backward'
        )

        results = []
        for _, row in merged_df.iterrows():
            res = self.process_tick(row['window_end'], row['CCI'], row['OFI'])
            results.append(res)
            
        return pd.DataFrame(results)