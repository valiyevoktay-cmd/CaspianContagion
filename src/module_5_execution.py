import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Optional

# Import risk constants
try:
    from src.config import LATENCY_THRESHOLD_MS, CCI_THRESHOLD
except ImportError:
    LATENCY_THRESHOLD_MS = 500
    CCI_THRESHOLD = 0.95

logger = logging.getLogger("CaspianContagion.Execution")

class MarketMakerEngine:
    """
    Module 5: HFT Execution & Risk Management Engine.
    Simulates order fills, inventory liquidation via MWAP (Mean Weighted Average Price),
    and monitors network latency for stale data protection.
    """
    def __init__(self, tick_size: float = 0.01, lot_size: int = 100, slippage_penalty: float = 0.0001):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.slippage_penalty = slippage_penalty # Deterministic penalty for market impact
        
        # Internal State
        self.inventory = 0      # Position in contracts
        self.cash = 0.0         # Realized PnL (Cash balance)
        self.is_active = True   # Engine connectivity status
        
    def reset(self):
        """Resets the engine state for new simulation/session."""
        self.inventory = 0
        self.cash = 0.0
        self.is_active = True

    def _walk_the_book(self, position_size: int, row: pd.Series, side: str) -> float:
        """
        Calculates MWAP (Mean Weighted Average Price) by consuming L2 liquidity.
        side: 'bids' to liquidate Long, 'asks' to liquidate Short.
        """
        remaining = abs(position_size)
        total_value = 0.0
        
        prefix = 'bid' if side == 'bids' else 'ask'
        
        for i in range(1, 6): # Iterate through 5 levels of L2 depth
            p_col = f'{prefix}_p_{i}'
            v_col = f'{prefix}_v_{i}'
            
            if p_col not in row or v_col not in row:
                break
                
            level_price = row[p_col]
            level_vol = row[v_col]
            
            fill = min(remaining, level_vol)
            total_value += fill * level_price
            remaining -= fill
            
            if remaining <= 0:
                break
        
        # If book is too thin, fill remaining at the last known price with extra penalty
        if remaining > 0:
            total_value += remaining * (level_price * (1 - self.slippage_penalty if side == 'bids' else 1 + self.slippage_penalty))
            
        return total_value / abs(position_size)

    def execute_emergency_exit(self, row: pd.Series):
        """
        Liquidates entire inventory immediately using L2 liquidity walk.
        Simulates the 'Killswitch' reaction.
        """
        if self.inventory == 0:
            return

        side = 'bids' if self.inventory > 0 else 'asks'
        mwap_price = self._walk_the_book(self.inventory, row, side)
        
        # Apply additional latency/panic penalty
        execution_price = mwap_price * (1 - self.slippage_penalty if side == 'bids' else 1 + self.slippage_penalty)
        
        # Close position
        self.cash += (execution_price * self.inventory) if self.inventory < 0 else (execution_price * self.inventory)
        # Note: Logic check - selling long increases cash, buying back short decreases cash
        if self.inventory > 0: # Closing Long
            self.cash = self.cash # Already handled by price * inv
        
        logger.warning(f"EMERGENCY EXIT: Closed {self.inventory} units at {execution_price:.2f}")
        self.inventory = 0

    def simulate_step(self, row: pd.Series) -> Dict:
        """
        Simulates one HFT step: Latency check -> Risk Evaluation -> Execution.
        """
        state = row.get('state', 'NORMAL')
        mid_price = row['mid_price']
        
        # 1. Latency Tracking (Stale Data Protection)
        # Compare current system time with exchange event time (T)
        current_ts_ms = time.time() * 1000
        exch_ts = row.get('exch_ts', current_ts_ms)
        latency = current_ts_ms - exch_ts
        
        if latency > LATENCY_THRESHOLD_MS and state != 'HALTED':
            state = 'WARNING'
            action = "LATENCY_LOB_STALE"
        else:
            action = "QUOTING"

        # 2. Risk Response & Auto-Halt
        if state == 'HALTED':
            if self.inventory != 0:
                self.execute_emergency_exit(row)
            action = "KILLSWITCH_ENGAGED"
            fill_prob = 0.0
        elif state == 'WARNING':
            fill_prob = 0.05 # Reduce exposure
            action = "WIDENING_SPREADS"
        else:
            fill_prob = 0.25 # Normal market making probability
            action = "ACTIVE_MARKET_MAKING"

        # 3. Market Making Fill Simulation (Passive)
        if state == 'NORMAL':
            # Simulate Bid Fill (We buy)
            if np.random.random() < fill_prob:
                self.inventory += self.lot_size
                self.cash -= (row['bid_p_1'] * self.lot_size)
            
            # Simulate Ask Fill (We sell)
            if np.random.random() < fill_prob:
                self.inventory -= self.lot_size
                self.cash += (row['ask_p_1'] * self.lot_size)

        # 4. PnL Attribution
        # Equity = Realized Cash + Unrealized value of inventory at Mid Price
        unrealized_pnl = (self.inventory * mid_price) + self.cash
        
        return {
            "inventory": self.inventory,
            "realized_pnl": self.cash,
            "equity": unrealized_pnl,
            "mm_action": action,
            "latency_ms": latency
        }

    def run_backtest(self, tick_df: pd.DataFrame, dynamics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the simulation over historical or live data buffers.
        """
        logger.info("Starting Execution Engine simulation...")
        self.reset()
        
        # Align risk dynamics with market ticks
        sim_df = pd.merge_asof(
            tick_df.sort_values('timestamp'),
            dynamics_df[['window_end', 'state']].sort_values('window_end'),
            left_on='timestamp',
            right_on='window_end',
            direction='backward'
        )
        
        results = []
        for _, row in sim_df.iterrows():
            results.append(self.simulate_step(row))
            
        return pd.DataFrame(results, index=sim_df.index)