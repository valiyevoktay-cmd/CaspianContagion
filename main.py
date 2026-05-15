import logging
import numpy as np
import pandas as pd
import streamlit as st
import time
import os
import threading
import asyncio
from collections import deque

# Import Caspian Architecture Modules
from src.module_1_ingestion import OrderBookProcessor
from src.module_3_dynamics import ContagionDynamicsEngine
from src.module_4_risk import KillswitchEngine
from src.module_5_execution import MarketMakerEngine
from src.utils.data_adapter import BinanceLiveAdapter, ExchangeDataAdapter
from ui.dashboard import ContagionDashboard
from src.config import (
    MAX_BUFFER_SIZE, 
    DEFAULT_SYMBOL, 
    OB_LEVELS, 
    CCI_THRESHOLD, 
    SLIPPAGE_PENALTY,
    LATENCY_THRESHOLD_MS
)

# Configure production logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CaspianContagion.Main")

# --- PERSISTENT GLOBAL BRIDGE ---
@st.cache_resource
def get_global_bridge():
    """Creates a persistent buffer and a lock that survives Streamlit reruns."""
    return deque(maxlen=MAX_BUFFER_SIZE), threading.Lock()

GLOBAL_RAW_BUFFER, BUFFER_LOCK = get_global_bridge()

def live_callback(data: dict):
    """
    Thread-safe callback executed for every new WebSocket packet.
    Handles data ingestion into the global buffer.
    """
    with BUFFER_LOCK:
        GLOBAL_RAW_BUFFER.append(data)

# --- SHARED STATE INITIALIZATION ---
if 'live_buffer' not in st.session_state:
    st.session_state.live_buffer = deque(maxlen=MAX_BUFFER_SIZE)
    st.session_state.missed_packets = 0
    st.session_state.ingestion_started = False

def start_live_ingestion():
    """
    Wrapper function to run the async adapter in a dedicated background thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    adapter = BinanceLiveAdapter(symbol=DEFAULT_SYMBOL, callback=live_callback)
    
    try:
        loop.run_until_complete(adapter.connect())
    except Exception as e:
        logger.error(f"Ingestion thread failed: {e}")

@st.cache_resource
def launch_ingestion_thread():
    """Guarantees that only ONE ingestion thread is ever launched per server session."""
    thread = threading.Thread(target=start_live_ingestion, daemon=True)
    thread.start()
    logger.info("CORE: Background Ingestion Thread Permanently Established.")
    return True

def load_or_generate_data(n_ticks: int = 6000) -> pd.DataFrame:
    """
    Generates a mathematically clean synthetic Level 2 Limit Order Book.
    Simulates a stable Mean-Reverting regime followed by a structural Flash Crash.
    """
    logger.info("Generating clean synthetic L2 data with baseline noise...")
    np.random.seed(42)
    levels = 5
    
    # 1. Base stable price (Random Walk)
    price_changes = np.random.normal(loc=0.0, scale=0.005, size=n_ticks)
    mid_price = 100.0 + np.cumsum(price_changes)
    
    # 2. Inject structural Flash Crash (Tick 4000 to 4500)
    crash_start, crash_end = 4000, 4500
    
    # Create a permanent price drop trend
    crash_price_drop = np.random.normal(loc=-0.03, scale=0.01, size=crash_end-crash_start)
    cumulative_drop = np.cumsum(crash_price_drop)
    
    # Apply the drop to the mid_price permanently
    mid_price[crash_start:crash_end] += cumulative_drop
    mid_price[crash_end:] += cumulative_drop[-1] 
    
    data_dict = {'timestamp': pd.date_range(start='2026-04-18 10:00:00', periods=n_ticks, freq='10ms')}
    
    # 3. Populate LOB volumes
    for i in range(1, levels + 1):
        spread = 0.05 * i
        data_dict[f'bid_p_{i}'] = mid_price - spread
        data_dict[f'ask_p_{i}'] = mid_price + spread
        
        # Stable baseline liquidity
        data_dict[f'bid_v_{i}'] = np.abs(np.random.normal(loc=100.0 + i*20, scale=5.0, size=n_ticks))
        data_dict[f'ask_v_{i}'] = np.abs(np.random.normal(loc=100.0 + i*20, scale=5.0, size=n_ticks))

    df = pd.DataFrame(data_dict)
    
    # Surgical Fix: Enforce nanosecond precision to match Live pipeline and avoid MergeError
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')

    # 4. Simulate Liquidity Pulling (The Contagion Trigger)
    df.loc[crash_start:crash_end, 'bid_v_1'] *= 0.01
    df.loc[crash_start:crash_end, 'bid_v_2'] *= 0.05
    df.loc[crash_start:crash_end, 'ask_v_1'] *= 5.0

    # 5. Inject background micro-events to maintain MLE stability
    safe_indices = list(range(0, crash_start - 100)) + list(range(crash_end + 100, n_ticks))
    background_events = np.random.choice(safe_indices, size=250, replace=False)
    df.loc[background_events, 'bid_v_1'] *= 3.0 
    
    return df

# SURGICAL FIX: Removed @st.cache_data so the live pipeline recalculates dynamically
def run_quant_pipeline_live(_data_list: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes the current LIVE buffer and returns results for the UI.
    Integrates MWAP execution and Alpha tracking.
    """
    raw_df = pd.DataFrame(_data_list)
    
    if len(raw_df) < 100:
        return raw_df, pd.DataFrame()

    # Surgical Fix: Enforce nanosecond precision to prevent MergeError in Risk/Execution modules
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp']).astype('datetime64[ns]')

    # Module 1: Ingestion & OFI
    processor = OrderBookProcessor(levels=OB_LEVELS)
    ofi_series = processor.calculate_ofi(raw_df)
    raw_df['OFI'] = ofi_series
    market_events = processor.extract_market_events(raw_df, ofi_series, threshold_z=1.5) 
    
    enriched_data = raw_df
    enriched_data['is_event'] = 0
    enriched_data.loc[enriched_data['timestamp'].isin(market_events['timestamp']), 'is_event'] = 1
    enriched_data['mid_price'] = (enriched_data['bid_p_1'] + enriched_data['ask_p_1']) / 2.0
    
    # Module 3: Rolling Hawkes Dynamics
    engine = ContagionDynamicsEngine(window_size_ms=10000, step_size_ms=1000, cci_threshold=CCI_THRESHOLD)
    dynamics_df = engine.process_rolling_dynamics(raw_df=enriched_data, events_df=market_events)
    
    # Surgical Fix: Ensure dynamics timestamps also match nanosecond precision
    dynamics_df['window_end'] = pd.to_datetime(dynamics_df['window_end']).astype('datetime64[ns]')

    # Market State Persistence
    cols_to_fill = ['hawkes_mu', 'hawkes_alpha', 'hawkes_beta', 'CCI']
    dynamics_df[cols_to_fill] = dynamics_df[cols_to_fill].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    # Module 4: Killswitch
    risk_engine = KillswitchEngine(warning_threshold=0.8, critical_threshold=CCI_THRESHOLD, cooldown_periods=5)
    risk_df = risk_engine.batch_evaluate(dynamics_df=dynamics_df, tick_df=enriched_data)
    
    dynamics_df = pd.merge(dynamics_df, risk_df[['timestamp', 'state']], 
                           left_on='window_end', right_on='timestamp', how='left').drop(columns=['timestamp'])
    
    # Module 5: Execution Simulation (Caspian with MWAP & Slippage)
    mm_engine = MarketMakerEngine(lot_size=10, slippage_penalty=SLIPPAGE_PENALTY)
    
    np.random.seed(123) # Sync random fills for benchmark comparison
    pnl_df = mm_engine.run_backtest(tick_df=enriched_data, dynamics_df=dynamics_df)
    
    # Benchmark: Naive Execution (No Killswitch)
    naive_dyn = dynamics_df.copy()
    naive_dyn['state'] = 'NORMAL'
    np.random.seed(123)
    naive_pnl_df = mm_engine.run_backtest(tick_df=enriched_data, dynamics_df=naive_dyn)

    # Final Merge for UI with full PnL Attribution
    dynamics_df['equity'] = pnl_df.reindex(dynamics_df.index)['equity'].ffill().values
    dynamics_df['inventory'] = pnl_df.reindex(dynamics_df.index)['inventory'].ffill().values
    dynamics_df['naive_equity'] = naive_pnl_df.reindex(dynamics_df.index)['equity'].ffill().values
    
    return enriched_data, dynamics_df

if __name__ == "__main__":
    # 1. Start Ingestion Thread (Singleton Resource)
    launch_ingestion_thread()

    # 2. Warm-up UI Barrier (Check Cached Global Buffer)
    with BUFFER_LOCK:
        buffer_len = len(GLOBAL_RAW_BUFFER)

    if buffer_len < 200:
        st.markdown("<h1 style='text-align: center;'>⚡ CASPIAN CONTAGION IS WARMING UP</h1>", unsafe_allow_html=True)
        st.progress(buffer_len / 200)
        st.info(f"Syncing market microstructure from global bridge... ({buffer_len}/200 ticks)")
        time.sleep(1)
        st.rerun()

    # 3. Static UI Shell (Rendered ONCE, never flickers)
    ui = ContagionDashboard(cci_threshold=CCI_THRESHOLD)
    st.markdown("<h1> CASPIAN LIVE | HFT CONTAGION TERMINAL</h1>", unsafe_allow_html=True)
    step_size = ui.render_sidebar()

    # 4. Autonomous Zero-Flicker Fragment (Updates every 1 second)
    @st.fragment(run_every=1)
    def live_terminal_loop():
        with BUFFER_LOCK:
            latest_snapshot = list(GLOBAL_RAW_BUFFER)
        
        # Heavy quant math runs cleanly inside the isolated fragment
        tick_df, dynamics_df = run_quant_pipeline_live(latest_snapshot)
        
        # Push fresh data to charts without reloading the whole page
        ui.render_visualization_fragment(tick_df, dynamics_df, step_size)

    # Engage the HFT Loop
    live_terminal_loop()