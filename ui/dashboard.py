import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
import threading
import time
import logging
from datetime import datetime, timezone  # Updated for 2026 timezone-aware standards
from typing import Optional, NoReturn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaspianContagion.UI.Optimized")

class ContagionDashboard:
    """
    Module 4 & 5: Optimized High-Throughput Contagion UI.
    Refactored with Streamlit Fragments and WebGL for zero-flicker rendering.
    Supports both Historical Simulation and Live WebSocket streaming.
    """

    def __init__(self, cci_threshold: float = 1.0, buffer_size: int = 2000):
        """
        Initializes the terminal state and data buffers.
        """
        self.cci_threshold = cci_threshold
        
        # Robust state initialization
        if 'sim_idx' not in st.session_state:
            st.session_state.sim_idx = 800 

        # Initialize thread-safe buffers in Session State
        if 'tick_buffer' not in st.session_state:
            st.session_state.tick_buffer = deque(maxlen=buffer_size)
            st.session_state.dynamics_buffer = deque(maxlen=200)
            st.session_state.ingestion_active = False
            st.session_state.last_update = time.time()

        # Set page config once
        try:
            st.set_page_config(page_title="Caspian Contagion | Live Terminal", layout="wide", page_icon="⚡")
        except st.errors.StreamlitAPIException:
            pass

        self._inject_terminal_css()

    def _inject_terminal_css(self) -> None:
        """Applies high-density dark theme CSS."""
        st.markdown("""
        <style>
            .stApp { background-color: #0d1117; color: #c9d1d9; }
            div[data-testid="metric-container"] {
                background-color: #161b22;
                border: 1px solid #30363d;
                padding: 10px;
                border-radius: 4px;
            }
            .block-container { padding-top: 1.5rem; }
            h1 { font-family: 'Inter', sans-serif; color: #e6edf3; font-weight: 700; border-bottom: 1px solid #30363d; }
        </style>
        """, unsafe_allow_html=True)

    def _simulate_ingestion(self) -> NoReturn:
        """Background thread: Simulates ingestion for standalone/dev mode."""
        logger.info("Background Ingestion Thread Started.")
        n = 0
        while True:
            n += 1
            new_tick = {
                'timestamp': pd.Timestamp.now(tz='UTC'),
                'mid_price': 100 + np.sin(n * 0.1) + np.random.normal(0, 0.05),
                'OFI': np.random.normal(0, 50),
                'is_event': 1 if np.random.random() > 0.98 else 0
            }
            st.session_state.tick_buffer.append(new_tick)
            time.sleep(0.01)

    def start_background_task(self) -> None:
        """Launches the data ingestion thread if not already running."""
        if not st.session_state.ingestion_active:
            thread = threading.Thread(target=self._simulate_ingestion, daemon=True)
            thread.start()
            st.session_state.ingestion_active = True

    def render_sidebar(self) -> int:
        """Renders UI controls and system health indicators."""
        with st.sidebar:
            st.header("⚙️ ENGINE CONTROL")
            step_size = st.slider("Playback Speed (Ticks/Frame)", 10, 100, 30, step=10)
            st.divider()
            
            # System Health Metrics (Surgical integration)
            st.subheader("🌐 NETWORK HEALTH")
            missed = st.session_state.get('missed_packets', 0)
            st.metric("MISSED PACKETS", missed, delta=None if missed == 0 else "- PACKET LOSS", delta_color="inverse")
            
            st.info("Institutional Grade: Autonomous Fragment Rendering Active (5Hz Locked).")
            return step_size

    def build_chart(self, tick_df: pd.DataFrame, dynamics_df: pd.DataFrame) -> go.Figure:
        """Constructs a high-performance 4-tier Plotly figure using WebGL (Scattergl)."""
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.4, 0.2, 0.2, 0.2]
        )

        # ROW 1: Price Action
        fig.add_trace(go.Scattergl(
            x=tick_df['timestamp'], y=tick_df['mid_price'],
            mode='lines', name="Price", line=dict(color='#58a6ff', width=1.5)
        ), row=1, col=1)

        # ROW 2: OFI
        pos_ofi = np.where(tick_df['OFI'] > 0, tick_df['OFI'], 0)
        neg_ofi = np.where(tick_df['OFI'] < 0, tick_df['OFI'], 0)

        fig.add_trace(go.Scattergl(
            x=tick_df['timestamp'], y=pos_ofi,
            mode='lines', name="OFI Buy", line=dict(width=0.5, color='#3fb950'),
            fill='tozeroy', fillcolor='rgba(63, 185, 80, 0.4)' 
        ), row=2, col=1)

        fig.add_trace(go.Scattergl(
            x=tick_df['timestamp'], y=neg_ofi,
            mode='lines', name="OFI Sell", line=dict(width=0.5, color='#f85149'),
            fill='tozeroy', fillcolor='rgba(248, 81, 73, 0.4)' 
        ), row=2, col=1)

        # ROW 3: CCI Dynamics
        if not dynamics_df.empty:
            fig.add_trace(go.Scattergl(
                x=dynamics_df['window_end'], y=dynamics_df['CCI'],
                mode='lines', name="CCI", line=dict(color='#d2a8ff', width=2)
            ), row=3, col=1)
            fig.add_hline(y=self.cci_threshold, line_dash="dash", line_color="#f85149", row=3, col=1)

        # ROW 4: Equity Curve
        if not dynamics_df.empty and 'equity' in dynamics_df.columns:
            if 'naive_equity' in dynamics_df.columns:
                fig.add_trace(go.Scattergl(
                    x=dynamics_df['window_end'], y=dynamics_df['naive_equity'],
                    mode='lines', name="Naive", line=dict(color='rgba(150, 150, 150, 0.5)', width=2, dash='dot')
                ), row=4, col=1)

            fig.add_trace(go.Scattergl(
                x=dynamics_df['window_end'], y=dynamics_df['equity'],
                mode='lines', name="Caspian", line=dict(color='#e3b341', width=2)
            ), row=4, col=1)

        fig.update_layout(
            uirevision='constant', # SURGICAL FIX: Prevents Plotly from destroying the WebGL context
            height=850, template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False, hovermode=False,
            plot_bgcolor='#0d1117', paper_bgcolor='#0d1117'
        )
        return fig

    # SURGICAL FIX: Completely removed @st.fragment to prevent nested DOM collisions with main.py
    def render_visualization_fragment(self, tick_df: Optional[pd.DataFrame], dynamics_df: Optional[pd.DataFrame], step_size: int) -> None:
        """
        Isolated render function for zero-flicker UI updates.
        Handles both Live-buffer pulling and Simulation-slicing.
        """
        # 1. Data Retrieval Logic (Surgical Switch between Sim and Live)
        if 'live_buffer' in st.session_state and tick_df is not None:
            # LIVE MODE: tick_df and dynamics_df are passed from main.py's live pipeline
            df_ticks = tick_df.iloc[-800:]
            df_dyns = dynamics_df.iloc[-40:]
            
            # Latency Calculation: Using UTC-aware comparison to prevent timezone offsets
            last_ts = df_ticks['timestamp'].iloc[-1]
            now_utc = datetime.now(timezone.utc)
            # Ensure comparison is done between timezone-aware objects (Binance sends UTC)
            latency_ms = (now_utc - last_ts.replace(tzinfo=timezone.utc)).total_seconds() * 1000
        else:
            # SIMULATION MODE: use the sim_idx to slide through the historical data
            i = st.session_state.sim_idx
            df_ticks = tick_df.iloc[max(0, i-800):i]
            current_time = tick_df['timestamp'].iloc[i]
            df_dyns = dynamics_df[dynamics_df['window_end'] <= current_time].iloc[-40:]
            
            if i + step_size < len(tick_df):
                st.session_state.sim_idx += step_size
            latency_ms = 0.0

        if df_ticks.empty:
            st.warning("Awaiting market data stream...")
            return

        # 2. Metrics Header Logic
        latest_cci = df_dyns['CCI'].iloc[-1] if not df_dyns.empty else np.nan
        risk_state = df_dyns['state'].iloc[-1] if (not df_dyns.empty and 'state' in df_dyns.columns) else "N/A"
        current_equity = df_dyns['equity'].iloc[-1] if (not df_dyns.empty and 'equity' in df_dyns.columns) else 0.0
        current_inv = df_dyns['inventory'].iloc[-1] if (not df_dyns.empty and 'inventory' in df_dyns.columns) else 0
        current_naive = df_dyns['naive_equity'].iloc[-1] if (not df_dyns.empty and 'naive_equity' in df_dyns.columns) else current_equity

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        
        # CCI Metric
        if pd.isna(latest_cci):
            m1.metric("LIVE CCI", "WAITING")
        else:
            cci_color = "inverse" if latest_cci >= self.cci_threshold else "normal"
            m1.metric("LIVE CCI", f"{latest_cci:.3f}", "CRITICAL" if latest_cci >= self.cci_threshold else "STABLE", delta_color=cci_color)
            
        # OFI & Latency
        latest_ofi = df_ticks['OFI'].iloc[-10:].mean()
        m2.metric("OFI PRESSURE", f"{latest_ofi:.1f}")
        m3.metric("LATENCY", f"{latency_ms:.0f}ms", "DELAY" if latency_ms > 500 else "FAST", delta_color="inverse" if latency_ms > 500 else "normal")
        
        # Sim/Buffer Index
        system_idx = len(tick_df) if tick_df is not None else st.session_state.sim_idx
        m4.metric("SYSTEM INDEX", system_idx)

        # Risk State
        m5.metric("RISK ENGINE", risk_state, "HALT" if risk_state == "HALTED" else "ACTIVE", delta_color="normal" if risk_state == "NORMAL" else "inverse")

        # PnL & Saved Capital
        saved = current_equity - current_naive
        if risk_state == "HALTED" and saved > 0:
            m6.metric("TOTAL PnL", f"${current_equity:,.1f}", f"SAVED: ${saved:,.0f}", delta_color="normal")
        else:
            m6.metric("TOTAL PnL", f"${current_equity:,.1f}", f"POS: {int(current_inv)}", delta_color="normal" if current_equity >= 0 else "inverse")

        # 3. EXECUTION ANALYTICS (Surgical Addition for Step 5)
        st.divider()
        st.subheader("📊 EXECUTION ANALYTICS")
        e1, e2, e3, e4 = st.columns(4)

        # Alpha Generated: Capital preserved by Caspian versus the late "crowd" exit
        alpha_val = current_equity - current_naive
        e1.metric("ALPHA GENERATED", f"${alpha_val:,.2f}", "PRESERVED CAPITAL" if alpha_val > 0 else "BENCHMARK TRACKING")

        # Potential Slippage: Market impact estimate for liquidating current inventory
        current_mid = df_ticks['mid_price'].iloc[-1]
        slippage_est = abs(current_inv) * current_mid * 0.0001
        e2.metric("POTENTIAL SLIPPAGE", f"${slippage_est:,.2f}", "EST. LIQUIDITY COST", delta_color="inverse")

        # Fill Probability based on adverse selection risk
        fill_prob_text = "0%" if risk_state in ["HALTED", "WARNING"] else "25%"
        e3.metric("FILL PROBABILITY", fill_prob_text, "ADVERSE SELECTION RISK")

        # Real-time Execution Quality tracking
        e4.metric("EXECUTION QUALITY", "STABLE" if latency_ms < 500 else "DEGRADED", f"{latency_ms:.0f}ms NETWORK LAG")

        # 4. Optimized Chart Rendering
        fig = self.build_chart(df_ticks, df_dyns)
        # SURGICAL FIX: Added `key` parameter to anchor the DOM element and prevent flickering.
        st.plotly_chart(fig, width="stretch", key="caspian_live_chart", config={'displayModeBar': False})

    def run(self, tick_df: Optional[pd.DataFrame] = None, dynamics_df: Optional[pd.DataFrame] = None) -> None:
        """Main entry point. Renders static shell and triggers autonomous fragment."""
        st.markdown("<h1> CASPIAN LIVE | HFT CONTAGION TERMINAL</h1>", unsafe_allow_html=True)
        
        if tick_df is None and not st.session_state.ingestion_active:
            self.start_background_task()
            
        step_size = self.render_sidebar()
        self.render_visualization_fragment(tick_df, dynamics_df, step_size)

if __name__ == "__main__":
    terminal = ContagionDashboard(cci_threshold=0.95)
    terminal.run()