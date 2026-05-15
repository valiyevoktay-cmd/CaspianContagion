"""
Caspian Contagion | Configuration Module
Defines global constants for WebSocket connectivity, market streams, and system limits.
"""

# --- CONNECTIVITY SETTINGS ---
# Binance Futures WebSocket Base URL
WSS_URL = "wss://fstream.binance.com/ws/"

# Market stream: 20 levels of depth updated every 100ms
# Format: {symbol}@depth20@100ms
STREAM_NAME = "{symbol}@depth20@100ms"

# Default trading pair (Binance format: lowercase)
DEFAULT_SYMBOL = "btcusdt"

# --- SYSTEM PERFORMANCE & MEMORY ---
# Maximum number of ticks stored in the rolling buffer (deque)
# 5000 ticks at 100ms intervals = ~8.3 minutes of high-density micro-history
MAX_BUFFER_SIZE = 5000

# Seconds to wait before attempting a reconnection
RECONNECT_DELAY = 5

# --- PERFORMANCE FLAGS ---
# Use 'orjson' for ultra-fast JSON parsing (requires: pip install orjson)
USE_ORJSON = True

# --- QUANTITATIVE DEFAULTS ---
# Number of LOB levels to process in the Hawkes engine
OB_LEVELS = 5

# --- RISK & EXECUTION (Step 5) ---
# Critical threshold for the Caspian Contagion Index
CCI_THRESHOLD = 0.95

# Maximum allowed network latency in milliseconds before triggering WARNING state
# If (Current Time - Exchange Event Time) > 500ms, data is considered stale.
LATENCY_THRESHOLD_MS = 500

# Base slippage penalty (0.01%) applied to the MWAP during emergency liquidation
# Mimics market impact and loss of liquidity during panic events.
SLIPPAGE_PENALTY = 0.0001

# Maximum institutional inventory limit for the market maker simulation
MAX_POSITION_SIZE = 1000