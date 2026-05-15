import pandas as pd
import logging
import asyncio
import websockets
import time
from datetime import datetime
from typing import Optional, Dict, List, Callable, NoReturn

# Import system config for WebSocket URLs and performance flags
try:
    from src.config import WSS_URL, STREAM_NAME, OB_LEVELS, USE_ORJSON, RECONNECT_DELAY
    if USE_ORJSON:
        import orjson as json
    else:
        import json
except ImportError:
    import json
    WSS_URL = "wss://fstream.binance.com/ws/"
    STREAM_NAME = "{symbol}@depth20@100ms"
    OB_LEVELS = 5
    RECONNECT_DELAY = 5

logger = logging.getLogger("CaspianContagion.DataAdapter")

class ExchangeDataAdapter:
    """
    Universal L2 Order Book Adapter.
    Transforms raw exchange CSV dumps (e.g., Binance, Tardis) into Caspian Alpha format.
    """
    def __init__(self, file_path: str, levels: int = 5):
        self.file_path = file_path
        self.levels = levels

    def load_and_format(self, limit_rows: Optional[int] = 10000) -> pd.DataFrame:
        """
        Loads the CSV and standardizes column names.
        """
        logger.info(f"Loading historical L2 data from: {self.file_path}")
        
        try:
            # Read CSV (adjust memory usage for massive files)
            df = pd.read_csv(self.file_path, nrows=limit_rows)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.file_path}")
            raise

        # 1. Standardize Timestamp
        time_col = next((col for col in ['timestamp', 'local_timestamp', 'time'] if col in df.columns), None)
        if time_col:
            if pd.api.types.is_numeric_dtype(df[time_col]):
                unit = 'us' if df[time_col].max() > 1e13 else 'ms'
                df['timestamp'] = pd.to_datetime(df[time_col], unit=unit)
            elif time_col != 'timestamp':
                df.rename(columns={time_col: 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("No valid timestamp column found in CSV.")

        # 2. Map Price and Volume Columns
        rename_map = {}
        for i in range(self.levels):
            rename_map[f'bids[{i}].price'] = f'bid_p_{i+1}'
            rename_map[f'bids[{i}].amount'] = f'bid_v_{i+1}'
            rename_map[f'bid_price_{i}'] = f'bid_p_{i+1}'
            rename_map[f'bid_size_{i}'] = f'bid_v_{i+1}'
            
            rename_map[f'asks[{i}].price'] = f'ask_p_{i+1}'
            rename_map[f'asks[{i}].amount'] = f'ask_v_{i+1}'
            rename_map[f'ask_price_{i}'] = f'ask_p_{i+1}'
            rename_map[f'ask_size_{i}'] = f'ask_v_{i+1}'

        df.rename(columns=rename_map, inplace=True)

        # 3. Validation
        required_cols = ['timestamp'] + [f'bid_p_{i}' for i in range(1, self.levels+1)] + \
                        [f'ask_p_{i}' for i in range(1, self.levels+1)]
                        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing expected columns after mapping: {missing_cols}")
            raise KeyError("CSV does not match L2 Order Book structure.")

        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        logger.info(f"Successfully loaded {len(df)} L2 snapshots.")
        return df

class BinanceLiveAdapter:
    """
    High-speed Async WebSocket Adapter for Binance Futures.
    Streams real-time L2 Partial Book Depth into the Caspian Alpha pipeline.
    """
    def __init__(self, symbol: str, callback: Callable[[Dict], None]):
        self.symbol = symbol.lower()
        self.callback = callback
        self.url = f"{WSS_URL}{STREAM_NAME.format(symbol=self.symbol)}"
        self.is_running = False

    async def connect(self) -> NoReturn:
        """
        Establishes and maintains the WebSocket connection with auto-reconnect logic.
        """
        self.is_running = True
        while self.is_running:
            try:
                # Surgical check for parser type to log diagnostics
                parser_type = "orjson" if USE_ORJSON else "standard json"
                logger.info(f"Connecting to Binance Stream ({parser_type}): {self.url}")
                async with websockets.connect(self.url) as websocket:
                    await self._handle_stream(websocket)
            except websockets.ConnectionClosed:
                logger.warning(f"Connection lost. Reconnecting in {RECONNECT_DELAY}s...")
                await asyncio.sleep(RECONNECT_DELAY)
            except Exception as e:
                logger.error(f"Unexpected WebSocket error: {e}")
                await asyncio.sleep(RECONNECT_DELAY)

    async def _handle_stream(self, websocket: websockets.WebSocketClientProtocol):
        """
        Listens to the stream, parses JSON, and routes data to the callback.
        """
        async for message in websocket:
            try:
                # Optimized parsing via orjson if available (mapped via config import)
                data = json.loads(message)
                
                # Standardize Binance L2 partial depth to internal format
                formatted_snap = self._map_binance_to_internal(data)
                
                # Pass to the main processing queue/buffer
                self.callback(formatted_snap)
                
            except Exception as e:
                logger.error(f"Error processing stream message: {e}")

    def _map_binance_to_internal(self, data: Dict) -> Dict:
        """
        Maps Binance [price, volume] lists into Caspian Alpha named fields.
        Includes exchange timestamp for latency tracking.
        """
        # Binance uses 'T' for Transaction Time (Engine) or 'E' for Event Time
        exch_ts_raw = data.get('T', data.get('E', time.time() * 1000))
        ts = pd.to_datetime(exch_ts_raw, unit='ms')
        
        # Adding 'exch_ts' for surgical latency tracking in Step 5
        internal_snap = {
            'timestamp': ts,
            'exch_ts': exch_ts_raw
        }
        
        # Map Bids (b) and Asks (a) up to OB_LEVELS
        for i in range(OB_LEVELS):
            # Binance Bids: [[price, qty], [price, qty]...]
            if i < len(data['b']):
                internal_snap[f'bid_p_{i+1}'] = float(data['b'][i][0])
                internal_snap[f'bid_v_{i+1}'] = float(data['b'][i][1])
            
            # Binance Asks: [[price, qty], [price, qty]...]
            if i < len(data['a']):
                internal_snap[f'ask_p_{i+1}'] = float(data['a'][i][0])
                internal_snap[f'ask_v_{i+1}'] = float(data['a'][i][1])
                
        return internal_snap

    def stop(self):
        """Gracefully stops the connection loop."""
        self.is_running = False