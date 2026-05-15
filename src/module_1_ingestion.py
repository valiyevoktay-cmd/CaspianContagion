import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CaspianContagion.Ingestion")

class OrderBookProcessor:
    """
    Module 1: Data Ingestion & Preprocessing Engine.
    
    Responsible for reconstructing the L2 Limit Order Book state, 
    calculating multi-level Order Flow Imbalance (OFI), and extracting 
    high-frequency aggressive market events (liquidity taking).
    """

    def __init__(self, levels: int = 5):
        """
        Initializes the Order Book Processor.

        Args:
            levels (int): The number of LOB levels to consider for OFI calculation.
        """
        self.levels = levels
        logger.info(f"Initialized OrderBookProcessor with {self.levels} LOB levels.")

    def calculate_ofi(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates the cross-level Order Flow Imbalance (OFI) using a vectorized approach.
        
        The OFI represents the net liquidity provision/depletion.
        Math: 
        e_t = I(d_P_bid >= 0) * d_V_bid - I(d_P_bid <= 0) * V_bid_{t-1}
            - I(d_P_ask <= 0) * d_V_ask + I(d_P_ask >= 0) * V_ask_{t-1}

        Args:
            df (pd.DataFrame): DataFrame containing 'bid_p_X', 'bid_v_X', 
                               'ask_p_X', 'ask_v_X' for X in range(1, levels+1).

        Returns:
            pd.Series: A pandas Series containing the aggregated OFI at each timestamp.
        """
        try:
            total_ofi = np.zeros(len(df))
            
            for level in range(1, self.levels + 1):
                bid_p = df[f'bid_p_{level}'].values
                bid_v = df[f'bid_v_{level}'].values
                ask_p = df[f'ask_p_{level}'].values
                ask_v = df[f'ask_v_{level}'].values

                # Shift arrays for vectorized t-1 comparison (padding with first value to maintain shape)
                prev_bid_p = np.roll(bid_p, 1)
                prev_bid_v = np.roll(bid_v, 1)
                prev_ask_p = np.roll(ask_p, 1)
                prev_ask_v = np.roll(ask_v, 1)
                
                # Handle edge case for the first row after rolling
                prev_bid_p[0] = bid_p[0]
                prev_bid_v[0] = bid_v[0]
                prev_ask_p[0] = ask_p[0]
                prev_ask_v[0] = ask_v[0]

                # Vectorized Bid-side OFI logic
                bid_ofi = np.where(
                    bid_p > prev_bid_p, bid_v,
                    np.where(bid_p == prev_bid_p, bid_v - prev_bid_v, -prev_bid_v)
                )

                # Vectorized Ask-side OFI logic
                ask_ofi = np.where(
                    ask_p < prev_ask_p, ask_v,
                    np.where(ask_p == prev_ask_p, ask_v - prev_ask_v, -prev_ask_v)
                )

                # Net OFI for the current level
                level_ofi = bid_ofi - ask_ofi
                total_ofi += level_ofi

            logger.info("Successfully calculated multi-level OFI via vectorization.")
            return pd.Series(total_ofi, index=df.index, name='OFI')

        except KeyError as e:
            logger.error(f"Missing expected LOB column: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during OFI calculation: {e}")
            raise

    def extract_market_events(self, df: pd.DataFrame, ofi_series: pd.Series, threshold_z: float = 2.0) -> pd.DataFrame:
        """
        Extracts timestamps of aggressive liquidity-taking events.
        
        Since raw tick-by-tick trades might be unavailable, we proxy aggressive 
        market orders by identifying statistically significant spikes in OFI.

        Args:
            df (pd.DataFrame): The main LOB dataframe with timestamps.
            ofi_series (pd.Series): The calculated OFI series.
            threshold_z (float): Z-score threshold to classify an OFI spike as a market event.

        Returns:
            pd.DataFrame: A structured DataFrame containing [timestamp, mid_price, OFI_state]
                          filtered only for aggressive event occurrences.
        """
        try:
            # Calculate rolling Z-score for OFI to adapt to different volatility regimes
            rolling_mean = ofi_series.rolling(window=100, min_periods=1).mean()
            rolling_std = ofi_series.rolling(window=100, min_periods=1).std().replace(0, 1e-9)
            z_scores = np.abs((ofi_series - rolling_mean) / rolling_std)

            # Identify events where OFI volatility exceeds the threshold
            event_mask = z_scores > threshold_z
            
            # Calculate mid price for L1
            mid_price = (df['bid_p_1'] + df['ask_p_1']) / 2.0

            events_df = pd.DataFrame({
                'timestamp': df['timestamp'][event_mask],
                'mid_price': mid_price[event_mask],
                'OFI_state': ofi_series[event_mask]
            }).reset_index(drop=True)

            logger.info(f"Extracted {len(events_df)} aggressive market events based on OFI z-score > {threshold_z}.")
            return events_df

        except Exception as e:
            logger.error(f"Error extracting market events: {e}")
            raise

    def process_pipeline(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the full Module 1 pipeline.

        Args:
            raw_data (pd.DataFrame): Raw L2 order book data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The enriched main dataframe and the extracted events dataframe.
        """
        logger.info("Starting Module 1 Pipeline...")
        ofi_series = self.calculate_ofi(raw_data)
        raw_data['OFI'] = ofi_series
        
        events_df = self.extract_market_events(raw_data, ofi_series)
        
        logger.info("Module 1 Pipeline execution completed successfully.")
        return raw_data, events_df


if __name__ == '__main__':
    # ---------------------------------------------------------
    # DUMMY DATA GENERATION FOR IMMEDIATE TESTING
    # ---------------------------------------------------------
    np.random.seed(42)
    n_ticks = 5000
    levels = 5
    
    logger.info(f"Generating synthetic L2 Order Book data for {n_ticks} ticks...")
    
    # Generate random walk for L1 Mid Price
    price_changes = np.random.normal(loc=0.0, scale=0.01, size=n_ticks)
    mid_price = 100.0 + np.cumsum(price_changes)
    
    data_dict = {'timestamp': pd.date_range(start='2026-04-18 09:30:00', periods=n_ticks, freq='10ms')}
    
    # Populate LOB levels
    for i in range(1, levels + 1):
        spread = 0.05 * i
        data_dict[f'bid_p_{i}'] = mid_price - spread
        data_dict[f'ask_p_{i}'] = mid_price + spread
        
        # Volumes follow a log-normal distribution, deeper levels have more volume
        data_dict[f'bid_v_{i}'] = np.random.lognormal(mean=2.0 + (i*0.1), sigma=0.5, size=n_ticks)
        data_dict[f'ask_v_{i}'] = np.random.lognormal(mean=2.0 + (i*0.1), sigma=0.5, size=n_ticks)

    df_dummy = pd.DataFrame(data_dict)
    
    # Inject synthetic aggressive market orders (Flash Crash proxy) at tick 2500
    df_dummy.loc[2500:2510, 'bid_v_1'] *= 0.1 # Massive liquidity drain on bid side
    df_dummy.loc[2500:2510, 'ask_v_1'] *= 5.0 # Accumulation on ask side
    
    # ---------------------------------------------------------
    # TEST PIPELINE
    # ---------------------------------------------------------
    processor = OrderBookProcessor(levels=5)
    enriched_data, market_events = processor.process_pipeline(df_dummy)
    
    print("\n--- TEST RESULTS ---")
    print(f"Enriched Data Shape: {enriched_data.shape}")
    print(f"Extracted Hawkes Events: {market_events.shape[0]}")
    print("\nSample of Extracted Events (Flash Crash Proxy at 09:30:25):")
    print(market_events[market_events['timestamp'] >= '2026-04-18 09:30:25'].head())