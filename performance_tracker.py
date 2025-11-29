# --- START OF FILE performance_tracker.py ---

import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import math

# Configure logging for this module
log = logging.getLogger('PerformanceTracker') # Use a specific logger name
log.setLevel(logging.INFO) # Set default level
log.propagate = False # Stop messages going to the root logger

# Ensure handlers are not added multiple times if the module is reloaded
if not log.handlers:
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler specific to performance tracker logs
    try:
        fh = logging.FileHandler('logs/performance.log', mode='a') # Append mode
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    except Exception as e:
         print(f"Error setting up performance file logger: {e}") # Use print as logger might fail

    # Optional: Stream handler for console output
    # sh = logging.StreamHandler()
    # sh.setLevel(logging.INFO)
    # sh.setFormatter(formatter)
    # log.addHandler(sh)


class PerformanceTracker:
    """
    Records closed trades and calculates performance metrics.
    Persists trade history to a CSV file.
    """

    # Define the columns expected in the trade history file and dictionaries
    # Ensure these match the keys provided when calling record_trade
    HISTORY_COLUMNS: List[str] = [
        'trade_id',         # Unique ID for the trade (e.g., timestamp + pair)
        'pair',             # Trading pair (e.g., 'EURUSDm')
        'direction',        # 'BUY' or 'SELL'
        'entry_time',       # ISO Format Timestamp (UTC or consistent timezone)
        'exit_time',        # ISO Format Timestamp
        'entry_price',      # Float
        'exit_price',       # Float
        'stop_loss',        # Original SL price (Float)
        'take_profit',      # Original TP price (Float)
        'position_size',    # Float (optional, depends on profit calculation)
        'profit_currency',  # Profit/Loss in account currency (Float) - Primary Metric
        'profit_pips',      # Profit/Loss in pips (Float, optional)
        'profit_percentage',# Profit/Loss as percentage (Float, optional)
        'exit_reason',      # String (e.g., 'TP', 'SL', 'Signal', 'Manual', 'Error')
        'signal_confidence',# Entry signal confidence (Float, optional)
        # Add other relevant fields if needed, e.g., 'strategy_details', 'commission', 'swap'
    ]

    # Define data types for robust loading (especially dates)
    CSV_DTYPES: Dict[str, Any] = {
        'trade_id': str,
        'pair': str,
        'direction': str,
        # Timestamps will be parsed separately
        'entry_price': float,
        'exit_price': float,
        'stop_loss': float,
        'take_profit': float,
        'position_size': float,
        'profit_currency': float,
        'profit_pips': float,
        'profit_percentage': float,
        'exit_reason': str,
        'signal_confidence': float,
    }
    TIMESTAMP_COLS = ['entry_time', 'exit_time'] # Columns to parse as datetime


    def __init__(self, history_file: str = 'data/trade_history.csv'):
        """
        Initializes the PerformanceTracker.

        Args:
            history_file (str): Path to the CSV file for storing trade history.
        """
        self.history_file = history_file
        self.trade_history: List[Dict] = [] # Holds trades in memory as list of dicts

        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

        log.info(f"Initializing PerformanceTracker with history file: {self.history_file}")
        self._load_history()

    def _load_history(self):
        """Loads trade history from the CSV file."""
        if not os.path.exists(self.history_file):
            log.warning(f"History file '{self.history_file}' not found. Starting with empty history.")
            self.trade_history = []
            # Create header if file is new
            self._save_history() # Save empty history to create file with header
            return

        log.info(f"Loading trade history from {self.history_file}...")
        try:
            # Read CSV, attempting dtype inference but parsing dates explicitly
            df = pd.read_csv(
                self.history_file,
                dtype=self.CSV_DTYPES, # Apply basic types
                parse_dates=self.TIMESTAMP_COLS, # Specify date columns
                date_parser=lambda x: pd.to_datetime(x, errors='coerce') # Robust date parsing
                # Use na_values to handle common placeholders for missing data if needed
            )

            # Check for and handle duplicate trade_ids
            if 'trade_id' in df.columns and df['trade_id'].duplicated().any():
                duplicate_count = df['trade_id'].duplicated().sum()
                log.warning(f"Found {duplicate_count} duplicate trade_ids in history file. Keeping only the first occurrence of each.")
                # Drop duplicates, keeping the first occurrence
                df = df.drop_duplicates(subset=['trade_id'], keep='first')
                # Save the de-duplicated file right away
                df.to_csv(self.history_file, index=False, encoding='utf-8')
                log.info(f"Saved de-duplicated history with {len(df)} unique trades.")

            # Validate columns after loading
            missing_cols = [col for col in self.HISTORY_COLUMNS if col not in df.columns]
            extra_cols = [col for col in df.columns if col not in self.HISTORY_COLUMNS]
            if missing_cols:
                log.warning(f"Loaded history is missing expected columns: {missing_cols}. They will be NaN.")
                # Add missing columns with NaN values to maintain structure
                for col in missing_cols:
                    df[col] = pd.NA
            if extra_cols:
                log.warning(f"Loaded history contains unexpected columns: {extra_cols}. They will be ignored.")
                # Optionally drop extra columns: df = df[self.HISTORY_COLUMNS]

            # Ensure correct column order for consistency
            df = df[self.HISTORY_COLUMNS]

            # Handle potential NaN values created by coerce or missing data
            # Log rows with NaN timestamps as they indicate loading issues
            nan_timestamp_rows = df[df[self.TIMESTAMP_COLS].isnull().any(axis=1)]
            if not nan_timestamp_rows.empty:
                log.warning(f"Found {len(nan_timestamp_rows)} rows with invalid timestamps during loading. Review history file.")
                # Consider filtering these rows or logging their indices

            # Convert DataFrame to list of dictionaries
            self.trade_history = df.to_dict('records')
            log.info(f"Successfully loaded {len(self.trade_history)} trades from history.")

        except pd.errors.EmptyDataError:
             log.warning(f"History file '{self.history_file}' is empty. Starting fresh.")
             self.trade_history = []
        except Exception as e:
            log.exception(f"Failed to load trade history from '{self.history_file}': {e}. Starting with empty history.")
            self.trade_history = [] # Ensure history is empty on error

    def _save_history(self):
        """Saves the current trade history to the CSV file."""
        log.debug(f"Attempting to save {len(self.trade_history)} trades to {self.history_file}...")
        try:
            # Create DataFrame from the list of dicts, ensuring column order
            df_to_save = pd.DataFrame(self.trade_history, columns=self.HISTORY_COLUMNS)

            # Convert datetime objects to ISO format strings for CSV consistency
            for col in self.TIMESTAMP_COLS:
                if col in df_to_save.columns:
                    # Ensure the column is actually datetime before formatting
                    if pd.api.types.is_datetime64_any_dtype(df_to_save[col]):
                         df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%dT%H:%M:%S%z') # ISO format with TZ
                    else:
                         # Attempt conversion if not already datetime (e.g., if loaded incorrectly)
                         df_to_save[col] = pd.to_datetime(df_to_save[col], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S%z')


            # Save to CSV, overwriting the file
            df_to_save.to_csv(self.history_file, index=False, encoding='utf-8')
            log.debug(f"Trade history saved successfully to {self.history_file}.")

        except Exception as e:
            log.exception(f"Failed to save trade history to '{self.history_file}': {e}")

    def record_trade(self, trade_details: Dict):
        """
        Adds a closed trade to the history and saves it.

        Args:
            trade_details (Dict): A dictionary containing all details for the closed trade.
                                  Keys must match PerformanceTracker.HISTORY_COLUMNS.
        """
        # --- Validation ---
        if not isinstance(trade_details, dict):
            log.error("Failed to record trade: Input 'trade_details' must be a dictionary.")
            return

        missing_keys = [key for key in self.HISTORY_COLUMNS if key not in trade_details]
        if missing_keys:
            log.error(f"Failed to record trade: Missing required keys: {missing_keys}. Provided keys: {list(trade_details.keys())}")
            return

        extra_keys = [key for key in trade_details if key not in self.HISTORY_COLUMNS]
        if extra_keys:
             log.warning(f"Recording trade: Extra keys found and ignored: {extra_keys}")
             # Create a new dict with only expected keys
             validated_trade = {key: trade_details[key] for key in self.HISTORY_COLUMNS}
        else:
             validated_trade = trade_details

        # Optional: Further type validation (e.g., ensure prices are floats, times are datetime or valid strings)
        try:
            # Attempt conversions for robustness
            validated_trade['entry_price'] = float(validated_trade['entry_price'])
            validated_trade['exit_price'] = float(validated_trade['exit_price'])
            validated_trade['profit_currency'] = float(validated_trade['profit_currency'])
            # Coerce timestamps to datetime objects if they aren't already
            for col in self.TIMESTAMP_COLS:
                if not isinstance(validated_trade[col], datetime):
                    validated_trade[col] = pd.to_datetime(validated_trade[col], errors='coerce')
                if pd.isna(validated_trade[col]): # Check if conversion failed
                     raise ValueError(f"Invalid timestamp format for '{col}'")

        except (ValueError, TypeError) as e:
            log.error(f"Failed to record trade: Invalid data type for key value. Error: {e}. Details: {validated_trade}")
            return

        # Check for duplicate trade_id to avoid duplicates
        if any(trade.get('trade_id') == validated_trade.get('trade_id') for trade in self.trade_history):
            log.warning(f"Trade with ID {validated_trade.get('trade_id')} already exists in history. Skipping.")
            return

        # --- Add and Save ---
        self.trade_history.append(validated_trade)
        log.info(f"Recorded trade: ID={validated_trade.get('trade_id', 'N/A')}, Pair={validated_trade['pair']}, P/L={validated_trade['profit_currency']:.2f} {validated_trade.get('account_currency', '')}")
        try:
                self._save_history()
        except Exception as e:
            log.error(f"Failed to save trade history after recording trade {validated_trade.get('trade_id', 'N/A')}: {e}")
            # The trade is still in memory, just not saved to disk yet


    def get_performance_metrics(self, lookback_trades: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculates performance metrics based on the recorded trade history.

        Args:
            lookback_trades (Optional[int]): If provided, calculate metrics only for the
                                             last N trades. Defaults to using all trades.

        Returns:
            Dict[str, Any]: A dictionary containing calculated performance metrics.
        """
        log.debug(f"Calculating performance metrics (Lookback: {lookback_trades or 'All'})...")

        default_metrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_profit_currency': 0.0,
            'total_profit_pips': 0.0,
            'profit_factor': 0.0,
            'average_win_currency': 0.0,
            'average_loss_currency': 0.0,
            'expectancy_currency': 0.0, # Avg P/L per trade
            'max_drawdown_currency': 0.0, # Requires equity curve simulation, placeholder for now
            'sharpe_ratio': 0.0, # Requires risk-free rate & return std dev, placeholder
            'lookback_period': lookback_trades if lookback_trades else 'All',
            'error': None
        }

        if not self.trade_history:
            log.warning("Cannot calculate metrics: No trade history available.")
            default_metrics['error'] = "No trade history"
            return default_metrics

        try:
            # Convert history to DataFrame for easier calculation
            df = pd.DataFrame(self.trade_history)

            # Filter by lookback period if specified
            if lookback_trades is not None and isinstance(lookback_trades, int) and lookback_trades > 0:
                if lookback_trades < len(df):
                    df = df.tail(lookback_trades).copy() # Use copy to avoid SettingWithCopyWarning
                else:
                     log.debug(f"Lookback ({lookback_trades}) is >= total trades ({len(df)}). Using all trades.")
            elif lookback_trades is not None:
                 log.warning(f"Invalid lookback_trades value: {lookback_trades}. Using all trades.")


            # --- Basic Counts & Sums ---
            total_trades = len(df)
            if total_trades == 0:
                 log.warning("No trades found within the specified lookback period.")
                 default_metrics['error'] = "No trades in lookback period"
                 return default_metrics

            # Use 'profit_currency' as the primary P/L metric
            profit_col = 'profit_currency'
            if profit_col not in df.columns or df[profit_col].isnull().all():
                 log.error(f"Cannot calculate metrics: Missing or all-NaN '{profit_col}' column.")
                 default_metrics['error'] = f"Missing/Invalid '{profit_col}' data"
                 return default_metrics

            # Calculate total profit
            total_profit_currency = df[profit_col].sum()

            # Calculate total pips if available
            pips_col = 'profit_pips'
            total_profit_pips = 0.0
            if pips_col in df.columns and not df[pips_col].isnull().all():
                 total_profit_pips = df[pips_col].sum()
            else:
                 log.debug("Profit pips column missing or empty, skipping pips calculation.")


            # --- Wins & Losses ---
            wins_df = df[df[profit_col] > 0]
            losses_df = df[df[profit_col] <= 0] # Include zero profit as loss/break-even

            win_count = len(wins_df)
            loss_count = len(losses_df) # total_trades - win_count

            # --- Ratios & Averages ---
            win_rate = win_count / total_trades if total_trades > 0 else 0.0

            total_gross_profit = wins_df[profit_col].sum()
            total_gross_loss = abs(losses_df[profit_col].sum()) # Sum of absolute losses

            # Profit Factor: Gross Profit / Gross Loss
            profit_factor = 0.0
            if total_gross_loss > 1e-9: # Avoid division by zero or near-zero
                profit_factor = total_gross_profit / total_gross_loss
            elif total_gross_profit > 0: # Only profits, no losses
                profit_factor = float('inf') # Technically infinite

            # Average Win / Loss
            average_win_currency = total_gross_profit / win_count if win_count > 0 else 0.0
            average_loss_currency = total_gross_loss / loss_count if loss_count > 0 else 0.0

            # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
            loss_rate = 1.0 - win_rate
            expectancy_currency = (win_rate * average_win_currency) - (loss_rate * average_loss_currency)


            # --- Placeholder for Advanced Metrics ---
            # max_drawdown_currency = self._calculate_max_drawdown_from_trades(df) # Needs implementation
            # sharpe_ratio = self._calculate_sharpe_ratio(df) # Needs implementation


            # --- Assemble Results ---
            metrics = {
                'total_trades': total_trades,
                'win_rate': round(win_rate, 4),
                'total_profit_currency': round(total_profit_currency, 2),
                'total_profit_pips': round(total_profit_pips, 2) if pips_col in df.columns else 0.0,
                'profit_factor': round(profit_factor, 2) if math.isfinite(profit_factor) else 999.99, # Cap infinite PF
                'average_win_currency': round(average_win_currency, 2),
                'average_loss_currency': round(average_loss_currency, 2),
                'expectancy_currency': round(expectancy_currency, 4),
                'max_drawdown_currency': 0.0, # Placeholder
                'sharpe_ratio': 0.0, # Placeholder
                'lookback_period': lookback_trades if lookback_trades else 'All',
                'error': None
            }

            log.info(f"Performance Metrics (Lookback: {metrics['lookback_period']}): "
                     f"Trades={metrics['total_trades']}, WinRate={metrics['win_rate']:.2%}, "
                     f"Total P/L={metrics['total_profit_currency']:.2f}, PF={metrics['profit_factor']:.2f}, "
                     f"Expectancy={metrics['expectancy_currency']:.4f}")

            return metrics

        except Exception as e:
            log.exception(f"Error calculating performance metrics: {e}")
            default_metrics['error'] = f"Calculation error: {e}"
            return default_metrics


    # --- Placeholder Methods for Advanced Metrics (Requires more complex calculation) ---

    def _calculate_max_drawdown_from_trades(self, df: pd.DataFrame, initial_equity: float = 10000) -> float:
        """
        Placeholder: Calculates maximum drawdown based on simulated equity curve from trades.
        NOTE: This requires simulating an equity curve, which depends on assumptions
              about position sizing relative to equity, which isn't stored directly here.
              A simpler approximation might use cumulative profit.
        """
        if 'profit_currency' not in df.columns or df.empty:
            return 0.0

        # Simple approximation using cumulative profit
        cumulative_profit = df['profit_currency'].cumsum()
        equity_curve = initial_equity + cumulative_profit
        peak_equity = equity_curve.expanding(min_periods=1).max()
        drawdown = peak_equity - equity_curve
        max_drawdown = drawdown.max()

        # Could also calculate percentage drawdown:
        # drawdown_pct = drawdown / peak_equity
        # max_drawdown_pct = drawdown_pct.max()

        return round(max_drawdown, 2)


    def _calculate_sharpe_ratio(self, df: pd.DataFrame, risk_free_rate_annual: float = 0.01) -> float:
        """
        Placeholder: Calculates the Sharpe ratio.
        NOTE: Requires assumptions about the frequency of returns (daily, hourly?)
              and needs the standard deviation of those returns.
        """
        # This requires converting trade profits into periodic returns (e.g., daily returns)
        # and then calculating the mean and std dev of those returns.
        # It's complex to do accurately just from a list of trades without knowing
        # the time period each return represents relative to the total backtest duration.
        log.warning("Sharpe ratio calculation is complex and not fully implemented.")
        return 0.0

# --- END OF FILE performance_tracker.py ---
