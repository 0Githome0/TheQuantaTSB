# --- START OF FILE Signal.py ---

from logging import config
from sysconfig import get_paths
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import os
from typing import Dict, Optional, List, Tuple, Union, Any
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, NotFittedError
from enum import Enum
from datetime import datetime
import pytz
from joblib import load, dump
import math
import warnings

# Suppress specific warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*") # Mute common numpy calculation warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom <= 0 for slice.*")

# Configure logging for this module
log = logging.getLogger('SignalGenerator') # Use a specific logger name
log.setLevel(logging.INFO) # Set default level
log.propagate = False # Stop messages going to the root logger

# Ensure handlers are not added multiple times if the module is reloaded
if not log.handlers:
    # Ensure logs and models directories exist
    logs_dir = 'logs'
    models_dir = 'models'
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler specific to signal generator logs
    try:
        fh = logging.FileHandler(os.path.join(logs_dir, 'signals.log'), mode='a') # Append mode
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    except Exception as e:
         print(f"Error setting up signals file logger: {e}") # Use print as logger might fail

    # Stream handler for console output (useful during development)
    # Comment out if running in production without console output needed
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    log.addHandler(sh)

# Define a configuration dictionary to replace the incorrectly used logging.config module
model_config = {
    'output_model_base_name': 'signal_MULTI_ASSET_H1_XGB_Auto'
}

class TimeFrames(Enum):
    """Enumeration for supported timeframes in minutes."""
    M5 = 5  # Add M5 timeframe
    M15 = 15
    H1 = 60
    H4 = 240
    D1 = 1440

# Import the NewsAPIIntegration at the top of the file

class SignalGenerator:
    """
    Generates trading signals using technical indicators, dynamic parameters,
    and machine learning integration. Loads configuration and models from files.
    Handles potential data issues and calculation errors gracefully.
    """
    # File paths for models and configuration data
    MODEL_DIR: str = "models"
    ML_MODEL_FILE: str = os.path.join(MODEL_DIR, "signal_gb_model.joblib")
    SCALER_FILE: str = os.path.join(MODEL_DIR, "signal_scaler.joblib")
    INDICATOR_PERF_FILE: str = os.path.join(MODEL_DIR, "indicator_performance.joblib")
    OPTIMAL_PERIODS_FILE: str = os.path.join(MODEL_DIR, "optimal_periods.joblib")

    # Default values (used if files are not found)
    DEFAULT_INDICATOR_PERFORMANCE: Dict[str, Dict[str, float]] = {
        'rsi': {'accuracy': 0.55, 'profit_factor': 1.2, 'win_rate': 0.52},
        'macd': {'accuracy': 0.57, 'profit_factor': 1.3, 'win_rate': 0.54},
        'bollinger': {'accuracy': 0.56, 'profit_factor': 1.25, 'win_rate': 0.53},
        'adx': {'accuracy': 0.54, 'profit_factor': 1.15, 'win_rate': 0.51},
        'ichimoku': {'accuracy': 0.58, 'profit_factor': 1.35, 'win_rate': 0.55},
        'fibonacci': {'accuracy': 0.53, 'profit_factor': 1.18, 'win_rate': 0.50},
        'stoch': {'accuracy': 0.55, 'profit_factor': 1.22, 'win_rate': 0.53},
        'ema_trend': {'accuracy': 0.56, 'profit_factor': 1.28, 'win_rate': 0.54},
        'parabolic_sar': {'accuracy': 0.52, 'profit_factor': 1.10, 'win_rate': 0.50},
        'obv': {'accuracy': 0.53, 'profit_factor': 1.15, 'win_rate': 0.51},
        'atr': {'accuracy': 0.50, 'profit_factor': 1.05, 'win_rate': 0.50}, # ATR not directly directional
        'volatility_rsi': {'accuracy': 0.51, 'profit_factor': 1.08, 'win_rate': 0.50}, # Vol RSI not directly directional
        'chaikin_oscillator': {'accuracy': 0.54, 'profit_factor': 1.17, 'win_rate': 0.52},
        'supply_demand': {'accuracy': 0.50, 'profit_factor': 1.0, 'win_rate': 0.50} # S/D zones not directional score
    }
    DEFAULT_OPTIMAL_PERIODS: Dict[str, Dict[str, int]] = {
        'trending': {'rsi': 14, 'macd_fast': 12, 'macd_slow': 26, 'bollinger': 20, 'stoch': 14},
        'volatile': {'rsi': 9, 'macd_fast': 9, 'macd_slow': 18, 'bollinger': 15, 'stoch': 9},
        'ranging': {'rsi': 21, 'macd_fast': 15, 'macd_slow': 30, 'bollinger': 25, 'stoch': 21},
        'volatile_trending': {'rsi': 11, 'macd_fast': 10, 'macd_slow': 20, 'bollinger': 18, 'stoch': 11}, # Added specific
        'volatile_ranging': {'rsi': 10, 'macd_fast': 10, 'macd_slow': 20, 'bollinger': 16, 'stoch': 10} # Added specific
    }

    # Constants for technical score calculation thresholds
    RSI_OVERBOUGHT: int = 70
    RSI_STRONG_REVERSAL_BUY: int = 65 # Near overbought, weakening (negative score)
    RSI_OVERSOLD: int = 30
    RSI_STRONG_REVERSAL_SELL: int = 35 # Near oversold, weakening (positive score)
    STOCH_OVERBOUGHT: int = 80
    STOCH_OVERSOLD: int = 20
    ADX_TREND_THRESHOLD: int = 25
    ADX_RANGE_THRESHOLD: int = 20
    VOL_RSI_HIGH: int = 70 # Threshold for high volatility via Volatility RSI
    VOL_RSI_LOW: int = 30  # Threshold for low volatility via Volatility RSI

    # Constants for dynamic period adjustment
    MIN_INDICATOR_PERIOD: int = 5
    MAX_PERIOD_MULTIPLIER: float = 2.5
    MIN_DATA_FOR_DYNAMIC_PERIOD: int = 30 # Need enough data for reliable ATR/ADX

    # Constants for signal generation
    MIN_DATA_FOR_SIGNAL: int = 50 # Minimum length for overall signal generation
    BUY_THRESHOLD_COMBINED: float = 0.60  # آمن للشراء
    SELL_THRESHOLD_COMBINED: float = 0.40  # آمن للبيع
    TECH_SCORE_WEIGHT: float = 0.60 # Weight for technical score in combined score
    ML_CONFIDENCE_WEIGHT: float = 0.40 # Weight for ML confidence in combined score
    ATR_SL_MULTIPLIER: float = 1.5
    ATR_TP_MULTIPLIER: float = 2.5
    MIN_RR_RATIO: float = 1.1 # Minimum required Risk:Reward ratio for SL/TP setting
    MIN_SL_DISTANCE_ATR_FACTOR: float = 0.3 # Min distance for SL in terms of ATR

    # Define the exact features the ML model was trained on
    # !! IMPORTANT !! This list MUST match the features used when training 'signal_gb_model.joblib'
    #                 and 'signal_scaler.joblib'. Order matters!
    EXPECTED_ML_FEATURES: List[str] = [
        'rsi', 'macd_histogram', 'atr_norm', 'bbands_squeeze', 'stoch_k', 'stoch_d',
        'adx', 'ema_trend', 'volume_sma_ratio', 'volatility_rsi'
    ]

    # Add new thresholds for early detection
    EARLY_RSI_OVERBOUGHT: int = 65
    EARLY_RSI_STRONG_REVERSAL_BUY: int = 60
    EARLY_RSI_OVERSOLD: int = 35
    EARLY_RSI_STRONG_REVERSAL_SELL: int = 40
    EARLY_STOCH_OVERBOUGHT: int = 75
    EARLY_STOCH_OVERSOLD: int = 25
    EARLY_ADX_TREND_THRESHOLD: int = 20
    EARLY_ADX_RANGE_THRESHOLD: int = 15
    EARLY_BUY_THRESHOLD_COMBINED: float = 0.55
    EARLY_SELL_THRESHOLD_COMBINED: float = 0.45
    EARLY_TECH_SCORE_WEIGHT: float = 0.70
    EARLY_ML_CONFIDENCE_WEIGHT: float = 0.30

    # Early warning parameters
    PRICE_ACCELERATION_THRESHOLD: float = 0.0002
    VOLUME_SPIKE_THRESHOLD: float = 2.0
    MOMENTUM_THRESHOLD: float = 0.0001

    # Signal confirmation parameters
    SIGNAL_CONFIRMATION_THRESHOLD: float = 0.8
    EARLY_SIGNAL_WEIGHT: float = 0.4
    CONFIRMED_SIGNAL_WEIGHT: float = 0.6

    # Constants for early warning detection
    EARLY_WARNING_RSI_THRESHOLD: int = 35  # More sensitive RSI threshold for early signals
    EARLY_WARNING_STOCH_THRESHOLD: int = 25  # More sensitive Stochastic threshold
    EARLY_WARNING_MACD_THRESHOLD: float = 0.0001  # Smaller MACD threshold for early detection
    EARLY_WARNING_VOLUME_THRESHOLD: float = 1.5  # Volume spike threshold
    EARLY_WARNING_WEIGHT: float = 0.3  # Weight for early warning signals

    def __init__(self, timezone: str = 'Asia/Riyadh'):
        """
        Initializes the SignalGenerator.

        Args:
            timezone (str): The timezone to use for timestamps (e.g., 'UTC', 'Asia/Riyadh').
        """
        log.info("Initializing SignalGenerator...")
        self.scaler = self._load_sklearn_object(self.SCALER_FILE, StandardScaler())
        self.model = self._load_sklearn_object(self.ML_MODEL_FILE, GradientBoostingClassifier())
        try:
            self.timezone = pytz.timezone(timezone)
        except pytz.UnknownTimeZoneError:
            log.warning(f"Unknown timezone '{timezone}'. Defaulting to UTC.")
            self.timezone = pytz.utc

        # Load historical performance and optimal periods
        self.indicator_performance = self._load_joblib_data(
            self.INDICATOR_PERF_FILE, self.DEFAULT_INDICATOR_PERFORMANCE
        )
        self.optimal_periods_history = self._load_joblib_data(
            self.OPTIMAL_PERIODS_FILE, self.DEFAULT_OPTIMAL_PERIODS
        )

        # Symbol-specific volatility baselines (in-memory cache)
        self.volatility_baselines: Dict[str, float] = {}
        log.info("SignalGenerator initialized successfully.")

    def _load_sklearn_object(self, filepath: str, default_object: Any) -> Any:
        """Helper to load scikit-learn objects (models/scalers) or return a default."""
        if not os.path.exists(filepath):
            log.warning(f"File '{filepath}' not found. Initializing default {type(default_object).__name__}.")
            return default_object
        try:
            loaded_object = load(filepath)
            log.info(f"Object loaded successfully from '{filepath}'.")
            # Critical check: Ensure loaded scalers/models are fitted BEFORE use
            check_is_fitted(loaded_object)
            log.info(f"Loaded object {type(loaded_object).__name__} from '{filepath}' is fitted.")
            return loaded_object
        except NotFittedError:
            log.error(f"Loaded object from '{filepath}' is NOT FITTED. "
                      f"This object cannot be used for predictions. "
                      f"Please provide a fitted model/scaler file. Using default object (likely unfitted).")
            # Return the default (which might also be unfitted), prediction will likely fail later
            return default_object
        except ImportError:
            log.error(f"Joblib not installed. Cannot load object '{filepath}'. Returning default.")
            return default_object
        except Exception as e:
            log.error(f"Error loading object from '{filepath}': {e}. Returning default.", exc_info=True)
            return default_object

    def _load_joblib_data(self, filepath: str, default_data: Dict) -> Dict:
        """Loads arbitrary data from a joblib file or returns default."""
        if not os.path.exists(filepath):
             log.warning(f"Data file '{filepath}' not found. Initializing default values.")
             return default_data
        try:
            data = load(filepath)
            log.info(f"Data loaded successfully from '{filepath}'.")
            # Basic type check
            if isinstance(data, dict):
                return data
            else:
                log.warning(f"Data loaded from '{filepath}' is not a dictionary ({type(data)}). Using default.")
                return default_data
        except ImportError:
             log.error(f"Joblib not installed. Cannot load data '{filepath}'. Returning default.")
             return default_data
        except Exception as e:
            log.error(f"Error loading data from '{filepath}': {e}. Returning default values.", exc_info=True)
            return default_data

    def _robust_rolling_vwap(self, df: pd.DataFrame, window: int = 20) -> Optional[pd.Series]:
        """
        Calculates a rolling Volume Weighted Average Price (VWAP) robustly.
        Returns None if calculation fails.
        """
        required_cols = ['high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_cols):
            log.warning("VWAP: Missing required columns.")
            return None
        if df['tick_volume'].isnull().all() or (df['tick_volume'] <= 0).all():
             log.warning("VWAP: Volume data is missing, zero, or negative.")
             return None
        # Ensure we have enough data for the window
        calc_window = min(window, len(df))
        if calc_window < 1: return None # Cannot calculate on empty df

        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            volume = df['tick_volume'].fillna(0).replace([np.inf, -np.inf], 0).clip(lower=0) # Ensure non-negative volume

            tpv = (typical_price * volume).reindex(df.index) # Ensure index alignment
            cumulative_tpv = tpv.rolling(window=calc_window, min_periods=1).sum()
            cumulative_volume = volume.reindex(df.index).rolling(window=calc_window, min_periods=1).sum()

            vwap = np.where(cumulative_volume > 1e-9, cumulative_tpv / cumulative_volume, np.nan) # Avoid division by near-zero
            vwap_series = pd.Series(vwap, index=df.index)

            # Optional: Backfill initial NaNs where calculation wasn't possible yet
            # vwap_series = vwap_series.fillna(method='bfill')

            return vwap_series

        except Exception as e:
            log.error(f"Error calculating robust rolling VWAP: {e}", exc_info=False)
            return None

    def _calculate_historical_volatility(self, df: pd.DataFrame, pair: str, lookback: int = 100) -> Optional[float]:
        """
        Calculates and caches historical volatility (std dev of log returns).
        Returns std dev, or None if calculation fails.
        """
        if df is None or df.empty or 'close' not in df.columns:
            log.warning(f"HistVol ({pair}): Invalid DataFrame.")
            return None
        min_required_rows = 5
        if len(df) < min_required_rows:
            log.debug(f"HistVol ({pair}): Not enough data ({len(df)} rows).")
            return None

        try:
            # Use pct_change for robustness against zero prices, then log
            # log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            returns = df['close'].pct_change().dropna()
            if len(returns) < 2:
                log.debug(f"HistVol ({pair}): Not enough returns ({len(returns)}).")
                return None

            actual_lookback = min(lookback, len(returns))
            if actual_lookback < 2: return None

            hist_vol_std = returns.iloc[-actual_lookback:].std()

            if pd.isna(hist_vol_std) or not np.isfinite(hist_vol_std) or hist_vol_std < 0:
                 log.warning(f"HistVol ({pair}): Calculated std dev is invalid ({hist_vol_std}).")
                 return None

            self.volatility_baselines[pair] = hist_vol_std
            log.debug(f"HistVol ({pair}): Calculated volatility (pct return std dev): {hist_vol_std:.6f}")
            return hist_vol_std

        except Exception as e:
            log.error(f"HistVol ({pair}): Error calculating: {e}", exc_info=False)
            return None

    def get_dynamic_period(self, df: pd.DataFrame, base_period: int, indicator_type: str,
                           volatility_factor: float = 0.5, adx_factor: float = 0.3,
                           back_test_enabled: bool = True, pair: str = None) -> int:
        """Dynamically adjusts indicator periods based on market conditions."""
        required_cols = ['high', 'low', 'close']
        if df is None or df.empty or not all(col in df.columns for col in required_cols) or len(df) < self.MIN_DATA_FOR_DYNAMIC_PERIOD:
            log.debug(f"DynamicPeriod ({indicator_type}): Insufficient data ({len(df)} < {self.MIN_DATA_FOR_DYNAMIC_PERIOD}). Using base {base_period}.")
            return base_period

        try:
            atr_period = 14
            adx_period = 14
            try:
                 # Ensure inputs to ta funcs are valid
                 if df['high'].isnull().all() or df['low'].isnull().all() or df['close'].isnull().all():
                     log.warning(f"DynamicPeriod ({indicator_type}): Input HLC data contains all NaNs. Using base {base_period}.")
                     return base_period
                 atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
                 adx_df = ta.adx(df['high'], df['low'], df['close'], length=adx_period)
            except Exception as ta_err:
                 log.error(f"DynamicPeriod ({indicator_type}): Error calculating ATR/ADX: {ta_err}. Using base {base_period}.", exc_info=False)
                 return base_period

            adx_col_name = f'ADX_{adx_period}'
            if atr_series is None or atr_series.empty or atr_series.isnull().all() or \
               adx_df is None or adx_df.empty or adx_col_name not in adx_df.columns or adx_df[adx_col_name].isnull().all():
                 log.warning(f"DynamicPeriod ({indicator_type}): ATR/ADX calculation invalid/NaN. Using base {base_period}.")
                 return base_period

            # Safely get last valid values using dropna().iloc[-1]
            last_atr = atr_series.dropna().iloc[-1] if not atr_series.dropna().empty else np.nan
            last_adx = adx_df[adx_col_name].dropna().iloc[-1] if not adx_df[adx_col_name].dropna().empty else np.nan
            last_close = df['close'].dropna().iloc[-1] if not df['close'].dropna().empty else np.nan


            if pd.isna(last_atr) or pd.isna(last_adx) or pd.isna(last_close) or last_close <= 0 or not np.isfinite(last_atr) or not np.isfinite(last_adx):
                 log.warning(f"DynamicPeriod ({indicator_type}): NaN/Inf values ATR({last_atr})/ADX({last_adx})/Close({last_close}). Using base {base_period}.")
                 return base_period

            normalized_volatility = (last_atr / last_close) * 100.0
            trend_strength = min(max(last_adx - 10, 0) / 40.0, 1.0)

            # Define thresholds
            # These could potentially be loaded from config or adapted per asset class
            vol_thresh_high = 1.0 # Example: 1% ATR/Price
            vol_thresh_low = 0.2  # Example: 0.2% ATR/Price
            adx_thresh_trend = self.ADX_TREND_THRESHOLD
            adx_thresh_range = self.ADX_RANGE_THRESHOLD

            # Determine Market Condition
            market_condition = 'normal'
            if normalized_volatility > vol_thresh_high and last_adx > adx_thresh_trend: market_condition = 'volatile_trending'
            elif normalized_volatility > vol_thresh_high: market_condition = 'volatile_ranging'
            elif normalized_volatility < vol_thresh_low and last_adx < adx_thresh_range: market_condition = 'ranging'
            elif last_adx > adx_thresh_trend: market_condition = 'trending'

            log.debug(f"DynamicPeriod ({indicator_type}, {pair}): Market={market_condition} (Vol={normalized_volatility:.2f}%, ADX={last_adx:.1f})")

            adjusted_period = base_period

            # 1. Check Historical Optimal Periods
            if back_test_enabled:
                 cond_to_check = [market_condition]
                 if market_condition.startswith('volatile'): cond_to_check.append('volatile') # Fallback

                 for cond in cond_to_check:
                     if cond in self.optimal_periods_history:
                         optimal_periods = self.optimal_periods_history[cond]
                         if indicator_type in optimal_periods:
                             adjusted_period = optimal_periods[indicator_type]
                             log.info(f"DynamicPeriod ({indicator_type}, {pair}): Using optimal period for '{cond}': {adjusted_period}")
                             final_period = max(self.MIN_INDICATOR_PERIOD, min(adjusted_period, int(base_period * self.MAX_PERIOD_MULTIPLIER)))
                             return final_period # Found optimal, return

            # 2. Adjust based on current conditions if no optimal found
            adjustment = 1.0
            if market_condition == 'volatile_ranging' or market_condition == 'volatile_trending':
                vol_scale = min(normalized_volatility / vol_thresh_high, 2.5)
                adjustment *= (1.0 - volatility_factor * (vol_scale - 1.0) * 0.7) # Adjust shortening factor
            elif market_condition == 'ranging':
                vol_scale = max(0, min(normalized_volatility / vol_thresh_low, 1.0))
                adjustment *= (1.0 + volatility_factor * (1.0 - vol_scale) * 0.7) # Adjust lengthening factor
                adjustment *= (1.0 + adx_factor * max(0, 1.0 - trend_strength * 1.5)) # More sensitive to weak trend
            elif market_condition == 'trending':
                 adjustment *= (1.0 - adx_factor * max(0, trend_strength - 0.55)) # Adjust trend sensitivity slightly

            adjustment = max(0.35, min(adjustment, 1.9)) # Adjust clamping range
            adjusted_period = int(base_period * adjustment)

            # Final clamping
            final_period = max(self.MIN_INDICATOR_PERIOD, min(adjusted_period, int(base_period * self.MAX_PERIOD_MULTIPLIER)))
            if final_period != base_period:
                log.info(f"DynamicPeriod ({indicator_type}, {pair}): Base={base_period}, Market={market_condition}, Adjusted={final_period}")

            return final_period

        except Exception as e:
            log.error(f"DynamicPeriod ({indicator_type}, {pair}): Error: {e}", exc_info=False)
            return base_period # Fallback

    def calculate_indicators(self, df: pd.DataFrame, pair: str = None) -> Dict[str, Union[pd.Series, Dict, float, None]]:
        """Calculates technical indicators using dynamic periods and robust methods."""
        required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        min_data_length = 30 # Need enough for dynamic periods + lookbacks

        if df is None or df.empty or not all(col in df.columns for col in required_columns):
            log.error(f"Indicators: Invalid/incomplete DataFrame. Need {required_columns}. Got {list(df.columns) if df is not None else 'None'}")
            return {}
        if len(df) < min_data_length:
             log.warning(f"Indicators: Insufficient data ({len(df)} rows, need {min_data_length}).")
             return {}

        # Ensure essential columns have numeric data
        for col in required_columns:
             if not pd.api.types.is_numeric_dtype(df[col]):
                 log.error(f"Indicators: Column '{col}' is not numeric. Cannot proceed.")
                 return {}
             if df[col].isnull().all():
                  log.warning(f"Indicators: Column '{col}' contains only NaN values.")
                  # Allow proceeding, but expect downstream failures

        log.debug(f"Calculating indicators for {pair or 'Unknown'}...")
        indicators: Dict[str, Union[pd.Series, Dict, float, None]] = {}

        try:
            # --- Dynamic Periods ---
            try:
                rsi_p = self.get_dynamic_period(df, 14, 'rsi', pair=pair)
                macd_f, macd_s, macd_sig = self.get_dynamic_period(df, 12, 'macd_fast', pair=pair), self.get_dynamic_period(df, 26, 'macd_slow', pair=pair), 9
                bb_p, bb_std = self.get_dynamic_period(df, 20, 'bollinger', pair=pair), 2.0
                stoch_k, stoch_d, stoch_sm = self.get_dynamic_period(df, 14, 'stoch', pair=pair), 3, 3
                atr_p, adx_p, obv_ma_p, vol_sma_p, vol_rsi_p = 14, 14, 20, 20, 14
                ema_s_p, ema_m_p, ema_l_p = 9, 21, 50
                psar_st, psar_max = 0.02, 0.2
                ichi_t = max(self.MIN_INDICATOR_PERIOD, min(self.get_dynamic_period(df, 9, 'ichimoku_tenkan', pair=pair), 20))
                ichi_k = max(self.MIN_INDICATOR_PERIOD, min(self.get_dynamic_period(df, 26, 'ichimoku_kijun', pair=pair), 40))
                ichi_s = max(self.MIN_INDICATOR_PERIOD, min(self.get_dynamic_period(df, 52, 'ichimoku_senkou', pair=pair), 70))
                ichi_c = ichi_k # Lagging span period = Kijun period
            except Exception as dyn_e:
                 log.error(f"Indicators: Error getting dynamic periods: {dyn_e}. Aborting.", exc_info=False)
                 return {}

            # --- Helper for Safe TA Calls ---
            def safe_ta_call(func, name: str, *args, **kwargs) -> Optional[Union[pd.DataFrame, pd.Series]]:
                input_series_valid = True
                # Check specific series needed by the function (common examples)
                for series_kw in ['close', 'high', 'low', 'volume', 'open']:
                    if series_kw in kwargs:
                        series = kwargs[series_kw]
                        if series is None or series.isnull().all():
                            log.debug(f"Skipping {name}: Input '{series_kw}' series is missing or all NaN.")
                            input_series_valid = False
                            break
                if not input_series_valid: return None

                try:
                    result = func(*args, **kwargs)
                    if result is None: return None
                    # Check for Series/DataFrame results that are all NaN
                    if isinstance(result, pd.Series) and result.isnull().all(): return None
                    if isinstance(result, pd.DataFrame) and result.isnull().all().all(): return None
                    # Check for excessive NaNs (e.g., >95%) which might indicate calculation issues
                    if isinstance(result, pd.Series) and (result.isnull().sum() / len(result) > 0.95 if len(result)>0 else False):
                        log.debug(f"Indicator '{name}' has >95% NaN values.")
                    elif isinstance(result, pd.DataFrame):
                         for col in result.columns:
                             if not result[col].empty and result[col].isnull().sum() / len(result[col]) > 0.95:
                                 log.debug(f"Indicator '{name}' column '{col}' has >95% NaN values.")

                    return result
                except IndexError as ie: # Catch index errors specifically if iloc fails inside TA-Lib/pandas-ta
                    log.warning(f"Indicators: Pandas-TA function {name} ({func.__name__}) raised IndexError: {ie}. Likely insufficient data for periods.", exc_info=False)
                    return None
                except Exception as e:
                    log.warning(f"Indicators: Pandas-TA function {name} ({func.__name__}) failed: {e}", exc_info=False)
                    return None

            # --- Calculations ---
            indicators['rsi'] = safe_ta_call(ta.rsi, 'RSI', close=df['close'], length=rsi_p)
            macd_df = safe_ta_call(ta.macd, 'MACD', close=df['close'], fast=macd_f, slow=macd_s, signal=macd_sig)
            if macd_df is not None:
                 mk, msk, mhk = f"MACD_{macd_f}_{macd_s}_{macd_sig}", f"MACDs_{macd_f}_{macd_s}_{macd_sig}", f"MACDh_{macd_f}_{macd_s}_{macd_sig}"
                 indicators['macd'] = macd_df.get(mk)
                 indicators['macd_signal'] = macd_df.get(msk)
                 indicators['macd_histogram'] = macd_df.get(mhk)
            else: indicators['macd'] = indicators['macd_signal'] = indicators['macd_histogram'] = None

            stoch_df = safe_ta_call(ta.stoch, 'Stochastic', high=df['high'], low=df['low'], close=df['close'], k=stoch_k, d=stoch_d, smooth_k=stoch_sm)
            if stoch_df is not None:
                 sk, sd = f'STOCHk_{stoch_k}_{stoch_d}_{stoch_sm}', f'STOCHd_{stoch_k}_{stoch_d}_{stoch_sm}'
                 indicators['stoch_k'] = stoch_df.get(sk)
                 indicators['stoch_d'] = stoch_df.get(sd)
            else: indicators['stoch_k'] = indicators['stoch_d'] = None

            indicators['atr'] = safe_ta_call(ta.atr, 'ATR', high=df['high'], low=df['low'], close=df['close'], length=atr_p)
            indicators['volatility_rsi'] = safe_ta_call(ta.rsi, 'Volatility RSI', close=indicators['atr'], length=vol_rsi_p) if indicators.get('atr') is not None else None

            bb_df = safe_ta_call(ta.bbands, 'Bollinger', close=df['close'], length=bb_p, std=bb_std)
            if bb_df is not None:
                 bbu, bbm, bbl = f'BBU_{bb_p}_{bb_std}', f'BBM_{bb_p}_{bb_std}', f'BBL_{bb_p}_{bb_std}'
                 bbw, bbp = f'BBB_{bb_p}_{bb_std}', f'BBP_{bb_p}_{bb_std}'
                 indicators['bollinger_upper'] = bb_df.get(bbu)
                 indicators['bollinger_middle'] = bb_df.get(bbm)
                 indicators['bollinger_lower'] = bb_df.get(bbl)
                 indicators['bollinger_width'] = bb_df.get(bbw)
                 indicators['bollinger_percent'] = bb_df.get(bbp)
                 # Squeeze Calculation
                 if indicators['bollinger_upper'] is not None and indicators['bollinger_lower'] is not None and indicators['bollinger_middle'] is not None:
                     bbm_safe = indicators['bollinger_middle'].replace(0, np.nan)
                     with np.errstate(divide='ignore', invalid='ignore'): # Suppress division warnings locally
                        indicators['bbands_squeeze'] = (indicators['bollinger_upper'] - indicators['bollinger_lower']) / bbm_safe
                 else: indicators['bbands_squeeze'] = None
            else:
                 indicators['bollinger_upper'] = indicators['bollinger_middle'] = indicators['bollinger_lower'] = None
                 indicators['bollinger_width'] = indicators['bollinger_percent'] = indicators['bbands_squeeze'] = None

            adx_df = safe_ta_call(ta.adx, 'ADX', high=df['high'], low=df['low'], close=df['close'], length=adx_p)
            if adx_df is not None:
                 adx, pdi, mdi = f'ADX_{adx_p}', f'DMP_{adx_p}', f'DMN_{adx_p}'
                 indicators['adx'] = adx_df.get(adx)
                 indicators['plus_di'] = adx_df.get(pdi)
                 indicators['minus_di'] = adx_df.get(mdi)
            else: indicators['adx'] = indicators['plus_di'] = indicators['minus_di'] = None

            psar_df = safe_ta_call(ta.psar, 'PSAR', high=df['high'], low=df['low'], close=df['close'], step=psar_st, max_step=psar_max)
            if psar_df is not None:
                 pl, ps, pr = f'PSARl_{psar_st}_{psar_max}', f'PSARs_{psar_st}_{psar_max}', f'PSARr_{psar_st}_{psar_max}'
                 indicators['parabolic_sar_long'] = psar_df.get(pl)
                 indicators['parabolic_sar_short'] = psar_df.get(ps)
                 indicators['parabolic_sar_reversal'] = psar_df.get(pr)
            else: indicators['parabolic_sar_long'] = indicators['parabolic_sar_short'] = indicators['parabolic_sar_reversal'] = None

            indicators['ema_short'] = safe_ta_call(ta.ema, 'EMA Short', close=df['close'], length=ema_s_p)
            indicators['ema_medium'] = safe_ta_call(ta.ema, 'EMA Medium', close=df['close'], length=ema_m_p)
            indicators['ema_long'] = safe_ta_call(ta.ema, 'EMA Long', close=df['close'], length=ema_l_p)
            indicators['ema_trend'] = (indicators['ema_short'] - indicators['ema_long']) if indicators.get('ema_short') is not None and indicators.get('ema_long') is not None else None

            indicators['obv'] = safe_ta_call(ta.obv, 'OBV', close=df['close'], volume=df['tick_volume'])
            indicators['obv_ema'] = safe_ta_call(ta.ema, 'OBV EMA', close=indicators['obv'], length=obv_ma_p) if indicators.get('obv') is not None else None

            indicators['volume_sma'] = safe_ta_call(ta.sma, 'Volume SMA', close=df['tick_volume'], length=vol_sma_p)
            if df['tick_volume'] is not None and indicators.get('volume_sma') is not None:
                 vol_sma_safe = indicators['volume_sma'].replace(0, np.nan)
                 with np.errstate(divide='ignore', invalid='ignore'):
                     indicators['volume_sma_ratio'] = df['tick_volume'] / vol_sma_safe
            else: indicators['volume_sma_ratio'] = None

            indicators['price_volume_trend'] = safe_ta_call(ta.pvt, 'PVT', close=df['close'], volume=df['tick_volume'])
            indicators['chaikin_oscillator'] = safe_ta_call(ta.cmf, 'CMF', high=df['high'], low=df['low'], close=df['close'], volume=df['tick_volume'], length=20)

            # Robust VWAP calculation
            indicators['vwap'] = self._robust_rolling_vwap(df, window=20)

            # Ichimoku
            ichi_tuple = safe_ta_call(ta.ichimoku, 'Ichimoku', high=df['high'], low=df['low'], close=df['close'],
                                       tenkan=ichi_t, kijun=ichi_k, senkou=ichi_s, chikou=ichi_c)
            indicators['ichimoku'] = {} # Initialize as empty dict
            if ichi_tuple is not None and isinstance(ichi_tuple, tuple) and len(ichi_tuple) > 0:
                 ichi_df = ichi_tuple[0]
                 if isinstance(ichi_df, pd.DataFrame):
                     # Standard pandas-ta Ichimoku names (use get for safety)
                     spa, spb, ten, kij, chi = f'ISA_{ichi_t}', f'ISB_{ichi_k}', f'ITS_{ichi_t}', f'IKS_{ichi_k}', f'ICS_{ichi_c}'
                     indicators['ichimoku'] = {
                         'tenkan_sen': ichi_df.get(ten),
                         'kijun_sen': ichi_df.get(kij),
                         'senkou_span_a': ichi_df.get(spa),
                         'senkou_span_b': ichi_df.get(spb),
                         'chikou_span': ichi_df.get(chi),
                         'cloud_thickness': None
                     }
                     span_a = indicators['ichimoku'].get('senkou_span_a')
                     span_b = indicators['ichimoku'].get('senkou_span_b')
                     if span_a is not None and span_b is not None:
                         indicators['ichimoku']['cloud_thickness'] = span_a - span_b
                 else: log.warning("Ichimoku calculation returned unexpected format.")
            else: log.warning("Ichimoku calculation failed or returned None.")


            # Fibonacci & Supply/Demand (These return dicts, not Series)
            indicators['fibonacci'] = self._calculate_fibonacci_levels(df)
            indicators['supply_demand'] = self._calculate_supply_demand_zones(df, vwap_series=indicators['vwap'])


            # --- Log Final Indicator Status ---
            valid_indicators_count = sum(1 for v in indicators.values() if v is not None and (not isinstance(v, dict) or v))
            log.info(f"Indicators calculated for {pair or 'Unknown'}. Valid: {valid_indicators_count}/{len(indicators)}")
            return indicators

        except Exception as e:
            log.exception(f"Critical error during indicator calculation pipeline for {pair}: {e}")
            return {} # Return empty dict on major failure

    def _find_recent_swing_points(self, df: pd.DataFrame, lookback: int = 60, window: int = 5) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Identifies the most recent significant swing high/low points."""
        min_len = 2 * window + 1
        if df is None or len(df) < min_len: return None, None

        try:
            data = df.iloc[-min(lookback, len(df)):].copy() # Work on a copy
            if len(data) < min_len: return None, None

            # Use pandas rolling window for efficient peak/trough detection
            data['is_high'] = (data['high'] == data['high'].rolling(window*2 + 1, center=True, min_periods=min_len).max())
            data['is_low'] = (data['low'] == data['low'].rolling(window*2 + 1, center=True, min_periods=min_len).min())

            swing_highs = data[data['is_high']]
            swing_lows = data[data['is_low']]

            latest_high = None
            if not swing_highs.empty:
                 high_point = swing_highs.iloc[-1]
                 latest_high = {'price': high_point['high'], 'index': high_point.name, 'time': high_point.name} # Assuming index is time

            latest_low = None
            if not swing_lows.empty:
                 low_point = swing_lows.iloc[-1]
                 latest_low = {'price': low_point['low'], 'index': low_point.name, 'time': low_point.name}

            # Basic validation: ensure high > low if both found and not identical points
            if latest_high and latest_low:
                 if latest_high['index'] == latest_low['index']:
                      # If same candle is both high and low, prioritize based on close? Or discard?
                      # For simplicity, let's keep both for now, Fib logic might handle it.
                      log.debug(f"Swing high and low occurred on the same candle: {latest_high['time']}")
                 elif latest_high['price'] <= latest_low['price']:
                      log.warning(f"Swing high price ({latest_high['price']}) <= swing low price ({latest_low['price']}). Invalid swings detected.")
                      # Potentially discard one or both? Returning as is for now.
                      # latest_high = None # Example: discard invalid high
                      pass

            # log.debug(f"Found swings: High={latest_high}, Low={latest_low}")
            return latest_high, latest_low

        except Exception as e:
            log.error(f"Error finding swing points: {e}", exc_info=False)
            return None, None

    def _calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 60, swing_window: int = 5) -> Dict:
        """Calculates Fibonacci retracement levels based on recent swing high/low."""
        default_fib = {f'level_{lvl}': np.nan for lvl in ['0', '236', '382', '500', '618', '786', '100']}
        default_fib.update({'direction': 'unknown', 'swing_high_price': np.nan, 'swing_low_price': np.nan})

        if df is None or df.empty or len(df) < 20: # Need some data for swings/range
            return default_fib

        try:
            swing_high, swing_low = self._find_recent_swing_points(df, lookback=lookback, window=swing_window)

            high_p, low_p = np.nan, np.nan
            high_time, low_time = None, None
            direction = 'unknown'

            if swing_high and swing_low and swing_high['price'] > swing_low['price']:
                high_p, low_p = swing_high['price'], swing_low['price']
                high_time, low_time = swing_high.get('time'), swing_low.get('time')
                # Determine direction based on which swing occurred later (dominant trend leading to current price)
                if high_time and low_time:
                    direction = 'bullish' if low_time < high_time else 'bearish' # If low came first, uptrend; if high came first, downtrend
            else:
                # Fallback to recent range if swings are invalid or missing
                log.debug("Fibonacci: Valid swing points not found, using recent range.")
                recent_data = df.iloc[-lookback:] # Use lookback period
                if not recent_data.empty:
                    high_p = recent_data['high'].max()
                    low_p = recent_data['low'].min()
                direction = 'ranging' # Or unknown, range implies ranging

            if pd.isna(high_p) or pd.isna(low_p) or high_p <= low_p:
                 log.warning("Fibonacci: Invalid high/low points for calculation.")
                 return default_fib

            diff = high_p - low_p
            levels = {
                'level_0': low_p,
                'level_236': low_p + 0.236 * diff,
                'level_382': low_p + 0.382 * diff,
                'level_500': low_p + 0.500 * diff,
                'level_618': low_p + 0.618 * diff,
                'level_786': low_p + 0.786 * diff, # Optional level
                'level_100': high_p,
                'direction': direction, # Direction of the move that created the range (low->high or high->low)
                'swing_high_price': swing_high['price'] if swing_high else high_p, # Report used high
                'swing_low_price': swing_low['price'] if swing_low else low_p    # Report used low
            }
            # Round levels for clarity
            for key in levels:
                if isinstance(levels[key], (float, np.number)):
                    levels[key] = round(levels[key], 5) # Adjust rounding based on pair precision

            return levels

        except Exception as e:
            log.error(f"Error calculating Fibonacci levels: {e}", exc_info=False)
            return default_fib


    def _calculate_supply_demand_zones(self, df: pd.DataFrame, vwap_series: Optional[pd.Series] = None, lookback: int = 30, vol_multiplier: float = 1.75) -> Dict:
        """
        Identifies potential supply and demand zones based on recent price action, pivots, and volume.
        """
        default_zones = {
            'pivot': np.nan, 'resistance1': np.nan, 'resistance2': np.nan, 'resistance3': np.nan,
            'support1': np.nan, 'support2': np.nan, 'support3': np.nan, 'vwap': np.nan,
            'closest_supply': {'top': np.nan, 'bottom': np.nan, 'strength': 0.0},
            'closest_demand': {'top': np.nan, 'bottom': np.nan, 'strength': 0.0},
            'all_supply_zones': [], 'all_demand_zones': []
        }
        required_cols = ['high', 'low', 'close', 'open', 'tick_volume']
        if df is None or df.empty or not all(c in df.columns for c in required_cols) or len(df) < 5:
             log.warning("S/D Zones: Insufficient data.")
             return default_zones

        try:
            # Use data within lookback, ensure enough rows remain
            data = df.iloc[-min(len(df), lookback):].copy()
            if len(data) < 5: return default_zones

            last_row = data.iloc[-1]
            prev_row = data.iloc[-2] if len(data) > 1 else last_row # Use previous day/period for classic pivots
            last_high, last_low, last_close = prev_row['high'], prev_row['low'], prev_row['close'] # Use previous period's HLC

            # --- Pivot Points (Classic - based on previous period) ---
            if pd.notna(last_high) and pd.notna(last_low) and pd.notna(last_close):
                pivot = (last_high + last_low + last_close) / 3.0
                r1 = (2.0 * pivot) - last_low
                s1 = (2.0 * pivot) - last_high
                r2 = pivot + (last_high - last_low)
                s2 = pivot - (last_high - last_low)
                r3 = last_high + 2.0 * (pivot - last_low)
                s3 = last_low - 2.0 * (last_high - pivot)
            else:
                pivot = r1 = s1 = r2 = s2 = r3 = s3 = np.nan

            # --- VWAP ---
            last_vwap = vwap_series.dropna().iloc[-1] if vwap_series is not None and not vwap_series.dropna().empty else pivot # Use calculated VWAP or fallback to pivot

            # --- Volume-Based Zones ---
            # Calculate average volume and body size in the lookback period
            avg_volume = data['tick_volume'].mean()
            data['body'] = abs(data['close'] - data['open'])
            avg_body = data['body'].mean()

            # Define high volume relative to recent average
            high_volume_threshold = avg_volume * vol_multiplier
            large_body_threshold = avg_body * 1.2 # Candle body larger than average

            # Identify potential zone candles (high volume, potentially large body)
            zone_candles = data[data['tick_volume'] > high_volume_threshold].copy()

            supply_zones_raw = []
            demand_zones_raw = []

            # Iterate through high-volume candles
            for index, candle in zone_candles.iterrows():
                # Strength based on volume ratio and body size (optional)
                strength = candle['tick_volume'] / avg_volume # Primary factor: volume

                # Supply Zone Criteria: Typically a strong down move preceded by a smaller up candle (base)
                # Simplified: Strong bearish candle (close < open) with high volume. Zone = open to high.
                if candle['close'] < candle['open']: # Bearish candle
                    # Zone is often considered the range of the candle body or full candle
                    zone_top = candle['high'] # Use candle high as top
                    zone_bottom = candle['open'] # Use candle open as bottom
                    # Add strength modifier based on body size?
                    if candle['body'] > large_body_threshold: strength *= 1.2
                    supply_zones_raw.append({'top': zone_top, 'bottom': zone_bottom, 'strength': round(strength, 2)})

                # Demand Zone Criteria: Typically a strong up move preceded by a smaller down candle (base)
                # Simplified: Strong bullish candle (close > open) with high volume. Zone = low to open.
                elif candle['close'] > candle['open']: # Bullish candle
                    zone_top = candle['open'] # Use candle open as top
                    zone_bottom = candle['low'] # Use candle low as bottom
                    if candle['body'] > large_body_threshold: strength *= 1.2
                    demand_zones_raw.append({'top': zone_top, 'bottom': zone_bottom, 'strength': round(strength, 2)})

            # --- Merge Overlapping Zones (Simplified) ---
            def merge_zones(zones: List[Dict]) -> List[Dict]:
                if not zones: return []
                zones.sort(key=lambda z: z['bottom']) # Sort by bottom price
                merged = [zones[0]]
                for current in zones[1:]:
                    previous = merged[-1]
                    # Check for overlap: current bottom < previous top
                    if current['bottom'] <= previous['top']:
                        # Merge: extend top, keep bottom, combine strength (e.g., max)
                        previous['top'] = max(previous['top'], current['top'])
                        previous['strength'] = max(previous['strength'], current['strength'])
                        # Keep the lower bottom (already sorted)
                    else:
                        merged.append(current)
                return merged

            merged_supply = merge_zones(supply_zones_raw)
            merged_demand = merge_zones(demand_zones_raw)

            # --- Determine Closest Relevant Zones ---
            current_price = df['close'].iloc[-1] # Use the actual latest close price

            # Filter zones relative to current price and find closest
            valid_supply = [zone for zone in merged_supply if zone['bottom'] > current_price]
            valid_demand = [zone for zone in merged_demand if zone['top'] < current_price]

            closest_supply = min(valid_supply, key=lambda z: z['bottom']) if valid_supply else None
            closest_demand = max(valid_demand, key=lambda z: z['top']) if valid_demand else None

            # --- Consolidate Results ---
            results = {
                'pivot': round(pivot, 5) if pd.notna(pivot) else np.nan,
                'resistance1': round(r1, 5) if pd.notna(r1) else np.nan,
                'resistance2': round(r2, 5) if pd.notna(r2) else np.nan,
                'resistance3': round(r3, 5) if pd.notna(r3) else np.nan,
                'support1': round(s1, 5) if pd.notna(s1) else np.nan,
                'support2': round(s2, 5) if pd.notna(s2) else np.nan,
                'support3': round(s3, 5) if pd.notna(s3) else np.nan,
                'vwap': round(last_vwap, 5) if pd.notna(last_vwap) else np.nan,
                'closest_supply': {'top': round(closest_supply['top'], 5), 'bottom': round(closest_supply['bottom'], 5), 'strength': closest_supply['strength']} if closest_supply else default_zones['closest_supply'],
                'closest_demand': {'top': round(closest_demand['top'], 5), 'bottom': round(closest_demand['bottom'], 5), 'strength': closest_demand['strength']} if closest_demand else default_zones['closest_demand'],
                # Optionally return all found zones (can be noisy)
                'all_supply_zones': sorted(valid_supply, key=lambda z: z['bottom'])[:3], # Top 3 closest
                'all_demand_zones': sorted(valid_demand, key=lambda z: z['top'], reverse=True)[:3] # Top 3 closest
            }
            return results

        except Exception as e:
            log.error(f"Error calculating supply/demand zones: {e}", exc_info=False)
            return default_zones # Return default structure on error


    def _detect_trend(self, df: pd.DataFrame, indicators: Dict) -> str:
        """Detects market trend ('bullish', 'bearish', 'ranging', 'weak_bullish', 'weak_bearish')."""
        # Required indicators for trend detection
        required_indicators = ['adx', 'plus_di', 'minus_di', 'ema_short', 'ema_long']
        missing_indicators = [ind for ind in required_indicators if ind not in indicators or indicators.get(ind) is None or indicators[ind].dropna().empty]
        
        if missing_indicators:
            log.warning(f"Trend detection skipped: Missing or invalid required indicators ({', '.join(missing_indicators)}).")
            return 'ranging' # Default if inputs are bad

        try:
            # Safely get the last valid value
            last_adx = indicators['adx'].dropna().iloc[-1] if not indicators['adx'].dropna().empty else np.nan
            last_pdi = indicators['plus_di'].dropna().iloc[-1] if not indicators['plus_di'].dropna().empty else np.nan
            last_mdi = indicators['minus_di'].dropna().iloc[-1] if not indicators['minus_di'].dropna().empty else np.nan
            last_ema_short = indicators['ema_short'].dropna().iloc[-1] if not indicators['ema_short'].dropna().empty else np.nan
            last_ema_long = indicators['ema_long'].dropna().iloc[-1] if not indicators['ema_long'].dropna().empty else np.nan

            # Check for NaN values after retrieval
            if pd.isna(last_adx) or pd.isna(last_pdi) or pd.isna(last_mdi) or pd.isna(last_ema_short) or pd.isna(last_ema_long):
                log.warning("Trend detection skipped: NaN values in required indicators.")
                return 'ranging'

            # --- Trend Logic ---
            adx_trend = self.ADX_TREND_THRESHOLD
            adx_range = self.ADX_RANGE_THRESHOLD

            is_adx_trending = last_adx > adx_trend
            is_adx_ranging = last_adx < adx_range

            di_bullish = last_pdi > last_mdi
            di_bearish = last_mdi > last_pdi
            ema_bullish = last_ema_short > last_ema_long
            ema_bearish = last_ema_short < last_ema_long

            # Combine signals
            if is_adx_trending:
                if di_bullish and ema_bullish: return 'bullish'
                elif di_bearish and ema_bearish: return 'bearish'
                else:
                    log.debug(f"Trend: Strong ADX ({last_adx:.1f}) but conflicting DI/EMA signals.")
                    return 'ranging' # Treat strong conflict as uncertainty
            elif is_adx_ranging:
                return 'ranging'
            else: # ADX between range and trend thresholds (weak/developing)
                if di_bullish and ema_bullish: return 'weak_bullish'
                elif di_bearish and ema_bearish: return 'weak_bearish'
                else:
                    return 'ranging' # Weak conflict or indeterminate

        except IndexError:
             log.warning("Trend detection failed: IndexError (likely insufficient indicator data length).")
             return 'ranging'
        except Exception as e:
            log.error(f"Error detecting trend: {e}", exc_info=False)
            return 'ranging'


    def _calculate_indicator_weights(self, df: pd.DataFrame, indicators: Dict, pair: str = None) -> Dict[str, float]:
        """Calculates dynamic weights for indicators based on market conditions and historical performance."""
        # Base Weights (Define relative importance - should sum roughly to 1)
        # Adjusted to give more balanced initial weights
        base_weights = {
            # Oscillators
            'rsi': 0.11, 'stoch': 0.09, 'bollinger': 0.10,
            # Trend / Momentum
            'macd': 0.12, 'adx': 0.11, 'ema_trend': 0.09, 'parabolic_sar': 0.07, 'ichimoku': 0.10,
            # Volume
            'obv': 0.06, 'chaikin_oscillator': 0.05, #'volume_sma_ratio': 0.03,
            # Volatility (less direct signal, lower weight)
            'atr': 0.03, 'volatility_rsi': 0.04,
            # Other (Fib/SD used contextually, not direct score contribution usually)
            'fibonacci': 0.0, 'supply_demand': 0.0
        }
        # Ensure base weights sum close to 1, normalize if not
        current_sum = sum(base_weights.values())
        if not math.isclose(current_sum, 1.0):
            log.debug(f"Normalizing base weights (Sum was {current_sum:.3f})")
            base_weights = {k: v / current_sum for k, v in base_weights.items()} if current_sum > 0 else base_weights


        adjusted_weights = base_weights.copy()
        min_weight = 0.01 # Minimum weight an indicator can have after adjustment

        try:
            # --- Get Market Context ---
            trend_state = self._detect_trend(df, indicators)
            is_trending = 'bullish' in trend_state or 'bearish' in trend_state
            is_ranging = trend_state == 'ranging'

            # Get Last ATR and Close safely
            last_atr = indicators.get('atr', pd.Series(dtype=float)).dropna().iloc[-1] if indicators.get('atr') is not None and not indicators['atr'].dropna().empty else np.nan
            last_close = df['close'].dropna().iloc[-1] if not df['close'].dropna().empty else np.nan

            normalized_volatility = np.nan
            if pd.notna(last_atr) and pd.notna(last_close) and last_close > 0:
                 normalized_volatility = (last_atr / last_close) * 100.0
                 vol_thresh_high = 1.0 # Example threshold %
                 vol_thresh_low = 0.2 # Example threshold %
                 is_volatile = normalized_volatility > vol_thresh_high
                 is_low_volatility = normalized_volatility < vol_thresh_low
            else:
                 is_volatile = is_low_volatility = False
                 log.warning("Weights: Cannot determine volatility for adjustment due to missing ATR/Close.")


            # --- Adjust Weights Based on Market Conditions ---
            log.debug(f"Adjusting weights. Trend={trend_state}, Volatile={is_volatile}, LowVol={is_low_volatility}")
            trend_boost = 1.4
            range_boost = 1.4
            trend_reduce = 0.65
            range_reduce = 0.65
            vol_boost = 1.3
            vol_reduce = 0.8

            if is_trending:
                for key in ['macd', 'adx', 'ema_trend', 'parabolic_sar', 'ichimoku']:
                    if key in adjusted_weights: adjusted_weights[key] *= trend_boost
                for key in ['rsi', 'stoch', 'bollinger']: # Oscillators less reliable in strong trends
                    if key in adjusted_weights: adjusted_weights[key] *= trend_reduce
            elif is_ranging:
                for key in ['rsi', 'stoch', 'bollinger']: # Oscillators more reliable
                     if key in adjusted_weights: adjusted_weights[key] *= range_boost
                for key in ['macd', 'adx', 'ema_trend', 'parabolic_sar']: # Trend indicators less reliable
                    if key in adjusted_weights: adjusted_weights[key] *= range_reduce
            # Volatility adjustments (can overlay trend/range)
            if is_volatile:
                 for key in ['atr', 'volatility_rsi', 'bollinger']: # Volatility indicators
                      if key in adjusted_weights: adjusted_weights[key] *= vol_boost
                 # Potentially reduce weight of slower indicators?
                 for key in ['ema_long', 'ichimoku']: # Example
                       if key in adjusted_weights and 'ema_long' in adjusted_weights: adjusted_weights[key] *= vol_reduce # Reduce EMA Long weight more directly?
            elif is_low_volatility:
                 # Increase ranging indicators slightly more if low vol confirmed
                 for key in ['rsi', 'stoch', 'bollinger']:
                      if key in adjusted_weights: adjusted_weights[key] *= 1.15 # Slightly higher boost


            # --- Adjust Based on Historical Performance ---
            perf_factor_strength = 0.25 # Reduced influence from historical performance
            for indicator, perf in self.indicator_performance.items():
                 if indicator in adjusted_weights and adjusted_weights[indicator] > 0: # Only adjust active indicators
                      win_rate_norm = perf.get('win_rate', 0.5)
                      profit_factor_capped = min(max(perf.get('profit_factor', 1.0), 0.5), 3.0) # Cap PF 0.5-3.0
                      profit_factor_norm = (profit_factor_capped - 0.5) / 2.5 # Scale to 0-1
                      perf_score = (win_rate_norm + profit_factor_norm) / 2.0 # Score 0-1

                      # Adjust weight: multiplier between (1-strength) and (1+strength)
                      multiplier = 1.0 + perf_factor_strength * (perf_score - 0.5) * 2.0 # Scale score 0-1 -> adjustment -strength to +strength
                      adjusted_weights[indicator] *= max(0.2, multiplier) # Apply multiplier, prevent going too low


            # --- Final Normalization ---
            adjusted_weights = {k: max(min_weight, v) if v > 0 else 0 for k, v in adjusted_weights.items()} # Ensure min weight only if active
            total_weight = sum(adjusted_weights.values())

            if total_weight <= 0:
                 log.error("Weights: Total calculated indicator weight is zero or negative. Falling back to base weights.")
                 total_base = sum(base_weights.values())
                 return {k: v / total_base for k, v in base_weights.items()} if total_base > 0 else {}

            normalized_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
            log.debug(f"Weights: Calculated normalized weights: { {k: round(v, 3) for k, v in normalized_weights.items() if v > 0} }")
            return normalized_weights

        except Exception as e:
            log.exception("Weights: Error calculating dynamic weights. Using base weights.")
            total_base = sum(base_weights.values())
            return {k: v / total_base for k, v in base_weights.items()} if total_base > 0 else {}


    def filter_signal_by_volatility(self, df: pd.DataFrame, signal_type: str, pair: str = None,
                                  timeframe: TimeFrames = TimeFrames.H1) -> Tuple[bool, float]:
        """
        Filters signals based on current volatility (ATR/Price) relative to thresholds.
        Adjusts position size factor based on volatility.

        Returns:
            Tuple[bool, float]: (is_valid_signal, position_size_factor)
        """
        # Base thresholds (ATR/Price percentage) - Consider making these configurable per timeframe/asset
        VOLATILITY_THRESHOLDS = {
            TimeFrames.M15: {'base': 0.05, 'high': 0.20}, # Adjusted example thresholds
            TimeFrames.H1:  {'base': 0.07, 'high': 0.30},
            TimeFrames.H4:  {'base': 0.12, 'high': 0.50},
            TimeFrames.D1:  {'base': 0.20, 'high': 0.80},
        }
        DEFAULT_THRESHOLDS = {'base': 0.08, 'high': 0.35}

        ADX_TREND_STRONG = 30
        ADX_TREND_WEAK = 20
        POS_SIZE_MIN_FACTOR = 0.3
        POS_SIZE_VOL_SENSITIVITY = 0.6 # Lower sensitivity

        if df is None or df.empty or not all(c in df.columns for c in ['high', 'low', 'close']) or len(df) < 15:
            log.warning(f"Volatility Filter ({pair} {timeframe.name}): Insufficient data. Allowing signal, default size factor 1.0.")
            return True, 1.0

        try:
            # --- Calculate Current Volatility & ADX ---
            atr_series = ta.atr(df['high'], df['low'], df['close'], length=14)
            adx_series_df = ta.adx(df['high'], df['low'], df['close'], length=14)

            if atr_series is None or atr_series.dropna().empty:
                 log.warning(f"Volatility Filter ({pair} {timeframe.name}): ATR calculation failed. Allowing signal.")
                 return True, 1.0

            last_atr = atr_series.dropna().iloc[-1]
            last_close = df['close'].dropna().iloc[-1]
            last_adx = np.nan
            if adx_series_df is not None and 'ADX_14' in adx_series_df and not adx_series_df['ADX_14'].dropna().empty:
                last_adx = adx_series_df['ADX_14'].dropna().iloc[-1]

            if pd.isna(last_atr) or pd.isna(last_close) or last_close <= 0:
                 log.warning(f"Volatility Filter ({pair} {timeframe.name}): NaN ATR/Close. Allowing signal.")
                 return True, 1.0

            current_volatility_pct = (last_atr / last_close) * 100.0

            # --- Determine Thresholds ---
            thresholds = VOLATILITY_THRESHOLDS.get(timeframe, DEFAULT_THRESHOLDS)
            base_thresh = thresholds['base']
            high_thresh = thresholds['high']

            # --- Adjust Thresholds Based on Context ---
            # 1. Historical Volatility (Example using baseline std dev)
            if pair and pair in self.volatility_baselines:
                # Comparing ATR% to baseline std dev directly isn't perfect. Needs careful calibration.
                # Simplified approach: If current ATR% is >> baseline std dev * some factor, increase threshold
                hist_vol_std = self.volatility_baselines[pair]
                # Heuristic factor to convert std dev to comparable % range
                factor = 50 # Very rough estimate, needs tuning
                if current_volatility_pct > hist_vol_std * factor * 2.0: # Significantly more volatile than baseline?
                    base_thresh *= 1.3
                    high_thresh *= 1.2
                    log.debug(f"Volatility Filter ({pair}): High relative vol, adjusting thresholds.")
                elif current_volatility_pct < hist_vol_std * factor * 0.5: # Significantly less volatile?
                    base_thresh *= 0.8
                    log.debug(f"Volatility Filter ({pair}): Low relative vol, adjusting base threshold.")

            # 2. Trend Strength (ADX)
            if pd.notna(last_adx):
                if last_adx > ADX_TREND_STRONG: # Strong trend - allow slightly lower vol entry?
                     base_thresh *= 0.90
                     log.debug(f"Volatility Filter ({pair}): Strong trend (ADX={last_adx:.1f}), adjusting base threshold lower.")
                elif last_adx < ADX_TREND_WEAK: # Ranging - require higher vol confirmation?
                     base_thresh *= 1.10
                     log.debug(f"Volatility Filter ({pair}): Weak trend (ADX={last_adx:.1f}), adjusting base threshold higher.")

            # --- Signal Validity Check ---
            is_valid_signal = current_volatility_pct >= base_thresh

            # --- Position Size Factor Calculation ---
            position_size_factor = 1.0
            if current_volatility_pct > high_thresh:
                # Reduce size if volatility exceeds high threshold
                excess_vol_ratio = max(0, (current_volatility_pct - high_thresh) / high_thresh) # Ratio above threshold
                # Reduction factor using smoother curve: 1 / (1 + sensitivity * ratio)
                reduction_factor = 1.0 / (1.0 + POS_SIZE_VOL_SENSITIVITY * excess_vol_ratio)
                position_size_factor = max(POS_SIZE_MIN_FACTOR, reduction_factor)
                log.info(f"Volatility Filter ({pair} {timeframe.name}): High volatility ({current_volatility_pct:.2f}% > {high_thresh:.2f}%). Reducing size factor to {position_size_factor:.2f}")
            elif not is_valid_signal:
                 # If signal is invalid due to low vol, maybe set size factor low? Or just filter out?
                 # Currently just filtering out. Could return (False, 0.0) for example.
                 pass


            log_level = logging.INFO if not is_valid_signal else logging.DEBUG
            log.log(log_level, f"Volatility Filter ({pair} {timeframe.name}): "
                      f"Current={current_volatility_pct:.3f}%, Threshold={base_thresh:.3f}%. "
                      f"Signal Valid: {is_valid_signal}. Pos Size Factor: {position_size_factor:.2f}")

            return is_valid_signal, position_size_factor

        except Exception as e:
            log.error(f"Error in volatility filter for {pair} {timeframe.name}: {e}", exc_info=False)
            return True, 1.0 # Default to allow signal on error

    def calculate_technical_score(self, df: pd.DataFrame, indicators: Dict,
                                 trend_bias: str = 'ranging', pair: str = None) -> Tuple[float, Dict]:
        """
        Calculates a technical score (0-1) based on weighted indicator signals.

        Returns:
            Tuple[float, Dict]: (final_score_0_to_1, details_dictionary_neg1_to_pos1)
        """
        if df is None or df.empty or len(df) < self.MIN_DATA_FOR_SIGNAL: # Use overall min data length
            log.warning(f"TechScore ({pair}): Insufficient data ({len(df)} < {self.MIN_DATA_FOR_SIGNAL}).")
            return 0.5, {} # Neutral score, empty details
        if not indicators:
            log.warning(f"TechScore ({pair}): Indicators dictionary is empty.")
            return 0.5, {}

        log.debug(f"Calculating technical score for {pair or 'Unknown'}...")
        details = {} # Stores individual indicator scores (-1 to +1)
        score_contributions = {} # Stores weighted contributions

        # --- Safely get latest values from indicator Series ---
        def get_latest(series: Optional[pd.Series], lookback: int = 1) -> Any:
             if series is None or series.dropna().empty: return np.nan
             try:
                 return series.dropna().iloc[-lookback]
             except IndexError:
                 return np.nan # Not enough non-NaN values

        # --- Get Dynamic Weights ---
        weights = self._calculate_indicator_weights(df, indicators, pair)

        # --- Get Latest Price/Indicator Data ---
        last_close = get_latest(df['close'])
        prev_close = get_latest(df['close'], 2)
        # last_high = get_latest(df['high'])
        # last_low = get_latest(df['low'])
        if pd.isna(last_close):
             log.error(f"TechScore ({pair}): Cannot get latest close price.")
             return 0.5, {}

        # --- Indicator Analysis & Scoring (-1 to +1) ---

        # RSI
        rsi = get_latest(indicators.get('rsi'))
        if pd.notna(rsi):
             if rsi > self.RSI_OVERBOUGHT: score = -1.0 # Strong Overbought
             elif rsi > self.RSI_STRONG_REVERSAL_BUY: score = -0.5 # Weakening Momentum Up / Potential Reversal
             elif rsi < self.RSI_OVERSOLD: score = 1.0 # Strong Oversold
             elif rsi < self.RSI_STRONG_REVERSAL_SELL: score = 0.5 # Weakening Momentum Down / Potential Reversal
             else: score = np.interp(rsi, [self.RSI_STRONG_REVERSAL_SELL, self.RSI_STRONG_REVERSAL_BUY], [-0.2, 0.2]) # Linear interp in neutral zone
             details['rsi'] = score
        else: details['rsi'] = 0.0

        # MACD
        macd = get_latest(indicators.get('macd'))
        sig = get_latest(indicators.get('macd_signal'))
        hist = get_latest(indicators.get('macd_histogram'))
        prev_hist = get_latest(indicators.get('macd_histogram'), 2)
        if pd.notna(macd) and pd.notna(sig) and pd.notna(hist):
            score = 0.0
            # Crossover / Position
            if macd > sig and hist > 0: score = 0.8 # Bullish crossover/state
            elif macd < sig and hist < 0: score = -0.8 # Bearish crossover/state
            elif macd > sig and hist <= 0: score = 0.2 # Bullish state, but histo weakening/crossed below zero
            elif macd < sig and hist >= 0: score = -0.2 # Bearish state, but histo weakening/crossed above zero
            # Momentum (Histogram change)
            if pd.notna(prev_hist):
                if hist > 0 and prev_hist >= 0 and hist > prev_hist: score += 0.2 # Increasing bullish momentum
                elif hist < 0 and prev_hist <= 0 and hist < prev_hist: score -= 0.2 # Increasing bearish momentum
                elif hist > 0 and prev_hist < 0: score += 0.3 # Bullish zero cross
                elif hist < 0 and prev_hist > 0: score -= 0.3 # Bearish zero cross
            details['macd'] = np.clip(score, -1.0, 1.0)
        else: details['macd'] = 0.0

        # Bollinger Bands
        bbu = get_latest(indicators.get('bollinger_upper'))
        bbl = get_latest(indicators.get('bollinger_lower'))
        bbp = get_latest(indicators.get('bollinger_percent')) # %B
        bbw = get_latest(indicators.get('bollinger_width')) # Bandwidth
        prev_bbw = get_latest(indicators.get('bollinger_width'), 2)
        if pd.notna(bbp) and pd.notna(last_close):
            score = 0.0
            # Mean Reversion based on %B
            if bbp > 1.05: score = -1.0 # Strongly above upper band
            elif bbp > 0.9: score = -0.6
            elif bbp < -0.05: score = 1.0 # Strongly below lower band
            elif bbp < 0.1: score = 0.6
            elif bbp > 0.5: score = -0.2 # Upper half
            else: score = 0.2 # Lower half
            # Squeeze / Breakout potential (check bandwidth change)
            if pd.notna(bbw) and pd.notna(prev_bbw):
                 if bbw < prev_bbw * 0.8 and bbw < 0.1: # Bandwidth tightening significantly and low -> Squeeze forming
                     # Neutral score, maybe slight bias towards breakout direction later?
                     score *= 0.5 # Reduce mean reversion signal during squeeze
                 elif bbw > prev_bbw * 1.2 and prev_bbw < 0.1: # Bandwidth expanding from low level -> Breakout?
                      # Enhance signal if price breaks out of band during expansion
                      if bbp > 1.0: score = -1.0 # Stronger sell if breaking high
                      elif bbp < 0.0: score = 1.0 # Stronger buy if breaking low
            details['bollinger'] = score
        else: details['bollinger'] = 0.0

        # Stochastic
        k = get_latest(indicators.get('stoch_k'))
        d = get_latest(indicators.get('stoch_d'))
        prev_k = get_latest(indicators.get('stoch_k'), 2)
        prev_d = get_latest(indicators.get('stoch_d'), 2)
        if pd.notna(k) and pd.notna(d):
            score = 0.0
            if k < self.STOCH_OVERSOLD and d < self.STOCH_OVERSOLD: score = 1.0 # Deep Oversold
            elif k > self.STOCH_OVERBOUGHT and d > self.STOCH_OVERBOUGHT: score = -1.0 # Deep Overbought
            elif pd.notna(prev_k) and pd.notna(prev_d):
                # Crossover detection
                if k > d and prev_k <= prev_d: score = 0.8 # Bullish crossover just happened
                elif k < d and prev_k >= prev_d: score = -0.8 # Bearish crossover just happened
                elif k > d: score = 0.3 # K above D (bullish momentum)
                else: score = -0.3 # K below D (bearish momentum)
            # Adjust score based on location (more confidence near extremes)
            if k < self.STOCH_OVERSOLD + 10: score = max(score, 0.6) # Boost bullish score if near oversold
            elif k > self.STOCH_OVERBOUGHT - 10: score = min(score, -0.6) # Boost bearish score if near overbought
            details['stoch'] = np.clip(score, -1.0, 1.0)
        else: details['stoch'] = 0.0

        # ADX & DI
        adx = get_latest(indicators.get('adx'))
        pdi = get_latest(indicators.get('plus_di'))
        mdi = get_latest(indicators.get('minus_di'))
        if pd.notna(adx) and pd.notna(pdi) and pd.notna(mdi):
            score = 0.0
            if adx > self.ADX_TREND_THRESHOLD: # Trending market
                 strength_factor = min(adx / 50.0, 1.2) # Scale score by ADX strength, allow > 1
                 if pdi > mdi: score = 1.0 * strength_factor # Bullish trend
                 else: score = -1.0 * strength_factor # Bearish trend
            elif adx < self.ADX_RANGE_THRESHOLD: # Ranging market
                 score = 0.0 # No directional bias from ADX/DI in range
            else: # Weak trend zone
                 strength_factor = max(0, (adx - self.ADX_RANGE_THRESHOLD) / (self.ADX_TREND_THRESHOLD - self.ADX_RANGE_THRESHOLD)) # 0-1 scale
                 if pdi > mdi: score = 0.4 * strength_factor # Weak bullish bias
                 else: score = -0.4 * strength_factor # Weak bearish bias
            details['adx'] = np.clip(score, -1.0, 1.0)
        else: details['adx'] = 0.0

        # EMA Trend
        ema_trend = get_latest(indicators.get('ema_trend'))
        atr = get_latest(indicators.get('atr'))
        if pd.notna(ema_trend) and pd.notna(atr) and atr > 1e-9:
             trend_strength = ema_trend / atr # EMA separation in ATR terms
             score = np.clip(trend_strength * 0.3, -1.0, 1.0) # Scale and clip, adjust multiplier (0.3 is moderate)
             details['ema_trend'] = score
        else: details['ema_trend'] = 0.0

        # Ichimoku Cloud
        ichi_data = indicators.get('ichimoku', {})
        if isinstance(ichi_data, dict) and ichi_data: # Check if dict and not empty
             tenkan = get_latest(ichi_data.get('tenkan_sen'))
             kijun = get_latest(ichi_data.get('kijun_sen'))
             span_a = get_latest(ichi_data.get('senkou_span_a'))
             span_b = get_latest(ichi_data.get('senkou_span_b'))
             chikou = get_latest(ichi_data.get('chikou_span'))
             # Chikou needs price from the past (lag period)
             chikou_lag = 26 # Standard lag, should ideally match kijun/chikou period used
             price_at_chikou_time = get_latest(df['close'], chikou_lag + 1) # Get close price 'chikou_lag' bars ago

             if all(pd.notna(v) for v in [tenkan, kijun, span_a, span_b, last_close]):
                 ichi_score = 0.0
                 # Price vs Cloud
                 cloud_top = max(span_a, span_b)
                 cloud_bottom = min(span_a, span_b)
                 if last_close > cloud_top: ichi_score += 0.4 # Above cloud (Bullish)
                 elif last_close < cloud_bottom: ichi_score -= 0.4 # Below cloud (Bearish)
                 else: ichi_score += 0.0 # Inside cloud (Neutral/Uncertain)

                 # Tenkan/Kijun Cross & Position
                 if tenkan > kijun: ichi_score += 0.25 # Bullish cross/position
                 else: ichi_score -= 0.25 # Bearish cross/position

                 # Price vs Kijun
                 if last_close > kijun: ichi_score += 0.10 # Price above baseline (Bullish)
                 else: ichi_score -= 0.10 # Price below baseline (Bearish)

                 # Future Cloud Direction
                 if span_a > span_b: ichi_score += 0.10 # Future cloud bullish
                 else: ichi_score -= 0.10 # Future cloud bearish

                 # Chikou Span vs Price (Lagged Confirmation)
                 if pd.notna(chikou) and pd.notna(price_at_chikou_time):
                      if chikou > price_at_chikou_time: ichi_score += 0.15 # Chikou above price (Bullish confirmation)
                      else: ichi_score -= 0.15 # Chikou below price (Bearish confirmation)

                 details['ichimoku'] = np.clip(ichi_score, -1.0, 1.0)
             else: details['ichimoku'] = 0.0
        else: details['ichimoku'] = 0.0

        # OBV vs Signal Line (OBV EMA)
        obv = get_latest(indicators.get('obv'))
        obv_ema = get_latest(indicators.get('obv_ema'))
        if pd.notna(obv) and pd.notna(obv_ema):
            score = 0.0
            if obv > obv_ema: score = 0.6 # OBV above signal (Bullish volume pressure)
            else: score = -0.6 # OBV below signal (Bearish volume pressure)
            # Add momentum? Check if OBV is trending up/down relative to EMA?
            # prev_obv = get_latest(indicators.get('obv'), 2)
            # prev_obv_ema = get_latest(indicators.get('obv_ema'), 2)
            # if pd.notna(prev_obv) and pd.notna(prev_obv_ema):
            #     obv_rising = (obv - obv_ema) > (prev_obv - prev_obv_ema)
            #     if obv_rising and score > 0: score += 0.2
            #     elif not obv_rising and score < 0: score -= 0.2
            details['obv'] = np.clip(score, -1.0, 1.0)
        else: details['obv'] = 0.0

        # Parabolic SAR
        psar_l = get_latest(indicators.get('parabolic_sar_long'))
        psar_s = get_latest(indicators.get('parabolic_sar_short'))
        # psar_r = get_latest(indicators.get('parabolic_sar_reversal')) # Reversal signal
        if pd.notna(psar_l) or pd.notna(psar_s): # If either is not NaN
            # If PSAR Long has a value, trend is up (SAR is below price) -> Buy signal
            # If PSAR Short has a value, trend is down (SAR is above price) -> Sell signal
            if pd.notna(psar_l) and pd.isna(psar_s): score = 1.0 # Bullish SAR
            elif pd.notna(psar_s) and pd.isna(psar_l): score = -1.0 # Bearish SAR
            else: score = 0.0 # Should not happen if calculated correctly, maybe at exact reversal point?
            details['parabolic_sar'] = score
        else: details['parabolic_sar'] = 0.0

        # Chaikin Money Flow (CMF)
        cmf = get_latest(indicators.get('chaikin_oscillator')) # Using CMF here
        if pd.notna(cmf):
            # Score based on CMF value (positive = buying pressure, negative = selling pressure)
            # Scale CMF from its typical range (e.g., -0.5 to 0.5) to -1 to +1
            score = np.clip(cmf * 3.0, -1.0, 1.0) # Multiplier 3 maps +/-0.33 to +/-1.0
            details['chaikin_oscillator'] = score
        else: details['chaikin_oscillator'] = 0.0

        # --- Calculate Final Weighted Score ---
        technical_score_raw = 0.0
        total_weight_used = 0.0

        for indicator, score in details.items():
            weight = weights.get(indicator, 0.0)
            if weight > 0 and pd.notna(score) and np.isfinite(score): # Check for NaN and infinity
                contribution = score * weight
                score_contributions[indicator] = contribution
                technical_score_raw += contribution
                total_weight_used += weight
            # else: log.debug(f"Excluding indicator '{indicator}' from score (Weight={weight}, Score={score})")

        # Normalize raw score based on the total weight actually used
        if total_weight_used > 1e-6: # Use small epsilon to avoid division by zero
             normalized_score_neg1_to_pos1 = np.clip(technical_score_raw / total_weight_used, -1.0, 1.0)
        else:
             log.warning(f"TechScore ({pair}): Total weight used is near zero. Result may be unreliable.")
             normalized_score_neg1_to_pos1 = 0.0

        # --- Apply Trend Bias ---
        # Boost score slightly if it aligns with the provided trend bias
        bias_factor = 0.10 # Max boost/penalty adjustment
        if 'bullish' in trend_bias and normalized_score_neg1_to_pos1 > 0: # Bias agrees with score direction (Buy)
            normalized_score_neg1_to_pos1 = min(normalized_score_neg1_to_pos1 + bias_factor, 1.0)
        elif 'bearish' in trend_bias and normalized_score_neg1_to_pos1 < 0: # Bias agrees with score direction (Sell)
            normalized_score_neg1_to_pos1 = max(normalized_score_neg1_to_pos1 - bias_factor, -1.0)
        # Optional: Reduce score slightly if it disagrees with bias?
        # elif 'bullish' in trend_bias and normalized_score_neg1_to_pos1 < 0: # Bias disagrees (Buy bias, Sell score)
        #      normalized_score_neg1_to_pos1 *= (1.0 - bias_factor * 0.5) # Reduce magnitude slightly
        # elif 'bearish' in trend_bias and normalized_score_neg1_to_pos1 > 0: # Bias disagrees (Sell bias, Buy score)
        #      normalized_score_neg1_to_pos1 *= (1.0 - bias_factor * 0.5) # Reduce magnitude slightly


        # --- Final Score (Convert -1 to +1 range to 0 to 1 range) ---
        final_score_0_to_1 = (normalized_score_neg1_to_pos1 + 1.0) / 2.0

        log.info(f"TechScore ({pair}): Calculated score: {final_score_0_to_1:.3f} (Raw: {normalized_score_neg1_to_pos1:.3f})")
        log.debug(f"TechScore ({pair}): Contributions: { {k: round(v, 3) for k, v in score_contributions.items()} }")

        # Return final 0-1 score and raw -1 to +1 details for individual indicators
        return final_score_0_to_1, details


    def generate_signal(self, df: pd.DataFrame, pair: str, timeframe: TimeFrames) -> Optional[Dict]:
        """
        Generates the final trading signal by combining technical score, ML prediction,
        volatility filter, and trend analysis.

        Returns:
            Optional[Dict]: Signal dictionary if a BUY/SELL signal is generated, else None.
        """
        try:
            if df is None or df.empty or len(df) < self.MIN_DATA_FOR_SIGNAL:
                log.warning(f"Signal ({pair} {timeframe.name}): Insufficient data ({len(df)} < {self.MIN_DATA_FOR_SIGNAL}).")
                return None

            log.info(f"Attempting to generate signal for {pair} on {timeframe.name}...")

            # 1. Calculate Indicators
            indicators = self.calculate_indicators(df, pair)
            if not indicators:
                log.error(f"Signal ({pair} {timeframe.name}): Failed to calculate indicators.")
                return None

            # 2. Calculate Early Warning Indicators and Patterns
            early_indicators = self._calculate_early_warning_indicators(df, indicators)
            patterns = self._detect_early_patterns(df, indicators)
            early_warning_score = self._calculate_early_warning_score(early_indicators, patterns)
            pattern_score = self._calculate_pattern_score(patterns)

            # 3. Calculate Technical Score
            technical_score, details = self.calculate_technical_score(df, indicators, pair)
            if technical_score is None:
                log.error(f"Signal ({pair} {timeframe.name}): Failed to calculate technical score.")
                return None

            # 4. Combine All Scores
            combined_score = (
                early_warning_score * self.EARLY_WARNING_WEIGHT * 0.6 +
                pattern_score * self.EARLY_WARNING_WEIGHT * 0.4 +
                technical_score * self.CONFIRMED_SIGNAL_WEIGHT
            )

            # 5. Machine Learning Prediction
            ml_confidence = 0.5  # Default neutral ML confidence
            ml_features_valid = False
            
            # ... [existing ML prediction code] ...

            # 6. Final Score Combination
            total_weight = self.TECH_SCORE_WEIGHT + self.ML_CONFIDENCE_WEIGHT
            tech_w = self.TECH_SCORE_WEIGHT / total_weight if total_weight > 0 else 0.5
            ml_w = self.ML_CONFIDENCE_WEIGHT / total_weight if total_weight > 0 else 0.5

            final_score = (tech_w * combined_score) + (ml_w * ml_confidence)
            combined_score = (tech_w * technical_score) + (ml_w * ml_confidence)
            combined_score = np.clip(combined_score, 0.0, 1.0) # Ensure final score is 0-1

            # Define final thresholds for BUY/SELL/HOLD
            if combined_score >= self.BUY_THRESHOLD_COMBINED:
                direction = 'BUY'
            elif combined_score <= self.SELL_THRESHOLD_COMBINED:
                direction = 'SELL'
            else:
                direction = 'HOLD'

            log.info(f"Signal ({pair} {timeframe.name}): Tech={technical_score:.3f}, ML={ml_confidence:.3f} -> Combined={combined_score:.3f} -> Direction={direction}")

            # 7. Generate Signal Output if BUY or SELL
            if direction == 'HOLD':
                log.debug(f"Signal ({pair} {timeframe.name}): Score ({combined_score:.3f}) within HOLD range.")
                return None

            # --- Calculate SL/TP ---
            last_close = df['close'].dropna().iloc[-1] if not df['close'].dropna().empty else np.nan
            last_atr = indicators.get('atr', pd.Series(dtype=float)).dropna().iloc[-1] if indicators.get('atr') is not None and not indicators['atr'].dropna().empty else np.nan

            if pd.isna(last_close) or pd.isna(last_atr) or last_atr <= 1e-9:
                log.error(f"Signal ({pair} {timeframe.name}): Cannot calculate SL/TP due to invalid Close ({last_close}) or ATR ({last_atr}).")
                return None # Cannot generate signal without SL/TP

            # Default SL/TP based on ATR
            sl_distance = self.ATR_SL_MULTIPLIER * last_atr
            tp_distance = self.ATR_TP_MULTIPLIER * last_atr

            # Refine using Supply/Demand and Pivots if available
            sd_zones = indicators.get('supply_demand', {})
            fib_levels = indicators.get('fibonacci', {})
            closest_supply_bottom = sd_zones.get('closest_supply', {}).get('bottom', np.nan)
            closest_demand_top = sd_zones.get('closest_demand', {}).get('top', np.nan)

            # Define SL
            if direction == 'BUY':
                sl_price = last_close - sl_distance
                # Consider demand zone or pivot support
                potential_sl_demand = closest_demand_top if pd.notna(closest_demand_top) and closest_demand_top < last_close else np.nan
                potential_sl_pivot = sd_zones.get('support1', np.nan) if pd.notna(sd_zones.get('support1')) and sd_zones['support1'] < last_close else np.nan
                # Use the lowest (safest) potential SL, but not lower than ATR SL
                sl_candidates = [sl_price]
                if pd.notna(potential_sl_demand): sl_candidates.append(potential_sl_demand)
                if pd.notna(potential_sl_pivot): sl_candidates.append(potential_sl_pivot)
                sl_price = min(sl_candidates) # Take the lowest price among valid candidates
            else: # SELL
                sl_price = last_close + sl_distance
                # Consider supply zone or pivot resistance
                potential_sl_supply = closest_supply_bottom if pd.notna(closest_supply_bottom) and closest_supply_bottom > last_close else np.nan
                potential_sl_pivot = sd_zones.get('resistance1', np.nan) if pd.notna(sd_zones.get('resistance1')) and sd_zones['resistance1'] > last_close else np.nan
                # Use the highest (safest) potential SL, but not higher than ATR SL
                sl_candidates = [sl_price]
                if pd.notna(potential_sl_supply): sl_candidates.append(potential_sl_supply)
                if pd.notna(potential_sl_pivot): sl_candidates.append(potential_sl_pivot)
                sl_price = max(sl_candidates) # Take the highest price among valid candidates

            # Ensure minimum SL distance
            min_sl_dist_val = self.MIN_SL_DISTANCE_ATR_FACTOR * last_atr
            if direction == 'BUY':
                 sl_price = min(sl_price, last_close - min_sl_dist_val)
            else: # SELL
                 sl_price = max(sl_price, last_close + min_sl_dist_val)

            # Define TP, ensuring minimum R:R
            required_profit = abs(last_close - sl_price)
            if required_profit < 1e-9: # Avoid division by zero if SL is somehow at entry
                 log.warning(f"Signal ({pair} {timeframe.name}): SL distance is zero. Cannot ensure R:R for TP.")
                 # Fallback to simple ATR TP or skip signal?
                 tp_price = (last_close + tp_distance) if direction == 'BUY' else (last_close - tp_distance)
            else:
                target_tp_based_on_rr = (last_close + required_profit * self.MIN_RR_RATIO) if direction == 'BUY' else (last_close - required_profit * self.MIN_RR_RATIO)
                atr_tp = (last_close + tp_distance) if direction == 'BUY' else (last_close - tp_distance)

                # Consider supply/demand/pivots for TP targets
                potential_tp_supply = closest_supply_bottom if pd.notna(closest_supply_bottom) and closest_supply_bottom > last_close else np.nan
                potential_tp_demand = closest_demand_top if pd.notna(closest_demand_top) and closest_demand_top < last_close else np.nan
                potential_tp_pivot_r = sd_zones.get('resistance1', np.nan) if pd.notna(sd_zones.get('resistance1')) and sd_zones['resistance1'] > last_close else np.nan
                potential_tp_pivot_s = sd_zones.get('support1', np.nan) if pd.notna(sd_zones.get('support1')) and sd_zones['support1'] < last_close else np.nan


                tp_candidates = [atr_tp, target_tp_based_on_rr] # Start with ATR and Min R:R targets

                if direction == 'BUY':
                    if pd.notna(potential_tp_supply): tp_candidates.append(potential_tp_supply)
                    if pd.notna(potential_tp_pivot_r): tp_candidates.append(potential_tp_pivot_r)
                    # Take the minimum of the potential TPs that satisfy min R:R
                    valid_tps = [tp for tp in tp_candidates if tp >= target_tp_based_on_rr]
                    tp_price = min(valid_tps) if valid_tps else target_tp_based_on_rr # Fallback to min R:R if no other valid TP found
                else: # SELL
                    if pd.notna(potential_tp_demand): tp_candidates.append(potential_tp_demand)
                    if pd.notna(potential_tp_pivot_s): tp_candidates.append(potential_tp_pivot_s)
                    # Take the maximum of the potential TPs that satisfy min R:R
                    valid_tps = [tp for tp in tp_candidates if tp <= target_tp_based_on_rr]
                    tp_price = max(valid_tps) if valid_tps else target_tp_based_on_rr # Fallback to min R:R

            # Final check for valid SL/TP relative to entry
            if (direction == 'BUY' and (sl_price >= last_close or tp_price <= last_close)) or \
               (direction == 'SELL' and (sl_price <= last_close or tp_price >= last_close)):
                log.error(f"Signal ({pair} {timeframe.name}): Invalid SL ({sl_price}) / TP ({tp_price}) relative to Entry ({last_close}). Signal aborted.")
                return None

            # Get position size factor from volatility filter
            _, position_size_factor = self.filter_signal_by_volatility(df, direction, pair, timeframe)

            # Detect market trend
            trend = self._detect_trend(df, indicators)

            current_volatility_pct = None
            def get_latest(series: Optional[pd.Series], lookback: int = 1) -> Any:
              if series is None or series.dropna().empty: return np.nan
              try:
                       return series.dropna().iloc[-lookback]
              except IndexError:
                  return np.nan # Not enough non-NaN values
            # --- Prepare Final Signal Dictionary ---
            latest_rsi = get_latest(indicators.get('rsi'))
            latest_macd_h = get_latest(indicators.get('macd_histogram'))
            latest_adx = get_latest(indicators.get('adx'))
            latest_bb_width = get_latest(indicators.get('bollinger_width'))
            # حساب atr_pct هنا إذا لم يتم حسابه في فلتر التقلب
            atr_pct_val = np.nan
            if 'current_volatility_pct' in locals():
                atr_pct_val = current_volatility_pct
            elif pd.notna(last_atr) and pd.notna(last_close) and last_close > 0:
                atr_pct_val = (last_atr / last_close) * 100.0

            signal_output = {
                'pair': pair,
                'timeframe': timeframe.value, # Store minutes value
                'direction': direction,
                'confidence': round(combined_score, 4),
                'technical_score': round(technical_score, 4),
                'ml_confidence': round(ml_confidence, 4) if ml_features_valid else None, # Indicate if ML was used
                'trend': trend,
                'timestamp': datetime.now(self.timezone).isoformat(timespec='seconds'),
                'entry_price': round(last_close, 5), # Adjust rounding per pair
                'stop_loss': round(sl_price, 5),
                'take_profit': round(tp_price, 5),
                'position_size_factor': round(position_size_factor, 3),
                'details': { # Include key values for context/debugging
                     # استخدم get_latest هنا بدلاً من get_paths
                     'rsi': round(latest_rsi, 2) if pd.notna(latest_rsi) else None,
                     'macd_h': round(latest_macd_h, 5) if pd.notna(latest_macd_h) else None,
                     'adx': round(latest_adx, 2) if pd.notna(latest_adx) else None,
                     'atr_pct': round(atr_pct_val, 3) if pd.notna(atr_pct_val) else None,
                     'bb_width': round(latest_bb_width, 4) if pd.notna(latest_bb_width) else None,
                }
            }

            log.info(f"*** Signal Generated ***: {pair} {timeframe.name} -> {signal_output['direction']} "
                     f"@ {signal_output['entry_price']} (Conf: {signal_output['confidence']:.3f}) "
                     f"SL={signal_output['stop_loss']}, TP={signal_output['take_profit']}, Trend={trend}, SizeFactor={signal_output['position_size_factor']}")

            return signal_output

        except Exception as e:
            log.exception(f"Critical error generating signal for {pair} {timeframe.name}: {e}")
            return None # Return None on any major error in the pipeline

    def _calculate_early_warning_indicators(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Calculate early warning indicators for potential signal detection."""
        early_indicators = {}
        
        try:
            # Price velocity and acceleration
            df['price_velocity'] = df['close'].diff()
            df['price_acceleration'] = df['price_velocity'].diff()
            early_indicators['price_acceleration'] = df['price_acceleration'].iloc[-1] if not df.empty else 0

            # Volume analysis
            df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            early_indicators['volume_spike'] = df['volume_ratio'].iloc[-1] if not df.empty else 0

            # Momentum indicators
            df['momentum'] = df['close'].diff(5)
            early_indicators['momentum'] = df['momentum'].iloc[-1] if not df.empty else 0

            # Volatility expansion
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_ma'] = df['atr'].rolling(window=20).mean()
            df['volatility_ratio'] = df['atr'] / df['atr_ma']
            early_indicators['volatility_expansion'] = df['volatility_ratio'].iloc[-1] if not df.empty else 0

            # Add signal timing assessment
            early_indicators['signal_timing'] = self._assess_signal_timing(df, indicators)
            
            return early_indicators
        except Exception as e:
            log.error(f"Error calculating early warning indicators: {e}")
            # Return a default dictionary with zeros to prevent further errors
            return {
                'price_acceleration': 0,
                'volume_spike': 0,
                'momentum': 0,
                'volatility_expansion': 0,
                'signal_timing': 0.5  # Neutral timing
            }

    def _detect_early_patterns(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Detect early price patterns that might indicate upcoming signals."""
        patterns = {}
        
        # Get recent price data
        recent_high = df['high'].tail(5)
        recent_low = df['low'].tail(5)
        recent_close = df['close'].tail(5)
        
        # Double Bottom/Top Pattern
        if len(recent_low) >= 5:
            # Double Bottom
            if (recent_low.iloc[-1] > recent_low.iloc[-2] and 
                recent_low.iloc[-2] < recent_low.iloc[-3] and 
                abs(recent_low.iloc[-1] - recent_low.iloc[-3]) / recent_low.iloc[-3] < 0.01):
                patterns['double_bottom'] = True
            
            # Double Top
            if (recent_high.iloc[-1] < recent_high.iloc[-2] and 
                recent_high.iloc[-2] > recent_high.iloc[-3] and 
                abs(recent_high.iloc[-1] - recent_high.iloc[-3]) / recent_high.iloc[-3] < 0.01):
                patterns['double_top'] = True
        
        # Bullish/Bearish Engulfing
        if len(recent_close) >= 2:
            prev_close = recent_close.iloc[-2]
            curr_close = recent_close.iloc[-1]
            prev_open = df['open'].iloc[-2]
            curr_open = df['open'].iloc[-1]
            
            # Bullish Engulfing
            if (curr_close > curr_open and  # Current candle is bullish
                prev_close < prev_open and  # Previous candle is bearish
                curr_open < prev_close and  # Current open below previous close
                curr_close > prev_open):    # Current close above previous open
                patterns['bullish_engulfing'] = True
            
            # Bearish Engulfing
            if (curr_close < curr_open and  # Current candle is bearish
                prev_close > prev_open and  # Previous candle is bullish
                curr_open > prev_close and  # Current open above previous close
                curr_close < prev_open):    # Current close below previous open
                patterns['bearish_engulfing'] = True
        
        # RSI Divergence
        rsi = indicators.get('rsi')
        if rsi is not None and not rsi.empty and len(rsi) >= 5:
            recent_rsi = rsi.tail(5)
            # Bullish Divergence
            if (recent_low.iloc[-1] < recent_low.iloc[-2] and 
                recent_rsi.iloc[-1] > recent_rsi.iloc[-2]):
                patterns['bullish_rsi_divergence'] = True
            # Bearish Divergence
            if (recent_high.iloc[-1] > recent_high.iloc[-2] and 
                recent_rsi.iloc[-1] < recent_rsi.iloc[-2]):
                patterns['bearish_rsi_divergence'] = True
        
        return patterns

    def _calculate_pattern_score(self, patterns: Dict) -> float:
        """Calculate a score based on detected patterns."""
        if not patterns:
            return 0.5
        
        # Define pattern weights
        pattern_weights = {
            'double_bottom': 0.3,
            'double_top': -0.3,
            'bullish_engulfing': 0.2,
            'bearish_engulfing': -0.2,
            'bullish_rsi_divergence': 0.15,
            'bearish_rsi_divergence': -0.15
        }
        
        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0
        
        for pattern, detected in patterns.items():
            if detected and pattern in pattern_weights:
                total_score += pattern_weights[pattern]
                total_weight += abs(pattern_weights[pattern])
        
        if total_weight == 0:
            return 0.5
        
        # Normalize score between 0 and 1
        normalized_score = (total_score / total_weight + 1) / 2
        return normalized_score

    def _calculate_early_warning_score(self, early_indicators: Dict, patterns: Dict) -> float:
        """Calculate early warning score based on indicators and patterns."""
        score = 0.0
        
        # Price acceleration
        if early_indicators['price_acceleration'] > self.PRICE_ACCELERATION_THRESHOLD:
            score += 0.2
            
        # Volume spike
        if early_indicators['volume_spike'] > self.VOLUME_SPIKE_THRESHOLD:
            score += 0.2
            
        # Momentum
        if early_indicators['momentum'] > self.MOMENTUM_THRESHOLD:
            score += 0.2
            
        # Patterns
        if patterns.get('bullish_rsi_divergence'):
            score += 0.2
        if patterns.get('double_bottom'):
            score += 0.2
        if patterns.get('bullish_engulfing'):
            score += 0.2
            
        return score

    def generate_enhanced_signal(self, df: pd.DataFrame, pair: str, timeframe: TimeFrames) -> Optional[Dict]:
        """
        Generate enhanced trading signals using both early and confirmed detection methods.
        """
        if len(df) < 50:
            log.warning(f"Insufficient data for {pair} on {timeframe}")
            return None

        try:
            # Calculate technical indicators first
            indicators = self.calculate_indicators(df, pair)
            
            # Then use indicators in subsequent calculations
            early_indicators = self._calculate_early_warning_indicators(df, indicators)
            patterns = self._detect_early_patterns(df, indicators)
            
            # Detect market regime
            trend = self._detect_trend(df, indicators)
            
            # Calculate both early and confirmed technical scores
            early_tech_score, early_details = self._calculate_early_technical_score(df, indicators, trend, pair)
            confirmed_tech_score, confirmed_details = self.calculate_technical_score(df, indicators, trend, pair)
            
            # Calculate early warning score
            early_score = self._calculate_early_warning_score(early_indicators, patterns)
            
            # Combine scores with weighted approach
            combined_score = (
                early_tech_score * self.EARLY_SIGNAL_WEIGHT +
                confirmed_tech_score * self.CONFIRMED_SIGNAL_WEIGHT
            )
            
            # Determine signal type and confidence
            signal_type, confidence = self._determine_signal_type(combined_score, early_score)
            
            # Calculate stop loss and take profit levels
            atr = indicators.get('atr', pd.Series([0])).iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if signal_type in ['BUY', 'SELL']:
                sl_distance = atr * 1.5
                tp_distance = atr * 2.5
                
                if signal_type == 'BUY':
                    sl = current_price - sl_distance
                    tp = current_price + tp_distance
                else:
                    sl = current_price + sl_distance
                    tp = current_price - tp_distance

                return {
                    'pair': pair,
                    'timeframe': timeframe.name,
                    'direction': signal_type,
                    'confidence': confidence,
                    'combined_score': combined_score,
                    'early_score': early_score,
                    'early_tech_score': early_tech_score,
                    'confirmed_tech_score': confirmed_tech_score,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'timestamp': datetime.now(self.timezone),
                    'details': {
                        'trend': trend,
                        'early_indicators': early_indicators,
                        'patterns': patterns,
                        'early_details': early_details,
                        'confirmed_details': confirmed_details
                    }
                }
            
            return None

        except Exception as e:
            log.error(f"Error generating enhanced signal for {pair}: {str(e)}", exc_info=True)
            return None

    def _determine_signal_type(self, combined_score: float, early_score: float) -> Tuple[str, float]:
        """Determine signal type and confidence level."""
        confidence = 0.0
        
        if combined_score >= self.EARLY_BUY_THRESHOLD_COMBINED:
            if early_score >= 0.6:  # Strong early confirmation
                return 'BUY', 0.9
            elif combined_score >= self.BUY_THRESHOLD_COMBINED:
                return 'BUY', 0.8
            else:
                return 'BUY', 0.7
                
        elif combined_score <= self.EARLY_SELL_THRESHOLD_COMBINED:
            if early_score >= 0.6:  # Strong early confirmation
                return 'SELL', 0.9
            elif combined_score <= self.SELL_THRESHOLD_COMBINED:
                return 'SELL', 0.8
            else:
                return 'SELL', 0.7
                
        return 'HOLD', 0.0
        
    def _assess_signal_timing(self, df: pd.DataFrame, indicators: Dict) -> float:
        """Assess the timing of potential signals based on various factors."""
        factors = {}
        timing_score = 0.5  # Neutral base score
        
        try:
            # Get required indicators
            rsi = indicators.get('rsi')
            macd_hist = indicators.get('macd_histogram')
            market_condition = self._detect_trend(df, indicators)
            
            # Safety check - ensure we have enough data
            if rsi is None or len(rsi) < 3:
                log.warning("Insufficient RSI data for signal timing assessment")
                return timing_score
                
            # RSI timing factor - different optimal zones for different market conditions
            if market_condition == 'trending':
                # In trending markets, entry on pullbacks is optimal
                rsi_timing = 0.0
                if rsi.iloc[-1] < 30:  # Potential buy in uptrend pullback
                    rsi_timing = 1.0 - (rsi.iloc[-1] / 30)  # Higher score for lower RSI
                elif rsi.iloc[-1] > 70:  # Potential sell in downtrend pullback
                    rsi_timing = (rsi.iloc[-1] - 70) / 30  # Higher score for higher RSI
                factors['rsi_timing'] = rsi_timing
            
            elif market_condition in ['ranging', 'volatile_ranging']:
                # In ranging markets, entry near extremes is optimal
                rsi_timing = 0.0
                if rsi.iloc[-1] < 30:  # Potential buy at support
                    rsi_timing = 1.0 - (rsi.iloc[-1] / 30)
                elif rsi.iloc[-1] > 70:  # Potential sell at resistance
                    rsi_timing = (rsi.iloc[-1] - 70) / 30
                factors['rsi_timing'] = rsi_timing
            
            elif market_condition in ['volatile', 'volatile_trending']:
                # In volatile markets, confirmation is more important than exact timing
                rsi_timing = 0.5  # Neutral base score
                # Add momentum confirmation - use iloc instead of direct indexing
                if len(rsi) >= 3 and rsi.iloc[-1] > rsi.iloc[-2] > rsi.iloc[-3]:  # Rising RSI
                    rsi_timing += 0.2
                elif len(rsi) >= 3 and rsi.iloc[-1] < rsi.iloc[-2] < rsi.iloc[-3]:  # Falling RSI
                    rsi_timing += 0.2
                factors['rsi_timing'] = min(1.0, rsi_timing)
                
            # Calculate final timing score
            if factors:
                timing_score = sum(factors.values()) / len(factors)
                
            return timing_score
            
        except Exception as e:
            log.warning(f"Error in signal timing assessment: {e}")
            return timing_score  # Return neutral score on error

    def _calculate_early_technical_score(self, df: pd.DataFrame, indicators: Dict, trend: str, pair: str) -> Tuple[float, Dict]:
        """Calculate technical score using early detection parameters."""
        # Store original thresholds
        original_rsi_overbought = self.RSI_OVERBOUGHT
        original_rsi_oversold = self.RSI_OVERSOLD
        original_stoch_overbought = self.STOCH_OVERBOUGHT
        original_stoch_oversold = self.STOCH_OVERSOLD
        original_adx_threshold = self.ADX_TREND_THRESHOLD
        
        # Use early thresholds
        self.RSI_OVERBOUGHT = self.EARLY_RSI_OVERBOUGHT
        self.RSI_OVERSOLD = self.EARLY_RSI_OVERSOLD
        self.STOCH_OVERBOUGHT = self.EARLY_STOCH_OVERBOUGHT
        self.STOCH_OVERSOLD = self.EARLY_STOCH_OVERSOLD
        self.ADX_TREND_THRESHOLD = self.EARLY_ADX_TREND_THRESHOLD
        
        # Calculate score
        score, details = self.calculate_technical_score(df, indicators, trend, pair)
        
        # Restore original thresholds
        self.RSI_OVERBOUGHT = original_rsi_overbought
        self.RSI_OVERSOLD = original_rsi_oversold
        self.STOCH_OVERBOUGHT = original_stoch_overbought
        self.STOCH_OVERSOLD = original_stoch_oversold
        self.ADX_TREND_THRESHOLD = original_adx_threshold
        
        return score, details
    def _calculate_early_warning_indicators(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Calculate early warning indicators for potential signal detection."""
        early_indicators = {}
        
        try:
            # Price velocity and acceleration
            df['price_velocity'] = df['close'].diff()
            df['price_acceleration'] = df['price_velocity'].diff()
            early_indicators['price_acceleration'] = df['price_acceleration'].iloc[-1] if not df.empty else 0

            # Volume analysis
            df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            early_indicators['volume_spike'] = df['volume_ratio'].iloc[-1] if not df.empty else 0

            # Momentum indicators
            df['momentum'] = df['close'].diff(5)
            early_indicators['momentum'] = df['momentum'].iloc[-1] if not df.empty else 0

            # Volatility expansion
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_ma'] = df['atr'].rolling(window=20).mean()
            df['volatility_ratio'] = df['atr'] / df['atr_ma']
            early_indicators['volatility_expansion'] = df['volatility_ratio'].iloc[-1] if not df.empty else 0

            # Add signal timing assessment
            early_indicators['signal_timing'] = self._assess_signal_timing(df, indicators)
            
            return early_indicators
        except Exception as e:
            log.error(f"Error calculating early warning indicators: {e}")
            # Return a default dictionary with zeros to prevent further errors
            return {
                'price_acceleration': 0,
                'volume_spike': 0,
                'momentum': 0,
                'volatility_expansion': 0,
                'signal_timing': 0.5  # Neutral timing
            }


# --- END OF FILE Signal.py ---
