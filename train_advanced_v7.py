# --- [0] Imports & Setup ---
from sklearn.base import clone
from typing import Self
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, roc_auc_score, f1_score, precision_score, recall_score,
                             accuracy_score, confusion_matrix, roc_curve, average_precision_score,
                             precision_recall_curve, brier_score_loss)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV # <<< لمعايرة الاحتمالات
from joblib import dump, load
import os
import logging
import warnings
import gc
import json
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.integration import XGBoostPruningCallback
from functools import partial
import time
import pickle
from datetime import datetime
import traceback

print("Script starting - imports loaded successfully")

# --- Directory Constants ---
MODELS_DIR = 'models'
RESULTS_DIR = 'training_results_auto_v7'  # Default, will be overridden by config

# --- Signal Generator Import & Feature Lists ---
try:
    print("Attempting to import SignalGenerator and TimeFrames from Signal...")
    from Signal import SignalGenerator, TimeFrames
    print("Import successful!")
    # --- Feature Lists V7 (يمكن تعديلها حسب الحاجة وتحليل الأهمية) ---
    BASE_ML_FEATURES = [
        'rsi', 'macd_histogram', 'atr_norm', 'bbands_squeeze', 'stoch_k', 'stoch_d',
        'adx', 'ema_trend', 'volume_sma_ratio', 'volatility_rsi', 'bbands_pct',
        'mfi', 'willr', 'adosc'
    ]
    TIME_FEATURES = ['hour', 'day_of_week', 'week_of_year', 'month', 'day_of_year', 'hour_sin', 'hour_cos']
    ROLLING_FEATURES = [
        'return_1_std_10', 'return_1_skew_10', 'return_1_kurt_10',
        'return_1_std_30', 'return_1_skew_30', 'return_1_kurt_30',
        'rsi_std_14', 'volatility_ratio_sma_5'
    ]
    INTERACTION_FEATURES = ['rsi_x_adx', 'stoch_x_volatility_ratio', 'hour_sin_x_volatility_ratio', 'hour_cos_x_volatility_ratio']
    LAGGED_FEATURES = [
        'close_pct_change_1', 'close_pct_change_3', 'close_pct_change_5', 'close_pct_change_10',
        'atr_norm_lag_1','atr_norm_lag_3', 'volume_sma_ratio_lag_1',
        'target_lag_1', 'target_lag_2', 'target_lag_3'
    ]
    DISTANCE_FEATURES = ['dist_from_ema_50_atr', 'dist_from_ema_200_atr']
    NUMERIC_FEATURES = sorted(list(set(BASE_ML_FEATURES + TIME_FEATURES + ROLLING_FEATURES +
                                  INTERACTION_FEATURES + LAGGED_FEATURES + DISTANCE_FEATURES)))
    CATEGORICAL_FEATURES = ['pair_id']
    ALL_FEATURES_INITIAL = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    TARGET_COLUMN = 'target'
except ImportError: print("Error: Signal.py not found."); exit()
except Exception as e: print(f"Error during initial setup: {e}"); exit()

# --- Configuration Loading ---
CONFIG_FILE = 'training_config.json'
print(f"Attempting to load configuration from {CONFIG_FILE}...")
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    print(f"Successfully loaded configuration from {CONFIG_FILE}")
    logging.info(f"Loaded configuration from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"ERROR: Configuration file {CONFIG_FILE} not found. Please create it.")
    logging.error(f"Configuration file {CONFIG_FILE} not found. Please create it."); exit()
except json.JSONDecodeError:
    print(f"ERROR: Error decoding JSON from {CONFIG_FILE}.")
    logging.error(f"Error decoding JSON from {CONFIG_FILE}."); exit()

# --- Setup Logging and Directories ---
warnings.filterwarnings("ignore")
RESULTS_DIR = config.get('results_dir', 'training_results_auto_v7')
print(f"Creating results directory: {RESULTS_DIR}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)
log_file_path = os.path.join(RESULTS_DIR, f"{config.get('output_model_base_name', 'model')}_training.log")
print(f"Log file will be: {log_file_path}")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s] - %(message)s',
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
print("Logging configured")
logging.info("--- Starting Autonomous Training Script V7 ---")
logging.info(f"Configuration:\n{json.dumps(config, indent=2)}")
print("Proceeding with script execution...")


# --- [Function Definitions: load_and_combine_data, engineer_advanced_features_v6, etc.] ---
# (استخدم نفس تعريفات الدوال من V6 أو V5، مع التأكد من أنها تستخدم المتغيرات من config)

# --- [1] Load Data Function (using config) ---
def load_and_combine_data_v7(config_or_asset_files):
    """
    Load and combine data from multiple files.
    Takes either a config dict or an asset_files dict directly.
    """
    # Extract asset_files from config if needed
    asset_files_config = config_or_asset_files
    if isinstance(config_or_asset_files, dict) and 'asset_files' in config_or_asset_files:
        asset_files_config = config_or_asset_files.get('asset_files', {})
    
    # Continue with the original function logic
    all_data_dfs = []
    logging.info(f"Loading and processing data from {len(asset_files_config)} files...")
    for pair, filename in asset_files_config.items(): # Use config dict
        try:
            df_raw = pd.read_csv(filename, parse_dates=['time'])
            # ... (rest of the loading and pip size logic) ...
            df_raw.set_index('time', inplace=True)
            required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df_raw.columns for col in required_cols):
                logging.warning(f"Skipping {pair}: Missing columns in {filename}")
                continue
            df_raw.dropna(subset=required_cols, inplace=True)
            if df_raw.empty: continue
            df_raw['pair_id'] = pair
            if 'JPY' in pair: df_raw['pip_size'] = 0.01
            elif 'XAU' in pair: df_raw['pip_size'] = 0.1
            elif 'BTC' in pair: df_raw['pip_size'] = 0.1
            else: df_raw['pip_size'] = 0.0001
            all_data_dfs.append(df_raw)
            logging.info(f"Loaded {pair}: {df_raw.shape[0]} rows")
            del df_raw; gc.collect()
        except Exception as e: logging.warning(f"Skipping {pair}: Error loading '{filename}': {e}")
    if not all_data_dfs: return None
    df_combined = pd.concat(all_data_dfs, sort=True).sort_index()
    del all_data_dfs; gc.collect()
    logging.info(f"Combined data shape: {df_combined.shape}")
    if 'tick_volume' in df_combined.columns:
        df_combined['tick_volume'] = df_combined['tick_volume'].astype(np.float64)
    return df_combined


# --- [2] Feature Engineering Function V7 (using config for potential future options) ---
def engineer_advanced_features_v7(df_combined, sg_instance, config):
    logging.info("Calculating advanced features (V7)...")
    all_features_dfs = []
    grouped_by_pair = df_combined.groupby('pair_id')
    historical_atr = {}

    # Calculate ATR baselines first (this part seems fine)
    for pair_name, df_pair_temp in grouped_by_pair:
        df_pair_temp = df_pair_temp.sort_index()
        # Use calculate_indicators to get ATR, handle potential None
        indic_temp = sg_instance.calculate_indicators(df_pair_temp.iloc[-500:], pair=pair_name)
        atr_long = indic_temp.get('atr') if isinstance(indic_temp, dict) else None
        historical_atr[pair_name] = atr_long.mean() if atr_long is not None and not atr_long.dropna().empty else None
        del df_pair_temp, atr_long, indic_temp; gc.collect()

    # Calculate features per pair
    for pair_name, df_pair in grouped_by_pair:
        logging.debug(f"Calculating features for {pair_name}...")
        df_pair = df_pair.sort_index() # Ensure sorted
        indicators = sg_instance.calculate_indicators(df_pair, pair=pair_name)

        # --- START OF FIX ---
        # Initialize features_df WITH essential columns from df_pair
        essential_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'pip_size']
        # Make sure all essential columns exist in the input df_pair
        if not all(col in df_pair.columns for col in essential_cols):
            logging.error(f"Essential columns missing for {pair_name}. Skipping feature engineering for this pair.")
            continue # Skip this pair if essential data is missing

        features_df = df_pair[essential_cols].copy() # Start with essential columns
        features_df['pair_id'] = pair_name          # Add pair_id
        # --- END OF FIX ---

        missing = [] # Track missing calculated features

        # --- Base Features + New TA ---
        # (Calculations remain the same, assigning to features_df)
        base_features_current = list(set(BASE_ML_FEATURES) | {'mfi', 'willr', 'adosc'})
        for fname in base_features_current:
            value_series = None
            if fname == 'atr_norm':
               atr=indicators.get('atr'); close=df_pair['close'] # Use df_pair for close
               if atr is not None and close is not None and not (close == 0).all():
                   value_series = atr / close.replace(0, np.nan)
               else: missing.append(fname)
            elif fname == 'bbands_squeeze':
                bbu=indicators.get('bollinger_upper'); bbl=indicators.get('bollinger_lower'); bbm=indicators.get('bollinger_middle')
                if all(s is not None for s in [bbu, bbl, bbm]) and not (bbm == 0).all():
                    value_series = (bbu - bbl) / bbm.replace(0, np.nan)
                else: missing.append(fname)
            elif fname == 'bbands_pct':
                value_series = indicators.get('bollinger_percent')
                if value_series is None: missing.append(fname)
            elif fname == 'ema_trend':
                ema_s = indicators.get('ema_short'); ema_l = indicators.get('ema_long')
                if ema_s is not None and ema_l is not None: value_series = ema_s - ema_l
                else: missing.append(fname)
            else:
               value_series = indicators.get(fname)
               if value_series is None: missing.append(fname)
            features_df[fname] = value_series if value_series is not None else np.nan

        # --- Time Features ---
        # (Calculations remain the same, assigning to features_df)
        idx = df_pair.index # Use index from df_pair
        features_df['hour']=idx.hour; features_df['day_of_week']=idx.dayofweek
        features_df['week_of_year']=idx.isocalendar().week.astype(int); features_df['month']=idx.month
        features_df['day_of_year']=idx.dayofyear; features_df['hour_sin']=np.sin(2*np.pi*features_df['hour']/24.0)
        features_df['hour_cos']=np.cos(2*np.pi*features_df['hour']/24.0)

        # --- Rolling Features ---
        # (Calculations remain the same, assigning to features_df)
        returns_1 = df_pair['close'].pct_change(periods=1) # Use df_pair for close
        for window in [10, 30]:
            features_df[f'return_1_std_{window}'] = returns_1.rolling(window,min_periods=window//2).std()
            features_df[f'return_1_skew_{window}'] = returns_1.rolling(window,min_periods=window//2).skew()
            features_df[f'return_1_kurt_{window}'] = returns_1.rolling(window,min_periods=window//2).kurt()
        rsi = indicators.get('rsi')
        if rsi is not None: features_df['rsi_std_14'] = rsi.rolling(14, min_periods=10).std()
        else: features_df['rsi_std_14'] = np.nan; missing.append('rsi_std_14')

        # --- Extra / Interaction / Distance Features ---
        # (Calculations remain the same, using the corrected logic from previous fix)
        atr = indicators.get('atr')
        hist_atr = historical_atr.get(pair_name)
        vol_ratio = pd.Series(np.nan, index=features_df.index)
        if atr is not None and not atr.isnull().all() and hist_atr is not None and hist_atr > 1e-9:
            vol_ratio = atr / hist_atr
            features_df['volatility_ratio'] = vol_ratio
            features_df['volatility_ratio_sma_5'] = vol_ratio.rolling(5, min_periods=3).mean()
        else:
            features_df['volatility_ratio'] = np.nan
            features_df['volatility_ratio_sma_5'] = np.nan
            missing.extend(['volatility_ratio', 'volatility_ratio_sma_5'])
        adx = indicators.get('adx')
        if rsi is not None and adx is not None: features_df['rsi_x_adx'] = rsi * adx.clip(0, 100)
        else: features_df['rsi_x_adx'] = np.nan; missing.append('rsi_x_adx')
        stoch_k = indicators.get('stoch_k')
        if stoch_k is not None: features_df['stoch_x_volatility_ratio'] = stoch_k * vol_ratio
        else: features_df['stoch_x_volatility_ratio'] = np.nan; missing.append('stoch_x_volatility_ratio')
        if 'hour_sin' in features_df.columns: features_df['hour_sin_x_volatility_ratio'] = features_df['hour_sin'] * vol_ratio
        else: features_df['hour_sin_x_volatility_ratio'] = np.nan; missing.append('hour_sin_x_volatility_ratio')
        if 'hour_cos' in features_df.columns: features_df['hour_cos_x_volatility_ratio'] = features_df['hour_cos'] * vol_ratio
        else: features_df['hour_cos_x_volatility_ratio'] = np.nan; missing.append('hour_cos_x_volatility_ratio')
        ema50=indicators.get('ema_medium'); ema200=indicators.get('ema_long')
        if atr is not None and not atr.isnull().all() and ema50 is not None and (atr > 1e-9).any():
            features_df['dist_from_ema_50_atr']=(df_pair['close'] - ema50) / atr.where(atr > 1e-9) # Use df_pair for close
        else: features_df['dist_from_ema_50_atr']=np.nan; missing.append('dist_from_ema_50_atr')
        if atr is not None and not atr.isnull().all() and ema200 is not None and (atr > 1e-9).any():
            features_df['dist_from_ema_200_atr']=(df_pair['close'] - ema200) / atr.where(atr > 1e-9) # Use df_pair for close
        else: features_df['dist_from_ema_200_atr']=np.nan; missing.append('dist_from_ema_200_atr')

        # --- Lagged Features ---
        # (Calculations remain the same, assigning to features_df)
        for lag in [1, 3, 5, 10]: features_df[f'close_pct_change_{lag}']=df_pair['close'].pct_change(periods=lag) * 100 # Use df_pair for close
        if 'atr_norm' in features_df:
             features_df['atr_norm_lag_1']=features_df['atr_norm'].shift(1); features_df['atr_norm_lag_3']=features_df['atr_norm'].shift(3)
        else: features_df['atr_norm_lag_1']=np.nan; features_df['atr_norm_lag_3']=np.nan; missing.extend(['atr_norm_lag_1','atr_norm_lag_3'])
        if 'volume_sma_ratio' in features_df: features_df['volume_sma_ratio_lag_1']=features_df['volume_sma_ratio'].shift(1)
        else: features_df['volume_sma_ratio_lag_1']=np.nan; missing.append('volume_sma_ratio_lag_1')
        # --- END FEATURE CALCULATIONS ---

        if missing: logging.warning(f"Missing features calculated for {pair_name}: {list(set(missing))}")
        all_features_dfs.append(features_df)
        del df_pair, indicators, features_df; gc.collect() # Clean up memory

    # Check if any features were generated
    if not all_features_dfs:
        logging.error("No feature DataFrames were generated. Aborting.")
        return None

    features_combined = pd.concat(all_features_dfs, sort=True).sort_index()
    del all_features_dfs; gc.collect() # Clean up memory
    features_combined.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace infinities
    logging.info(f"Combined features calculated (V7). Shape: {features_combined.shape}")

    # --- Final check for essential columns ---
    if 'close' not in features_combined.columns:
        logging.error("FATAL: 'close' column is missing from features_combined after engineering!")
        return None

    return features_combined

# --- [3] Target Definition Functions (Triple Barrier V2 as default) ---
def compute_daily_volatility_v7(close_series, lookback=50):
    # ... (same as compute_daily_volatility in V5) ...
    log_returns = np.log(close_series / close_series.shift(1))
    daily_vol = log_returns.rolling(window=lookback, min_periods=lookback//2).std()
    return daily_vol

def get_triple_barrier_targets_v7(prices_df, daily_vol_series, config):
    """
    Create triple barrier method target variables.
    Parameterized version for V7 that takes settings from config.
    """
    # Extract settings from config dict (with fallbacks)
    try:
        lookahead = config.get('lookahead', 15)
        tp_mult = config.get('tp_mult', 1.8)
        sl_mult = config.get('sl_mult', 1.0)
        min_vol = config.get('min_vol', 0.0005)
        cost_pips = config.get('cost_pips', 1.5)
    except (TypeError, KeyError, AttributeError):
        # If config is not a dict or missing keys, use defaults
        lookahead = 15
        tp_mult = 1.8
        sl_mult = 1.0
        min_vol = 0.0005
        cost_pips = 1.5
        logging.warning("Triple barrier settings not found in config, using defaults")

    # Rest of existing implementation...
    try:
        # Ensure we have clean data
        if prices_df.empty or daily_vol_series.empty or (daily_vol_series < min_vol).all():
            logging.warning("Insufficient data for triple barrier targets")
            return None

        # Create target df with close prices
        target_df = pd.DataFrame(index=prices_df.index)
        target_df['close'] = prices_df['close']
        target_df['vol'] = daily_vol_series
        
        # Filter out rows with NaN volatility or below minimum
        target_df = target_df[target_df['vol'] >= min_vol].copy()
        if target_df.empty:
            logging.warning("No rows with sufficient volatility for triple barrier targets")
            return None

        # Set up price barriers (take profit and stop loss)
        target_df['tp'] = target_df['close'] * (1 + tp_mult * target_df['vol'])
        target_df['sl'] = target_df['close'] * (1 - sl_mult * target_df['vol'])
        
        # Prepare to store results
        target_df['target'] = np.nan
        target_df['target_date'] = pd.NaT
        target_df['return_pct'] = np.nan
        
        # Set up barrier crossing check
        for i in range(len(target_df) - lookahead):
            current_idx = target_df.index[i]
            future_idx = target_df.index[i+1:i+lookahead+1]
            
            if future_idx.empty:  # Skip if we're near the end of the data
                continue
                
            # Get the future prices for barrier check
            future_prices = prices_df.loc[future_idx]
            
            # Current values
            curr_close = target_df.loc[current_idx, 'close']
            tp_price = target_df.loc[current_idx, 'tp']
            sl_price = target_df.loc[current_idx, 'sl']
            
            # Find if/when the price hits our barriers
            hits_tp = future_prices[future_prices['high'] >= tp_price].index
            hits_sl = future_prices[future_prices['low'] <= sl_price].index
            
            # Determine which barrier was hit first (if any)
            if not hits_tp.empty and not hits_sl.empty:
                if hits_tp[0] <= hits_sl[0]:  # TP hit first
                    target_date = hits_tp[0]
                    target_value = 1  # Success
                    return_pct = (tp_price / curr_close - 1) * 100
                else:  # SL hit first
                    target_date = hits_sl[0]
                    target_value = 0  # Failure
                    return_pct = (sl_price / curr_close - 1) * 100
            elif not hits_tp.empty:  # Only TP hit
                target_date = hits_tp[0]
                target_value = 1  # Success
                return_pct = (tp_price / curr_close - 1) * 100
            elif not hits_sl.empty:  # Only SL hit
                target_date = hits_sl[0]
                target_value = 0  # Failure
                return_pct = (sl_price / curr_close - 1) * 100
            else:  # No barriers hit, use the last price
                last_idx = future_idx[-1]
                last_close = future_prices.loc[last_idx, 'close']
                return_pct = (last_close / curr_close - 1) * 100
                
                # Account for transaction costs (in percentage)
                pip_size = prices_df['pip_size'].iloc[0] if 'pip_size' in prices_df.columns else 0.0001
                cost_pct = cost_pips * pip_size / curr_close * 100
                
                # Adjust return by transaction costs
                net_return_pct = return_pct - cost_pct
                
                # Determine if trade was profitable after costs
                target_value = 1 if net_return_pct > 0 else 0
                target_date = last_idx
            
            # Store the result
            target_df.loc[current_idx, 'target'] = target_value
            target_df.loc[current_idx, 'target_date'] = target_date
            target_df.loc[current_idx, 'return_pct'] = return_pct
        
        # Clean up rows without targets
        target_df = target_df.dropna(subset=['target']).copy()
        
        # Convert target to int
        target_df['target'] = target_df['target'].astype(int)
        
        return target_df
    except Exception as e:
        logging.error(f"Error in triple barrier target generation: {e}")
        return None

# --- [4] Prepare Final Data Function V7 (Handles feature lists dynamically) ---
def prepare_final_data_v7(features_df, target_series, initial_numeric, initial_categorical):
    """
    Prepares the final dataset for modeling.
    V7 with explicit target passing and returning of filtered feature lists.
    Handles potential duplicate indexes by ensuring index uniqueness.
    """
    logging.info("Preparing final data for modeling")

    # Make a copy of the dataset to avoid modifying original
    features_df = features_df.copy()
    
    # Check for and handle duplicate indices in features_df
    if features_df.index.duplicated().any():
        logging.warning(f"Found {features_df.index.duplicated().sum()} duplicate indices in features_df - keeping first occurrence")
        features_df = features_df.loc[~features_df.index.duplicated(keep='first')]
    
    # Check for and handle duplicate indices in target_series
    if isinstance(target_series, pd.Series) and target_series.index.duplicated().any():
        logging.warning(f"Found {target_series.index.duplicated().sum()} duplicate indices in target_series - keeping first occurrence")
        target_series = target_series.loc[~target_series.index.duplicated(keep='first')]

    # Now merge target into features safely
    try:
        common_indices = features_df.index.intersection(target_series.index)
        logging.info(f"Using {len(common_indices)} common indices between features and targets")
        
        # Filter both DataFrames to common indices
        features_df = features_df.loc[common_indices]
        target_series = target_series.loc[common_indices]
        
        # Add target column to features DataFrame
        features_df[TARGET_COLUMN] = target_series
    except Exception as e:
        logging.error(f"Error adding target to features: {e}")
        raise

    # Remove rows with NaN in target
    features_df = features_df.dropna(subset=[TARGET_COLUMN])
    
    # Separate features and target
    X = features_df.drop(columns=[TARGET_COLUMN])
    y = features_df[TARGET_COLUMN]
    
    # Filter out features with too many NaNs
    original_feature_count = len(initial_numeric) + len(initial_categorical)
    usable_numeric = []
    for col in initial_numeric:
        if col in X.columns:
            missing_pct = X[col].isna().mean()
            if missing_pct <= 0.2:  # Allow up to 20% missing (arbitrary threshold)
                usable_numeric.append(col)
            else:
                logging.info(f"Dropping feature '{col}' due to high missingness ({missing_pct:.1%})")
        else:
            logging.warning(f"Feature '{col}' not found in dataset")
    
    # Keep all categorical features (likely to be hand-crafted and important)
    usable_categorical = [col for col in initial_categorical if col in X.columns]
    missing_categorical = set(initial_categorical) - set(usable_categorical)
    if missing_categorical:
        logging.warning(f"Missing categorical features: {missing_categorical}")
    
    # Check if we've dropped too many features
    if len(usable_numeric) + len(usable_categorical) < 0.5 * original_feature_count:
        logging.warning(f"More than 50% of features were dropped due to missing values")
    
    # Final features
    final_feature_list = usable_numeric + usable_categorical
    X = X[final_feature_list]
    
    # Fill remaining NaNs strategically
    # 1. Numeric columns: fill with median
    for col in usable_numeric:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            logging.debug(f"Filled NaN in '{col}' with median: {median_val}")
    
    # 2. Categorical columns: fill with mode or specific value
    for col in usable_categorical:
        if X[col].isna().any():
            mode_val = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode_val)
            logging.debug(f"Filled NaN in '{col}' with mode: {mode_val}")
    
    logging.info(f"Final data shape: X: {X.shape}, y: {y.shape}")
    logging.info(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y, usable_numeric, usable_categorical


# --- [5] Build Pipeline Function V7 (Adds Calibrator) ---
def build_pipeline_v7(numeric_features, categorical_features, config):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    # Always use XGBoost now
    classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                   random_state=42, n_jobs=-1)

    steps = [('preprocessor', preprocessor), ('classifier', classifier)]

    # Add calibration step if configured
    if config.get('calibrate_probabilities', False):
        logging.info(f"Adding probability calibration (Method: {config.get('calibration_method', 'isotonic')})")
        # Use an estimator for calibration (the XGBoost classifier itself)
        # CORRECTED PARAMETER NAME: estimator= instead of base_estimator=
        calibrator = CalibratedClassifierCV(
            estimator=classifier, # <--- CORRECTED
            method=config.get('calibration_method', 'isotonic'),
            cv='prefit' # Use the already trained classifier from the main fit
        )
        # We will calibrate the FINAL best model after Optuna search.
        logging.warning("Calibration step will be applied to the FINAL model after Optuna, not during HPO.")

    # Return the pipeline WITHOUT the calibrator for HPO stage
    return Pipeline(steps)

# --- [6] Optuna Objective Function V7 (as before, uses build_pipeline_v7 structure) ---
def objective_v7(trial, X_train, y_train, X_val, y_val, numeric_features, categorical_features, handle_imbalance, is_imbalanced, config):
    """
    Optuna objective function that uses CV if X_val/y_val are None, or a single validation set otherwise.
    Includes better error handling and early stopping support.
    """
    # Get configuration parameters
    cv_splits = config.get('cv_splits', 5)
    use_xgboost = config.get('use_xgboost', True)
    early_stopping_rounds = config.get('early_stopping_optuna', 50)
    
    # Determine if we're using CV or a single validation set
    use_cv = X_val is None or y_val is None
    
    try:
        # Suggested parameter space (similar for XGBoost and sklearn GBM)
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
            'reg_lambda': trial.suggest_float('lambda', 0.1, 10.0, log=True),
        }
        
        # Add class weight parameter if handling imbalance
        if handle_imbalance and is_imbalanced:
            if use_xgboost:
                # For XGBoost: scale_pos_weight for binary classification
                minority_class_ratio = y_train.value_counts(normalize=True).min()
                params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 
                                                               1.0, 
                                                               min(10.0, 1.0/minority_class_ratio), 
                                                               log=True)
            else:
                # For sklearn: class_weight='balanced' is handled separately
                pass
        
        # Create base pipeline
        pipeline = build_pipeline_v7(numeric_features, categorical_features, config)
        
        # Set classifier parameters
        for param_name, param_value in params.items():
            pipeline.named_steps['classifier'].set_params(**{param_name: param_value})
        
        # For cross-validation
        if use_cv:
            cv_scores = []
            cv_splitter = TimeSeriesSplit(n_splits=cv_splits)
            
            for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train)):
                # Split data
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Skip fold if not enough samples or classes
                if X_fold_train.shape[0] < 20 or y_fold_train.nunique() < 2:
                    continue
                
                # Fit preprocessor
                preprocessor = pipeline.named_steps['preprocessor']
                preprocessor.fit(X_fold_train, y_fold_train)
                X_fold_train_transformed = preprocessor.transform(X_fold_train)
                X_fold_val_transformed = preprocessor.transform(X_fold_val)
                
                # Handle class weights for imbalanced data
                fit_params = {}
                if handle_imbalance and is_imbalanced:
                    sample_weights = compute_sample_weight(class_weight='balanced', y=y_fold_train)
                    fit_params['sample_weight'] = sample_weights
                
                # Extract classifier from pipeline and fit
                classifier = pipeline.named_steps['classifier']
                
                # Add early stopping for XGBoost
                if early_stopping_rounds and use_xgboost and isinstance(classifier, xgb.XGBClassifier):
                    eval_set = [(X_fold_val_transformed, y_fold_val)]
                    # For XGBoost classifiers only, don't pass early_stopping_rounds directly
                    classifier.fit(
                        X_fold_train_transformed, 
                        y_fold_train,
                        eval_set=eval_set,
                        verbose=False,
                        **fit_params
                    )
                else:
                    # For other classifiers
                    classifier.fit(X_fold_train_transformed, y_fold_train, **fit_params)
                
                # Predict and score
                try:
                    y_pred_proba = classifier.predict_proba(X_fold_val_transformed)[:, 1]
                    fold_score = roc_auc_score(y_fold_val, y_pred_proba)
                    cv_scores.append(fold_score)
                except Exception as e:
                    logging.warning(f"Error calculating ROC-AUC for fold {fold}: {e}")
            
            # Return mean CV score or a penalty if failed
            if not cv_scores:
                return 0.0  # Penalty for failed CV
            return np.mean(cv_scores)
        
        # For single validation set
        else:
            # Fit preprocessor
            preprocessor = pipeline.named_steps['preprocessor']
            preprocessor.fit(X_train, y_train)
            X_train_transformed = preprocessor.transform(X_train)
            X_val_transformed = preprocessor.transform(X_val)
            
            # Handle class weights for imbalanced data
            fit_params = {}
            if handle_imbalance and is_imbalanced:
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                fit_params['sample_weight'] = sample_weights
            
            # Extract classifier from pipeline and fit
            classifier = pipeline.named_steps['classifier']
            
            # Add early stopping for XGBoost
            if early_stopping_rounds and use_xgboost and isinstance(classifier, xgb.XGBClassifier):
                eval_set = [(X_val_transformed, y_val)]
                # For XGBoost classifiers only, don't pass early_stopping_rounds directly
                classifier.fit(
                    X_train_transformed, 
                    y_train,
                    eval_set=eval_set,
                    verbose=False,
                    **fit_params
                )
            else:
                # For other classifiers
                classifier.fit(X_train_transformed, y_train, **fit_params)
            
            # Predict and score
            y_pred_proba = classifier.predict_proba(X_val_transformed)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
    
    except Exception as e:
        logging.warning(f"Trial raised exception: {e}")
        raise optuna.exceptions.TrialPruned() from e

# --- [7] Execute Optuna Study Function (as before) ---
def run_optuna_optimization_v7(X, y, numeric_features, categorical_features, config):
    """
    Runs Optuna hyperparameter optimization with V7 improvements.
    Has better error handling and logs progress more clearly.
    """
    logging.info(f"Starting Optuna optimization with {config.get('optuna_trials', 100)} trials")
    
    # Get configuration parameters
    n_trials = config.get('optuna_trials', 100)
    timeout = config.get('optuna_timeout_seconds', None)
    handle_imbalance = config.get('handle_imbalance', True)
    use_xgboost = config.get('use_xgboost', True)
    early_stopping_optuna = config.get('early_stopping_optuna', 50)
    
    # Check for class imbalance
    class_dist = y.value_counts(normalize=True)
    is_imbalanced = class_dist.min() < 0.25
    if is_imbalanced and handle_imbalance:
        logging.info(f"Class imbalance detected: {dict(class_dist)}. Will use class weights.")
    
    # Create study
    try:
        study = optuna.create_study(direction="maximize", 
                                    sampler=optuna.samplers.TPESampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))

        # Create a partial function with our parameters
        objective_partial = partial(
            objective_v7,
            X_train=X, 
            y_train=y,
            X_val=None,  # We'll use CV within objective
            y_val=None,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            handle_imbalance=handle_imbalance,
            is_imbalanced=is_imbalanced,
            config=config
        )

        # Run optimization
        start_time = time.time()
        study.optimize(objective_partial, n_trials=n_trials, timeout=timeout, 
                      show_progress_bar=True, catch=(Exception,))
        end_time = time.time()

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Log results
        logging.info(f"Optuna optimization completed in {end_time - start_time:.1f} seconds")
        logging.info(f"Best CV ROC-AUC: {best_value:.4f}")
        logging.info(f"Best parameters: {best_params}")
        
        return best_params
    
    except Exception as e:
        logging.error(f"Error during Optuna optimization: {e}", exc_info=True)
        # Return sensible defaults on error
        default_params = {
            'learning_rate': 0.05,
            'min_child_weight': 3,
            'max_depth': 5,
            'gamma': 0.2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 0.01,
            'lambda': 1.0,
            'scale_pos_weight': 1.0 if not (is_imbalanced and handle_imbalance) else (1.0 / class_dist.min())
        }
        logging.warning(f"Using default parameters due to error: {default_params}")
        return default_params

# --- [8] Final Model Training, Calibration & Evaluation Function V7 ---
def train_calibrate_evaluate_final_model_v7(X, y, numeric_features, categorical_features, best_params, config):
    """Final model training with CV evaluation."""
    logging.info("Training final model with cross-validation...")
    
    # Get parameters from config
    cv_splits = config.get('cv_splits', 5)
    handle_imbalance = config.get('handle_imbalance', True)
    use_xgboost = config.get('use_xgboost', True)
    calibrate_prob = config.get('calibrate_probabilities', True)
    calibration_method = config.get('calibration_method', 'isotonic')
    
    # Check for class imbalance
    class_dist = y.value_counts(normalize=True)
    is_imbalanced = class_dist.min() < 0.25
    if is_imbalanced:
        logging.info(f"Class imbalance detected: {dict(class_dist)}. {'Using' if handle_imbalance else 'Not using'} class weights.")
    
    # For CV 
    cv_splitter = TimeSeriesSplit(n_splits=cv_splits)
    
    # Build base pipeline
    base_pipeline_structure = build_pipeline_v7(numeric_features, categorical_features, config)
    
    # Set best parameters
    for param, value in best_params.items():
        # Handle different parameter naming conventions
        if param in ['alpha', 'lambda'] and use_xgboost:
            param = f'reg_{param}'  # XGBoost uses reg_alpha and reg_lambda
        base_pipeline_structure.named_steps['classifier'].set_params(**{param: value})
    
    # Placeholders for CV results
    cv_scores_roc_auc = []
    cv_scores_f1 = []
    cv_oof_preds = pd.Series(dtype=int, index=X.index)
    cv_oof_probas = pd.Series(dtype=float, index=X.index)
    feature_importances_folds = []
    
    # Train CV folds
    logging.info(f"Starting {cv_splits}-fold cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        logging.info(f"Training fold {fold+1}")
        
        try:
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Clone pipeline for this fold
            fold_pipeline = clone(base_pipeline_structure)
            
            # Prepare fit parameters
            fit_params = {}
            
            # Add sample weights for imbalanced data
            if handle_imbalance and is_imbalanced:
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
                fit_params['classifier__sample_weight'] = sample_weights
            
            # Fit the pipeline
            try:
                fold_pipeline.fit(X_train, y_train, **fit_params)
            except Exception as e:
                logging.error(f"Fold {fold+1}: Error training model: {e}")
                continue
            
            # Predict probabilities
            try:
                y_proba = fold_pipeline.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)
                
                # Store OOF predictions
                cv_oof_preds.loc[X_test.index] = y_pred
                cv_oof_probas.loc[X_test.index] = y_proba
                
                # Calculate metrics
                fold_roc_auc = roc_auc_score(y_test, y_proba)
                fold_f1 = f1_score(y_test, y_pred)
                
                cv_scores_roc_auc.append(fold_roc_auc)
                cv_scores_f1.append(fold_f1)
                
                logging.info(f"Fold {fold+1} metrics - ROC-AUC: {fold_roc_auc:.4f}, F1: {fold_f1:.4f}")
                
                # Get feature importances if available
                if hasattr(fold_pipeline.named_steps['classifier'], 'feature_importances_'):
                    feature_importances_folds.append(fold_pipeline.named_steps['classifier'].feature_importances_)
                
            except Exception as e:
                logging.error(f"Fold {fold+1}: Error evaluating model: {e}")
        
        except Exception as e:
            logging.error(f"Fold {fold+1}: Error processing fold: {e}")
    
    # Calculate mean metrics
    if cv_scores_roc_auc:
        mean_roc_auc = np.mean(cv_scores_roc_auc)
        mean_f1 = np.mean(cv_scores_f1)
        logging.info(f"Mean CV ROC-AUC: {mean_roc_auc:.4f}, Mean CV F1: {mean_f1:.4f}")
    else:
        mean_roc_auc = np.nan
        mean_f1 = np.nan
        logging.warning("No valid CV folds completed. Cannot calculate metrics.")
    
    # Get OOF values where predictions were made
    valid_indices = cv_oof_probas.dropna().index
    if not valid_indices.empty:
        oof_y_true = y.loc[valid_indices]
        oof_y_pred = cv_oof_preds.loc[valid_indices].astype(int)
        oof_y_proba = cv_oof_probas.loc[valid_indices]
    else:
        logging.warning("No valid OOF predictions. Using empty series.")
        oof_y_true = pd.Series(dtype=int)
        oof_y_pred = pd.Series(dtype=int)
        oof_y_proba = pd.Series(dtype=float)
    
    # Calculate feature stability
    feature_stability_df = None
    if feature_importances_folds and len(feature_importances_folds) > 1:
        try:
            # Get feature names
            preprocessor = base_pipeline_structure.named_steps['preprocessor']
            feature_names = []
            
            # Get numeric feature names (transformed)
            if numeric_features:
                for feature in numeric_features:
                    feature_names.append(feature)
            
            # Get one-hot encoded categorical feature names
            if categorical_features:
                for feature in categorical_features:
                    values = X[feature].unique()
                    for value in values:
                        feature_names.append(f"{feature}_{value}")
            
            # Create feature importance DataFrame
            if len(feature_names) == feature_importances_folds[0].shape[0]:
                importance_df = pd.DataFrame(feature_importances_folds, columns=feature_names)
                
                # Calculate stability metrics
                feature_stability_df = pd.DataFrame({
                    'mean_importance': importance_df.mean(),
                    'std_importance': importance_df.std()
                })
                
                # Calculate coefficient of variation
                feature_stability_df['stability_cv'] = feature_stability_df['std_importance'] / (feature_stability_df['mean_importance'] + 1e-9)
                
                # Sort by importance
                feature_stability_df = feature_stability_df.sort_values('mean_importance', ascending=False)
                
                logging.info(f"Feature importance stability calculation complete. Top features:\n{feature_stability_df.head(10).to_string()}")
            else:
                logging.warning(f"Feature name length mismatch: {len(feature_names)} names vs {feature_importances_folds[0].shape[0]} importance values")
        except Exception as e:
            logging.error(f"Error in feature stability calculation: {e}")
    
    # Train final model on all data
    final_model = None
    try:
        logging.info("Training final model on all data...")
        
        # Create new pipeline
        final_pipeline = clone(base_pipeline_structure)
        
        # Prepare fit parameters
        fit_params = {}
        
        # Handle class imbalance
        if handle_imbalance and is_imbalanced:
            sample_weights = compute_sample_weight(class_weight='balanced', y=y)
            fit_params['classifier__sample_weight'] = sample_weights
        
        # Fit the model
        final_pipeline.fit(X, y, **fit_params)
        logging.info("Final model trained successfully")
        
        # Calibrate if requested
        if calibrate_prob:
            try:
                logging.info(f"Calibrating probabilities using {calibration_method}...")
                
                # Create a new set for calibration (use a small portion of the data)
                calibration_frac = 0.2
                n_samples = len(X)
                cal_size = int(n_samples * calibration_frac)
                
                # Get indices for calibration (use most recent data)
                cal_indices = X.index[-cal_size:]
                X_cal = X.loc[cal_indices]
                y_cal = y.loc[cal_indices]
                
                # Create and fit calibrator
                calibrator = CalibratedClassifierCV(
                    estimator=clone(final_pipeline.named_steps['classifier']),
                    method=calibration_method,
                    cv=3  # Use small number of CV folds for calibration
                )
                
                # Prepare calibration data
                X_cal_transformed = final_pipeline.named_steps['preprocessor'].transform(X_cal)
                
                # Fit calibrator
                calibrator.fit(X_cal_transformed, y_cal)
                
                # Create new pipeline with calibrator
                calibrated_pipeline = Pipeline([
                    ('preprocessor', final_pipeline.named_steps['preprocessor']),
                    ('classifier', calibrator)
                ])
                
                final_model = calibrated_pipeline
                logging.info("Probability calibration complete")
            except Exception as e:
                logging.error(f"Error during calibration: {e}")
                logging.warning("Using uncalibrated model")
                final_model = final_pipeline
        else:
            final_model = final_pipeline
            
    except Exception as e:
        logging.error(f"Error training final model: {e}")
    
    return final_model, mean_roc_auc, mean_f1, oof_y_true, oof_y_pred, oof_y_proba, feature_stability_df

# --- Function to load configuration ---
def load_config():
    """
    Load configuration from CONFIG_FILE.
    Returns the configuration dictionary or uses already loaded config.
    """
    try:
        # Check if config is already loaded globally
        global config
        if 'config' in globals() and config is not None:
            return config
            
        # Otherwise load from file
        config_path = CONFIG_FILE
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

# --- [MAIN SCRIPT EXECUTION] ---
if __name__ == "__main__":
    try:
        logging.info("-" * 80)
        logging.info("STARTING ADVANCED MODEL TRAINING PIPELINE V7")
        logging.info("-" * 80)
        
        # --- 1. Initialize Components ---
        config = load_config()
        asset_code = config.get('asset_code', 'BTC')
        logging.info(f"Training model for asset: {asset_code}")
        
        # Create SignalGenerator instance
        sg_instance = SignalGenerator()
        logging.info("SignalGenerator initialized")
        
        # --- 2. Load and Prepare Data ---
        logging.info("Loading data...")
        trading_data = load_and_combine_data_v7(config)
        
        if trading_data is None or trading_data.empty:
            logging.error("No data loaded or data is empty")
            raise ValueError("Failed to load trading data")
        
        logging.info("Engineering features...")
        features_df = engineer_advanced_features_v7(trading_data, sg_instance, config)
        
        logging.info("Calculating target labels...")
        try:
            # For each pair, calculate volatility and targets
            grouped_by_pair = features_df.groupby('pair_id')
            all_targets = []
            
            for pair_name, pair_df in grouped_by_pair:
                logging.info(f"Processing triple barrier targets for {pair_name}...")
                # Compute volatility for this pair
                daily_vol = compute_daily_volatility_v7(pair_df['close'], lookback=50)
                # Get triple barrier targets
                triple_barrier_params = config.get('triple_barrier_settings', {})
                targets = get_triple_barrier_targets_v7(pair_df, daily_vol, triple_barrier_params)
                
                if targets is not None and not targets.empty:
                    targets['pair_id'] = pair_name
                    all_targets.append(targets)
                    logging.info(f"Generated {len(targets)} targets for {pair_name}")
                else:
                    logging.warning(f"No targets generated for {pair_name}")
            
            if not all_targets:
                logging.error("No targets generated for any pair")
                raise ValueError("No valid targets could be generated")
            
            # Combine all targets
            features_with_targets = pd.concat(all_targets, axis=0)
            logging.info(f"Combined {len(features_with_targets)} total target entries")
            
            # Ensure target column exists and is properly formatted
            if 'target' not in features_with_targets.columns:
                logging.error("'target' column missing from features_with_targets")
                raise ValueError("Missing target column in processed data")
            
            # Convert target to integer
            features_with_targets['target'] = features_with_targets['target'].astype(int)
            
            logging.info("Preparing final dataset...")
            X, y, numeric_features, categorical_features = prepare_final_data_v7(
                features_with_targets, features_with_targets['target'], NUMERIC_FEATURES, CATEGORICAL_FEATURES)
            
            logging.info(f"Final dataset shape: X: {X.shape}, y: {y.shape}")
            logging.info(f"Target distribution: {dict(y.value_counts(normalize=True).round(3))}")
            
        except Exception as e:
            logging.error(f"Error in data preparation: {e}")
            logging.error(traceback.format_exc())
            raise
        
        # --- 3. Optimize Model Parameters ---
        logging.info("Starting hyperparameter optimization...")
        try:
            best_params = run_optuna_optimization_v7(X, y, numeric_features, categorical_features, config)
            logging.info(f"Best parameters: {best_params}")
        except Exception as e:
            logging.error(f"Error in hyperparameter optimization: {e}")
            logging.warning("Using default parameters")
            best_params = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'n_estimators': 100
            }
        
        # --- 4. Train and Evaluate Final Model ---
        logging.info("Training final model...")
        try:
            final_model, cv_roc_auc, cv_f1, oof_y_true, oof_y_pred, oof_y_proba, feature_stability_df = (
                train_calibrate_evaluate_final_model_v7(
                    X, y, numeric_features, categorical_features, best_params, config
                )
            )
            
            if final_model is not None:
                # Save the model
                output_model_name = config.get('output_model_base_name', f"{asset_code}_model_v7")
                model_path = os.path.join(MODELS_DIR, f"{output_model_name}.pkl")
                
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(final_model, f)
                    logging.info(f"Model saved to {model_path}")
                except Exception as e:
                    logging.error(f"Error saving model: {e}")
            else:
                logging.error("Final model training failed. No model to save.")
                
            # --- 5. Create Results Summary ---
            logging.info("Creating results summary...")
            # Get top features if available
            top_features = None
            if feature_stability_df is not None:
                try:
                    top_features = feature_stability_df.head(20).to_dict(orient='index')
                except Exception as e:
                    logging.error(f"Error extracting top features: {e}")
            
            # Create summary dictionary
            summary = {
                'model_name': config.get('output_model_base_name', f"{asset_code}_model_v7"),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cv_splits': config.get('cv_splits', 5),
                'performance': {
                    'cv_roc_auc': float(cv_roc_auc) if not np.isnan(cv_roc_auc) else None,
                    'cv_f1': float(cv_f1) if not np.isnan(cv_f1) else None
                },
                'best_params': best_params,
                'top_features': top_features
            }
            
            # Save summary
            summary_path = os.path.join(RESULTS_DIR, f"{config.get('output_model_base_name', f'{asset_code}_model_v7')}_summary.json")
            try:
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                logging.info(f"Results summary saved to {summary_path}")
            except Exception as e:
                logging.error(f"Error saving results summary: {e}")
                
        except Exception as final_model_error:
            logging.error(f"Error in final model training and evaluation: {final_model_error}")
            
        logging.info("-" * 80)
        logging.info("ADVANCED MODEL TRAINING PIPELINE V7 COMPLETED")
        logging.info("-" * 80)
        
    except Exception as e:
        logging.error(f"ERROR in main execution: {e}")
        logging.error(traceback.format_exc())
        raise