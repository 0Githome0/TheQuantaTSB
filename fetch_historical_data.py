# --- START OF FILE fetch_historical_data.py ---

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz # Usually installed with pandas
import os
import json
import logging
import time

# --- Configuration ---
CONFIG_FILE = "config.json"
TRAINING_CONFIG_FILE = "training_config.json"
# --- IMPORTANT: Set where to save the CSV files ---
# By default, saves in the same directory as the script.
# Change this if you want to save them elsewhere (e.g., a 'data' folder)
OUTPUT_DIR = "."

# --- How many bars to fetch? ---
# REDUCED FROM 100,000 to 10,000 as the large number likely caused the
# 'Invalid params' error (-2) on the Exness Trial server.
# You might need to adjust this value further based on your broker's limits.
BARS_TO_FETCH = 10000

# --- Target Timeframe (Must match training script expectation) ---
TARGET_TIMEFRAME = mt5.TIMEFRAME_H1 # H1 timeframe

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s] - %(message)s',
                    handlers=[logging.StreamHandler()]) # Log to console

# --- Load Config Files ---
try:
    with open(CONFIG_FILE, 'r') as f:
        mt5_config = json.load(f).get("mt5_details", {})
    with open(TRAINING_CONFIG_FILE, 'r') as f:
        training_config = json.load(f)
        asset_files_to_fetch = training_config.get("asset_files", {})
    if not asset_files_to_fetch:
        logging.error(f"No assets found in 'asset_files' section of {TRAINING_CONFIG_FILE}")
        exit()
except FileNotFoundError as e:
    logging.error(f"Error: Configuration file not found: {e}. Please ensure {CONFIG_FILE} and {TRAINING_CONFIG_FILE} exist.")
    exit()
except json.JSONDecodeError as e:
    logging.error(f"Error decoding JSON configuration: {e}")
    exit()
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    exit()

# --- Get MT5 Connection Details ---
login = mt5_config.get("login")
password = mt5_config.get("password")
server = mt5_config.get("server")
path = mt5_config.get("path")

if not all([login, password, server]):
    logging.error(f"MT5 login, password, or server missing in {CONFIG_FILE}")
    exit()

# --- Main Data Fetching Function ---
def fetch_mt5_data():
    """Connects to MT5, fetches historical data for specified assets, and saves to CSV."""

    # Initialize MT5 connection
    logging.info("Initializing MetaTrader 5...")
    init_success = False
    if path and os.path.exists(path):
        init_success = mt5.initialize(path=path)
        if not init_success:
             logging.warning(f"Initialization with path '{path}' failed (Error: {mt5.last_error()}). Trying default path...")
             init_success = mt5.initialize() # Try default path if specific fails
    else:
        logging.warning(f"MT5 path ('{path}') not found or not specified. Trying default path...")
        init_success = mt5.initialize()

    if not init_success:
        logging.error(f"MT5 initialize() failed, error code = {mt5.last_error()}. Check MT5 installation/path and ensure terminal is running.")
        mt5.shutdown()
        return

    logging.info(f"MetaTrader 5 Initialized Successfully. Version: {mt5.version()}")

    # Login
    logging.info(f"Logging into account {login} on server {server}...")
    # Ensure login is treated as an integer if it's numeric in the JSON
    try:
        login_val = int(login)
    except ValueError:
        logging.error(f"Invalid MT5 login value '{login}' in config. It should be numeric.")
        mt5.shutdown()
        return

    if not mt5.login(login=login_val, password=password, server=server):
        logging.error(f"MT5 login failed, error code = {mt5.last_error()}")
        mt5.shutdown()
        return
    logging.info("MT5 Login Successful.")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fetch data for each asset defined in training_config
    for asset_key, target_filename in asset_files_to_fetch.items():
        logging.info(f"--- Processing: {asset_key} ---")

        # --- Determine the correct MT5 symbol ---
        mt5_symbol_to_try = asset_key
        symbol_info = mt5.symbol_info(mt5_symbol_to_try)

        if not symbol_info and mt5_symbol_to_try.endswith('m'):
            symbol_without_m = mt5_symbol_to_try[:-1]
            logging.info(f"Symbol '{mt5_symbol_to_try}' not found, trying '{symbol_without_m}'...")
            symbol_info_no_m = mt5.symbol_info(symbol_without_m)
            if symbol_info_no_m:
                mt5_symbol_to_try = symbol_without_m
                symbol_info = symbol_info_no_m
                logging.info(f"Using MT5 symbol: '{mt5_symbol_to_try}'")
            else:
                 logging.warning(f"Neither '{asset_key}' nor '{symbol_without_m}' found by info request.")
                 # Proceed with original key, rely on select below

        elif not symbol_info:
             logging.warning(f"Symbol '{mt5_symbol_to_try}' not initially found by info request.")
             # Proceed with original key, rely on select below
        else:
            logging.info(f"Using MT5 symbol: '{mt5_symbol_to_try}'")

        # Ensure symbol is available in MarketWatch
        logging.debug(f"Selecting symbol '{mt5_symbol_to_try}'...")
        if not mt5.symbol_select(mt5_symbol_to_try, True):
            logging.warning(f"Could not select symbol '{mt5_symbol_to_try}' in MarketWatch. Fetch attempt might fail. Error: {mt5.last_error()}")
        else:
            time.sleep(0.5) # Brief pause after selection

        # Check symbol info again after select attempt
        symbol_info = mt5.symbol_info(mt5_symbol_to_try)
        if not symbol_info:
             logging.error(f"Could not get info for symbol '{mt5_symbol_to_try}' even after select attempt. Skipping. Error: {mt5.last_error()}")
             continue

        # Fetch historical rates
        logging.info(f"Fetching up to {BARS_TO_FETCH} bars for {mt5_symbol_to_try} (Timeframe: H1)...")
        try:
            # Fetch data from the current position backwards
            rates = mt5.copy_rates_from_pos(mt5_symbol_to_try, TARGET_TIMEFRAME, 0, BARS_TO_FETCH)
        except Exception as e:
             logging.error(f"Exception during copy_rates_from_pos for {mt5_symbol_to_try}: {e}")
             rates = None

        if rates is None:
            # Check last error specifically for the 'invalid params' case after changing BARS_TO_FETCH
            last_err_code, last_err_msg = mt5.last_error()
            if last_err_code == -2:
                logging.error(f"Failed to fetch rates for {mt5_symbol_to_try}. Error: {mt5.last_error()}. "
                              f"Even with {BARS_TO_FETCH} bars, the parameters are invalid. "
                              f"Check symbol name ('{mt5_symbol_to_try}') and timeframe (H1) carefully for this broker. Skipping.")
            else:
                logging.error(f"Failed to fetch rates for {mt5_symbol_to_try}. MT5 Error: {mt5.last_error()}. Skipping.")
            continue

        if len(rates) == 0:
            logging.warning(f"Fetched 0 bars for {mt5_symbol_to_try}. Does your broker have H1 data for this symbol? Skipping.")
            continue

        logging.info(f"Successfully fetched {len(rates)} bars for {mt5_symbol_to_try}.")

        # Convert to pandas DataFrame
        df = pd.DataFrame(rates)
        # Convert time in seconds into datetime format
        df['time'] = pd.to_datetime(df['time'], unit='s')
        # Rename volume column to match training script expectation
        df.rename(columns={'tick_volume': 'tick_volume'}, inplace=True) # Already correct in original

        # Select and order required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_cols):
             # Some brokers might not provide tick_volume, handle this
             if 'tick_volume' not in df.columns:
                 logging.warning(f"Column 'tick_volume' not found for {mt5_symbol_to_try}. Creating it with NaN values.")
                 df['tick_volume'] = pd.NA
             else:
                 logging.error(f"Fetched data for {mt5_symbol_to_try} is missing other required columns. Found: {list(df.columns)}. Skipping.")
                 continue

        df_final = df[required_cols]

        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, target_filename)
        try:
            df_final.to_csv(output_path, index=False)
            logging.info(f"Successfully saved data for {asset_key} to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save data for {asset_key} to {output_path}: {e}")

        time.sleep(1) # Add a small delay between requests

    # Shutdown MT5 connection
    logging.info("Data fetching complete. Shutting down MetaTrader 5 connection.")
    mt5.shutdown()

# --- Run the Data Fetching Process ---
if __name__ == "__main__":
    logging.info("--- Starting Historical Data Fetch Script ---")
    # Add reminder about prerequisites
    logging.info("Ensure MetaTrader 5 terminal is running and config files are correct.")
    fetch_mt5_data()
    logging.info("--- Data Fetch Script Finished ---")

# --- END OF FILE fetch_historical_data.py ---