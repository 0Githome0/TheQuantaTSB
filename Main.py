# --- START OF FILE Main.py ---

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import threading
import time
import logging
import pandas as pd
import queue
from datetime import datetime, date, timedelta # Added date and timedelta
import asyncio
import pytz
import MetaTrader5 as mt5
import warnings
import os
import json
from typing import Dict, Optional, List, Any, Tuple
from ENTRY import AdvancedEntryStrategies

# Import your custom modules
try:
    from Signal import SignalGenerator, TimeFrames
    from advanced_analysis import AdvancedAnalysis
    # ---> ADDED IMPORTS <---
    from performance_tracker import PerformanceTracker
    from profitability_enhancer import ProfitabilityEnhancer
    from execution_manager import ExecutionManager
    # ---> END ADDED IMPORTS <---
except ImportError as e:
    # Updated error message to include new modules
    print(f"FATAL ERROR: Could not import required modules (Signal, AdvancedAnalysis, PerformanceTracker, ProfitabilityEnhancer, ExecutionManager): {e}")
    print("Please ensure Signal.py, advanced_analysis.py, performance_tracker.py, profitability_enhancer.py, and execution_manager.py are in the same directory.")
    exit() # Exit if core modules are missing

# Suppress specific warnings (optional)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Constants ---
APP_NAME = "FXL V1 - Signal Bot"
WINDOW_WIDTH = 1350 # Slightly wider
WINDOW_HEIGHT = 820
UPDATE_INTERVAL_MS = 250  # Slightly faster UI updates
DASHBOARD_UPDATE_INTERVAL_MS = 5000 # How often to update Perf/Positions tabs (5 seconds)
CONFIG_FILE = "config.json"
LOG_FILE = 'logs/main_app.log' # Centralized log file definition

# --- UI Theme ---
try:
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
except Exception as e:
    print(f"Warning: Could not set CustomTkinter theme: {e}")


# --- Logging Setup ---
log_queue = queue.Queue() # Queue for thread-safe logging to UI

class QueueHandler(logging.Handler):
    """Sends log records to a queue for the UI thread."""
    def __init__(self, log_queue_instance):
        super().__init__()
        self.log_queue = log_queue_instance

    def emit(self, record):
        self.log_queue.put(self.format(record))

# Configure Root Logger
logger = logging.getLogger()
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s')
if logger.hasHandlers(): logger.handlers.clear()
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(log_formatter)
logger.addHandler(queue_handler)
try:
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"CRITICAL: Failed to set up main file logger ({LOG_FILE}): {e}")
logger.setLevel(logging.INFO) # Default level

# --- Data Management Class (Keep as is) ---
class DataManager:
    """Handles connection and data fetching from MetaTrader 5."""
    def __init__(self, config: Dict):
        self.mt5_config = config.get("mt5_details", {})
        self.mt5_path = self.mt5_config.get("path") or None # Explicitly None if empty
        self.is_initialized = False
        self.connection_lock = threading.Lock() # Prevent simultaneous connection attempts
        if not self.mt5_path:
             logger.warning("MT5 path not specified in config. Initialization might rely on system path.")

    def connect_mt5(self) -> bool:
        """Initializes and logs into MetaTrader 5. Returns True on success."""
        with self.connection_lock: # Ensure only one thread tries to connect/disconnect
            if self.is_initialized:
                logger.info("MT5 already initialized and connected.")
                return True

            logger.info("Attempting to initialize MetaTrader 5...")
            init_success = False
            try:
                # Initialize with path if provided
                start_time = time.time()
                if self.mt5_path and os.path.exists(self.mt5_path):
                     init_success = mt5.initialize(path=self.mt5_path)
                else:
                     init_success = mt5.initialize() # Try default path
                duration = time.time() - start_time
                logger.debug(f"mt5.initialize() call took {duration:.2f}s")

                if not init_success:
                    logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}. Check MT5 installation/path.")
                    self.is_initialized = False
                    return False
            except Exception as e:
                 # Catch potential errors like terminal not found, etc.
                 logger.error(f"Exception during MT5 initialization: {e}. Check MT5 path in config ('{self.mt5_path}').")
                 self.is_initialized = False
                 return False

            logger.info(f"MetaTrader 5 version: {mt5.version()} initialized successfully.")

            # --- Login Attempt ---
            login = self.mt5_config.get("login")
            password = self.mt5_config.get("password")
            server = self.mt5_config.get("server")

            if not all([login, password, server]):
                logger.error("MT5 login details (login, password, server) missing in configuration.")
                mt5.shutdown() # Shutdown if details are missing
                self.is_initialized = False
                return False

            logger.info(f"Attempting MT5 login for account {login} on server {server}...")
            start_time = time.time()
            authorized = mt5.login(login, password=password, server=server)
            duration = time.time() - start_time
            logger.debug(f"mt5.login() call took {duration:.2f}s")

            if authorized:
                acc_info = mt5.account_info()
                currency = acc_info.currency if acc_info else "N/A"
                logger.info(f"MT5 Login Successful! Account: {login}, Name: {acc_info.name if acc_info else 'N/A'}, Currency: {currency}")
                self.is_initialized = True
                return True
            else:
                logger.error(f"MT5 Login Failed for account {login} on {server}. Error code: {mt5.last_error()}")
                mt5.shutdown() # Shutdown if login failed
                self.is_initialized = False
                return False

    def disconnect_mt5(self):
        """Shuts down MetaTrader 5 connection if initialized."""
        with self.connection_lock:
            if self.is_initialized:
                logger.info("Disconnecting from MetaTrader 5...")
                mt5.shutdown()
                self.is_initialized = False
                logger.info("MT5 Connection Closed.")
            else:
                 logger.debug("MT5 already disconnected or not initialized.")

    def fetch_data(self, pair: str, timeframe_enum: TimeFrames, count: int = 100) -> Optional[pd.DataFrame]:
        """Fetches historical OHLCV data from MT5 for a given pair and timeframe."""
        if not self.is_initialized:
            logger.warning(f"Cannot fetch data for {pair}: MT5 not connected/initialized.")
            return None

        # Use the ExecutionManager's mapping to get the correct MT5 symbol
        # Need access to execution_manager, pass it to DataManager or access globally (less ideal)
        # For now, assume the pair passed is the MT5 symbol or handle mapping elsewhere before calling fetch.
        mt5_symbol = pair # Assume pair is the MT5 symbol for now

        mt5_timeframe_map = {
            TimeFrames.M15: mt5.TIMEFRAME_M15, TimeFrames.H1: mt5.TIMEFRAME_H1,
            TimeFrames.H4: mt5.TIMEFRAME_H4, TimeFrames.D1: mt5.TIMEFRAME_D1,
        }
        mt5_tf = mt5_timeframe_map.get(timeframe_enum)
        if not mt5_tf:
            logger.error(f"Unsupported TimeFrames enum member: {timeframe_enum}. Cannot map to MT5 timeframe.")
            return None

        logger.debug(f"Fetching {count} bars for {mt5_symbol} on timeframe {timeframe_enum.name} ({mt5_tf})...")
        try:
            symbol_info = self.get_symbol_info(mt5_symbol) # Use helper method
            if symbol_info is None: return None # Error handled in helper

            fetch_count = count + 50
            rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, fetch_count)

            if rates is None:
                logger.error(f"MT5 copy_rates_from_pos returned None for {mt5_symbol}/{timeframe_enum.name}. Error: {mt5.last_error()}")
                return None
            if len(rates) < 20: # Check minimum data length
                logger.warning(f"Fetched only {len(rates)} bars for {mt5_symbol}/{timeframe_enum.name} (requested {fetch_count}). May be insufficient.")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={'tick_volume': 'tick_volume', 'real_volume': 'real_volume', 'spread': 'spread'}, inplace=True)
            required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            if not all(col in df.columns for col in required_cols):
                 logger.error(f"Fetched data for {mt5_symbol}/{timeframe_enum.name} is missing required columns.")
                 return None

            logger.debug(f"Successfully fetched and processed {len(df)} bars for {mt5_symbol}/{timeframe_enum.name}. Returning last {count} bars.")
            return df.iloc[-count:].copy()

        except Exception as e:
            logger.exception(f"Unexpected exception fetching data for {mt5_symbol} on {timeframe_enum.name}: {e}")
            return None

    def get_symbol_info(self, mt5_symbol: str) -> Optional[mt5.SymbolInfo]:
        """Safely gets symbol info from MT5, handling selection if needed."""
        if not self.is_initialized: return None
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            logger.warning(f"Symbol '{mt5_symbol}' not found in MarketWatch. Attempting to select...")
            if mt5.symbol_select(mt5_symbol, True):
                time.sleep(0.5)
                info = mt5.symbol_info(mt5_symbol)
                if info: logger.info(f"Successfully selected symbol '{mt5_symbol}'.")
                else: logger.error(f"Failed to get info for symbol '{mt5_symbol}' even after selecting."); return None
            else:
                logger.error(f"Failed to select symbol '{mt5_symbol}' (Error: {mt5.last_error()})."); return None
        # if not info.visible: logger.warning(f"Symbol '{mt5_symbol}' is available but not visible/tradable in MarketWatch.")
        return info
# --- Main Application Class ---
class TradingApp(ctk.CTk):
    """Main application window for the Trading Signal Bot UI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title(APP_NAME)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(1000, 600)

        # --- Bot State ---
        self.is_running = False
        self.bot_thread: Optional[threading.Thread] = None
        self.last_signals: Dict[str, Tuple[ctk.CTkFrame, ctk.CTkLabel, ctk.CTkLabel, ctk.CTkLabel, ctk.CTkLabel]] = {} # Updated tuple size
        self.signal_widgets: List[ctk.CTkFrame] = []

        # --- Load Config ---
        self.config = self._load_config()
        self._apply_log_level()

        # --- Core Components Initialization ---
        try:
            logger.info("Initializing Data Manager...")
            self.data_manager = DataManager(self.config)

            logger.info("Initializing Signal Generator...")
            self.signal_generator = SignalGenerator() # Assumes models/config are local

            logger.info("Initializing Advanced Analysis...")
            self.advanced_analyzer = AdvancedAnalysis(mt5_instance=None) # MT5 passed later

            # ---> NEW COMPONENT INITIALIZATION <---
            logger.info("Initializing Performance Tracker...")
            tracker_config = self.config.get("performance_tracker", {})
            self.tracker = PerformanceTracker(history_file=tracker_config.get("history_file", "data/trade_history.csv"))

            logger.info("Initializing Profitability Enhancer...")
            self.profit_enhancer = ProfitabilityEnhancer(tracker=self.tracker, mt5_instance=None) # MT5 passed later

            logger.info("Initializing Entry Strategies...")
            self.entry_strategies = AdvancedEntryStrategies(signal_generator=self.signal_generator)

            logger.info("Initializing Execution Manager...")
            exec_config = self.config.get("execution_manager", {})
            self.execution_manager = ExecutionManager(
                mt5_instance=None, # MT5 passed later
                tracker=self.tracker,
                profit_enhancer=self.profit_enhancer,
                config=exec_config
            )
            # Initialize symbol_map for MT5 symbol mapping if it doesn't exist
            if not hasattr(self.execution_manager, 'symbol_map'):
                self.execution_manager.symbol_map = {}
                logger.info("Created symbol mapping dictionary in ExecutionManager")
            # ---> END NEW COMPONENT INITIALIZATION <---

            logger.info("Core components initialized successfully.")
        except Exception as e:
            logger.exception(f"FATAL ERROR: Failed to initialize core components: {e}")
            messagebox.showerror("Initialization Error",
                                 f"Core component initialization failed:\n{e}\n\n"
                                 f"Please check logs and required files. Application will exit.")
            try: self.destroy()
            except Exception: pass
            exit()

        # --- UI Setup ---
        self._create_widgets()
        self._update_config_display()
        # Start the UI update loops
        self.after(UPDATE_INTERVAL_MS, self._process_ui_updates)
        self.after(DASHBOARD_UPDATE_INTERVAL_MS, self._update_dashboard_tabs) # Start dashboard update loop

        # --- Handle Window Closing Gracefully ---
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Auto-Connect MT5 (Optional) ---
        if self.config.get("mt5_details", {}).get("auto_connect_mt5", False):
            logger.info("Auto-connect MT5 enabled. Attempting connection...")
            self.after(1500, self._connect_mt5)
        else:
             self.status_bar.set_status("Ready. Connect to MT5 manually.")


    def _load_config(self) -> Dict:
        """Loads configuration from CONFIG_FILE, returning defaults on failure."""
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                conf = json.load(f)
                logger.info(f"Configuration loaded successfully from {CONFIG_FILE}")
                # --- Add Validation for new sections ---
                conf.setdefault("performance_tracker", {})
                conf.setdefault("profitability_enhancer", {})
                conf.setdefault("execution_manager", ExecutionManager.DEFAULT_CONFIG) # Use default from class
                conf["performance_tracker"].setdefault("history_file", "data/trade_history.csv")
                conf["profitability_enhancer"].setdefault("enable_adaptation_check", True)
                # Merge default execution config with loaded to ensure all keys exist
                conf["execution_manager"] = {**ExecutionManager.DEFAULT_CONFIG, **conf.get("execution_manager", {})}
                return conf
        except FileNotFoundError:
            logger.warning(f"Configuration file '{CONFIG_FILE}' not found. Using default settings.")
            # Return structure with defaults for ALL components
            return {
                "mt5_details": {"login": 0, "password": "YOUR_PASSWORD", "server": "YOUR_SERVER", "path": "", "auto_connect_mt5": False},
                "trading_pairs": ["EURUSD", "GBPUSD", "XAUUSD"],
                "timeframes": ["H1", "H4"],
                "bot_settings": {"loop_sleep_seconds": 60, "summary_interval_minutes": 30, "data_fetch_count": 250, "signal_display_limit": 30, "monitor_interval_seconds": 10, "performance_check_interval_seconds": 300},
                "logging": {"level": "INFO"},
                "performance_tracker": {"history_file": "data/trade_history.csv"},
                "profitability_enhancer": {"enable_adaptation_check": True},
                "execution_manager": ExecutionManager.DEFAULT_CONFIG.copy() # Use default from class
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{CONFIG_FILE}': {e}. Please check the file syntax.")
            messagebox.showerror("Config Error", f"Could not parse {CONFIG_FILE}. Invalid JSON:\n{e}")
            return {}
        except Exception as e:
            logger.exception(f"Unexpected error loading configuration from '{CONFIG_FILE}': {e}")
            return {}

    def _save_config(self):
        """Saves current UI settings back to CONFIG_FILE."""
        logger.debug("Updating configuration dictionary from UI elements...")
        try:
            # --- Get UI values ---
            pairs_str = self.entry_pairs.get()
            tfs_str = self.entry_timeframes.get()

            # --- Update Trading Pairs ---
            pairs_list = [p.strip().upper() for p in pairs_str.split(',') if p.strip()]
            if not pairs_list: messagebox.showwarning("Config Warning", "Trading Pairs list is empty.")
            self.config["trading_pairs"] = pairs_list

            # --- Update Timeframes ---
            tfs_list = [tf.strip().upper() for tf in tfs_str.split(',') if tf.strip()]
            valid_tfs = [tf for tf in tfs_list if tf in TimeFrames.__members__]
            invalid_tfs = [tf for tf in tfs_list if tf not in TimeFrames.__members__]
            if invalid_tfs: messagebox.showwarning("Config Warning", f"Invalid timeframes ignored: {', '.join(invalid_tfs)}")
            if not valid_tfs: messagebox.showwarning("Config Warning", "Timeframes list is empty or only contains invalid entries.")
            self.config["timeframes"] = valid_tfs

            # --- Update Bot Settings (Example - add UI elements for these if needed) ---
            self.config.setdefault("bot_settings", {})
            # self.config["bot_settings"]["loop_sleep_seconds"] = int(self.entry_loop_sleep.get())

            # --- Update Execution Manager Settings (Example - add UI elements if needed) ---
            self.config.setdefault("execution_manager", ExecutionManager.DEFAULT_CONFIG.copy())
            # self.config["execution_manager"]["risk_per_trade_pct"] = float(self.entry_risk_pct.get())
            # self.config["execution_manager"]["max_concurrent_trades"] = int(self.entry_max_concurrent.get())

            # --- Save ---
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration successfully saved to {CONFIG_FILE}")
            self.status_bar.set_status("Configuration Saved.", duration=5000)

        except ValueError as ve:
             logger.error(f"Configuration save error: Invalid value entered. {ve}")
             messagebox.showerror("Config Error", f"Invalid value entered. Please check numeric fields.\n{ve}")
        except Exception as e:
            logger.exception(f"Error saving configuration to '{CONFIG_FILE}': {e}")
            messagebox.showerror("Config Error", f"Could not save configuration to {CONFIG_FILE}:\n{e}")

    def _apply_log_level(self):
        """Sets the root logger's level based on config file setting."""
        try:
            log_level_str = self.config.get("logging", {}).get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            logger.setLevel(log_level) # Set root logger level
            for handler in logger.handlers: handler.setLevel(log_level)
            logger.info(f"Global logging level set to: {log_level_str}")
        except Exception as e:
            logger.error(f"Error applying log level from config: {e}. Using default INFO.")
            logger.setLevel(logging.INFO)

    def _update_config_display(self):
        """Populates UI input fields from the loaded config dictionary."""
        logger.debug("Populating UI configuration fields...")
        self.entry_pairs.delete(0, tk.END)
        self.entry_pairs.insert(0, ",".join(self.config.get("trading_pairs", [])))
        self.entry_timeframes.delete(0, tk.END)
        self.entry_timeframes.insert(0, ",".join(self.config.get("timeframes", [])))
        # Update other UI fields linked to config here...

    def _create_widgets(self):
        """Creates and arranges all UI elements."""
        # Main layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Frame (Keep mostly as is)
        self.left_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.left_frame.grid(row=0, column=0, rowspan=2, sticky="nsw")
        self.left_frame.grid_propagate(False)
        self.left_frame.grid_rowconfigure(20, weight=1) # Push content up

        current_row = 0
        # Title
        self.label_title = ctk.CTkLabel(self.left_frame, text=APP_NAME, font=ctk.CTkFont(size=20, weight="bold"))
        self.label_title.grid(row=current_row, column=0, columnspan=2, padx=20, pady=(20, 15), sticky="ew")
        current_row += 1
        # Separator
        ctk.CTkFrame(self.left_frame, height=2, fg_color="gray50").grid(row=current_row, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        current_row += 1
        # MT5 Controls
        ctk.CTkLabel(self.left_frame, text="MetaTrader 5", font=ctk.CTkFont(weight="bold")).grid(row=current_row, column=0, columnspan=2, padx=20, pady=(0, 5), sticky="w")
        current_row += 1
        self.mt5_status_label = ctk.CTkLabel(self.left_frame, text="Status: Unknown", text_color="gray")
        self.mt5_status_label.grid(row=current_row, column=0, padx=(20, 5), pady=5, sticky="w")
        self.connect_mt5_button = ctk.CTkButton(self.left_frame, text="Connect", width=90, command=self._connect_mt5)
        self.connect_mt5_button.grid(row=current_row, column=1, padx=(5, 20), pady=5, sticky="e")
        current_row += 1
        # Bot Controls
        ctk.CTkLabel(self.left_frame, text="Bot Control", font=ctk.CTkFont(weight="bold")).grid(row=current_row, column=0, columnspan=2, padx=20, pady=(15, 5), sticky="w")
        current_row += 1
        self.bot_status_label = ctk.CTkLabel(self.left_frame, text="Status: Stopped", text_color="gray")
        self.bot_status_label.grid(row=current_row, column=0, padx=(20, 5), pady=10, sticky="w")
        button_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        button_frame.grid(row=current_row, column=1, padx=(5, 20), pady=10, sticky="e")
        self.start_button = ctk.CTkButton(button_frame, text="Start", width=60, command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_button = ctk.CTkButton(button_frame, text="Stop", width=60, command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        current_row += 1
        # Configuration Section
        ctk.CTkLabel(self.left_frame, text="Configuration", font=ctk.CTkFont(weight="bold")).grid(row=current_row, column=0, columnspan=2, padx=20, pady=(20, 5), sticky="w")
        current_row += 1
        ctk.CTkLabel(self.left_frame, text="Trading Pairs (comma-separated):").grid(row=current_row, column=0, columnspan=2, padx=20, pady=(5,0), sticky="w")
        current_row += 1
        self.entry_pairs = ctk.CTkEntry(self.left_frame, placeholder_text="e.g., EURUSD,XAUUSD")
        self.entry_pairs.grid(row=current_row, column=0, columnspan=2, padx=20, pady=(0,5), sticky="ew")
        current_row += 1
        ctk.CTkLabel(self.left_frame, text="Timeframes (comma-separated):").grid(row=current_row, column=0, columnspan=2, padx=20, pady=(5,0), sticky="w")
        current_row += 1
        self.entry_timeframes = ctk.CTkEntry(self.left_frame, placeholder_text="e.g., H1,H4")
        self.entry_timeframes.grid(row=current_row, column=0, columnspan=2, padx=20, pady=(0,10), sticky="ew")
        current_row += 1
        self.save_config_button = ctk.CTkButton(self.left_frame, text="Save Configuration", command=self._save_config)
        self.save_config_button.grid(row=current_row, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        current_row += 1
        # Actions Section
        ctk.CTkLabel(self.left_frame, text="Display Actions", font=ctk.CTkFont(weight="bold")).grid(row=current_row, column=0, columnspan=2, padx=20, pady=(20, 5), sticky="w")
        current_row += 1
        self.clear_signals_button = ctk.CTkButton(self.left_frame, text="Clear Signals Display", command=self._clear_signals_display)
        self.clear_signals_button.grid(row=current_row, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        current_row += 1
        self.clear_logs_button = ctk.CTkButton(self.left_frame, text="Clear Logs Display", command=self._clear_logs_display)
        self.clear_logs_button.grid(row=current_row, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        current_row += 1

        # ---> UPDATED RIGHT FRAME (Tabs) <---
        self.tab_view = ctk.CTkTabview(self, corner_radius=8)
        self.tab_view.grid(row=0, column=1, padx=(10, 20), pady=(20, 10), sticky="nsew")

        self.tab_view.add("Signals")
        self.tab_view.add("Open Positions") # New Tab
        self.tab_view.add("Performance")   # New Tab
        self.tab_view.add("Market Summary")
        self.tab_view.add("Logs")
        self.tab_view.set("Signals") # Default tab

        # Signals Tab (Keep as is)
        self.signals_scroll_frame = ctk.CTkScrollableFrame(self.tab_view.tab("Signals"), label_text="Generated Signals")
        self.signals_scroll_frame.pack(expand=True, fill="both", padx=5, pady=5)
        self.signals_scroll_frame.grid_columnconfigure(0, weight=1)
        self.signals_placeholder = ctk.CTkLabel(self.signals_scroll_frame, text="Waiting for signals...", text_color="gray60")
        self.signals_placeholder.pack(pady=20)

        # Open Positions Tab
        self.positions_textbox = ctk.CTkTextbox(self.tab_view.tab("Open Positions"), wrap=tk.NONE, corner_radius=6, font=("Consolas", 10)) # Monospaced, no wrap
        self.positions_textbox.pack(expand=True, fill="both", padx=5, pady=5)
        self.positions_textbox.insert("1.0", "Fetching open positions...\n")
        self.positions_textbox.configure(state=tk.DISABLED) # Read-only

        # Performance Tab
        self.performance_frame = ctk.CTkFrame(self.tab_view.tab("Performance"))
        self.performance_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add controls frame at the top of Performance tab
        self.perf_controls_frame = ctk.CTkFrame(self.performance_frame)
        self.perf_controls_frame.pack(fill="x", padx=5, pady=5)
        
        # Add Import Historical Trades button to Performance tab
        self.import_history_button = ctk.CTkButton(
            self.perf_controls_frame,
            text="📥 Import Historical Trades",
            command=self._import_historical_trades,
            fg_color="#2a52be",
            hover_color="#1e3c8c",
            width=200
        )
        self.import_history_button.pack(side="left", padx=5, pady=5)
        
        # Add Export Trade History button to Performance tab
        self.export_history_button = ctk.CTkButton(
            self.perf_controls_frame,
            text="📤 Export Trade History",
            command=self._export_trade_history,
            fg_color="#2a52be",
            hover_color="#1e3c8c",
            width=200
        )
        self.export_history_button.pack(side="left", padx=5, pady=5)
        
        # Create performance content frame below controls
        self.perf_content_frame = ctk.CTkFrame(self.performance_frame)
        self.perf_content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.performance_textbox = ctk.CTkTextbox(self.perf_content_frame, wrap=tk.WORD, corner_radius=6, font=("Consolas", 11))
        self.performance_textbox.pack(expand=True, fill="both", padx=5, pady=5)
        self.performance_textbox.insert("1.0", "Fetching performance metrics...\n")
        self.performance_textbox.configure(state=tk.DISABLED) # Read-only

        # Market Summary Tab (Keep as is)
        self.summary_textbox = ctk.CTkTextbox(self.tab_view.tab("Market Summary"), wrap=tk.WORD, corner_radius=6, font=("Consolas", 11))
        self.summary_textbox.pack(expand=True, fill="both", padx=5, pady=5)
        self.summary_textbox.insert("1.0", "Market summary will update periodically while bot is running...\n")
        self.summary_textbox.configure(state=tk.DISABLED)

        # Log Tab (Keep as is)
        self.log_textbox = ctk.CTkTextbox(self.tab_view.tab("Logs"), wrap=tk.WORD, corner_radius=6, font=("Consolas", 10))
        self.log_textbox.pack(expand=True, fill="both", padx=5, pady=5)
        self.log_textbox.configure(state=tk.DISABLED)

        # Status Bar (Keep as is)
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="sew", padx=5, pady=(0,5))
        # ---> END UPDATED RIGHT FRAME <---


    # --- MT5 Connection Handling ---
    def _connect_mt5(self):
        """Initiates MT5 connection via DataManager in a background thread."""
        self.status_bar.set_status("Connecting to MT5...", priority=True)
        self.connect_mt5_button.configure(state=tk.DISABLED)
        threading.Thread(target=self._mt5_connection_worker, daemon=True).start()

    def _mt5_connection_worker(self):
        """Worker thread function for establishing MT5 connection."""
        connected = self.data_manager.connect_mt5()
        self.after(0, self._update_mt5_ui, connected)

    def _update_mt5_ui(self, connected: bool):
        """Updates UI elements and passes MT5 instance to components."""
        if connected:
            self.mt5_status_label.configure(text="Status: Connected", text_color="lightgreen")
            self.status_bar.set_status("MT5 Connected.", duration=10000, priority=False)
            self.connect_mt5_button.configure(text="Disconnect", command=self._disconnect_mt5)
            # ---> PASS MT5 INSTANCE & INITIALIZE LOCK <---
            if hasattr(self, 'advanced_analyzer'):
                self.advanced_analyzer.mt5 = mt5
                logger.info("Passed valid MT5 instance to AdvancedAnalysis.")
            if hasattr(self, 'profit_enhancer'):
                self.profit_enhancer.mt5 = mt5
                logger.info("Passed valid MT5 instance to ProfitabilityEnhancer.")
            if hasattr(self, 'execution_manager'):
                self.execution_manager.mt5 = mt5
                # Initialize lock if it wasn't created during __init__
                if self.execution_manager.order_lock is None and hasattr(mt5, 'lock'):
                    try:
                        self.execution_manager.order_lock = mt5.lock()
                        logger.info("Initialized order lock for ExecutionManager.")
                    except Exception as lock_e:
                         logger.error(f"Failed to initialize order lock for ExecutionManager: {lock_e}")
                elif self.execution_manager.order_lock:
                     logger.debug("Order lock for ExecutionManager already initialized.")
                logger.info("Passed valid MT5 instance to ExecutionManager.")
            # ---> END PASS MT5 INSTANCE & LOCK INIT <---
        else:
            self.mt5_status_label.configure(text="Status: Failed", text_color="#FF5050")
            self.status_bar.set_status("MT5 Connection Failed.", alert=True, duration=15000)
            self.connect_mt5_button.configure(text="Connect", command=self._connect_mt5)
            # ---> SET MT5 INSTANCE TO NONE & RESET LOCK <---
            if hasattr(self, 'advanced_analyzer'): self.advanced_analyzer.mt5 = None
            if hasattr(self, 'profit_enhancer'): self.profit_enhancer.mt5 = None
            if hasattr(self, 'execution_manager'):
                self.execution_manager.mt5 = None
                self.execution_manager.order_lock = None # Reset lock
            logger.warning("Set MT5 instance in components to None and reset lock due to connection failure/disconnect.")
            # ---> END SET MT5 INSTANCE TO NONE <---

        self.connect_mt5_button.configure(state=tk.NORMAL)

    def _disconnect_mt5(self):
        """Initiates MT5 disconnection and updates components."""
        # ---> ADDED: Prevent disconnect if running? Or handled by on_closing?
        # if self.is_running:
        #    if not messagebox.askyesno("Confirm Disconnect", "The bot is running. Disconnecting MT5 will stop it. Proceed?"):
        #        return
        #    self.stop_bot() # Stop bot first if disconnecting while running
        #    # Need to wait for bot to fully stop before proceeding? Add check.
        # ---

        self.status_bar.set_status("Disconnecting MT5...", priority=True)
        self.data_manager.disconnect_mt5() # Disconnect
        self.after(0, self._update_mt5_ui, False) # Update UI and set component mt5 to None
        self.status_bar.set_status("MT5 Disconnected.", duration=5000)

    # --- UI Update Processing ---
    def _process_ui_updates(self):
        """Periodically processes updates from queues (like logs) for the UI."""
        # Process Log Queue
        log_count = 0
        max_logs_per_cycle = 150
        try:
            while not log_queue.empty() and log_count < max_logs_per_cycle:
                message = log_queue.get_nowait()
                self.log_textbox.configure(state=tk.NORMAL)
                self.log_textbox.insert(tk.END, message + "\n")
                self.log_textbox.configure(state=tk.DISABLED)
                log_count += 1
            if log_count > 0: self.log_textbox.yview(tk.END)
        except queue.Empty: pass
        except Exception as e: print(f"Error processing log queue for UI: {e}")

        # Reschedule the next UI update check
        self.after(UPDATE_INTERVAL_MS, self._process_ui_updates)

    # ---> NEW DASHBOARD UPDATE FUNCTION <---
    def _update_dashboard_tabs(self):
        """Periodically updates the Open Positions and Performance tabs."""
        logger.debug("Updating dashboard tabs (Positions, Performance)...")
        try:
            # Update Open Positions Tab
            self._update_positions_tab()

            # Update Performance Tab
            if hasattr(self, 'tracker') and self.tracker:
                metrics = self.tracker.get_performance_metrics() # Get overall metrics
                perf_text = f"--- Performance Metrics ({datetime.now().strftime('%H:%M:%S')}) ---\n\n"
                
                # Update the performance textbox with metrics
                if hasattr(self, 'performance_textbox'):
                    # Format metrics for display
                    perf_text += f"Total Trades: {metrics.get('total_trades', 0)}\n"
                    perf_text += f"Win Rate: {metrics.get('win_rate', 0.0):.2f}%\n"
                    perf_text += f"Profit Factor: {metrics.get('profit_factor', 0.0):.2f}\n"
                    perf_text += f"Total Profit: {metrics.get('total_profit_currency', 0.0):.2f}\n"
                    perf_text += f"Average Win: {metrics.get('average_win_currency', 0.0):.2f}\n"
                    perf_text += f"Average Loss: {metrics.get('average_loss_currency', 0.0):.2f}\n"
                    perf_text += f"Max Drawdown: {metrics.get('max_drawdown_currency', 0.0):.2f}\n"
                    
                    # Display trade breakdown by exit reason if available
                    if 'exit_reason_breakdown' in metrics:
                        perf_text += "\n--- Trade Breakdown by Exit Reason ---\n"
                        for reason, stats in metrics.get('exit_reason_breakdown', {}).items():
                            count = stats.get('count', 0)
                            if count > 0:
                                win_rate = stats.get('win_rate', 0.0)
                                profit = stats.get('total_profit', 0.0)
                                perf_text += f"{reason}: {count} trades, {win_rate:.2f}% win rate, {profit:.2f} profit\n"
                    
                    # Update the textbox
                self.performance_textbox.configure(state=tk.NORMAL)
                self.performance_textbox.delete("1.0", tk.END)
                self.performance_textbox.insert("1.0", perf_text)
                self.performance_textbox.configure(state=tk.DISABLED)

        except Exception as e:
             logger.exception(f"Error updating dashboard tabs: {e}")
        
        # Reschedule the next dashboard update
        self.after(DASHBOARD_UPDATE_INTERVAL_MS, self._update_dashboard_tabs)
    # ---> END NEW DASHBOARD UPDATE FUNCTION <---


    # --- Signal Display Handling (Keep mostly as is) ---
    def _update_entry_time(self, signal_key, signal_data):
        """Updates the entry time for a signal with 0.1-second precision"""
        if not signal_data.get('entry_time_thread_active', False):
            return
            
        # Update entry time with current time
        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-4]  # Format with 0.1s precision
        signal_data['entry_time'] = current_time
        
        # Update the display if the signal is still in the last_signals dictionary
        if signal_key in self.last_signals:
            if len(self.last_signals[signal_key]) >= 6:  # Make sure we have enough elements
                frame, label_pair_tf, label_line1, label_line2, label_line3, label_line4, *rest = self.last_signals[signal_key]
                
                # Update line3 with new entry time
                entry = f"{signal_data.get('entry_price', 0.0):.5f}"
                sl = f"{signal_data.get('stop_loss', 0.0):.5f}"
                tp = f"{signal_data.get('take_profit', 0.0):.5f}"
                line3_text = f"Entry: {entry} | Entry Time: {current_time} | SL: {sl} | TP: {tp}"
                label_line3.configure(text=line3_text)
        
        # Schedule the next update in 0.1 seconds if still active
        if signal_data.get('entry_time_thread_active', False):
            self.after(100, lambda: self._update_entry_time(signal_key, signal_data))

    def _update_signals_display(self, signal_key: str, signal_data: Dict):
        """Adds or updates a signal frame in the Signals scrollable frame."""
        if hasattr(self, 'signals_placeholder') and self.signals_placeholder.winfo_exists():
            self.signals_placeholder.pack_forget()
            del self.signals_placeholder
        details = signal_data.get('details', {})
        # Example formatting for line 5:
        details_str = f"RSI:{details.get('rsi','?')} | MACD_H:{details.get('macd_h','?')} | ADX:{details.get('adx','?')} | ATR%:{details.get('atr%','?')}"
        direction = signal_data.get('direction', 'HOLD')
        recommendation = signal_data.get('advanced_analysis', {}).get('recommendation', direction)
        if "STRONG_BUY" in recommendation: bg_color = "#006400"
        elif "BUY" in recommendation: bg_color = "#008000"
        elif "WEAK_BUY" in recommendation: bg_color = "#558B2F"
        elif "STRONG_SELL" in recommendation: bg_color = "#8B0000"
        elif "SELL" in recommendation: bg_color = "#B22222"
        elif "WEAK_SELL" in recommendation: bg_color = "#CD5C5C"
        elif "AVOID" in recommendation or "REVERSE" in recommendation: bg_color = "#FFA500"
        else: bg_color = "gray25"

        pair_tf_text = f"{signal_data.get('pair', '?')} ({signal_data.get('timeframe', '?')}m)"
        timestamp = signal_data.get('timestamp', datetime.now(pytz.utc).isoformat())
        try:
            ts_dt = datetime.fromisoformat(timestamp).astimezone(self.signal_generator.timezone)
            ts_str = ts_dt.strftime('%H:%M:%S')
        except: ts_str = timestamp.split('T')[-1].split('.')[0]

        # Add entry time tracking (initialize with empty value)
        if 'entry_time' not in signal_data:
            signal_data['entry_time'] = ""
            signal_data['entry_time_thread_active'] = False

        entry = f"{signal_data.get('entry_price', 0.0):.5f}"
        sl = f"{signal_data.get('stop_loss', 0.0):.5f}"
        tp = f"{signal_data.get('take_profit', 0.0):.5f}"
        conf = signal_data.get('confidence', 0.0)
        tech_score = signal_data.get('technical_score', 0.0)
        ml_conf = signal_data.get('ml_confidence') # Can be None
        ml_conf_str = f"{ml_conf:.2f}" if ml_conf is not None else "N/A"
        adv_analysis = signal_data.get('advanced_analysis', {})
        adv_strength = adv_analysis.get('signal_strength', 'N/A')
        size_factor = signal_data.get('position_size_factor', 1.0)

        # Get entry strategy information with better defaults
        best_strategy = signal_data.get('entry_recommendation', {}).get('best_strategy', 'No Strategy')
        action = signal_data.get('action', 'WAIT')
        if action is None or action == 'N/A':
            action = 'WAIT'  # Replace None or N/A with WAIT
        action_confidence = signal_data.get('entry_recommendation', {}).get('confidence', 0.0)
        action_confidence_str = f"{action_confidence:.2f}" if isinstance(action_confidence, (int, float)) else action_confidence
        risk_level = signal_data.get('entry_recommendation', {}).get('quality', 'UNKNOWN')
        
        # Format the entry information
        entry_info = f"Strategy: {best_strategy} | Action: {action} | Conf: {action_confidence_str} | Risk: {risk_level}"

        line1_text = f"{ts_str} | {recommendation.ljust(12)} | Strength: {adv_strength: >5}"
        line2_text = (f"Conf: {conf:.2f} (T:{tech_score:.2f}, M:{ml_conf_str}) | SizeF: {size_factor:.2f}") # Added Size Factor
        
        # Add entry time to line3 if available
        entry_time_str = signal_data.get('entry_time', "")
        if entry_time_str:
            line3_text = f"Entry: {entry} | Entry Time: {entry_time_str} | SL: {sl} | TP: {tp}"
        else:
            line3_text = f"Entry: {entry} | SL: {sl} | TP: {tp}"
        
        # Add entry strategy information to line4
        line4_text = entry_info
        
        # Add technical details to line5
        line5_text = f"Details: {details_str}"

        # Determine if this signal should get a manual execution button
        show_manual_execution = "STRONG_BUY" not in recommendation and "STRONG_SELL" not in recommendation

        if signal_key in self.last_signals:
            if len(self.last_signals[signal_key]) == 5:  # Old format without button
                frame, label_pair_tf, label_line1, label_line2, label_line3 = self.last_signals[signal_key]
                # Remove old frame and create a new one with the button if needed
                frame.destroy()
                frame = ctk.CTkFrame(self.signals_scroll_frame, fg_color=bg_color, corner_radius=5, border_width=1, border_color="gray40")
                frame.pack(fill="x", padx=5, pady=(3, 0))
                frame.grid_columnconfigure(0, weight=1)
                
                label_pair_tf = ctk.CTkLabel(frame, text=pair_tf_text, font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
                label_pair_tf.grid(row=0, column=0, padx=8, pady=(4, 0), sticky="ew")
                label_line1 = ctk.CTkLabel(frame, text=line1_text, anchor="w", justify="left", font=ctk.CTkFont(size=11))
                label_line1.grid(row=1, column=0, padx=8, pady=(0, 0), sticky="ew")
                label_line2 = ctk.CTkLabel(frame, text=line2_text, anchor="w", justify="left", font=ctk.CTkFont(size=11))
                label_line2.grid(row=2, column=0, padx=8, pady=(0, 0), sticky="ew")
                label_line3 = ctk.CTkLabel(frame, text=line3_text, anchor="w", justify="left", font=ctk.CTkFont(size=11))
                label_line3.grid(row=3, column=0, padx=8, pady=(0, 0), sticky="ew")
                label_line4 = ctk.CTkLabel(frame, text=line4_text, anchor="w", justify="left", font=ctk.CTkFont(size=11))
                label_line4.grid(row=4, column=0, padx=8, pady=(0, 0), sticky="ew")
                label_line5 = ctk.CTkLabel(frame, text=line5_text, anchor="w", justify="left", font=ctk.CTkFont(size=10), text_color="gray80")
                label_line5.grid(row=5, column=0, padx=8, pady=(0, 4), sticky="ew")
                
                # Add execute button if needed
                if show_manual_execution:
                    exec_button = ctk.CTkButton(
                        frame, 
                        text="Execute Trade", 
                        width=110, 
                        height=24,
                        font=ctk.CTkFont(size=10),
                        command=lambda s=signal_data: self._manual_execute_trade(s)
                    )
                    exec_button.grid(row=6, column=0, padx=8, pady=(0, 6), sticky="e")
                    self.last_signals[signal_key] = (frame, label_pair_tf, label_line1, label_line2, label_line3, label_line4, label_line5, exec_button)
                else:
                    self.last_signals[signal_key] = (frame, label_pair_tf, label_line1, label_line2, label_line3, label_line4, label_line5)
            else:  # Updated format with or without button
                frame, label_pair_tf, label_line1, label_line2, label_line3, label_line4, *rest = self.last_signals[signal_key]
                
                # Check if we have label_line5 in the tuple
                if len(rest) > 0 and isinstance(rest[0], ctk.CTkLabel):
                    label_line5 = rest[0]
                    rest = rest[1:]
                else:
                    # Create label_line5 if it doesn't exist
                    label_line5 = ctk.CTkLabel(frame, text=line5_text, anchor="w", justify="left", font=ctk.CTkFont(size=10), text_color="gray80")
                    label_line5.grid(row=5, column=0, padx=8, pady=(0, 4), sticky="ew")
                
                frame.configure(fg_color=bg_color)
                label_pair_tf.configure(text=pair_tf_text)
                label_line1.configure(text=line1_text)
                label_line2.configure(text=line2_text)
                label_line3.configure(text=line3_text)
                label_line4.configure(text=line4_text)
                label_line5.configure(text=line5_text)
                    
                # Handle button if it exists
                if rest and show_manual_execution:
                    # Button already exists, update command with new signal data
                    rest[0].configure(command=lambda s=signal_data: self._manual_execute_trade(s))
                elif rest and not show_manual_execution:
                    # Button exists but should be removed
                    rest[0].destroy()
                    self.last_signals[signal_key] = (frame, label_pair_tf, label_line1, label_line2, label_line3, label_line4, label_line5)
                elif not rest and show_manual_execution:
                    # Need to add a button
                    exec_button = ctk.CTkButton(
                        frame, 
                        text="Execute Trade", 
                        width=110, 
                        height=24,
                        font=ctk.CTkFont(size=10),
                        command=lambda s=signal_data: self._manual_execute_trade(s)
                    )
                    exec_button.grid(row=6, column=0, padx=8, pady=(0, 6), sticky="e")
                    self.last_signals[signal_key] = (frame, label_pair_tf, label_line1, label_line2, label_line3, label_line4, label_line5, exec_button)
                
                frame.pack_forget()
                frame.pack(fill="x", padx=5, pady=(3, 0))
        else:
            frame = ctk.CTkFrame(self.signals_scroll_frame, fg_color=bg_color, corner_radius=5, border_width=1, border_color="gray40")
            frame.pack(fill="x", padx=5, pady=(3, 0))
            frame.grid_columnconfigure(0, weight=1)
            
            label_pair_tf = ctk.CTkLabel(frame, text=pair_tf_text, font=ctk.CTkFont(size=14, weight="bold"), anchor="w")
            label_pair_tf.grid(row=0, column=0, padx=8, pady=(4, 0), sticky="ew")
            
            label_line1 = ctk.CTkLabel(frame, text=line1_text, anchor="w", justify="left", font=ctk.CTkFont(size=11))
            label_line1.grid(row=1, column=0, padx=8, pady=(0, 0), sticky="ew")
            
            label_line2 = ctk.CTkLabel(frame, text=line2_text, anchor="w", justify="left", font=ctk.CTkFont(size=11))
            label_line2.grid(row=2, column=0, padx=8, pady=(0, 0), sticky="ew")
            
            label_line3 = ctk.CTkLabel(frame, text=line3_text, anchor="w", justify="left", font=ctk.CTkFont(size=11))
            label_line3.grid(row=3, column=0, padx=8, pady=(0, 0), sticky="ew")
            
            label_line4 = ctk.CTkLabel(frame, text=line4_text, anchor="w", justify="left", font=ctk.CTkFont(size=11))
            label_line4.grid(row=4, column=0, padx=8, pady=(0, 0), sticky="ew")
            
            label_line5 = ctk.CTkLabel(frame, text=line5_text, anchor="w", justify="left", font=ctk.CTkFont(size=10), text_color="gray80")
            label_line5.grid(row=5, column=0, padx=8, pady=(0, 4), sticky="ew")
            
            if show_manual_execution:
                exec_button = ctk.CTkButton(
                    frame, 
                    text="Execute Trade", 
                    width=110, 
                    height=24,
                    font=ctk.CTkFont(size=10),
                    command=lambda s=signal_data: self._manual_execute_trade(s)
                )
                exec_button.grid(row=6, column=0, padx=8, pady=(0, 6), sticky="e")
                self.last_signals[signal_key] = (frame, label_pair_tf, label_line1, label_line2, label_line3, label_line4, label_line5, exec_button)
            else:
                self.last_signals[signal_key] = (frame, label_pair_tf, label_line1, label_line2, label_line3, label_line4, label_line5)

            self.signal_widgets.append(frame)

            limit = self.config.get("bot_settings", {}).get("signal_display_limit", 30)
            if len(self.signal_widgets) > limit:
                widget_to_remove = self.signal_widgets.pop(0)
                key_to_remove = None
                for key, (frm, *_) in self.last_signals.items():
                     if frm == widget_to_remove: key_to_remove = key; break
                if key_to_remove: del self.last_signals[key_to_remove]
                widget_to_remove.destroy()

    def _manual_execute_trade(self, signal_data: Dict):
        """Handle manual execution of a trade from the signal UI"""
        if not self.data_manager.is_initialized:
            messagebox.showerror("MT5 Error", "Cannot execute trade.\nPlease connect to MetaTrader 5 first.")
            return
            
        if not self.execution_manager:
            messagebox.showerror("Execution Error", "Execution Manager is not available.")
            return
            
        try:
            pair = signal_data.get('pair', 'Unknown')
            direction = signal_data.get('direction', 'Unknown')
            
            # Verify symbol is available in MT5
            symbol_info = mt5.symbol_info(pair)
            if symbol_info is None:
                # Try to add symbol prefix/suffix if applicable (based on broker)
                modified_pairs = []
                
                # For crypto pairs like BTCUSD
                if "BTC" in pair or "ETH" in pair or "XRP" in pair:
                    # Common crypto symbol formats
                    modified_pairs = [
                        f"{pair}m",           # BTCUSDm 
                        f"{pair}.m",          # BTCUSD.m
                        f"{pair}-m",          # BTCUSD-m
                        pair.replace("USD", ""),  # BTC (some brokers)
                        f"{pair.replace('USD', '')}/USD",  # BTC/USD
                        f"XBT{pair[3:]}",     # XBTUSD (if using BTC)
                        f"{pair}:m",          # BTCUSD:m
                        f"{pair}.a",          # BTCUSD.a (some brokers)
                        f"{pair}cash",        # BTCUSDcash
                        pair                  # Try the original again
                    ]
                else:
                    # For forex/other pairs
                    modified_pairs = [
                        f"{pair}m",           # EURUSDm
                        f"{pair}.m",          # EURUSD.m
                        f"{pair}-m",          # EURUSD-m
                        f"{pair}.a",          # EURUSD.a
                        pair.replace("USD", ""), # For gold like XAUUSD -> XAU
                        pair                  # Try the original again
                    ]
                
                found_symbol = None
                
                for mod_pair in modified_pairs:
                    if mt5.symbol_info(mod_pair) is not None:
                        found_symbol = mod_pair
                        logger.info(f"Found modified symbol: {mod_pair} for {pair}")
                        break
                
                if found_symbol:
                    confirm_symbol = messagebox.askyesno(
                        "Symbol Modified", 
                        f"Original symbol '{pair}' not found in MT5.\nUse '{found_symbol}' instead?"
                    )
                    if confirm_symbol:
                        pair = found_symbol
                        signal_data['pair'] = found_symbol
                    else:
                        messagebox.showinfo("Execution Cancelled", "Trade execution cancelled.")
                        return
                else:
                    messagebox.showerror("Symbol Error", f"Symbol '{pair}' not found in MT5 and no alternative found.\nPlease check if the symbol is available in your MT5 terminal.")
                    return
            
            # Ensure symbol is selected in Market Watch
            if not mt5.symbol_select(pair, True):
                logger.warning(f"Failed to select symbol {pair} in Market Watch. Error: {mt5.last_error()}")
                messagebox.showwarning("Symbol Selection", f"Failed to select {pair} in Market Watch. The trade may fail.")
            
            confirm = messagebox.askyesno(
                "Confirm Manual Execution", 
                f"Execute {direction} trade for {pair}?\n\n" +
                f"Entry: {signal_data.get('entry_price', 0.0):.5f}\n" +
                f"Stop Loss: {signal_data.get('stop_loss', 0.0):.5f}\n" +
                f"Take Profit: {signal_data.get('take_profit', 0.0):.5f}"
            )
            
            if confirm:
                # Start entry time tracking for this signal
                signal_key = f"{signal_data.get('pair')}_{signal_data.get('timeframe')}"
                signal_data['entry_time'] = datetime.now().strftime('%H:%M:%S.%f')[:-4]  # Initial value with 0.1s precision
                signal_data['entry_time_thread_active'] = True
                
                # Start the entry time update thread
                self.after(100, lambda: self._update_entry_time(signal_key, signal_data))
                
                logger.info(f"Manual execution requested for {pair} {direction}")
                self.status_bar.set_status(f"Executing manual trade: {pair} {direction}...", priority=True)
                
                # Skip ExecutionManager and place order directly via MT5
                logger.debug(f"Placing order directly through MT5 for {pair} {direction}")
                    
                try:
                    # Get market info for proper order parameters
                    symbol_info = mt5.symbol_info(pair)
                    if symbol_info is None:
                        messagebox.showerror("Symbol Error", f"Failed to get symbol info for {pair}.")
                        return
                        
                    # Get current price
                    price = mt5.symbol_info_tick(pair).ask if direction == "BUY" else mt5.symbol_info_tick(pair).bid
                    
                    # Default volume if not specified
                    volume = float(signal_data.get("volume", 0.01))
                    
                    # Get SL/TP from signal or calculate based on ATR if available
                    sl = signal_data.get("stop_loss", 0.0)
                    tp = signal_data.get("take_profit", 0.0)
                    
                    # Basic order parameters
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": pair,
                        "volume": volume,
                        "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
                        "price": price,
                        "deviation": 20,
                        "magic": 234000,  # Use a specific magic number for manual trades for filtering
                        "comment": "ANOXILAL Manual",
                        "type_time": mt5.ORDER_TIME_GTC,
                    }
                    
                    # Add SL/TP if provided
                    if sl > 0:
                        request["sl"] = sl
                    if tp > 0:
                        request["tp"] = tp
                        
                    # Try to determine proper filling type
                    filling_modes = mt5.symbol_info(pair).filling_mode
                    if filling_modes == 1:  # FOK
                        request["type_filling"] = mt5.ORDER_FILLING_FOK
                    elif filling_modes == 2:  # IOC
                        request["type_filling"] = mt5.ORDER_FILLING_IOC
                    else:  # Try return or IOC as fallback
                        try:
                            request["type_filling"] = mt5.ORDER_FILLING_RETURN
                        except Exception:
                            request["type_filling"] = mt5.ORDER_FILLING_IOC
                    
                    # Better filling mode detection - fix for error 10030
                    logger.debug(f"Symbol {pair} filling mode value: {mt5.symbol_info(pair).filling_mode}")
                    
                    # First attempt: don't specify filling type at all (let MT5 choose default)
                    if "type_filling" in request:
                        del request["type_filling"]
                    
                    # Add more detailed debugging for filling modes
                    # Better filling mode detection - fix for error 10030
                    filling_mode_val = mt5.symbol_info(pair).filling_mode
                    logger.debug(f"Symbol {pair} filling mode value: {filling_mode_val}")
                    
                    # Log more detailed information about possible filling modes
                    logger.debug(f"Available filling modes for {pair}:")
                    if filling_mode_val & mt5.ORDER_FILLING_FOK:
                        logger.debug(f" - FOK (Fill or Kill) is supported")
                    if filling_mode_val & mt5.ORDER_FILLING_IOC:
                        logger.debug(f" - IOC (Immediate or Cancel) is supported")
                    
                    # First attempt: don't specify filling type at all (let MT5 choose default)
                    if "type_filling" in request:
                        logger.debug(f"Removing explicit type_filling parameter (was: {request['type_filling']})")
                        del request["type_filling"]
                    
                    logger.debug("Letting MT5 automatically select the appropriate filling mode")
                    
                    # Send order
                    logger.debug(f"Sending direct MT5 order: {request}")
                    result = mt5.order_send(request)
                    
                    # Handle filling mode errors with retry logic
                    if result and result.retcode == 10030:  # TRADE_RETCODE_INVALID_FILL
                        logger.warning(f"Received unsupported filling mode error. Retrying with explicit filling modes.")
                        
                        # Try with FOK filling
                        request["type_filling"] = mt5.ORDER_FILLING_FOK
                        logger.debug(f"Retrying with ORDER_FILLING_FOK")
                        result = mt5.order_send(request)
                        
                        # If still failing, try with IOC
                        if result and result.retcode == 10030:
                            request["type_filling"] = mt5.ORDER_FILLING_IOC
                            logger.debug(f"Retrying with ORDER_FILLING_IOC")
                            result = mt5.order_send(request)
                    
                    # Process result
                    if result is None:
                        logger.error(f"MT5 order_send returned None. Error: {mt5.last_error()}")
                        messagebox.showerror("Order Failed", f"MT5 returned no result. Error: {mt5.last_error()}")
                        return
                        
                    logger.info(f"MT5 order result: retcode={result.retcode}, comment={result.comment}")
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        position_ticket = result.order
                        logger.info(f"Direct MT5 order successfully placed: Ticket {position_ticket}")
                        
                        # Force immediate update of the positions tab
                        self._update_positions_tab()
                        
                        # Add the position to ExecutionManager for tracking/closing
                        try:
                            # Get position info for more details
                            position_info = mt5.positions_get(ticket=position_ticket)
                            if position_info and len(position_info) > 0:
                                position = position_info[0]
                                
                                # Make sure ExecutionManager tracks this position for proper recording when closed
                                if self.execution_manager:
                                    entry_time = datetime.now()
                                    identifier = position.identifier if hasattr(position, 'identifier') else position.ticket
                                    
                                    # Track with more detailed information
                                    self.execution_manager.open_positions[position_ticket] = {
                                        'ticket': position_ticket,
                                        'pair': pair,
                                        'mt5_symbol': pair,
                                        'direction': direction,
                                        'entry_time': entry_time,
                                        'entry_price': price,
                                        'sl': sl,
                                        'tp': tp,
                                        'volume': volume,
                                        'identifier': identifier,
                                        'magic': position.magic,
                                        'account_equity_on_entry': mt5.account_info().equity,
                                        'signal_confidence': signal_data.get('confidence', 0.5),
                                        'is_manual': True  # Tag as manual trade for reporting
                                    }
                                    logger.info(f"Added manual trade to execution manager for tracking: Ticket {position_ticket}")
                                    
                                    # Set up a dedicated MT5 position monitor for this manual trade
                                    threading.Thread(
                                        target=self._monitor_manual_trade, 
                                        args=(position_ticket, identifier, pair, direction, price, sl, tp, volume, entry_time),
                                        daemon=True
                                    ).start()
                                    
                                    # Switch to the Positions tab to show the new position
                                    self.dashboard_tab_view.set("Positions")
                                else:
                                    logger.warning("Could not add position to execution manager (not available)")
                                    messagebox.showwarning("Trade Not Tracked", 
                                       "Trade was placed successfully but will not be tracked in performance metrics.\n"
                                       "The ExecutionManager is not available.")
                            else:
                                logger.warning(f"Could not get position info for ticket {position_ticket}")
                        except Exception as e:
                            logger.exception(f"Error tracking manual trade: {e}")
                        
                        # Show success message
                        messagebox.showinfo("Trade Executed", 
                                          f"Successfully placed {direction} order for {pair}\n"
                                          f"Ticket: {position_ticket}\n"
                                          f"Entry: {price:.5f}\n"
                                          f"Volume: {volume}")
                        
                        # Switch to the Positions tab
                        self.tab_view.set("Open Positions")
                    else:
                        error_msg = f"MT5 Error: {result.retcode} - {result.comment}"
                        logger.warning(f"Direct MT5 order failed: {error_msg}")
                        
                        messagebox.showerror("Order Failed", 
                                           f"Failed to place order:\n\n{error_msg}\n\n"
                                           f"Check MT5 terminal for more details.")
                except Exception as mt5_e:
                    logger.exception(f"Error in direct MT5 order placement: {mt5_e}")
                    messagebox.showerror("Direct Order Error", f"Error: {mt5_e}")
            else:
                logger.info(f"Manual execution cancelled for {pair} {direction}")
                self.status_bar.set_status("Manual execution cancelled", duration=5000)
                
        except Exception as e:
            logger.exception(f"Error during manual trade execution: {e}")
            messagebox.showerror("Execution Error", f"An error occurred while executing the trade:\n{e}")
    
    def _monitor_manual_trade(self, ticket: int, identifier: int, pair: str, direction: str, 
                             entry_price: float, sl_price: float, tp_price: float, 
                             volume: float, entry_time: datetime):
        """
        Dedicated monitor for a manual trade to ensure it's properly recorded when closed.
        This runs in a separate thread and checks the position periodically.
        """
        if not hasattr(self, 'execution_manager') or not self.execution_manager:
            logger.warning(f"Cannot monitor manual trade {ticket}: No execution manager")
            return
            
        logger.info(f"Starting dedicated monitor for manual trade {ticket}")
        
        # Set initial state
        is_closed = False
        check_interval = 5  # seconds
        total_attempts = 0
        max_attempts = 2880  # 4 hours of monitoring at 5-second intervals
        
        # Find the signal key for this trade to stop entry time updates later
        signal_key = None
        for key in self.last_signals.keys():
            if key.startswith(pair) and self.last_signals[key][0].winfo_exists():
                signal_key = key
                break
        
        try:
            while not is_closed and total_attempts < max_attempts:
                # Sleep first to allow time for MT5 to process the order
                time.sleep(check_interval)
                total_attempts += 1
                
                # Check if trade is still in our tracking
                if ticket not in self.execution_manager.open_positions:
                    logger.info(f"Manual trade {ticket} no longer in tracking - likely already processed")
                    return
                
                # Check if trade still exists in MT5
                position = None
                try:
                    positions = mt5.positions_get(ticket=ticket)
                    if positions and len(positions) > 0:
                        position = positions[0]
                except Exception as e:
                    logger.warning(f"Error checking manual trade {ticket} status: {e}")
                    continue
                
                # If position not found, it's been closed
                if not position:
                    logger.info(f"Manual trade {ticket} closed - checking history for details")
                    is_closed = True
                    
                    # Stop entry time updates if we found the signal
                    if signal_key and signal_key in self.last_signals:
                        # Get the signal data from the execution manager
                        if ticket in self.execution_manager.open_positions:
                            signal_data = self.execution_manager.open_positions[ticket].get('original_signal', {})
                            if signal_data:
                                signal_data['entry_time_thread_active'] = False
                                logger.debug(f"Stopped entry time updates for signal {signal_key}")
                    
                    # Get trade details from history
                    try:
                        # Get history of deals for this position
                        from_date = entry_time - timedelta(minutes=5)
                        to_date = datetime.now() + timedelta(minutes=5)
                        
                        # Try to get deals by position ID first
                        deals = mt5.history_deals_get(position=identifier)
                        
                        if not deals:
                            # Fallback to time range if position ID didn't work
                            deals = mt5.history_deals_get(from_date, to_date)
                            # Filter by ticket manually
                            deals = [d for d in deals if hasattr(d, 'position_id') and d.position_id == identifier]
                        
                        if not deals:
                            logger.warning(f"No history found for manual trade {ticket} - cannot record closure")
                            # Mark as processed by removing from tracking if not already removed
                            if ticket in self.execution_manager.open_positions:
                                del self.execution_manager.open_positions[ticket]
                            return
                        
                        # Process deals to find entry and exit
                        entry_deal = None
                        exit_deal = None
                        profit_currency = 0.0
                        commission = 0.0
                        swap = 0.0
                        
                        for deal in deals:
                            # Skip unrelated deals
                            if not hasattr(deal, 'position_id') or deal.position_id != identifier:
                                continue
                                
                            if hasattr(deal, 'entry'):
                                # Entry deal
                                if deal.entry == mt5.DEAL_ENTRY_IN:
                                    entry_deal = deal
                                    profit_currency += deal.profit
                                    commission += deal.commission
                                    swap += deal.swap
                                # Exit deal
                                elif deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                                    exit_deal = deal
                                    profit_currency += deal.profit
                                    commission += deal.commission
                                    swap += deal.swap
                        
                        # Record trade if we found both entry and exit
                        if entry_deal and exit_deal:
                            # Determine pip value
                            symbol_info = mt5.symbol_info(pair)
                            pip_size = symbol_info.point * (10 if 'JPY' not in pair else 100)
                            
                            # Calculate profit in pips
                            price_diff = exit_deal.price - entry_price
                            profit_pips = (price_diff / pip_size) if direction == 'BUY' else (-price_diff / pip_size)
                            
                            # Determine exit reason
                            exit_reason = "Manual Close"
                            exit_comment = str(exit_deal.comment).lower() if hasattr(exit_deal, 'comment') else ""
                            
                            if 'sl' in exit_comment or (sl_price > 0 and abs(exit_deal.price - sl_price) < symbol_info.point * 5):
                                exit_reason = "SL"
                            elif 'tp' in exit_comment or (tp_price > 0 and abs(exit_deal.price - tp_price) < symbol_info.point * 5):
                                exit_reason = "TP"
                            
                            # Create trade record
                            trade_details = {
                                'trade_id': f"manual_{ticket}_{identifier}",
                                'pair': pair,
                                'direction': direction,
                                'entry_time': entry_time,
                                'exit_time': datetime.fromtimestamp(exit_deal.time, tz=pytz.utc),
                                'entry_price': entry_price,
                                'exit_price': exit_deal.price,
                                'stop_loss': sl_price,
                                'take_profit': tp_price,
                                'position_size': volume,
                                'profit_currency': profit_currency + commission + swap,
                                'profit_pips': round(profit_pips, 1),
                                'profit_percentage': 0.0,  # Not calculated
                                'exit_reason': exit_reason,
                                'signal_confidence': self.execution_manager.open_positions[ticket].get('signal_confidence', 0.5),
                                'is_manual': True
                            }
                            
                            # Record the trade
                            logger.info(f"Recording manual trade closure: {ticket}, Profit: {profit_currency}, Reason: {exit_reason}")
                            self.execution_manager.tracker.record_trade(trade_details)
                            
                            # Mark as processed by removing from tracking if not already removed
                            if ticket in self.execution_manager.open_positions:
                                del self.execution_manager.open_positions[ticket]
                                
                            # Update UI
                            self.after(1000, self._update_dashboard_tabs)
                        else:
                            logger.warning(f"Incomplete history data for manual trade {ticket} - cannot record properly")
                            # Clean up tracking even if we couldn't fully process
                            if ticket in self.execution_manager.open_positions:
                                del self.execution_manager.open_positions[ticket]
                    except Exception as e:
                        logger.exception(f"Error processing manual trade {ticket} history: {e}")
                        # Clean up tracking even if we had an error
                        if ticket in self.execution_manager.open_positions:
                            del self.execution_manager.open_positions[ticket]
                    
                    # Exit the monitoring loop
                    return
        except Exception as e:
            logger.exception(f"Error in manual trade monitor for ticket {ticket}: {e}")
            # Clean up tracking on error
            if hasattr(self, 'execution_manager') and self.execution_manager and ticket in self.execution_manager.open_positions:
                del self.execution_manager.open_positions[ticket]
                
    def _update_positions_tab(self):
        """Update the open positions tab with current positions from MT5"""
        try:
            if not self.execution_manager:
                return
            
            # Direct MT5 call to get positions with more debug information
            if mt5 is None or not mt5.initialize():
                # تقليل تكرار رسائل التحذير - عدم تسجيل تحذير في كل مرة
                if not hasattr(self, '_last_mt5_warning_time') or time.time() - self._last_mt5_warning_time > 60:
                    logger.warning("MT5 not initialized for position check")
                    self._last_mt5_warning_time = time.time()
                
                pos_text = f"--- Open Positions ({datetime.now().strftime('%H:%M:%S')}) ---\n\n"
                pos_text += "MetaTrader 5 غير متصل. اضغط على 'Connect' للاتصال بـ MT5.\n\n"
                pos_text += "⚠️ لتنفيذ الصفقات وعرض المراكز المفتوحة، يجب الاتصال بـ MT5.\n"
                pos_text += "   1. تأكد من أن برنامج MetaTrader 5 مفتوح\n"
                pos_text += "   2. اضغط على زر 'Connect' أعلى يسار الشاشة\n"
                pos_text += "   3. تأكد من صحة بيانات الدخول في الإعدادات\n"
            else:
                # Get positions directly from MT5 instead of through execution_manager
                logger.debug("Fetching positions directly from MT5")
                mt5_positions = mt5.positions_get()
                
                if mt5_positions is None:
                    mt5_error = mt5.last_error()
                    logger.warning(f"Failed to get positions from MT5: {mt5_error}")
                    pos_text = f"--- Open Positions ({datetime.now().strftime('%H:%M:%S')}) ---\n\nError fetching positions: {mt5_error}"
                elif len(mt5_positions) == 0:
                    logger.info("No open positions found in MT5")
                    pos_text = f"--- Open Positions ({datetime.now().strftime('%H:%M:%S')}) ---\n\n(No open positions)"
                else:
                    # Format positions data from MT5 directly
                    pos_text = f"--- Open Positions ({datetime.now().strftime('%H:%M:%S')}) ---\n\n"
                    pos_text += "{:<10} {:<12} {:<6} {:<8} {:<12} {:<12} {:<12} {:<12}\n".format(
                        "Ticket", "Symbol", "Type", "Volume", "Entry Price", "SL", "TP", "Profit"
                    )
                    pos_text += "-" * 95 + "\n"
                    
                    for pos in mt5_positions:
                        pos_type = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
                        entry_time = datetime.fromtimestamp(pos.time).strftime('%Y-%m-%d %H:%M')
                        
                        pos_text += "{:<10} {:<12} {:<6} {:<8.2f} {:<12.5f} {:<12.5f} {:<12.5f} {:<12.2f}\n".format(
                            pos.ticket,
                            pos.symbol,
                            pos_type,
                            pos.volume,
                            pos.price_open,
                            pos.sl,
                            pos.tp,
                            pos.profit
                        )
                        
                    logger.info(f"Found {len(mt5_positions)} open positions in MT5")
            
            # Update the UI with position data
            if hasattr(self, 'positions_textbox'):
                self.positions_textbox.configure(state=tk.NORMAL)
                self.positions_textbox.delete("1.0", tk.END)
                self.positions_textbox.insert("1.0", pos_text)
                self.positions_textbox.configure(state=tk.DISABLED)
                
            logger.debug("Positions tab updated with direct MT5 data")
        except Exception as e:
            logger.exception(f"Error updating positions tab: {e}")

    def _clear_signals_display(self):
        """Removes all signal frames from the display area."""
        logger.info("Clearing all signals from display...")
        for widget in self.signal_widgets:
            try: widget.destroy()
            except tk.TclError: pass
        self.signal_widgets.clear()
        self.last_signals.clear()
        if not hasattr(self, 'signals_placeholder') or not self.signals_placeholder.winfo_exists():
             self.signals_placeholder = ctk.CTkLabel(self.signals_scroll_frame, text="Waiting for signals...", text_color="gray60")
             self.signals_placeholder.pack(pady=20)
        self.status_bar.set_status("Signals display cleared.", duration=3000)

    def _clear_logs_display(self):
        """Clears the content of the log display textbox."""
        logger.info("Clearing logs display...")
        try:
            self.log_textbox.configure(state=tk.NORMAL)
            self.log_textbox.delete("1.0", tk.END)
            self.log_textbox.configure(state=tk.DISABLED)
            self.status_bar.set_status("Logs display cleared.", duration=3000)
        except Exception as e: logger.error(f"Error clearing logs display: {e}")

    # --- Market Summary Display (Keep as is) ---
    def _update_summary_display(self, summary_data: Dict):
        logger.debug("Updating market summary display.")
        try:
            self.summary_textbox.configure(state=tk.NORMAL)
            self.summary_textbox.delete("1.0", tk.END)
            ts = summary_data.get('timestamp', 'N/A')
            try: ts_dt = datetime.fromisoformat(ts).astimezone(self.signal_generator.timezone); ts_str = ts_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            except: ts_str = ts
            summary_str = f"--- Market Summary ({ts_str}) ---\n\n"
            mood = summary_data.get('market_mood', 'N/A').replace('_', ' ').title()
            summary_str += f"Overall Market Mood:  {mood}\n"
            if summary_data.get('crypto_fear_greed'):
                fg = summary_data['crypto_fear_greed']
                summary_str += f"Crypto Fear & Greed:  {fg.get('value')} ({fg.get('interpretation', 'N/A')})\n"
            summary_str += "\n--- Upcoming High Impact Events (Next ~2 Days) ---\n"
            events = summary_data.get('high_impact_events', [])
            if events:
                for event in events:
                     impact_stars = "*" * event.get('impact', 0); country_code = event.get('country', '?')[:3].upper()
                     summary_str += (f"- {event.get('date')} {event.get('time_utc')} UTC | {country_code.ljust(3)} | {event.get('event', '?')} ({impact_stars})\n")
            else: summary_str += "(None significant found)\n"
            summary_str += "\n--- Pair Sentiment & News Analysis ---\n"
            pairs_data = summary_data.get('pairs', {})
            if pairs_data:
                 for pair, data in pairs_data.items():
                     sentiment_info = data.get('sentiment', {})
                     sent_label = sentiment_info.get('sentiment', 'N/A').upper(); sent_score = sentiment_info.get('sentiment_score', 'N/A'); sent_conf = sentiment_info.get('confidence', 'N/A')
                     news_list = sentiment_info.get('recent_news', [])
                     summary_str += f"\n[{pair}]\n  Sentiment: {sent_label} (Score: {sent_score}, Conf: {sent_conf})\n"
                     if news_list:
                         summary_str += "  Recent News:\n"
                         for news in news_list:
                             news_sent = news.get('sentiment', {}); sent_news_label = news_sent.get('sentiment_label', 'neu')[:3]; sent_news_score = news_sent.get('combined_score', 0)
                             source = news.get('source', 'Src')[:10]; title = news.get('title', '?')[:75] + ('...' if len(news.get('title', '')) > 75 else '')
                             summary_str += f"    - [{source}][{sent_news_label}:{sent_news_score:+.1f}] {title}\n"
                     else: summary_str += "  Recent News: (None relevant found)\n"
                     if data.get('error'): summary_str += f"  ! Error fetching sentiment: {data['error']}\n"
            else: summary_str += "(No specific pair analysis available)\n"
            if summary_data.get('error'): summary_str += f"\n\n*** SUMMARY GENERATION ERROR: {summary_data['error']} ***"
            self.summary_textbox.insert("1.0", summary_str)
            self.summary_textbox.configure(state=tk.DISABLED)
        except Exception as e:
             logger.exception(f"Error updating market summary display: {e}")
             try:
                 self.summary_textbox.delete("1.0", tk.END); self.summary_textbox.insert("1.0", f"Error rendering summary:\n{e}"); self.summary_textbox.configure(state=tk.DISABLED)
             except: pass


    # --- Bot Control Logic (Main Loop) ---

    async def _analyze_pair_tf(self, pair: str, timeframe: TimeFrames, data_count: int) -> Optional[Dict]:
        """Async helper to fetch data, generate signal, and run advanced analysis."""
        signal_key = f"{pair}_{timeframe.name}"
        logger.debug(f"Starting analysis task for {signal_key}...")
        signal = None

        try:
            # Fetch data for the current timeframe
            market_data = await asyncio.to_thread(self.data_manager.fetch_data, pair, timeframe, data_count)
            # market_data = self.data_manager.fetch_data(pair, timeframe, count=data_count) # Sync alternative

            if market_data is None or market_data.empty:
                logger.warning(f"No market data returned for {signal_key}. Skipping analysis.")
                return None

            # Generate signal for the current timeframe
            signal = await asyncio.to_thread(self.signal_generator.generate_signal, market_data, pair, timeframe)
            # signal = self.signal_generator.generate_signal(market_data, pair, timeframe) # Sync alternative

            # Always display the signal even if it's HOLD
            if signal:
                # If it's a HOLD signal, we still want to display it
                if signal.get('direction') == 'HOLD':
                    logger.info(f"HOLD signal generated for {signal_key}.")
                    self.after(0, self._update_signals_display, signal_key, signal)
                    return None  # Don't execute HOLD signals
                
                # For BUY/SELL signals, proceed with advanced analysis
                elif signal.get('direction') in ['BUY', 'SELL']:
                    logger.info(f"Base signal [{signal['direction']}] generated for {signal_key}. Running advanced analysis...")
                    # Indicators might be recalculated inside advanced analysis if not passed,
                    # but passing them avoids redundant work if SignalGenerator already calculated them
                    indicators = self.signal_generator.calculate_indicators(market_data, pair) # Recalculate or get from signal if stored?
                    analysis_result = await self.advanced_analyzer.analyze_trading_signal(
                        pair=pair, signal_type=signal['direction'], data=market_data, indicators=indicators
                    )
                    signal['advanced_analysis'] = analysis_result
                    logger.info(f"Advanced analysis for {signal_key}: Recommend={analysis_result.get('recommendation', 'N/A')}, Str={analysis_result.get('signal_strength', 'N/A')}")
                    
                    # Add entry strategy analysis with multi-timeframe data
                    try:
                        # Create timeframes_data dictionary with data for all configured timeframes
                        timeframes_data = {}
                        
                        # Add current timeframe data first
                        timeframes_data[timeframe] = market_data
                        
                        # Fetch data for other timeframes
                        for tf in TimeFrames:
                            # Skip the current timeframe as we already have it
                            if tf == timeframe:
                                continue
                                
                            # Only include timeframes configured in config.json
                            if tf.name in self.config.get('timeframes', []):
                                tf_data = await asyncio.to_thread(self.data_manager.fetch_data, pair, tf, data_count)
                                if tf_data is not None and not tf_data.empty:
                                    timeframes_data[tf] = tf_data
                        
                        # Run entry analysis with multi-timeframe data
                        entry_analysis = self.entry_strategies.analyze_entry_opportunity(
                            pair=pair,
                            timeframes_data=timeframes_data,
                            primary_timeframe=timeframe
                        )
                        
                        # Add entry analysis to signal data
                        signal['entry_recommendation'] = entry_analysis.get('entry_recommendation', {})
                        signal['action'] = entry_analysis.get('action', 'WAIT')
                        if signal['action'] is None:
                            signal['action'] = 'WAIT'  # Replace None with WAIT
                        signal['action_details'] = entry_analysis.get('action_details', {})
                    
                        logger.info(f"Entry analysis for {signal_key}: Strategy={entry_analysis.get('entry_recommendation', {}).get('best_strategy', 'No Strategy')}, Action={entry_analysis.get('action', 'WAIT')}")
                    except Exception as e:
                        logger.error(f"Error during entry analysis for {signal_key}: {e}")
                        # Set better default values if entry analysis fails
                        signal['entry_recommendation'] = {}
                        signal['action'] = 'WAIT'
                        signal['action_details'] = {}

                    # Always update UI with the signal
                    self.after(0, self._update_signals_display, signal_key, signal)
                    
                    # Check for STRONG_BUY and STRONG_SELL to always allow execution
                    recommendation = analysis_result.get('recommendation', 'NEUTRAL')
                    if "STRONG_BUY" in recommendation or "STRONG_SELL" in recommendation:
                        # Force execution for strong signals
                        signal['execution_allowed'] = True
                        logger.info(f"Strong signal '{recommendation}' for {signal_key} will execute automatically.")
                    # Mark signals with certain recommendations as not executable
                    elif "AVOID" in recommendation or "NEUTRAL" in recommendation or "ERROR" in recommendation:
                         logger.info(f"Signal for {signal_key} ignored for execution based on advanced analysis recommendation: '{recommendation}'.")
                         signal['execution_allowed'] = False
                         # Return None to prevent execution but signal has already been sent to UI
                         return None

            return signal # Return the signal (or None if invalidated)

        except Exception as e:
            logger.exception(f"Error during analysis task for {signal_key}: {e}")
            return None


    def _run_bot_loop(self):
        """The main analysis and execution loop for the bot."""
        thread_name = threading.current_thread().name
        logger.info(f"Bot thread '{thread_name}' started.")
        last_summary_time = 0
        last_monitor_time = 0
        last_perf_check_time = 0

        # --- Get Settings from Config ---
        bot_config = self.config.get("bot_settings", {})
        loop_sleep_sec = bot_config.get("loop_sleep_seconds", 60)
        summary_interval_sec = bot_config.get("summary_interval_minutes", 30) * 60
        monitor_interval_sec = bot_config.get("monitor_interval_seconds", 10)
        perf_check_interval_sec = bot_config.get("performance_check_interval_seconds", 300)
        data_count = bot_config.get("data_fetch_count", 250)
        should_check_performance = self.config.get("profitability_enhancer", {}).get("enable_adaptation_check", True)

        # --- Create and Set Asyncio Event Loop for this Thread ---
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info(f"Asyncio event loop created and set for thread '{thread_name}'.")

        try:
            # Add symbol mapping to fix crypto pairs
            self._ensure_symbol_mappings()
            
            while self.is_running:
                cycle_start_time = time.time()
                # --- Monitor Open Positions (Run frequently) ---
                current_time_monitor = time.time()
                if current_time_monitor - last_monitor_time > monitor_interval_sec:
                     if self.execution_manager and self.data_manager.is_initialized:
                         logger.debug("Running position monitoring...")
                         self.status_bar.set_status("Monitoring open positions...", duration=monitor_interval_sec*900) # Show briefly
                         try:
                              # This method contains sync MT5 calls, run in thread if it blocks too long
                              # For now, assuming it's acceptable within the bot loop pause
                              self.execution_manager.monitor_and_close_positions()
                         except Exception as mon_e:
                              logger.exception(f"Error during position monitoring: {mon_e}")
                         last_monitor_time = current_time_monitor
                     else:
                         logger.debug("Skipping position monitoring: Execution Manager not ready or MT5 disconnected.")

                # --- Performance Check & Adaptation ---
                current_time_perf = time.time()
                if should_check_performance and current_time_perf - last_perf_check_time > perf_check_interval_sec:
                    if self.profit_enhancer and self.tracker:
                        logger.info("Checking performance for strategy adaptation...")
                        try:
                             needs_adapt = self.profit_enhancer.should_adapt_strategy(lookback_trades=50) # Check last 50 trades
                             if needs_adapt:
                                  # ---> Implement Adaptation Action <---
                                  logger.warning("ADAPTATION SUGGESTED! - Taking default action: Logging only.")
                                  self.status_bar.set_status("Warning: Recent performance suggests strategy review.", alert=True, duration=60000)
                                  # Example Action: Reduce risk temporarily
                                  # current_risk = self.config["execution_manager"]["risk_per_trade_pct"]
                                  # new_risk = max(0.1, current_risk * 0.75) # Reduce by 25%, minimum 0.1%
                                  # self.config["execution_manager"]["risk_per_trade_pct"] = new_risk
                                  # logger.warning(f"ADAPTATION: Temporarily reduced risk per trade to {new_risk:.2f}%")
                                  # self.status_bar.set_status(f"Action: Risk reduced to {new_risk:.2f}% due to performance.", alert=True, duration=60000)
                                  # self._save_config() # Optionally save the adapted config
                                  # ------------------------------------
                             else:
                                  logger.info("Performance check complete. No immediate adaptation flagged.")
                        except Exception as perf_e:
                             logger.exception(f"Error during performance check / adaptation: {perf_e}")
                    last_perf_check_time = current_time_perf


                # --- Signal Analysis Cycle ---
                logger.info("Starting new analysis cycle...")
                self.status_bar.set_status("Running analysis cycle...", priority=True)
                self.status_bar.set_progress(0)

                current_pairs = self.config.get("trading_pairs", [])
                current_tf_names = self.config.get("timeframes", [])
                if not current_pairs or not current_tf_names:
                    logger.warning("No trading pairs or timeframes configured. Skipping analysis cycle.")
                    time.sleep(loop_sleep_sec); continue

                analysis_tasks = []; valid_tf_enums = {name: tf for name, tf in TimeFrames.__members__.items()}
                for pair in current_pairs:
                    for tf_name in current_tf_names:
                        if tf_name in valid_tf_enums:
                            timeframe_enum = valid_tf_enums[tf_name]
                            task = self._analyze_pair_tf(pair, timeframe_enum, data_count)
                            analysis_tasks.append(task)
                        else: logger.warning(f"Invalid timeframe '{tf_name}' configured for {pair}. Skipping.")

                valid_signals_for_execution = []
                if analysis_tasks:
                     logger.info(f"Running analysis for {len(analysis_tasks)} pair/timeframe combinations...")
                     results = loop.run_until_complete(asyncio.gather(*analysis_tasks, return_exceptions=True))
                     logger.info("Finished concurrent analysis tasks.")

                     processed_count = 0
                     fail_count = 0
                     for i, res in enumerate(results):
                         if isinstance(res, dict) and res is not None: # Check if it's a valid signal dict
                             valid_signals_for_execution.append(res)
                             processed_count += 1
                         elif isinstance(res, Exception):
                              logger.error(f"Analysis task failed with exception: {res}")
                              fail_count += 1
                         elif res is None:
                              # This is expected if no signal or filtered out
                              processed_count += 1 # Count as processed, but not a failure
                         else:
                              logger.warning(f"Analysis task returned unexpected type: {type(res)}")
                              fail_count += 1

                     logger.debug(f"Analysis results: {len(valid_signals_for_execution)} valid signals generated, {fail_count} failures.")
                     self.status_bar.set_progress(50) # Progress after analysis

                else: logger.warning("No valid analysis tasks were created.")


                # --- Trade Execution ---
                if valid_signals_for_execution:
                     logger.info(f"Processing {len(valid_signals_for_execution)} valid signals for potential execution...")
                     executed_count = 0
                     for signal in valid_signals_for_execution:
                         # Check again if bot is running before placing order
                         if not self.is_running:
                              logger.info("Bot stopped during execution phase. Aborting further orders.")
                              break
                         # Call Execution Manager to place the order
                         if self.execution_manager and self.data_manager.is_initialized:
                              logger.info(f"Attempting execution for signal: {signal.get('pair')} {signal.get('direction')}")
                              self.status_bar.set_status(f"Attempting order: {signal.get('pair')} {signal.get('direction')}...", priority=True)
                              try:
                                  # place_order is synchronous but contains MT5 calls
                                  position_ticket = self.execution_manager.place_order(signal)
                                  if position_ticket:
                                       executed_count += 1
                                       logger.info(f"Successfully placed order for {signal.get('pair')}. Position Ticket: {position_ticket}")
                                       self.status_bar.set_status(f"Order placed: {signal.get('pair')} Ticket: {position_ticket}", duration=15000)
                                  else:
                                       logger.warning(f"Order placement failed or denied for signal: {signal.get('pair')} {signal.get('direction')}")
                                       self.status_bar.set_status(f"Order failed/denied: {signal.get('pair')}", alert=True, duration=10000)
                                  time.sleep(0.5) # Small delay between order attempts
                              except Exception as exec_e:
                                   logger.exception(f"Error during execution_manager.place_order call: {exec_e}")
                         else:
                              logger.warning(f"Skipping execution for {signal.get('pair')}: Execution Manager not ready or MT5 disconnected.")
                     logger.info(f"Finished execution phase. Placed {executed_count} new orders.")
                     self.status_bar.set_progress(75) # Progress after execution
                else:
                    logger.debug("No valid signals found for execution in this cycle.")
                    self.status_bar.set_progress(75) # Still update progress


                # --- Update Market Summary ---
                current_time = time.time()
                if current_time - last_summary_time > summary_interval_sec:
                     logger.info("Updating market summary...")
                     self.status_bar.set_status("Updating market summary...")
                     try:
                         summary_data = loop.run_until_complete(self.advanced_analyzer.generate_market_summary(current_pairs))
                         if summary_data and not summary_data.get('error'):
                             self.after(0, self._update_summary_display, summary_data)
                             last_summary_time = current_time
                             logger.info("Market summary updated successfully.")
                         elif summary_data and summary_data.get('error'): logger.error(f"Failed to generate market summary: {summary_data['error']}")
                         else: logger.warning("Market summary generation returned empty or invalid data.")
                     except Exception as sum_e: logger.exception(f"Unhandled exception during market summary generation: {sum_e}")


                # --- Cycle Sleep ---
                self.status_bar.set_progress(100) # Finished cycle
                cycle_elapsed_time = time.time() - cycle_start_time
                sleep_duration = max(1.0, loop_sleep_sec - cycle_elapsed_time)
                logger.info(f"Analysis cycle finished in {cycle_elapsed_time:.2f}s. Sleeping for {sleep_duration:.1f}s.")
                self.status_bar.set_status(f"Idle. Next cycle in {sleep_duration:.0f}s.", duration=int(sleep_duration * 1000))
                self.status_bar.hide_progress()

                sleep_end_time = time.time() + sleep_duration
                while time.time() < sleep_end_time and self.is_running:
                     time.sleep(0.5)

                if not self.is_running:
                    logger.info("Stop signal received during sleep. Exiting bot loop.")
                    break

        except Exception as e:
            logger.exception(f"CRITICAL ERROR in bot loop: {e}")
            self.after(0, messagebox.showerror, "Bot Loop Error", f"A critical error occurred in the bot's main loop:\n{e}\n\nBot stopped. Check logs.")
            self.is_running = False # Ensure flag is set to false on crash
        finally:
            logger.info(f"Bot thread '{thread_name}' beginning cleanup...")
            try:
                 if loop.is_running():
                      logger.debug("Requesting asyncio loop stop...")
                      loop.call_soon_threadsafe(loop.stop)
                 logger.info(f"Asyncio event loop for thread '{thread_name}' cleanup initiated.")
            except Exception as loop_close_e: logger.error(f"Error stopping/closing asyncio loop: {loop_close_e}")
            logger.info(f"Bot thread '{thread_name}' finished.")
            self.after(0, self._update_ui_stopped) # Schedule UI update

    def _ensure_symbol_mappings(self):
        """Set up symbol mappings for different brokers"""
        if not hasattr(self.execution_manager, 'symbol_map'):
            return
            
        # Try to detect if this is a crypto-friendly broker
        has_crypto = False
        crypto_symbols = ["BTCUSD", "BTCUSDm", "BTC/USD", "ETHUSD", "XBT/USD"]
        
        for sym in crypto_symbols:
            if mt5.symbol_info(sym) is not None:
                has_crypto = True
                break
                
        if has_crypto:
            logger.info("Detected crypto-enabled broker, setting up symbol mappings")
            
            # Check which format of crypto pairs works
            crypto_formats = []
            test_pairs = ["BTCUSDm", "BTCUSD.m", "BTCUSD", "BTC/USD"]
            
            for test in test_pairs:
                if mt5.symbol_info(test) is not None:
                    crypto_formats.append(test)
                    logger.info(f"Found working crypto format: {test}")
            
            if crypto_formats:
                preferred_format = crypto_formats[0]
                suffix = ""
                
                if "m" in preferred_format:
                    suffix = "m" if preferred_format.endswith("m") else ".m"
                    
                # Update symbol map for common crypto pairs
                self.execution_manager.symbol_map.update({
                    "BTCUSD": f"BTCUSD{suffix}",
                    "ETHUSD": f"ETHUSD{suffix}",
                    "XRPUSD": f"XRPUSD{suffix}",
                    "LTCUSD": f"LTCUSD{suffix}",
                    # Add other crypto pairs as needed
                })
                
                logger.info(f"Updated symbol mapping for crypto pairs with suffix: '{suffix}'")
            else:
                logger.warning("No working crypto format detected")

    def start_bot(self):
        """Starts the bot's main analysis loop in a separate thread."""
        if self.is_running:
            logger.warning("Bot start requested, but it is already running.")
            messagebox.showwarning("Already Running", "The bot is already running.")
            return

        if not self.data_manager.is_initialized:
            messagebox.showerror("MT5 Error", "Cannot start bot.\nPlease connect to MetaTrader 5 first.")
            return

        current_pairs = self.config.get("trading_pairs", [])
        current_tfs = self.config.get("timeframes", [])
        if not current_pairs or not current_tfs:
             messagebox.showerror("Configuration Error", "Cannot start bot.\nPlease configure and save at least one trading pair and timeframe.")
             return

        # ---> Added Check: Execution Manager Ready? ---
        if not self.execution_manager or not self.execution_manager.mt5:
             messagebox.showerror("Execution Error", "Cannot start bot.\nExecution Manager is not ready (MT5 might be disconnected).")
             return
        # ---> End Check ---

        self.is_running = True
        self.bot_status_label.configure(text="Status: Running", text_color="lightgreen")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.entry_pairs.configure(state=tk.DISABLED)
        self.entry_timeframes.configure(state=tk.DISABLED)
        self.save_config_button.configure(state=tk.DISABLED)

        logger.info("Starting bot analysis loop thread...")
        self.status_bar.set_status("Bot starting...", priority=True)

        self.bot_thread = threading.Thread(target=self._run_bot_loop, name="BotLoopThread", daemon=True)
        self.bot_thread.start()
        logger.info(f"Bot thread '{self.bot_thread.name}' started.")

    def stop_bot(self):
        """Signals the bot's analysis loop to stop."""
        if not self.is_running:
            logger.warning("Bot stop requested, but it is not currently running.")
            return

        logger.info("Stop requested. Signaling bot thread to exit...")
        self.status_bar.set_status("Bot stopping...", priority=True)
        self.stop_button.configure(state=tk.DISABLED)
        self.is_running = False

    def _update_ui_stopped(self):
         """Updates UI elements after the bot thread has confirmed it stopped."""
         self.bot_status_label.configure(text="Status: Stopped", text_color="gray")
         self.start_button.configure(state=tk.NORMAL)
         self.stop_button.configure(state=tk.DISABLED)
         self.entry_pairs.configure(state=tk.NORMAL)
         self.entry_timeframes.configure(state=tk.NORMAL)
         self.save_config_button.configure(state=tk.NORMAL)
         self.status_bar.set_status("Bot stopped.")
         self.status_bar.hide_progress()
         logger.info("Bot stop confirmed and UI updated.")


    def on_closing(self):
        """Handles the window closing event to ensure graceful shutdown."""
        logger.info("Window closing requested.")

        close_confirmed = True # Assume yes if bot isn't running
        if self.is_running:
             close_confirmed = messagebox.askyesno("Confirm Exit",
                                                  "The bot is currently running. Closing will attempt to stop the bot and close open positions managed by it.\n\nAre you sure you want to exit?")
             if close_confirmed:
                 logger.info("Stopping bot before closing application...")
                 self.stop_bot() # Signal the bot thread to stop
                 # Wait briefly for the thread to exit
                 if self.bot_thread and self.bot_thread.is_alive():
                      logger.info("Waiting up to 10 seconds for bot thread to finish...")
                      self.bot_thread.join(timeout=10.0)
                      if self.bot_thread.is_alive():
                           logger.warning("Bot thread did not exit gracefully within the timeout.")
                      else: logger.info("Bot thread finished.")
                 else: logger.info("Bot thread was already finished or not started.")
             else:
                 logger.info("Exit cancelled by user.")
                 return # Cancel closing

        # ---> Close Open Positions Managed by Bot <---
        if self.execution_manager and self.data_manager.is_initialized:
            try:
                open_pos_count = len(self.execution_manager.get_open_positions())
                if open_pos_count > 0:
                     if messagebox.askyesno("Close Positions?", f"Found {open_pos_count} open positions managed by the bot.\nDo you want to attempt to close them now?"):
                         logger.info(f"Attempting to close {open_pos_count} open positions before exit...")
                         close_results = self.execution_manager.close_all_positions("App Closing")
                         successful_closes = sum(1 for _, success, _ in close_results if success)
                         failed_closes = len(close_results) - successful_closes
                         logger.info(f"Close all result: {successful_closes} successful, {failed_closes} failed.")
                         if failed_closes > 0:
                              messagebox.showwarning("Close Failed", f"Failed to close {failed_closes} positions. Please check your MT5 terminal.")
                     else:
                         logger.warning("User chose not to close open positions on exit.")
            except Exception as close_e:
                 logger.exception(f"Error attempting to close positions on exit: {close_e}")
                 messagebox.showerror("Close Error", f"An error occurred while trying to close positions:\n{close_e}")
        # ---> END Close Positions <---

        # Cleanup other resources
        if hasattr(self, 'advanced_analyzer') and hasattr(self.advanced_analyzer, 'close'):
             logger.info("Closing Advanced Analysis resources...")
             try: asyncio.run(self.advanced_analyzer.close())
             except Exception as e: logger.error(f"Error during Advanced Analysis cleanup: {e}")

        logger.info("Disconnecting from MetaTrader 5...")
        self.data_manager.disconnect_mt5() # Disconnect MT5

        logger.info("Destroying UI and exiting application.")
        self.destroy() # Close the Tkinter window

    def _import_historical_trades(self):
        """Import historical closed trades from MT5 and add them to the performance tracker"""
        if not self.data_manager.is_initialized:
            messagebox.showerror("MT5 Error", "Cannot import historical trades.\nPlease connect to MetaTrader 5 first.")
            logger.error("Import historical trades failed: MT5 not initialized")
            return
            
        # Verify MT5 is actually connected and responsive
        if not mt5.terminal_info():
            messagebox.showerror("MT5 Error", "MetaTrader 5 connection not active. Please check connection and try again.")
            logger.error("Import historical trades failed: MT5 terminal_info returned False")
            return
            
        try:
            # Create a dialog for configuring the import
            import_dialog = tk.Toplevel(self)
            import_dialog.title("Import Historical Trades")
            import_dialog.geometry("500x350")
            import_dialog.resizable(False, False)
            import_dialog.transient(self)  # Make dialog modal
            import_dialog.grab_set()
            
            # Apply theme colors
            import_dialog.configure(bg="#2b2b2b")
            
            # Add dialog content
            header_label = ttk.Label(import_dialog, text="Import Historical Trades from MT5", 
                                    font=("Segoe UI", 14, "bold"), background="#2b2b2b", foreground="white")
            header_label.pack(pady=(15, 20))
            
            # Create frame for options
            options_frame = ttk.Frame(import_dialog, padding=10)
            options_frame.pack(fill="x", padx=20, pady=10)
            options_frame.configure(style="TFrame")
            
            # Style for the dialog
            style = ttk.Style(import_dialog)
            style.configure("TFrame", background="#2b2b2b")
            style.configure("TLabel", background="#2b2b2b", foreground="white")
            style.configure("TCheckbutton", background="#2b2b2b", foreground="white")
            style.configure("TButton", background="#2a52be", foreground="white")
            
            # Date range selection
            date_frame = ttk.Frame(options_frame)
            date_frame.pack(fill="x", pady=5)
            
            # Option for days or custom date range
            import_option = tk.StringVar(value="days")
            
            # Days option
            days_frame = ttk.Frame(options_frame)
            days_frame.pack(fill="x", pady=5)
            
            days_radio = ttk.Radiobutton(days_frame, text="Import trades from the last:", 
                                       variable=import_option, value="days")
            days_radio.pack(side="left", padx=5)
            
            days_var = tk.IntVar(value=30)
            days_entry = ttk.Entry(days_frame, textvariable=days_var, width=5)
            days_entry.pack(side="left", padx=5)
            
            days_label = ttk.Label(days_frame, text="days")
            days_label.pack(side="left", padx=5)
            
            # Custom date range option
            date_range_frame = ttk.Frame(options_frame)
            date_range_frame.pack(fill="x", pady=5)
            
            date_radio = ttk.Radiobutton(date_range_frame, text="Use custom date range:", 
                                       variable=import_option, value="custom")
            date_radio.pack(side="left", padx=5)
            
            # Date inputs for custom range
            custom_dates_frame = ttk.Frame(options_frame)
            custom_dates_frame.pack(fill="x", pady=5)
            
            # Start date
            start_label = ttk.Label(custom_dates_frame, text="Start date:")
            start_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
            
            start_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
            start_date_entry = ttk.Entry(custom_dates_frame, textvariable=start_date_var, width=15)
            start_date_entry.grid(row=0, column=1, padx=5, pady=5)
            
            # Format hint
            start_format = ttk.Label(custom_dates_frame, text="(YYYY-MM-DD)")
            start_format.grid(row=0, column=2, padx=5, pady=5, sticky="w")
            
            # End date
            end_label = ttk.Label(custom_dates_frame, text="End date:")
            end_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
            
            end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
            end_date_entry = ttk.Entry(custom_dates_frame, textvariable=end_date_var, width=15)
            end_date_entry.grid(row=1, column=1, padx=5, pady=5)
            
            # Format hint
            end_format = ttk.Label(custom_dates_frame, text="(YYYY-MM-DD)")
            end_format.grid(row=1, column=2, padx=5, pady=5, sticky="w")
            
            # Symbol filter option
            filter_frame = ttk.Frame(options_frame)
            filter_frame.pack(fill="x", pady=10)
            
            symbol_filter_enabled = tk.BooleanVar(value=False)
            filter_check = ttk.Checkbutton(filter_frame, text="Filter by symbol:", 
                                         variable=symbol_filter_enabled)
            filter_check.pack(side="left", padx=5)
            
            symbol_var = tk.StringVar(value="")
            symbol_entry = ttk.Entry(filter_frame, textvariable=symbol_var, width=15)
            symbol_entry.pack(side="left", padx=5)
            
            # Info frame
            info_frame = ttk.Frame(options_frame)
            info_frame.pack(fill="x", pady=10)
            
            info_label = ttk.Label(info_frame, text="This will import historical closed trades from MT5.\n"
                                               "Trades will be added to your performance metrics.")
            info_label.pack(pady=5)
            
            # Buttons frame
            button_frame = ttk.Frame(import_dialog)
            button_frame.pack(fill="x", padx=20, pady=15)
            
            # Function to handle import
            def do_import():
                try:
                    import_dialog.config(cursor="wait")
                    
                    # Get parameters from the dialog
                    if import_option.get() == "days":
                        days = days_var.get()
                        if days <= 0:
                            messagebox.showerror("Input Error", "Please enter a positive number of days.", parent=import_dialog)
                            import_dialog.config(cursor="")
                            return
                        from_date = datetime.now() - timedelta(days=days)
                        to_date = datetime.now()
                    else:  # custom date range
                        try:
                            from_date = datetime.strptime(start_date_var.get(), "%Y-%m-%d")
                            to_date = datetime.strptime(end_date_var.get(), "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
                        except ValueError:
                            messagebox.showerror("Date Error", "Invalid date format. Use YYYY-MM-DD format.", parent=import_dialog)
                            import_dialog.config(cursor="")
                            return
                    
                    # Check if dates are valid
                    if from_date >= to_date:
                        messagebox.showerror("Date Error", "Start date must be before end date.", parent=import_dialog)
                        import_dialog.config(cursor="")
                        return
                    
                    # Get symbol filter
                    symbol_filter = symbol_var.get().strip() if symbol_filter_enabled.get() else None
                    
                    # Close dialog
                    import_dialog.destroy()
                    
                    # Show progress
                    date_range_str = f"{from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}"
                    self.status_bar.set_status(f"Importing historical trades from {date_range_str}...", priority=True)
                    self.status_bar.set_progress(10)
                    
                    # Verify MT5 is still connected
                    if not mt5.terminal_info():
                        self.status_bar.hide_progress()
                        messagebox.showerror("MT5 Error", "MetaTrader 5 connection lost. Please reconnect and try again.")
                        logger.error("Import historical trades failed: MT5 connection lost before importing")
                        return
                    
                    logger.info(f"Requesting historical deals from MT5 for date range: {date_range_str}")
                    # Get history of deals from MT5
                    history_deals = mt5.history_deals_get(from_date, to_date)
                    
                    if history_deals is None:
                        self.status_bar.hide_progress()
                        error_code, error_message = mt5.last_error()
                        error_str = f"Failed to get history deals from MT5: ({error_code}) {error_message}"
                        messagebox.showerror("Import Error", error_str)
                        logger.error(error_str)
                        return
                        
                    if len(history_deals) == 0:
                        self.status_bar.hide_progress()
                        messagebox.showinfo("No Trades", f"No historical trades found in the selected date range.")
                        logger.info(f"No historical trades found in date range: {date_range_str}")
                        return
                    
                    logger.info(f"Found {len(history_deals)} historical deals from MT5")
                    self.status_bar.set_progress(30)
                    
                    # Process deals to reconstruct trades
                    positions = {}  # Dictionary to hold position data keyed by position ID
                    
                    for deal in history_deals:
                        position_id = deal.position_id
                        
                        # Skip deals with position_id 0 (like deposits, withdrawals)
                        if position_id == 0:
                            continue
                            
                        # Apply symbol filter if specified
                        if symbol_filter and deal.symbol != symbol_filter:
                            continue
                            
                        # Create or update position data
                        if position_id not in positions:
                            positions[position_id] = {
                                'entry_deals': [],
                                'exit_deals': [],
                                'position_id': position_id,
                                'pair': deal.symbol,
                                'magic': deal.magic,
                                'volume': 0.0,
                                'direction': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                                'profit': 0.0,
                                'commission': 0.0,
                                'swap': 0.0
                            }
                        
                        # Add deal to appropriate list based on entry type
                        if deal.entry == mt5.DEAL_ENTRY_IN:
                            positions[position_id]['entry_deals'].append(deal)
                            positions[position_id]['volume'] += deal.volume
                        elif deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                            positions[position_id]['exit_deals'].append(deal)
                            
                        # Accumulate profit and costs
                        positions[position_id]['profit'] += deal.profit
                        positions[position_id]['commission'] += deal.commission
                        positions[position_id]['swap'] += deal.swap
                    
                    logger.info(f"Processed deals into {len(positions)} positions")
                    self.status_bar.set_progress(60)
                    
                    # Get current trade IDs to avoid duplicates
                    existing_trade_ids = set()
                    if hasattr(self, 'execution_manager') and self.execution_manager and hasattr(self.execution_manager, 'tracker'):
                        for trade in self.execution_manager.tracker.trade_history:
                            if 'trade_id' in trade:
                                existing_trade_ids.add(trade['trade_id'])
                    
                    # Create trade records for completed trades
                    trades_added = 0
                    trades_skipped = 0
                    for position_id, position in positions.items():
                        trade_id = f"hist_{position_id}"
                        
                        # Skip already imported trades
                        if trade_id in existing_trade_ids:
                            logger.info(f"Skipping already imported trade with ID {trade_id}")
                            trades_skipped += 1
                            continue
                            
                        if position['entry_deals'] and position['exit_deals']:
                            # Get first entry and last exit
                            first_entry = min(position['entry_deals'], key=lambda d: d.time)
                            last_exit = max(position['exit_deals'], key=lambda d: d.time)
                            
                            # Calculate pip difference based on symbol point
                            symbol_info = mt5.symbol_info(position['pair'])
                            if symbol_info:
                                # Determine pip size based on pair
                                point = symbol_info.point
                                if 'JPY' in position['pair']:
                                    pip_size = point * 100  # For JPY pairs
                                else:
                                    pip_size = point * 10  # For non-JPY pairs
                                
                                price_diff = last_exit.price - first_entry.price
                                profit_pips = (price_diff / pip_size) if position['direction'] == 'BUY' else (-price_diff / pip_size)
                            else:
                                profit_pips = 0.0
                                logger.warning(f"Could not get symbol info for {position['pair']} to calculate pips")
                            
                            # Determine exit reason
                            exit_reason = "Historical"
                            if any('sl' in str(deal.comment).lower() for deal in position['exit_deals']):
                                exit_reason = "SL (Historical)"
                            elif any('tp' in str(deal.comment).lower() for deal in position['exit_deals']):
                                exit_reason = "TP (Historical)"
                            elif any('close' in str(deal.comment).lower() for deal in position['exit_deals']):
                                exit_reason = "Manual Close (Historical)"
                            
                            # Create trade record
                            trade_details = {
                                'trade_id': trade_id,
                                'pair': position['pair'],
                                'direction': position['direction'],
                                'entry_time': datetime.fromtimestamp(first_entry.time, tz=pytz.utc),
                                'exit_time': datetime.fromtimestamp(last_exit.time, tz=pytz.utc),
                                'entry_price': first_entry.price,
                                'exit_price': last_exit.price,
                                'stop_loss': 0.0,  # Not available from history
                                'take_profit': 0.0,  # Not available from history
                                'position_size': position['volume'],
                                'profit_currency': position['profit'] + position['commission'] + position['swap'],
                                'profit_pips': round(profit_pips, 1) if profit_pips else 0.0,
                                'profit_percentage': 0.0,  # Not calculable from history
                                'exit_reason': exit_reason,
                                'signal_confidence': 0.5,  # Default value
                            }
                            
                            # Record the trade
                            if hasattr(self, 'execution_manager') and self.execution_manager and hasattr(self.execution_manager, 'tracker'):
                                logger.info(f"Recording historical trade: {trade_id}, {position['pair']}, {exit_reason}, P/L: {trade_details['profit_currency']}")
                                self.execution_manager.tracker.record_trade(trade_details)
                                trades_added += 1
                    
                    self.status_bar.set_progress(100)
                    
                    if trades_added > 0:
                        message = f"Successfully imported {trades_added} historical trades."
                        if trades_skipped > 0:
                            message += f" ({trades_skipped} trades were already imported and skipped.)"
                        messagebox.showinfo("Import Successful", message)
                        self.status_bar.set_status(f"Imported {trades_added} historical trades", priority=True)
                        logger.info(message)
                        
                        # Update performance tab immediately
                        self._update_dashboard_tabs()
                    elif trades_skipped > 0:
                        messagebox.showinfo("No New Trades", f"All {trades_skipped} trades were already imported previously.")
                        logger.info(f"No new trades imported. {trades_skipped} trades were already in the history.")
                    else:
                        messagebox.showinfo("No Complete Trades", "No complete historical trades were found to import.")
                        logger.info("No complete historical trades were found to import.")
                    
                    self.status_bar.hide_progress()
                    
                except Exception as e:
                    import_dialog.config(cursor="")
                    error_message = f"Error importing historical trades: {str(e)}"
                    messagebox.showerror("Import Error", error_message)
                    logger.exception(error_message)
                    self.status_bar.hide_progress()
            
            # Cancel button
            cancel_button = ttk.Button(button_frame, text="Cancel", command=import_dialog.destroy)
            cancel_button.pack(side="right", padx=5)
            
            # Import button
            import_button = ttk.Button(button_frame, text="Import", command=do_import)
            import_button.pack(side="right", padx=5)
            
            # Center dialog on parent window
            import_dialog.update_idletasks()
            width = import_dialog.winfo_width()
            height = import_dialog.winfo_height()
            x = self.winfo_rootx() + (self.winfo_width() - width) // 2
            y = self.winfo_rooty() + (self.winfo_height() - height) // 2
            import_dialog.geometry(f"{width}x{height}+{x}+{y}")
            
            # Wait for the dialog to close
            import_dialog.wait_window()
            
        except Exception as e:
            error_message = f"Error creating import dialog: {str(e)}"
            messagebox.showerror("Error", error_message)
            logger.exception(error_message)

    def _export_trade_history(self):
        """Export trade history to a CSV file selected by the user"""
        if not hasattr(self, 'tracker') or not self.tracker:
            messagebox.showerror("Export Error", "Performance tracker is not available.")
            return
            
        try:
            # Let user select export destination
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Trade History",
                initialfile="trade_history_export.csv"
            )
            
            if not file_path:  # User cancelled
                return
                
            # Create a copy of the trade history
            if hasattr(self.tracker, 'trade_history') and self.tracker.trade_history:
                import pandas as pd
                import shutil
                
                # Create a DataFrame from trade history
                df = pd.DataFrame(self.tracker.trade_history)
                
                # Save to the selected location
                df.to_csv(file_path, index=False)
                
                messagebox.showinfo("Export Successful", f"Trade history exported to:\n{file_path}")
                logger.info(f"Trade history exported to {file_path}")
            else:
                messagebox.showinfo("No Data", "No trade history available to export.")
                
        except Exception as e:
            error_message = f"Error exporting trade history: {str(e)}"
            messagebox.showerror("Export Error", error_message)
            logger.exception(error_message)


# --- Custom Status Bar Class (Keep as is) ---
class StatusBar(ctk.CTkFrame):
    """A custom status bar widget with text label and optional progress bar."""
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, fg_color="transparent", **kwargs) # Transparent background
        self.configure(height=25) # Fixed height
        self.status_label = ctk.CTkLabel(self, text="Ready", anchor="w", font=ctk.CTkFont(size=12))
        self.status_label.grid(row=0, column=0, sticky="ew", padx=(10, 5))
        self.progress_bar = ctk.CTkProgressBar(self, width=150, height=15, corner_radius=8)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=(5, 10)); self.progress_bar.grid_remove()
        self.grid_columnconfigure(0, weight=1)
        self._status_clear_timer: Optional[str] = None
        self._permanent_message: str = "Ready"

    def set_status(self, text: str, duration: int = 0, alert: bool = False, priority: bool = False):
        """Sets the status text."""
        if self._status_clear_timer: self.after_cancel(self._status_clear_timer); self._status_clear_timer = None
        text_color = "#FF8C00" if alert else "gray80"
        self.status_label.configure(text=text, text_color=text_color)
        if duration > 0 and not priority:
            self._status_clear_timer = self.after(duration, lambda: self.set_status(self._permanent_message))
        elif duration == 0: self._permanent_message = text

    def set_progress(self, value: float):
        """Sets the progress bar value (0-100) and ensures it's visible."""
        if not self.progress_bar.winfo_ismapped(): self.progress_bar.grid()
        progress_value = max(0.0, min(1.0, value / 100.0))
        self.progress_bar.set(progress_value)

    def hide_progress(self):
        """Hides the progress bar."""
        if self.progress_bar.winfo_ismapped(): self.progress_bar.grid_remove()


# --- Main Execution Guard ---
if __name__ == "__main__":
    threading.current_thread().name = "MainUIThread"
    logger.info(f"Starting {APP_NAME}...")
    app = TradingApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Initiating shutdown...")
        app.on_closing()
    except Exception as e:
         logger.exception(f"Unhandled exception occurred in main application scope: {e}")
         try: app.on_closing()
         except Exception as close_e: logger.error(f"Error during emergency shutdown: {close_e}")
         finally: logger.critical("Forcing exit after unhandled mainloop exception."); exit(1)
    logger.info(f"{APP_NAME} finished.")

# --- END OF FILE Main.py ---
