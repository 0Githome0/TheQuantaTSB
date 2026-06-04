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
from src.core.entry import AdvancedEntryStrategies

# Import your custom modules
try:
    from src.core.signal import SignalGenerator, TimeFrames
    from src.core.advanced_analysis import AdvancedAnalysis
    # ---> ADDED IMPORTS <---
    from src.core.performance_tracker import PerformanceTracker
    from src.core.profitability_enhancer import ProfitabilityEnhancer
    from src.core.execution_manager import ExecutionManager
    from src.gui.insider_ui import InsiderTradingFrame
    from src.gui.settings_ui import SettingsDialog
    from src.gui.backtester_ui import BacktesterFrame
    from src.gui.strategy_ui import StrategyControlFrame
    from src.core.strategy_manager import StrategyManager
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
APP_NAME = "TheQuanta"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 850
UPDATE_INTERVAL_MS = 250
DASHBOARD_UPDATE_INTERVAL_MS = 500
CONFIG_FILE = "config/config.json"
LOG_FILE = 'logs/main_app.log'
LOGO_PATH = "assets/logo.png"

# --- Modern UI Theme Colors (Cyber/Teal Palette) ---
# Background Colors
BG_DARK = "#0b0e11"           # Main background (Deep Dark)
BG_CARD = "#151a21"           # Card/panel background (Soft Dark)
BG_SIDEBAR = "#0b0e11"        # Sidebar background
BG_HOVER = "#1e2630"          # Hover state background

# Border & Accent Colors
BORDER_COLOR = "#232d3b"      # Subtle borders
ACCENT_TEAL = "#00f2ea"       # Primary accent (Cyber Cyan)
ACCENT_BLUE = "#3b82f6"       # Secondary accent (Bright Blue)

# Status Colors
SUCCESS_GREEN = "#10b981"     # Profit/success/BUY (Emerald)
DANGER_RED = "#ef4444"        # Loss/error/SELL (Bright Red)
WARNING_YELLOW = "#f59e0b"    # Warnings (Amber)

# Text Colors
TEXT_PRIMARY = "#ffffff"      # Main text (White)
TEXT_SECONDARY = "#94a3b8"    # Muted text (Cool Gray)
TEXT_MUTED = "#64748b"        # Very muted text

# Fonts
MAIN_FONT = ("Roboto Medium", 13)
HEADER_FONT = ("Roboto Medium", 20)
SUBHEADER_FONT = ("Roboto Medium", 16)
BOLD_FONT = ("Roboto Bold", 13)
MONO_FONT = ("Consolas", 12)

# --- UI Theme Setup ---
try:
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue") # Switching to built-in dark-blue as base
    ctk.set_widget_scaling(1.0)
except Exception as e:
    print(f"Warning: Could not set CustomTkinter theme: {e}")
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
            
            # Connection retry logic with exponential backoff
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(1, max_retries + 1):
                start_time = time.time()
                authorized = mt5.login(login, password=password, server=server)
                duration = time.time() - start_time
                logger.debug(f"mt5.login() attempt {attempt} took {duration:.2f}s")

                if authorized:
                    acc_info = mt5.account_info()
                    currency = acc_info.currency if acc_info else "N/A"
                    logger.info(f"MT5 Login Successful! Account: {login}, Name: {acc_info.name if acc_info else 'N/A'}, Currency: {currency}")
                    self.is_initialized = True
                    return True
                else:
                    error = mt5.last_error()
                    logger.warning(f"MT5 Login attempt {attempt}/{max_retries} failed. Error: {error}")
                    
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"MT5 Login Failed after {max_retries} attempts for account {login} on {server}.")
                        mt5.shutdown()
                        self.is_initialized = False
                        return False
            
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
        """Saves current configuration dictionary to CONFIG_FILE."""
        try:
            # Note: self.config is updated by SettingsDialog before calling this.
            
            # --- Save ---
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration successfully saved to {CONFIG_FILE}")
            
            # Update internal state if needed (e.g. valid TFs)
            # This ensures app state is consistent with new config immediately
            self.status_bar.set_status("Configuration Saved.", duration=5000)
            
            # Refresh config display (if any other parts rely on it)
            self._update_config_display()

        except Exception as e:
            logger.exception(f"Error saving configuration to '{CONFIG_FILE}': {e}")
            messagebox.showerror("Save Error", f"Failed to save settings: {e}")
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
        # No-op: Config is now handled by SettingsDialog and not displayed on main UI
        pass

    def _create_stat_card(self, parent, title: str, value: str, value_color: str) -> ctk.CTkFrame:
        """Creates a styled stat card with title and value. Returns frame containing value_label."""
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=16, border_width=1, border_color=BORDER_COLOR)
        
        # Title label
        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=MAIN_FONT,
            text_color=TEXT_SECONDARY
        )
        title_label.pack(anchor="w", padx=16, pady=(12, 4))
        
        # Value label (big number)
        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=HEADER_FONT,
            text_color=value_color
        )
        value_label.pack(anchor="w", padx=12, pady=(0, 10))
        
        # Store value_label reference on the card for later updates
        card.value_label = value_label
        card.title_label = title_label
        
        return card
    
    def _update_stat_card(self, card: ctk.CTkFrame, value: str, color: str = None):
        """Updates the value displayed on a stat card."""
        if hasattr(card, 'value_label'):
            card.value_label.configure(text=value)
            if color:
                card.value_label.configure(text_color=color)

    def _create_widgets(self):
        """Creates and arranges all UI elements with modern dark theme."""
        # Configure main window with dark background
        self.configure(fg_color=BG_DARK)
        
        # Main layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ═══════════════════════════════════════════════════════════════════
        # LEFT SIDEBAR - Modern Dark Theme
        # ═══════════════════════════════════════════════════════════════════
        self.left_frame = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=BG_SIDEBAR, border_width=0)
        self.left_frame.grid(row=0, column=0, rowspan=2, sticky="nsw")
        self.left_frame.grid_propagate(False)
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(20, weight=1)

        current_row = 0
        
        # ─── Logo & App Name ───
        logo_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        logo_frame.grid(row=current_row, column=0, columnspan=2, padx=24, pady=(30, 8), sticky="ew")
        
        # Try to load logo image
        try:
            from PIL import Image
            logo_img = Image.open(LOGO_PATH)
            self.logo_image = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=(42, 42))
            logo_label = ctk.CTkLabel(logo_frame, image=self.logo_image, text="")
            logo_label.pack(side="left", padx=(0, 12))
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")
        
        self.label_title = ctk.CTkLabel(
            logo_frame, 
            text=APP_NAME, 
            font=("Roboto Medium", 24),
            text_color=ACCENT_TEAL
        )
        self.label_title.pack(side="left")
        current_row += 1
        
        # Subtitle
        ctk.CTkLabel(
            self.left_frame, 
            text="AI Trading Assistant", 
            font=MAIN_FONT,
            text_color=TEXT_SECONDARY
        ).grid(row=current_row, column=0, columnspan=2, padx=24, pady=(0, 20), sticky="w")
        current_row += 1
        
        # Separator line (Subtle)
        ctk.CTkFrame(self.left_frame, height=1, fg_color=BORDER_COLOR).grid(
            row=current_row, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew"
        )
        current_row += 1

        # ─── MT5 Connection Section ───
        mt5_section = ctk.CTkFrame(self.left_frame, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        mt5_section.grid(row=current_row, column=0, columnspan=2, padx=16, pady=(0, 12), sticky="ew")
        
        ctk.CTkLabel(
            mt5_section, 
            text="⚡ MetaTrader 5", 
            font=BOLD_FONT,
            text_color=TEXT_PRIMARY
        ).pack(anchor="w", padx=14, pady=(12, 6))
        
        mt5_inner = ctk.CTkFrame(mt5_section, fg_color="transparent")
        mt5_inner.pack(fill="x", padx=14, pady=(0, 12))
        
        self.mt5_status_label = ctk.CTkLabel(
            mt5_inner, 
            text="● Disconnected", 
            text_color=TEXT_SECONDARY,
            font=MAIN_FONT
        )
        self.mt5_status_label.pack(side="left")
        
        self.connect_mt5_button = ctk.CTkButton(
            mt5_inner, 
            text="Connect", 
            width=85, 
            height=30,
            corner_radius=8,
            fg_color=ACCENT_TEAL,
            hover_color="#00dbc4", # Slightly darker cyan
            text_color=BG_DARK,
            font=BOLD_FONT,
            command=self._connect_mt5
        )
        self.connect_mt5_button.pack(side="right")
        current_row += 1

        # ─── Bot Control Section ───
        bot_section = ctk.CTkFrame(self.left_frame, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        bot_section.grid(row=current_row, column=0, columnspan=2, padx=16, pady=(0, 12), sticky="ew")
        
        ctk.CTkLabel(
            bot_section, 
            text="🤖 Bot Control", 
            font=BOLD_FONT,
            text_color=TEXT_PRIMARY
        ).pack(anchor="w", padx=14, pady=(12, 6))
        
        bot_inner = ctk.CTkFrame(bot_section, fg_color="transparent")
        bot_inner.pack(fill="x", padx=14, pady=(0, 12))
        
        self.bot_status_label = ctk.CTkLabel(
            bot_inner, 
            text="● Stopped", 
            text_color=TEXT_SECONDARY,
            font=MAIN_FONT
        )
        self.bot_status_label.pack(side="left")
        
        button_frame = ctk.CTkFrame(bot_inner, fg_color="transparent")
        button_frame.pack(side="right")
        
        self.start_button = ctk.CTkButton(
            button_frame, 
            text="▶ Start", 
            width=70, 
            height=30,
            corner_radius=8,
            fg_color=SUCCESS_GREEN,
            hover_color="#10b981", # Emerald
            text_color="white",
            font=BOLD_FONT,
            command=self.start_bot
        )
        self.start_button.pack(side="left", padx=(0, 6))
        
        self.stop_button = ctk.CTkButton(
            button_frame, 
            text="■ Stop", 
            width=70, 
            height=30,
            corner_radius=8,
            fg_color=DANGER_RED,
            hover_color="#ef4444", # Red
            text_color="white",
            font=BOLD_FONT,
            command=self.stop_bot, 
            state=tk.DISABLED
        )
        self.stop_button.pack(side="left")
        current_row += 1

        # ─── Settings Button ───
        settings_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        settings_frame.grid(row=current_row, column=0, columnspan=2, padx=16, pady=20, sticky="ew")
        
        self.settings_btn = ctk.CTkButton(
            settings_frame,
            text="⚙️ Settings",
            fg_color=BG_CARD,
            hover_color=BG_HOVER,
            width=200,
            command=self.open_settings
        )
        self.settings_btn.pack()
        
        current_row += 1

        # ─── Quick Actions Section ───
        actions_section = ctk.CTkFrame(self.left_frame, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        actions_section.grid(row=current_row, column=0, columnspan=2, padx=16, pady=(5, 12), sticky="ew")
        
        ctk.CTkLabel(
            actions_section, 
            text="🎯 Quick Actions", 
            font=BOLD_FONT,
            text_color=TEXT_PRIMARY
        ).pack(anchor="w", padx=14, pady=(12, 8))
        
        self.clear_signals_button = ctk.CTkButton(
            actions_section, 
            text="🗑️ Clear Signals", 
            height=32,
            corner_radius=8,
            fg_color=BG_HOVER,
            hover_color=BORDER_COLOR,
            border_width=1,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            font=MAIN_FONT,
            command=self._clear_signals_display
        )
        self.clear_signals_button.pack(fill="x", padx=14, pady=(0, 6))
        
        self.clear_logs_button = ctk.CTkButton(
            actions_section, 
            text="📋 Clear Logs", 
            height=32,
            corner_radius=8,
            fg_color=BG_HOVER,
            hover_color=BORDER_COLOR,
            border_width=1,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            font=MAIN_FONT,
            command=self._clear_logs_display
        )
        self.clear_logs_button.pack(fill="x", padx=14, pady=(0, 12))
        current_row += 1

        # ─── RISK MANAGEMENT DASHBOARD ───
        risk_section = ctk.CTkFrame(self.left_frame, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        risk_section.grid(row=current_row, column=0, columnspan=2, padx=16, pady=(5, 12), sticky="ew")
        
        ctk.CTkLabel(
            risk_section, 
            text="🛡️ Risk Management", 
            font=BOLD_FONT,
            text_color=TEXT_PRIMARY
        ).pack(anchor="w", padx=14, pady=(12, 8))
        
        # Daily P&L
        pnl_row = ctk.CTkFrame(risk_section, fg_color="transparent")
        pnl_row.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(pnl_row, text="Daily P&L:", font=MAIN_FONT, text_color=TEXT_SECONDARY).pack(side="left")
        self.daily_pnl_label = ctk.CTkLabel(pnl_row, text="$0.00", font=BOLD_FONT, text_color=TEXT_PRIMARY)
        self.daily_pnl_label.pack(side="right")
        
        # Open Positions
        pos_row = ctk.CTkFrame(risk_section, fg_color="transparent")
        pos_row.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(pos_row, text="Open Positions:", font=MAIN_FONT, text_color=TEXT_SECONDARY).pack(side="left")
        self.open_positions_label = ctk.CTkLabel(pos_row, text="0", font=BOLD_FONT, text_color=TEXT_PRIMARY)
        self.open_positions_label.pack(side="right")
        
        # Exposure / Risk
        risk_row = ctk.CTkFrame(risk_section, fg_color="transparent")
        risk_row.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(risk_row, text="Total Exposure:", font=MAIN_FONT, text_color=TEXT_SECONDARY).pack(side="left")
        self.exposure_label = ctk.CTkLabel(risk_row, text="0.00 lots", font=BOLD_FONT, text_color=TEXT_PRIMARY)
        self.exposure_label.pack(side="right")
        
        # Auto-Trade Status
        auto_row = ctk.CTkFrame(risk_section, fg_color="transparent")
        auto_row.pack(fill="x", padx=14, pady=2)
        ctk.CTkLabel(auto_row, text="Auto-Trade:", font=MAIN_FONT, text_color=TEXT_SECONDARY).pack(side="left")
        self.auto_trade_label = ctk.CTkLabel(auto_row, text="● OFF", font=BOLD_FONT, text_color=TEXT_SECONDARY)
        self.auto_trade_label.pack(side="right")
        
        # Guardian Status
        guardian_row = ctk.CTkFrame(risk_section, fg_color="transparent")
        guardian_row.pack(fill="x", padx=14, pady=(2, 8))
        ctk.CTkLabel(guardian_row, text="Guardian:", font=MAIN_FONT, text_color=TEXT_SECONDARY).pack(side="left")
        self.guardian_label = ctk.CTkLabel(guardian_row, text="● INACTIVE", font=BOLD_FONT, text_color=TEXT_SECONDARY)
        self.guardian_label.pack(side="right")
        
        # Emergency Button
        self.emergency_close_btn = ctk.CTkButton(
            risk_section, 
            text="🚨 CLOSE ALL POSITIONS", 
            height=32,
            corner_radius=8,
            fg_color=DANGER_RED,
            hover_color="#dc2626",
            text_color="white",
            font=BOLD_FONT,
            command=self._emergency_close_all
        )
        self.emergency_close_btn.pack(fill="x", padx=14, pady=(0, 12))
        current_row += 1

        # ═══════════════════════════════════════════════════════════════════
        # RIGHT CONTENT AREA - Tabs with Modern Theme
        # ═══════════════════════════════════════════════════════════════════
        self.tab_view = ctk.CTkTabview(
            self, 
            corner_radius=10,
            fg_color=BG_CARD,
            segmented_button_fg_color=BG_DARK,
            segmented_button_selected_color=ACCENT_TEAL,
            segmented_button_selected_hover_color="#00b894",
            segmented_button_unselected_color=BG_DARK,
            segmented_button_unselected_hover_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            border_width=1,
            border_color=BORDER_COLOR
        )
        self.tab_view.grid(row=0, column=1, padx=(10, 15), pady=(15, 10), sticky="nsew")

        self.tab_view.add("📊 Signals")
        self.tab_view.add("📈 Positions")
        self.tab_view.add("🏆 Performance")
        self.tab_view.add("🎯 Strategies")  # NEW: Strategy Control
        self.tab_view.add("🌍 Market")
        self.tab_view.add("👁️ Insider") # New Tab
        self.tab_view.add("🧪 Backtest") # Backtester Hub
        self.tab_view.add("📜 Logs")
        self.tab_view.set("📊 Signals")


        # ─── Signals Tab ───
        self.signals_scroll_frame = ctk.CTkScrollableFrame(
            self.tab_view.tab("📊 Signals"), 
            label_text="Live Trading Signals",
            label_fg_color=BG_CARD,
            fg_color=BG_DARK,
            corner_radius=8
        )
        self.signals_scroll_frame.pack(expand=True, fill="both", padx=5, pady=5)
        self.signals_scroll_frame.grid_columnconfigure(0, weight=1)
        self.signals_placeholder = ctk.CTkLabel(
            self.signals_scroll_frame, 
            text="⏳ Waiting for signals...", 
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=13)
        )
        self.signals_placeholder.pack(pady=40)

        # ─── Positions Tab - Visual Cards Layout ───
        self.positions_frame = ctk.CTkScrollableFrame(
            self.tab_view.tab("📈 Positions"),
            fg_color=BG_DARK,
            corner_radius=8
        )
        self.positions_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Header row for positions
        self.positions_header = ctk.CTkFrame(self.positions_frame, fg_color=BG_CARD, corner_radius=8)
        self.positions_header.pack(fill="x", padx=0, pady=(0, 10))
        
        ctk.CTkLabel(
            self.positions_header,
            text="📈 Open Positions",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(side="left", padx=12, pady=10)
        
        self.positions_count_label = ctk.CTkLabel(
            self.positions_header,
            text="0 positions",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY
        )
        self.positions_count_label.pack(side="right", padx=12, pady=10)
        
        # Summary stats row
        self.positions_stats_frame = ctk.CTkFrame(self.positions_frame, fg_color="transparent")
        self.positions_stats_frame.pack(fill="x", padx=0, pady=(0, 10))
        self.positions_stats_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Total P/L card
        self.pos_total_pnl_card = self._create_stat_card(self.positions_stats_frame, "💰 Total P/L", "$0.00", TEXT_SECONDARY)
        self.pos_total_pnl_card.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
        
        # Buy positions
        self.pos_buy_card = self._create_stat_card(self.positions_stats_frame, "🟢 Long Positions", "0", SUCCESS_GREEN)
        self.pos_buy_card.grid(row=0, column=1, padx=5, pady=0, sticky="nsew")
        
        # Sell positions
        self.pos_sell_card = self._create_stat_card(self.positions_stats_frame, "🔴 Short Positions", "0", DANGER_RED)
        self.pos_sell_card.grid(row=0, column=2, padx=(5, 0), pady=0, sticky="nsew")
        
        # Container for individual position cards
        self.positions_list_frame = ctk.CTkFrame(self.positions_frame, fg_color="transparent")
        self.positions_list_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Placeholder when no positions
        self.positions_placeholder = ctk.CTkLabel(
            self.positions_list_frame,
            text="📭 No open positions",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_SECONDARY
        )
        self.positions_placeholder.pack(pady=30)

        # ─── Performance Tab - Visual Cards Layout ───
        self.performance_frame = ctk.CTkScrollableFrame(
            self.tab_view.tab("🏆 Performance"),
            fg_color=BG_DARK,
            corner_radius=8
        )
        self.performance_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Top controls row
        self.perf_controls_frame = ctk.CTkFrame(self.performance_frame, fg_color=BG_CARD, corner_radius=8)
        self.perf_controls_frame.pack(fill="x", padx=0, pady=(0, 10))
        
        ctk.CTkLabel(
            self.perf_controls_frame,
            text="📊 Performance Dashboard",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(side="left", padx=12, pady=10)
        
        self.import_history_button = ctk.CTkButton(
            self.perf_controls_frame,
            text="📥 Import",
            command=self._import_historical_trades,
            fg_color=ACCENT_BLUE,
            hover_color="#4c8ed9",
            corner_radius=6,
            height=28,
            width=90
        )
        self.import_history_button.pack(side="right", padx=5, pady=8)
        
        self.export_history_button = ctk.CTkButton(
            self.perf_controls_frame,
            text="📤 Export",
            command=self._export_trade_history,
            fg_color=BG_HOVER,
            hover_color=BORDER_COLOR,
            corner_radius=6,
            height=28,
            width=90
        )
        self.export_history_button.pack(side="right", padx=5, pady=8)
        
        # === MT5 Account Stats Row ===
        self.mt5_stats_frame = ctk.CTkFrame(self.performance_frame, fg_color="transparent")
        self.mt5_stats_frame.pack(fill="x", padx=0, pady=(0, 10))
        self.mt5_stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Balance Card
        self.balance_card = self._create_stat_card(self.mt5_stats_frame, "💰 Balance", "$0.00", SUCCESS_GREEN)
        self.balance_card.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
        
        # Equity Card
        self.equity_card = self._create_stat_card(self.mt5_stats_frame, "💎 Equity", "$0.00", ACCENT_BLUE)
        self.equity_card.grid(row=0, column=1, padx=5, pady=0, sticky="nsew")
        
        # Profit/Loss Card
        self.pnl_card = self._create_stat_card(self.mt5_stats_frame, "📈 Today's P/L", "$0.00", TEXT_SECONDARY)
        self.pnl_card.grid(row=0, column=2, padx=5, pady=0, sticky="nsew")
        
        # Win Rate Card
        self.winrate_card = self._create_stat_card(self.mt5_stats_frame, "🎯 Win Rate", "0%", ACCENT_TEAL)
        self.winrate_card.grid(row=0, column=3, padx=(5, 0), pady=0, sticky="nsew")
        
        # === Today's Trading Stats Row ===
        self.today_stats_frame = ctk.CTkFrame(self.performance_frame, fg_color="transparent")
        self.today_stats_frame.pack(fill="x", padx=0, pady=(0, 10))
        self.today_stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Trades Today
        self.trades_today_card = self._create_stat_card(self.today_stats_frame, "📊 Trades Today", "0", TEXT_PRIMARY)
        self.trades_today_card.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
        
        # Wins
        self.wins_card = self._create_stat_card(self.today_stats_frame, "✅ Wins", "0", SUCCESS_GREEN)
        self.wins_card.grid(row=0, column=1, padx=5, pady=0, sticky="nsew")
        
        # Losses
        self.losses_card = self._create_stat_card(self.today_stats_frame, "❌ Losses", "0", DANGER_RED)
        self.losses_card.grid(row=0, column=2, padx=5, pady=0, sticky="nsew")
        
        # Expectancy
        self.expectancy_card = self._create_stat_card(self.today_stats_frame, "💹 Expectancy", "$0.00", ACCENT_TEAL)
        self.expectancy_card.grid(row=0, column=3, padx=(5, 0), pady=0, sticky="nsew")
        
        # === Bot Performance Section ===
        bot_perf_header = ctk.CTkFrame(self.performance_frame, fg_color=BG_CARD, corner_radius=8)
        bot_perf_header.pack(fill="x", padx=0, pady=(0, 10))
        ctk.CTkLabel(
            bot_perf_header,
            text="🤖 Bot Tracked Performance",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(side="left", padx=12, pady=10)
        
        self.bot_stats_frame = ctk.CTkFrame(self.performance_frame, fg_color="transparent")
        self.bot_stats_frame.pack(fill="x", padx=0, pady=(0, 10))
        self.bot_stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Total Trades
        self.total_trades_card = self._create_stat_card(self.bot_stats_frame, "📊 Total Trades", "0", TEXT_PRIMARY)
        self.total_trades_card.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
        
        # Profit Factor
        self.profit_factor_card = self._create_stat_card(self.bot_stats_frame, "📈 Profit Factor", "0.00", ACCENT_BLUE)
        self.profit_factor_card.grid(row=0, column=1, padx=5, pady=0, sticky="nsew")
        
        # Total P/L
        self.total_pnl_card = self._create_stat_card(self.bot_stats_frame, "💵 Total P/L", "$0.00", TEXT_SECONDARY)
        self.total_pnl_card.grid(row=0, column=2, padx=5, pady=0, sticky="nsew")
        
        # Avg Win
        self.avg_win_card = self._create_stat_card(self.bot_stats_frame, "📊 Avg Win", "$0.00", SUCCESS_GREEN)
        self.avg_win_card.grid(row=0, column=3, padx=(5, 0), pady=0, sticky="nsew")

        # ─── Market Summary Tab - Visual Cards Layout ───
        self.market_frame = ctk.CTkScrollableFrame(
            self.tab_view.tab("🌍 Market"),
            fg_color=BG_DARK,
            corner_radius=8
        )
        self.market_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Header row
        self.market_header = ctk.CTkFrame(self.market_frame, fg_color=BG_CARD, corner_radius=8)
        self.market_header.pack(fill="x", padx=0, pady=(0, 10))
        
        ctk.CTkLabel(
            self.market_header,
            text="🌍 Market Overview",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(side="left", padx=12, pady=10)
        
        self.market_update_label = ctk.CTkLabel(
            self.market_header,
            text="Last update: —",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY
        )
        self.market_update_label.pack(side="right", padx=12, pady=10)
        
        # Market Mood Stats Row
        self.market_stats_frame = ctk.CTkFrame(self.market_frame, fg_color="transparent")
        self.market_stats_frame.pack(fill="x", padx=0, pady=(0, 10))
        self.market_stats_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Overall Mood Card
        self.market_mood_card = self._create_stat_card(self.market_stats_frame, "📊 Market Mood", "—", TEXT_SECONDARY)
        self.market_mood_card.grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")
        
        # Sentiment Score Card
        self.sentiment_score_card = self._create_stat_card(self.market_stats_frame, "📈 Sentiment", "0.00", TEXT_SECONDARY)
        self.sentiment_score_card.grid(row=0, column=1, padx=5, pady=0, sticky="nsew")
        
        # News Count Card
        self.news_count_card = self._create_stat_card(self.market_stats_frame, "📰 News Items", "0", ACCENT_BLUE)
        self.news_count_card.grid(row=0, column=2, padx=(5, 0), pady=0, sticky="nsew")
        
        # Section header for pairs
        pairs_header = ctk.CTkFrame(self.market_frame, fg_color=BG_CARD, corner_radius=8)
        pairs_header.pack(fill="x", padx=0, pady=(0, 10))
        ctk.CTkLabel(
            pairs_header,
            text="💱 Currency Pairs Analysis",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(side="left", padx=12, pady=10)
        
        # Container for currency pair cards
        self.market_pairs_frame = ctk.CTkFrame(self.market_frame, fg_color="transparent")
        self.market_pairs_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Placeholder when no data
        self.market_placeholder = ctk.CTkLabel(
            self.market_pairs_frame,
            text="⏳ Waiting for market data...\nStart the bot to generate market summary.",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_SECONDARY
        )
        self.market_placeholder.pack(pady=30)

        # ─── Strategies Tab ─── NEW
        try:
            self.strategy_manager = StrategyManager()
            self.strategy_control_frame = StrategyControlFrame(
                self.tab_view.tab("🎯 Strategies"),
                strategy_manager=self.strategy_manager,
                on_deploy_strategy=self._on_strategy_deployed
            )
            self.strategy_control_frame.pack(fill="both", expand=True)
        except Exception as e:
            # Fallback if StrategyControlFrame fails to load
            self.strategy_manager = None
            self.strategy_control_frame = None
            ctk.CTkLabel(
                self.tab_view.tab("🎯 Strategies"),
                text=f"Strategy Manager not available: {e}",
                text_color=TEXT_SECONDARY
            ).pack(pady=50)

        # ─── Insiders Tab ───
        self.insider_frame = InsiderTradingFrame(
            self.tab_view.tab("👁️ Insider"),
            execute_callback=lambda pair, direction, volume, use_market_price: self._manual_execute_trade({
                'pair': pair,
                'direction': direction,
                'volume': volume,
                'entry_price': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'timeframe': "Insider"
            })
        )
        self.insider_frame.pack(fill="both", expand=True)

        # ─── Backtest Tab ───
        self.backtester_frame = BacktesterFrame(
            self.tab_view.tab("🧪 Backtest"),
            data_manager=self.data_manager,
            signal_generator=self.signal_generator
        )
        self.backtester_frame.pack(fill="both", expand=True)

        # ─── Logs Tab ───
        self.log_textbox = ctk.CTkTextbox(
            self.tab_view.tab("📜 Logs"), 
            wrap=tk.WORD, 
            corner_radius=8, 
            font=("Consolas", 10),
            fg_color=BG_DARK,
            text_color=TEXT_SECONDARY,
            border_width=1,
            border_color=BORDER_COLOR
        )
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
            self.mt5_status_label.configure(text="● Connected", text_color=SUCCESS_GREEN)
            self.connect_mt5_button.configure(text="Disconnect", fg_color=DANGER_RED, hover_color="#da3633")
            self.status_bar.set_status("✅ MT5 Connected.", duration=10000, priority=False)
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
        """Periodically updates the Open Positions and Performance tabs with visual cards."""
        logger.debug("Updating dashboard tabs (Positions, Performance)...")
        try:
            # Update Open Positions Tab
            self._update_positions_tab()

            # Update Performance Cards with MT5 Account Data
            if self.data_manager and self.data_manager.is_initialized:
                try:
                    acc_info = mt5.account_info()
                    if acc_info:
                        currency = acc_info.currency
                        
                        # Today's stats from MT5 deals
                        today_stats = self._get_mt5_today_stats()

                        # Update MT5 Account Cards
                        self._update_stat_card(self.balance_card, f"${acc_info.balance:,.2f}", SUCCESS_GREEN)
                        self._update_stat_card(self.equity_card, f"${acc_info.equity:,.2f}", ACCENT_BLUE)
                        
                        # P/L color based on positive/negative (Using Realized Day Profit)
                        realized_day_pl = today_stats['profit']
                        pnl_color = SUCCESS_GREEN if realized_day_pl >= 0 else DANGER_RED
                        self._update_stat_card(self.pnl_card, f"${realized_day_pl:+,.2f}", pnl_color)
                        
                        # Win Rate
                        if today_stats['trades_count'] > 0:
                            today_wr = (today_stats['wins'] / today_stats['trades_count']) * 100
                            wr_color = SUCCESS_GREEN if today_wr >= 50 else WARNING_YELLOW
                            self._update_stat_card(self.winrate_card, f"{today_wr:.0f}%", wr_color)
                        else:
                            self._update_stat_card(self.winrate_card, "—", TEXT_SECONDARY)
                        
                        # Today Stats Cards
                        self._update_stat_card(self.trades_today_card, str(today_stats['trades_count']), TEXT_PRIMARY)
                        self._update_stat_card(self.wins_card, str(today_stats['wins']), SUCCESS_GREEN)
                        self._update_stat_card(self.losses_card, str(today_stats['losses']), DANGER_RED)
                        
                        # Expectancy
                        if today_stats['trades_count'] > 0:
                            today_exp = today_stats['profit'] / today_stats['trades_count']
                            exp_color = SUCCESS_GREEN if today_exp >= 0 else DANGER_RED
                            self._update_stat_card(self.expectancy_card, f"${today_exp:+,.2f}", exp_color)
                        else:
                            self._update_stat_card(self.expectancy_card, "—", TEXT_SECONDARY)
                            
                except Exception as mt5_e:
                    logger.warning(f"MT5 card update error: {mt5_e}")
            else:
                # MT5 not connected - show placeholder
                self._update_stat_card(self.balance_card, "—", TEXT_SECONDARY)
                self._update_stat_card(self.equity_card, "—", TEXT_SECONDARY)
                self._update_stat_card(self.pnl_card, "—", TEXT_SECONDARY)
                self._update_stat_card(self.winrate_card, "—", TEXT_SECONDARY)
            
            # Update Bot Tracker Cards
            if hasattr(self, 'tracker') and self.tracker:
                metrics = self.tracker.get_performance_metrics()
                total_trades = metrics.get('total_trades', 0)
                
                if total_trades > 0:
                    win_rate = metrics.get('win_rate', 0.0) * 100
                    profit_factor = metrics.get('profit_factor') # Can be None (Inf)
                    total_profit = metrics.get('total_profit_currency', 0.0)
                    avg_win = metrics.get('average_win_currency', 0.0)
                    
                    self._update_stat_card(self.total_trades_card, str(total_trades), TEXT_PRIMARY)
                    
                    if profit_factor is None:
                         self._update_stat_card(self.profit_factor_card, "Inf", SUCCESS_GREEN)
                    else:
                         pf_color = SUCCESS_GREEN if profit_factor >= 1.0 else DANGER_RED
                         self._update_stat_card(self.profit_factor_card, f"{profit_factor:.2f}", pf_color)
                    
                    pnl_color = SUCCESS_GREEN if total_profit >= 0 else DANGER_RED
                    self._update_stat_card(self.total_pnl_card, f"${total_profit:+,.2f}", pnl_color)
                    
                    self._update_stat_card(self.avg_win_card, f"${avg_win:+,.2f}", SUCCESS_GREEN)
                else:
                    self._update_stat_card(self.total_trades_card, "0", TEXT_SECONDARY)
                    self._update_stat_card(self.profit_factor_card, "—", TEXT_SECONDARY)
                    self._update_stat_card(self.total_pnl_card, "$0.00", TEXT_SECONDARY)
                    self._update_stat_card(self.avg_win_card, "$0.00", TEXT_SECONDARY)

        except Exception as e:
             logger.exception(f"Error updating dashboard tabs: {e}")
        
        # Reschedule the next dashboard update
        self.after(DASHBOARD_UPDATE_INTERVAL_MS, self._update_dashboard_tabs)
    
    def _on_strategy_deployed(self, strategy_name: str):
        """Callback when a strategy is deployed for live trading."""
        logger.info(f"🚀 Strategy deployed for live trading: {strategy_name}")
        if hasattr(self, 'status_bar') and self.status_bar:
            self.status_bar.set_status(f"✅ Strategy '{strategy_name}' deployed", duration=5000)
    
    def _get_mt5_today_stats(self) -> Dict:
        """Gets today's trading statistics from MT5 deal history."""
        stats = {'trades_count': 0, 'wins': 0, 'losses': 0, 'profit': 0.0}
        try:
            # Get today's date range
            from datetime import timezone
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            # Get today's deals
            deals = mt5.history_deals_get(today, tomorrow)
            if deals is None or len(deals) == 0:
                return stats
            
            # Track positions we've seen
            positions_profit = {}
            
            for deal in deals:
                # Skip deposits/withdrawals (position_id == 0)
                if deal.position_id == 0:
                    continue
                    
                # Only count closing deals
                if deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                    pos_id = deal.position_id
                    if pos_id not in positions_profit:
                        positions_profit[pos_id] = 0.0
                    positions_profit[pos_id] += deal.profit + deal.commission + deal.swap
            
            # Count wins/losses
            for pos_id, profit in positions_profit.items():
                stats['trades_count'] += 1
                stats['profit'] += profit
                if profit >= 0:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1
                    
        except Exception as e:
            logger.warning(f"Error fetching today's MT5 stats: {e}")
        
        return stats
    # ---> END NEW DASHBOARD UPDATE FUNCTION <---
    
    def _update_positions_tab(self):
        """Updates the positions tab with visual cards - updates in-place to prevent flickering."""
        try:
            # Initialize position cards dict if not exists
            if not hasattr(self, '_position_cards'):
                self._position_cards = {}  # ticket -> (card_frame, pnl_label, details_label, current_price_label)
            
            # Check MT5 connection
            if not self.data_manager or not self.data_manager.is_initialized:
                self._update_stat_card(self.pos_total_pnl_card, "—", TEXT_SECONDARY)
                self._update_stat_card(self.pos_buy_card, "—", TEXT_SECONDARY)
                self._update_stat_card(self.pos_sell_card, "—", TEXT_SECONDARY)
                return
            
            if not mt5.terminal_info():
                return
            
            # Get open positions from MT5
            positions = mt5.positions_get()
            
            if positions is None or len(positions) == 0:
                # No positions - clear all cards
                for ticket, card_data in list(self._position_cards.items()):
                    if card_data[0].winfo_exists():
                        card_data[0].destroy()
                self._position_cards.clear()
                
                self.positions_count_label.configure(text="0 positions")
                self._update_stat_card(self.pos_total_pnl_card, "$0.00", TEXT_SECONDARY)
                self._update_stat_card(self.pos_buy_card, "0", TEXT_SECONDARY)
                self._update_stat_card(self.pos_sell_card, "0", TEXT_SECONDARY)
                
                if hasattr(self, 'positions_placeholder') and self.positions_placeholder.winfo_exists():
                    if not self.positions_placeholder.winfo_ismapped():
                        self.positions_placeholder.pack(pady=30)
                return
            
            # Hide placeholder if we have positions
            if hasattr(self, 'positions_placeholder') and self.positions_placeholder.winfo_exists():
                if self.positions_placeholder.winfo_ismapped():
                    self.positions_placeholder.pack_forget()
            
            # Track current position tickets
            current_tickets = {pos.ticket for pos in positions}
            
            # Remove cards for closed positions
            for ticket in list(self._position_cards.keys()):
                if ticket not in current_tickets:
                    card_data = self._position_cards.pop(ticket)
                    if card_data[0].winfo_exists():
                        card_data[0].destroy()
            
            # Calculate stats and update/create cards
            total_pnl = 0.0
            buy_count = 0
            sell_count = 0
            
            for pos in positions:
                total_pnl += pos.profit
                if pos.type == 0:
                    buy_count += 1
                else:
                    sell_count += 1
                
                if pos.ticket in self._position_cards:
                    # Update existing card
                    self._update_position_card_values(pos)
                else:
                    # Create new card
                    self._create_position_card(pos)
            
            # Update summary cards
            self.positions_count_label.configure(text=f"{len(positions)} position{'s' if len(positions) != 1 else ''}")
            pnl_color = SUCCESS_GREEN if total_pnl >= 0 else DANGER_RED
            self._update_stat_card(self.pos_total_pnl_card, f"${total_pnl:+,.2f}", pnl_color)
            self._update_stat_card(self.pos_buy_card, str(buy_count), SUCCESS_GREEN if buy_count > 0 else TEXT_SECONDARY)
            self._update_stat_card(self.pos_sell_card, str(sell_count), DANGER_RED if sell_count > 0 else TEXT_SECONDARY)
            
        except Exception as e:
            logger.exception(f"Error updating positions tab: {e}")
    
    def _update_position_card_values(self, position):
        """Updates only the changing values (P/L, current price) of an existing position card."""
        if position.ticket not in self._position_cards:
            return
        
        card_data = self._position_cards[position.ticket]
        pnl_label = card_data[1]
        details_label = card_data[2]
        
        # Update P/L
        pnl_color = SUCCESS_GREEN if position.profit >= 0 else DANGER_RED
        pnl_label.configure(text=f"${position.profit:+,.2f}", text_color=pnl_color)
        
        # Update details with current price
        details_text = f"#{position.ticket} | Vol: {position.volume} | Open: {position.price_open:.5f} | Current: {position.price_current:.5f}"
        if position.sl > 0:
            details_text += f" | SL: {position.sl:.5f}"
        if position.tp > 0:
            details_text += f" | TP: {position.tp:.5f}"
        details_label.configure(text=details_text)
    
    def _create_position_card(self, position):
        """Creates a visual card for a single position with close button."""
        is_buy = position.type == 0
        direction = "BUY" if is_buy else "SELL"
        border_color = SUCCESS_GREEN if is_buy else DANGER_RED
        pnl_color = SUCCESS_GREEN if position.profit >= 0 else DANGER_RED
        ticket = position.ticket
        
        # Main card frame
        card = ctk.CTkFrame(self.positions_list_frame, fg_color=BG_CARD, corner_radius=16, border_width=1, border_color=BORDER_COLOR)
        card.pack(fill="x", padx=0, pady=(0, 10))
        
        # Left border indicator (Thicker and rounded)
        indicator = ctk.CTkFrame(card, width=5, fg_color=border_color, corner_radius=4)
        indicator.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 0), pady=10)
        
        # Content frame
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.grid(row=0, column=1, sticky="nsew", padx=12, pady=10)
        content.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)
        
        # Top row: Symbol, Direction, P/L, Close button
        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x")
        top_row.grid_columnconfigure(0, weight=1)
        
        symbol_label = ctk.CTkLabel(
            top_row,
            text=f"{position.symbol}",
            font=SUBHEADER_FONT,
            text_color=TEXT_PRIMARY
        )
        symbol_label.grid(row=0, column=0, sticky="w")
        
        direction_badge = ctk.CTkLabel(
            top_row,
            text=f"{'🟢' if is_buy else '🔴'} {direction}",
            font=BOLD_FONT,
            text_color=border_color
        )
        direction_badge.grid(row=0, column=1, sticky="e", padx=(12, 0))
        
        # Profit/Loss display
        pnl_label = ctk.CTkLabel(
            top_row,
            text=f"${position.profit:+,.2f}",
            font=HEADER_FONT,
            text_color=pnl_color
        )
        pnl_label.grid(row=0, column=2, sticky="e", padx=(16, 12))
        
        # Close button
        close_btn = ctk.CTkButton(
            top_row,
            text="✕ Close",
            width=80,
            height=28,
            corner_radius=8,
            fg_color=DANGER_RED,
            hover_color="#ef4444",
            text_color="white",
            font=BOLD_FONT,
            command=lambda t=ticket, s=position.symbol, v=position.volume, d=direction: self._close_position(t, s, v, d)
        )
        close_btn.grid(row=0, column=3, sticky="e", padx=(0, 0))
        
        # Bottom row: Details with ticket
        details_text = f"#{ticket} | Vol: {position.volume} | Open: {position.price_open:.5f} | Current: {position.price_current:.5f}"
        if position.sl > 0:
            details_text += f" | SL: {position.sl:.5f}"
        if position.tp > 0:
            details_text += f" | TP: {position.tp:.5f}"
        
        details_label = ctk.CTkLabel(
            content,
            text=details_text,
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY
        )
        details_label.pack(fill="x", anchor="w", pady=(4, 0))
        
        # Store references for in-place updates
        self._position_cards[ticket] = (card, pnl_label, details_label)
    
    def _close_position(self, ticket: int, symbol: str, volume: float, direction: str):
        """Closes an open position by ticket."""
        if not messagebox.askyesno("Confirm Close", 
            f"Close {direction} position for {symbol}?\n\nTicket: {ticket}\nVolume: {volume}"):
            return
        
        try:
            # Determine order type (opposite of position)
            order_type = mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                messagebox.showerror("Error", f"Could not get price for {symbol}")
                return
            
            price = tick.bid if direction == "BUY" else tick.ask
            
            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "TheQuanta close",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            
            logger.info(f"Closing position {ticket}: {request}")
            result = mt5.order_send(request)
            
            if result is None:
                messagebox.showerror("Error", f"Failed to close position.\nError: {mt5.last_error()}")
                return
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position {ticket} closed successfully")
                self.status_bar.set_status(f"Position {ticket} closed", duration=3000)
                # Force immediate update
                self._update_positions_tab()
            else:
                messagebox.showerror("Close Failed", 
                    f"Failed to close position.\n\nError: {result.retcode}\n{result.comment}")
                logger.error(f"Close position failed: {result.retcode} - {result.comment}")
                
        except Exception as e:
            logger.exception(f"Error closing position: {e}")
            messagebox.showerror("Error", f"Error closing position:\n{e}")



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
            widgets = self.last_signals[signal_key]
            
            # Check if this is the new dict format
            if isinstance(widgets, dict) and 'entry_time_label' in widgets:
                widgets['entry_time_label'].configure(text=current_time)
            
            # Legacy tuple check (fallback/temp)
            elif isinstance(widgets, tuple) and len(widgets) >= 5:
                 pass # Legacy update logic removed to force new format
        
        # Schedule the next update in 0.1 seconds if still active
        if signal_data.get('entry_time_thread_active', False):
            self.after(100, lambda: self._update_entry_time(signal_key, signal_data))

    def _update_signals_display(self, signal_key: str, signal_data: Dict):
        """Adds or updates a signal frame with a modern Cyber/Teal card design."""
        if hasattr(self, 'signals_placeholder') and self.signals_placeholder.winfo_exists():
            self.signals_placeholder.pack_forget()
            del self.signals_placeholder
            
        # Clean up old signal if it exists
        if signal_key in self.last_signals:
            widgets = self.last_signals[signal_key]
            if isinstance(widgets, tuple): widgets[0].destroy()
            elif isinstance(widgets, dict): widgets['frame'].destroy()
            del self.last_signals[signal_key]

        # Extract data
        pair = signal_data.get('pair', '?')
        timeframe = signal_data.get('timeframe', '?')
        direction = signal_data.get('direction', 'HOLD')
        recommendation = signal_data.get('advanced_analysis', {}).get('recommendation', direction)
        
        # Colors & Visuals
        if "BUY" in recommendation:
            accent_color = SUCCESS_GREEN
            badge_text = "BUY"
            if "STRONG" in recommendation: badge_text = "STRONG BUY"
            elif "WEAK" in recommendation: badge_text = "WEAK BUY"
        elif "SELL" in recommendation:
            accent_color = DANGER_RED
            badge_text = "SELL"
            if "STRONG" in recommendation: badge_text = "STRONG SELL"
            elif "WEAK" in recommendation: badge_text = "WEAK SELL"
        else:
            accent_color = WARNING_YELLOW
            badge_text = "WAIT"

        # --- CARD CONTAINER ---
        card = ctk.CTkFrame(self.signals_scroll_frame, fg_color=BG_CARD, corner_radius=16, border_width=1, border_color=BORDER_COLOR)
        card.pack(fill="x", padx=10, pady=(0, 15))
        
        # --- HEADER ROW (Pair + Badge) ---
        header_frame = ctk.CTkFrame(card, fg_color="transparent")
        header_frame.pack(fill="x", padx=16, pady=(16, 12))
        
        # Left: Pair & Timeframe
        pair_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        pair_frame.pack(side="left")
        
        ctk.CTkLabel(pair_frame, text=pair, font=("Roboto Medium", 18), text_color=TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(pair_frame, text=f"{timeframe}m Timeframe", font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY).pack(anchor="w")

        # Right: Signal Badge
        badge = ctk.CTkFrame(header_frame, fg_color="transparent", border_width=1, border_color=accent_color, corner_radius=20)
        badge.pack(side="right")
        
        ctk.CTkLabel(badge, text=f"  {badge_text}  ", font=BOLD_FONT, text_color=accent_color).pack(padx=8, pady=4)

        # --- GRID STATS ROW ---
        grid = ctk.CTkFrame(card, fg_color=BG_DARK, corner_radius=10) # Inner darker container
        grid.pack(fill="x", padx=16, pady=(0, 12))
        grid.grid_columnconfigure((0,1,2,3), weight=1)
        
        # Helper to create stat box
        def create_grid_item(col, label, value, color=TEXT_PRIMARY):
            f = ctk.CTkFrame(grid, fg_color="transparent")
            f.grid(row=0, column=col, pady=10, sticky="ew")
            ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=10), text_color=TEXT_SECONDARY).pack()
            l = ctk.CTkLabel(f, text=str(value), font=BOLD_FONT, text_color=color)
            l.pack()
            return l

        entry_val = f"{signal_data.get('entry_price', 0.0):.5f}"
        sl_val = f"{signal_data.get('stop_loss', 0.0):.5f}"
        tp_val = f"{signal_data.get('take_profit', 0.0):.5f}"
        conf_val = f"{signal_data.get('confidence', 0.0):.0%}"
        
        create_grid_item(0, "ENTRY", entry_val, TEXT_PRIMARY)
        create_grid_item(1, "STOP LOSS", sl_val, DANGER_RED)
        create_grid_item(2, "TAKE PROFIT", tp_val, SUCCESS_GREEN)
        create_grid_item(3, "CONFIDENCE", conf_val, ACCENT_BLUE)
        
        # --- DETAILS ROW ---
        details_frame = ctk.CTkFrame(card, fg_color="transparent")
        details_frame.pack(fill="x", padx=16, pady=(0, 12))
        
        # Entry Time (from Order Flow or --:--:--)
        time_row = ctk.CTkFrame(details_frame, fg_color="transparent")
        time_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(time_row, text="🕒 Entry Time: ", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(side="left")
        
        # Get entry time from signal data
        entry_time_val = signal_data.get('entry_time')
        if entry_time_val:
            try:
                from datetime import datetime
                if hasattr(entry_time_val, 'strftime'):
                    entry_time_str = entry_time_val.strftime('%H:%M:%S')
                else:
                    entry_time_str = str(entry_time_val)[:8]
            except:
                entry_time_str = "--:--:--"
        else:
            entry_time_str = "--:--:--"
        
        entry_time_label = ctk.CTkLabel(time_row, text=entry_time_str, font=ctk.CTkFont(size=11, family="Consolas"), text_color=TEXT_PRIMARY)
        entry_time_label.pack(side="left")

        # --- ORDER FLOW STATUS ---
        orderflow_data = signal_data.get('orderflow', {})
        of_status = orderflow_data.get('status', 'NEUTRAL')
        
        of_row = ctk.CTkFrame(details_frame, fg_color="transparent")
        of_row.pack(fill="x", pady=(0, 4))
        
        # Determine emoji and color
        if of_status == "CONFIRM":
            of_emoji = "🟢"
            of_color = SUCCESS_GREEN
            of_text = "Order Flow: CONFIRMS"
        elif of_status == "OPPOSE":
            of_emoji = "🔴"
            of_color = DANGER_RED
            of_text = "Order Flow: OPPOSES"
        else:
            of_emoji = "🟡"
            of_color = WARNING_YELLOW
            of_text = "Order Flow: NEUTRAL"
        
        ctk.CTkLabel(of_row, text=f"{of_emoji} {of_text}", font=ctk.CTkFont(size=11), text_color=of_color).pack(side="left")
        
        # Show reason if available
        of_reasons = orderflow_data.get('reasons', [])
        if of_reasons and len(of_reasons) > 0:
            reason_text = of_reasons[0][:40] + "..." if len(of_reasons[0]) > 40 else of_reasons[0]
            ctk.CTkLabel(of_row, text=f"  ({reason_text})", font=ctk.CTkFont(size=10), text_color=TEXT_MUTED).pack(side="left")

        # News/Sentiment
        adv = signal_data.get('advanced_analysis', {})
        sent_score = adv.get('sentiment', {}).get('sentiment_score', 0)
        sent_emoji = "📈" if sent_score > 0.1 else "📉" if sent_score < -0.1 else "➡️"
        news_text = f"{sent_emoji} News Sentiment: {adv.get('sentiment', {}).get('sentiment', 'neutral').upper()} ({sent_score:+.2f})"
        
        ctk.CTkLabel(details_frame, text=news_text, font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(anchor="w")
        
        # Strategy
        strategy = signal_data.get('entry_recommendation', {}).get('best_strategy', 'Standard')
        ctk.CTkLabel(details_frame, text=f"⚙️ Strategy: {strategy}", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED).pack(anchor="w", pady=(2, 6))

        # --- SIGNAL GRADE & ENHANCED INFO ---
        grade = signal_data.get('grade', 'C')
        grade_score = signal_data.get('grade_score', 0)
        grade_action = signal_data.get('grade_action', 'WAIT')
        
        grade_colors = {'A+': SUCCESS_GREEN, 'A': SUCCESS_GREEN, 'B': WARNING_YELLOW, 'C': DANGER_RED}
        grade_color = grade_colors.get(grade, TEXT_MUTED)
        
        grade_row = ctk.CTkFrame(details_frame, fg_color="transparent")
        grade_row.pack(fill="x", pady=(0, 4))
        
        ctk.CTkLabel(grade_row, text=f"📊 Signal Grade: ", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(side="left")
        ctk.CTkLabel(grade_row, text=f"{grade}", font=ctk.CTkFont(size=13, weight="bold"), text_color=grade_color).pack(side="left")
        ctk.CTkLabel(grade_row, text=f" ({grade_score:.0%}) → {grade_action}", font=ctk.CTkFont(size=10), text_color=TEXT_MUTED).pack(side="left")
        
        # Session & MTF Confluence
        session_info = signal_data.get('session', {})
        mtf_info = signal_data.get('mtf_confluence', {})
        
        session_row = ctk.CTkFrame(details_frame, fg_color="transparent")
        session_row.pack(fill="x", pady=(0, 4))
        
        active_sessions = session_info.get('active', [])
        session_text = ', '.join(active_sessions[:2]) if active_sessions else 'None'
        session_quality = session_info.get('quality', 0.5)
        session_emoji = "🌟" if session_quality >= 0.8 else ("✨" if session_quality >= 0.6 else "⭐")
        
        ctk.CTkLabel(session_row, text=f"{session_emoji} Session: {session_text}", font=ctk.CTkFont(size=10), text_color=TEXT_SECONDARY).pack(side="left")
        
        mtf_align = mtf_info.get('alignment', 'UNKNOWN')
        mtf_emoji = "✅" if 'ALIGNED' in mtf_align else ("⚠️" if 'PARTIAL' in mtf_align else "❌")
        ctk.CTkLabel(session_row, text=f"  |  {mtf_emoji} MTF: {mtf_align}", font=ctk.CTkFont(size=10), text_color=TEXT_SECONDARY).pack(side="left")
        
        # Market Structure & Trend
        struct_info = signal_data.get('market_structure', {})
        trend_strength = signal_data.get('trend_strength', 'UNKNOWN')
        
        struct_row = ctk.CTkFrame(details_frame, fg_color="transparent")
        struct_row.pack(fill="x", pady=(0, 4))
        
        structure = struct_info.get('structure', 'NEUTRAL')
        struct_emoji = "📈" if 'BULLISH' in structure else ("📉" if 'BEARISH' in structure else "➡️")
        
        ctk.CTkLabel(struct_row, text=f"{struct_emoji} Structure: {structure}", font=ctk.CTkFont(size=10), text_color=TEXT_SECONDARY).pack(side="left")
        ctk.CTkLabel(struct_row, text=f"  |  🔥 Trend: {trend_strength}", font=ctk.CTkFont(size=10), text_color=TEXT_SECONDARY).pack(side="left")
        
        # Multiple Take Profits Display
        take_profits = signal_data.get('take_profits', {})
        if take_profits:
            tp_row = ctk.CTkFrame(details_frame, fg_color=BG_HOVER, corner_radius=6)
            tp_row.pack(fill="x", pady=(4, 4))
            
            tp1 = take_profits.get('tp1', {}).get('price', 'N/A')
            tp2 = take_profits.get('tp2', {}).get('price', 'N/A')
            tp3 = take_profits.get('tp3', {}).get('price', 'N/A')
            
            tp_inner = ctk.CTkFrame(tp_row, fg_color="transparent")
            tp_inner.pack(fill="x", padx=8, pady=4)
            
            ctk.CTkLabel(tp_inner, text=f"🎯 TP1: {tp1} (50%)", font=ctk.CTkFont(size=10), text_color=SUCCESS_GREEN).pack(side="left", padx=(0, 8))
            ctk.CTkLabel(tp_inner, text=f"🎯 TP2: {tp2} (30%)", font=ctk.CTkFont(size=10), text_color=SUCCESS_GREEN).pack(side="left", padx=(0, 8))
            ctk.CTkLabel(tp_inner, text=f"🎯 TP3: {tp3} (20%)", font=ctk.CTkFont(size=10), text_color=SUCCESS_GREEN).pack(side="left")

        # --- TECHNICAL ANALYSIS ROW (Scrolling Tags) ---
        tech_scroll = ctk.CTkScrollableFrame(details_frame, height=35, orientation="horizontal", fg_color="transparent")
        tech_scroll.pack(fill="x", pady=(0, 2))
        
        # Extract technicals for display
        details = signal_data.get('details', {})
        tech_score = signal_data.get('technical_score', 0.0)
        ml_conf = signal_data.get('ml_confidence')
        
        # Helper for tech tags
        def add_tech_tag(label, value, color=BG_HOVER):
            t = ctk.CTkFrame(tech_scroll, fg_color=color, corner_radius=6)
            t.pack(side="left", padx=(0, 6))
            ctk.CTkLabel(t, text=f"{label}: {value}", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_SECONDARY).pack(padx=6, pady=2)

        # Add tags
        add_tech_tag("Strength", f"{tech_score:.1f}/5.0", color=BG_DARK)
        if ml_conf is not None:
             add_tech_tag("AI Conf", f"{ml_conf:.0%}", color=BG_DARK)
             
        add_tech_tag("RSI", details.get('rsi', '?'))
        add_tech_tag("MACD", details.get('macd_h', '?'))
        add_tech_tag("ADX", details.get('adx', '?'))
        add_tech_tag("ATR", details.get('atr%', '?'))

        # --- FOOTER (ACTION) ---
        show_exec = "STRONG" not in recommendation 
        
        if show_exec:
            footer = ctk.CTkFrame(card, fg_color="transparent")
            footer.pack(fill="x", padx=16, pady=(4, 16))
            
            exec_btn = ctk.CTkButton(
                footer,
                text="⚡ Execute Trade",
                font=BOLD_FONT,
                fg_color=accent_color,
                hover_color=accent_color,
                height=32,
                corner_radius=8,
                command=lambda s=signal_data: self._manual_execute_trade(s)
            )
            exec_btn.pack(fill="x")
        else:
            ctk.CTkFrame(card, fg_color="transparent", height=10).pack()

        # Save widget references for updates
        self.last_signals[signal_key] = {
            "frame": card,
            "entry_time_label": entry_time_label
        }
        
        if not hasattr(self, 'signal_widgets'):
            self.signal_widgets = []
            
        self.signal_widgets.append(card)

        # Enforce display limit
        limit = self.config.get("bot_settings", {}).get("signal_display_limit", 30)
        if len(self.signal_widgets) > limit:
            widget_to_remove = self.signal_widgets.pop(0)
            
            # Clean up from last_signals map too
            key_to_remove = None
            for key, widgets in list(self.last_signals.items()):
                # Handle both new dict and old tuple formats
                if isinstance(widgets, dict) and widgets.get('frame') == widget_to_remove:
                    key_to_remove = key
                    break
                elif isinstance(widgets, tuple) and widgets[0] == widget_to_remove:
                    key_to_remove = key
                    break
            
            if key_to_remove:
                del self.last_signals[key_to_remove]
                
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
            
            # Update entry price with live data if missing (for manual execution)
            if signal_data.get('entry_price', 0.0) == 0.0:
                tick = mt5.symbol_info_tick(pair)
                if tick:
                    current_price = tick.ask if direction == "BUY" else tick.bid
                    signal_data['entry_price'] = current_price

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
                                    self.tab_view.set("📈 Positions")
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
                        self.tab_view.set("📈 Positions")
                    
                    elif result.retcode == 10027: # AUTO_TRADING_DISABLED
                        logger.error("MT5 Error 10027: AutoTrading disabled by client")
                        messagebox.showerror("Algo Trading Disabled", 
                            "⚠️ Failed to execute trade: Algo Trading is disabled in MT5.\n\n"
                            "To fix this:\n"
                            "1. Go to your MetaTrader 5 terminal.\n"
                            "2. Look for the 'Algo Trading' button in the top toolbar.\n"
                            "3. Click it to enable it (should turn green/prominent).\n"
                            "4. Try executing the trade again.")
                            
                    elif result.retcode == 10016: # INVALID_STOPS
                        logger.error("MT5 Error 10016: Invalid Stops")
                        messagebox.showerror("Invalid Stops", 
                            f"⚠️ Failed to execute trade: Invalid Stop Loss or Take Profit.\n\n"
                            f"Entry: {price}\n"
                            f"SL: {sl}\n"
                            f"TP: {tp}\n\n"
                            "Ensure SL/TP levels are valid relative to current price and minimum stop distance.")

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
        """Updates the market summary tab with visual cards."""
        logger.debug("Updating market summary display with visual cards.")
        try:
            # Update timestamp
            ts = summary_data.get('timestamp', 'N/A')
            try: 
                ts_dt = datetime.fromisoformat(ts).astimezone(self.signal_generator.timezone)
                ts_str = ts_dt.strftime('%H:%M:%S')
            except: 
                ts_str = ts
            self.market_update_label.configure(text=f"Updated: {ts_str}")
            
            # Market Mood with color
            mood = summary_data.get('market_mood', 'neutral')
            mood_display = mood.replace('_', ' ').title()
            if 'bullish' in mood.lower():
                mood_color = SUCCESS_GREEN
            elif 'bearish' in mood.lower():
                mood_color = DANGER_RED
            else:
                mood_color = WARNING_YELLOW
            self._update_stat_card(self.market_mood_card, mood_display, mood_color)
            
            # Calculate overall sentiment from pairs
            pairs_data = summary_data.get('pairs', {})
            total_sentiment = 0.0
            total_news = 0
            for pair, data in pairs_data.items():
                sentiment_info = data.get('sentiment', {})
                total_sentiment += sentiment_info.get('sentiment_score', 0)
                total_news += sentiment_info.get('news_count', 0)
            
            avg_sentiment = total_sentiment / len(pairs_data) if pairs_data else 0.0
            sent_color = SUCCESS_GREEN if avg_sentiment > 0.05 else DANGER_RED if avg_sentiment < -0.05 else TEXT_SECONDARY
            self._update_stat_card(self.sentiment_score_card, f"{avg_sentiment:+.2f}", sent_color)
            self._update_stat_card(self.news_count_card, str(total_news), ACCENT_BLUE)
            
            # Clear existing pair cards
            for widget in self.market_pairs_frame.winfo_children():
                if widget != self.market_placeholder:
                    widget.destroy()
            
            if pairs_data:
                # Hide placeholder
                if self.market_placeholder.winfo_ismapped():
                    self.market_placeholder.pack_forget()
                
                # Create pair cards
                for pair, data in pairs_data.items():
                    self._create_market_pair_card(pair, data)
            else:
                # Show placeholder
                if not self.market_placeholder.winfo_ismapped():
                    self.market_placeholder.pack(pady=30)
                    
        except Exception as e:
            logger.exception(f"Error updating market summary display: {e}")
    
    def _create_market_pair_card(self, pair: str, data: Dict):
        """Creates a visual card for a currency pair's market analysis."""
        sentiment_info = data.get('sentiment', {})
        sent_label = sentiment_info.get('sentiment', 'neutral').upper()
        sent_score = sentiment_info.get('sentiment_score', 0)
        news_count = sentiment_info.get('news_count', 0)
        
        # Trade recommendation
        trade_rec = data.get('trade_recommendation', {})
        action = trade_rec.get('action', 'HOLD')
        
        # Determine colors
        if 'BUY' in action:
            border_color = SUCCESS_GREEN
            action_text = f"🟢 {action.replace('_', ' ')}"
        elif 'SELL' in action:
            border_color = DANGER_RED
            action_text = f"🔴 {action.replace('_', ' ')}"
        else:
            border_color = WARNING_YELLOW
            action_text = f"⚪ HOLD"
        
        # Main card
        card = ctk.CTkFrame(self.market_pairs_frame, fg_color=BG_CARD, corner_radius=16, border_width=1, border_color=BORDER_COLOR)
        card.pack(fill="x", padx=0, pady=(0, 10))
        
        # Left indicator
        indicator = ctk.CTkFrame(card, width=5, fg_color=border_color, corner_radius=4)
        indicator.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 0), pady=10)
        
        # Content
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.grid(row=0, column=1, sticky="nsew", padx=12, pady=10)
        content.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)
        
        # Top row: Pair and action
        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x")
        top_row.grid_columnconfigure(0, weight=1)
        
        pair_label = ctk.CTkLabel(
            top_row,
            text=pair,
            font=SUBHEADER_FONT,
            text_color=TEXT_PRIMARY
        )
        pair_label.grid(row=0, column=0, sticky="w")
        
        action_label = ctk.CTkLabel(
            top_row,
            text=action_text,
            font=BOLD_FONT,
            text_color=border_color
        )
        action_label.grid(row=0, column=1, sticky="e")
        
        # Sentiment info
        sent_emoji = "📈" if sent_score > 0.05 else "📉" if sent_score < -0.05 else "➡️"
        
        details_label = ctk.CTkLabel(
            content,
            text=f"{sent_emoji} Sentiment: {sent_label} ({sent_score:+.2f}) | 📰 {news_count} news",
            font=MAIN_FONT,
            text_color=TEXT_SECONDARY
        )
        details_label.pack(fill="x", anchor="w", pady=(4, 0))


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
        # self.entry_pairs.configure(state=tk.DISABLED) # Widget Removed
        # self.entry_timeframes.configure(state=tk.DISABLED) # Widget Removed
        # self.save_config_button.configure(state=tk.DISABLED) # Widget Removed

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
         # self.entry_pairs.configure(state=tk.NORMAL) # Widget Removed
         # self.entry_timeframes.configure(state=tk.NORMAL) # Widget Removed
         # self.save_config_button.configure(state=tk.NORMAL) # Widget Removed
         self.status_bar.set_status("Bot stopped.")
         self.status_bar.hide_progress()
         logger.info("Bot stop confirmed and UI updated.")


    def open_settings(self):
        """Opens the Settings Modal Dialog."""
        try:
            # Pass current config and callback to save
            SettingsDialog(self, self.config, self._save_config)
        except Exception as e:
             logger.error(f"Failed to open settings: {e}")
             messagebox.showerror("Error", f"Could not open settings: {e}")

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

    # ═══════════════════════════════════════════════════════════════════════════
    # RISK MANAGEMENT & TRADE CONTROL
    # ═══════════════════════════════════════════════════════════════════════════
    def _emergency_close_all(self):
        """Emergency close all open positions."""
        if not messagebox.askyesno("Emergency Close", 
            "⚠️ Are you sure you want to CLOSE ALL POSITIONS?\n\n"
            "This will immediately close all open trades."):
            return
        
        if not self.data_manager or not self.data_manager.is_initialized:
            messagebox.showerror("Error", "MT5 not connected")
            return
        
        try:
            positions = mt5.positions_get()
            if not positions:
                messagebox.showinfo("Info", "No open positions to close.")
                return
            
            closed = 0
            failed = 0
            
            for pos in positions:
                tick = mt5.symbol_info_tick(pos.symbol)
                if not tick:
                    failed += 1
                    continue
                    
                price = tick.bid if pos.type == 0 else tick.ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 30,
                    "magic": 999999,
                    "comment": "EMERGENCY_CLOSE",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    closed += 1
                else:
                    failed += 1
            
            messagebox.showinfo("Emergency Close", 
                f"✅ Closed: {closed} positions\n❌ Failed: {failed} positions")
            logger.warning(f"Emergency close: {closed} closed, {failed} failed")
            
            # Update dashboard
            self._update_risk_dashboard()
            
        except Exception as e:
            logger.exception(f"Emergency close error: {e}")
            messagebox.showerror("Error", f"Emergency close failed: {e}")

    def _update_risk_dashboard(self):
        """Update the risk dashboard with current data."""
        try:
            # Daily P&L
            if self.data_manager and self.data_manager.is_initialized:
                today_pnl = 0.0
                positions = mt5.positions_get()
                pos_count = len(positions) if positions else 0
                total_lots = sum(p.volume for p in positions) if positions else 0
                
                # Get floating P&L
                if positions:
                    today_pnl = sum(p.profit for p in positions)
                
                # Update labels
                pnl_color = SUCCESS_GREEN if today_pnl >= 0 else DANGER_RED
                self.daily_pnl_label.configure(
                    text=f"${today_pnl:+,.2f}", 
                    text_color=pnl_color
                )
                self.open_positions_label.configure(text=str(pos_count))
                self.exposure_label.configure(text=f"{total_lots:.2f} lots")
            
            # Auto-Trade status
            if hasattr(self, 'execution_manager') and self.execution_manager:
                config = self.execution_manager.config
                if config.get('auto_trade_enabled', False):
                    self.auto_trade_label.configure(text="● ON", text_color=SUCCESS_GREEN)
                else:
                    self.auto_trade_label.configure(text="● OFF", text_color=TEXT_SECONDARY)
            
            # Guardian status (placeholder - would need guardian instance)
            # self.guardian_label.configure(text="● ACTIVE", text_color=SUCCESS_GREEN)
            
        except Exception as e:
            logger.debug(f"Risk dashboard update error: {e}")


# --- Custom Status Bar Class - Modern Theme ---
class StatusBar(ctk.CTkFrame):
    """A custom status bar widget with text label and optional progress bar."""
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, fg_color=BG_CARD, corner_radius=8, **kwargs)
        self.configure(height=30)
        self.status_label = ctk.CTkLabel(
            self, 
            text="🟢 Ready", 
            anchor="w", 
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=(12, 5), pady=5)
        self.progress_bar = ctk.CTkProgressBar(
            self, 
            width=150, 
            height=12, 
            corner_radius=6,
            fg_color=BG_DARK,
            progress_color=ACCENT_TEAL
        )
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=(5, 12), pady=5)
        self.progress_bar.grid_remove()
        self.grid_columnconfigure(0, weight=1)
        self._status_clear_timer: Optional[str] = None
        self._permanent_message: str = "🟢 Ready"

    def set_status(self, text: str, duration: int = 0, alert: bool = False, priority: bool = False):
        """Sets the status text."""
        if self._status_clear_timer: self.after_cancel(self._status_clear_timer); self._status_clear_timer = None
        text_color = WARNING_YELLOW if alert else TEXT_SECONDARY
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
def main():
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

if __name__ == "__main__":
    main()

# --- END OF FILE Main.py ---
