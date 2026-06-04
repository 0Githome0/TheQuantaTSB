"""
Backtester Hub UI - Professional Backtesting Interface
Features:
- Strategy Editor with Code/Visual modes
- Candlestick chart with trade markers
- Performance metrics (Sharpe, Drawdown, etc.)
- Trade log with export
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
import threading
import logging
import json
import csv

import pandas as pd
import numpy as np

# Import backtester engine
from src.core.backtester_engine import BacktesterEngine

import numpy as np

# TradingView Lightweight Charts
try:
    from lightweight_charts import Chart
    HAS_LIGHTWEIGHT_CHARTS = True
except ImportError:
    HAS_LIGHTWEIGHT_CHARTS = False

# Theme colors (matching app.py)
BG_DARK = "#0a0a0f"
BG_CARD = "#12121a"
BG_HOVER = "#1a1a2e"
BORDER_COLOR = "#2a2a3e"
ACCENT_TEAL = "#00d9a5"
ACCENT_BLUE = "#4a9eff"
SUCCESS_GREEN = "#00c853"
DANGER_RED = "#ff4757"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#8b8b9e"

# Fonts defined as tuples (created lazily inside class)
CODE_FONT = ("Consolas", 12)

logger = logging.getLogger("BacktesterUI")


class BacktesterFrame(ctk.CTkFrame):
    """Main Backtester Hub Frame with all panels."""
    
    def __init__(self, parent, data_manager=None, signal_generator=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.data_manager = data_manager
        self.signal_generator = signal_generator
        self.backtest_results = None
        self.trades = []
        self.is_running = False
        
        # Initialize the backtester engine
        self.engine = BacktesterEngine(data_manager, signal_generator)
        
        # Default strategy code template
        self.default_strategy_code = '''# TheQuanta Strategy Script
# Available variables: rsi, macd, sma_20, sma_50, ema_20, close, open, high, low
# Use: signal = "BUY" or signal = "SELL" or signal = "HOLD"

def generate_signal(data, indicators):
    """Generate trading signal based on indicators."""
    rsi = indicators.get('rsi', 50)
    macd = indicators.get('macd', 0)
    sma_20 = indicators.get('sma_20', 0)
    close = data['close'].iloc[-1]
    
    signal = "HOLD"
    
    # RSI Oversold + Price above SMA = BUY
    if rsi < 30 and close > sma_20:
        signal = "BUY"
    
    # RSI Overbought + Price below SMA = SELL
    elif rsi > 70 and close < sma_20:
        signal = "SELL"
    
    return signal
'''
        
        self._create_layout()
        
    def _create_layout(self):
        """Create the main 3-column layout."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)  # Left - Strategy Editor
        self.grid_columnconfigure(1, weight=2)  # Center - Chart
        self.grid_columnconfigure(2, weight=1)  # Right - Trade Log
        self.grid_rowconfigure(0, weight=3)     # Top panels
        self.grid_rowconfigure(1, weight=1)     # Bottom - Metrics
        
        # ═══════════════════════════════════════════════════════════════════
        # LEFT PANEL - Strategy Editor
        # ═══════════════════════════════════════════════════════════════════
        self._create_strategy_panel()
        
        # ═══════════════════════════════════════════════════════════════════
        # CENTER PANEL - Chart
        # ═══════════════════════════════════════════════════════════════════
        self._create_chart_panel()
        
        # ═══════════════════════════════════════════════════════════════════
        # RIGHT PANEL - Trade Log
        # ═══════════════════════════════════════════════════════════════════
        self._create_trade_log_panel()
        
        # ═══════════════════════════════════════════════════════════════════
        # BOTTOM PANEL - Performance Metrics
        # ═══════════════════════════════════════════════════════════════════
        self._create_metrics_panel()
        
    def _create_strategy_panel(self):
        """Create the Strategy Editor panel (left)."""
        self.strategy_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        self.strategy_frame.grid(row=0, column=0, padx=(10, 5), pady=(10, 5), sticky="nsew")
        
        # Header
        header = ctk.CTkFrame(self.strategy_frame, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(12, 8))
        
        ctk.CTkLabel(header, text="📝 Strategy Editor", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT_PRIMARY).pack(side="left")
        
        # Load/Save Buttons
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right", padx=10)
        
        ctk.CTkButton(btn_frame, text="📂", width=30, height=24, font=ctk.CTkFont(size=14), 
                     fg_color=BG_DARK, hover_color=ACCENT_TEAL, command=self._load_strategy_file).pack(side="left", padx=2)
                     
        ctk.CTkButton(btn_frame, text="💾", width=30, height=24, font=ctk.CTkFont(size=14),
                     fg_color=BG_DARK, hover_color=ACCENT_TEAL, command=self._save_strategy_file).pack(side="left", padx=2)
        
        # NEW: Save to Live Trading button
        ctk.CTkButton(btn_frame, text="🚀", width=30, height=24, font=ctk.CTkFont(size=14),
                     fg_color=SUCCESS_GREEN, hover_color="#10b981", command=self._save_to_live).pack(side="left", padx=2)


        # Mode Switch
        self.mode_var = ctk.StringVar(value="code")
        mode_frame = ctk.CTkFrame(header, fg_color="transparent")
        mode_frame.pack(side="right")
        
        ctk.CTkRadioButton(
            mode_frame, text="Code", variable=self.mode_var, value="code",
            fg_color=ACCENT_TEAL, hover_color=ACCENT_BLUE, font=ctk.CTkFont(size=11),
            command=self._switch_mode
        ).pack(side="left", padx=5)
        
        ctk.CTkRadioButton(
            mode_frame, text="Visual", variable=self.mode_var, value="visual",
            fg_color=ACCENT_TEAL, hover_color=ACCENT_BLUE, font=ctk.CTkFont(size=11),
            command=self._switch_mode
        ).pack(side="left", padx=5)
        
        # Code Editor Container
        self.editor_container = ctk.CTkFrame(self.strategy_frame, fg_color=BG_DARK, corner_radius=8)
        self.editor_container.pack(fill="both", expand=True, padx=12, pady=8)
        
        # Code Editor (Text widget with syntax highlighting)
        self.code_editor = tk.Text(
            self.editor_container, 
            bg="#1e1e2e", fg="#d4d4d4",
            insertbackground=ACCENT_TEAL,
            font=CODE_FONT,
            wrap="none",
            padx=10, pady=10,
            relief="flat",
            highlightthickness=0
        )
        self.code_editor.pack(fill="both", expand=True, padx=2, pady=2)
        self.code_editor.insert("1.0", self.default_strategy_code)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.code_editor, orient="vertical", command=self.code_editor.yview)
        self.code_editor.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        
        # Visual Builder Container (hidden by default)
        self.visual_container = ctk.CTkFrame(self.strategy_frame, fg_color=BG_DARK, corner_radius=8)
        
        # ─── Strategy Parameters ───
        params_frame = ctk.CTkFrame(self.strategy_frame, fg_color="transparent")
        params_frame.pack(fill="x", padx=12, pady=8)
        
        ctk.CTkLabel(params_frame, text="⚙️ Parameters", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_PRIMARY).pack(anchor="w", pady=(0, 8))
        
        # Strategy Selector - Load from /strategies/ folder
        strat_frame = ctk.CTkFrame(params_frame, fg_color="transparent")
        strat_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(strat_frame, text="Strategy:", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY, width=120).pack(side="left")
        
        # Get strategies from /strategies/ folder
        strategy_names = self._get_available_strategies()
        
        self.strategy_combo = ctk.CTkComboBox(
            strat_frame, 
            values=strategy_names if strategy_names else ["No strategies found"],
            width=150, height=28, fg_color=BG_DARK, border_color=BORDER_COLOR,
            button_color=ACCENT_TEAL, button_hover_color=ACCENT_BLUE,
            command=self._on_strategy_selected
        )
        if strategy_names:
            self.strategy_combo.set(strategy_names[0])
        self.strategy_combo.pack(side="left", padx=5)
        
        # Refresh button
        ctk.CTkButton(strat_frame, text="🔄", width=28, height=28, font=ctk.CTkFont(size=12),
                     fg_color=BG_DARK, hover_color=ACCENT_TEAL, 
                     command=self._refresh_strategies).pack(side="left", padx=2)
        
        # Grid for parameters
        params_grid = ctk.CTkFrame(params_frame, fg_color="transparent")
        params_grid.pack(fill="x")
        
        # Stop Loss
        sl_frame = ctk.CTkFrame(params_grid, fg_color="transparent")
        sl_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(sl_frame, text="Stop Loss (pips):", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY, width=120).pack(side="left")
        self.sl_entry = ctk.CTkEntry(sl_frame, width=80, height=28, fg_color=BG_DARK, border_color=BORDER_COLOR)
        self.sl_entry.insert(0, "50")
        self.sl_entry.pack(side="left", padx=5)
        
        # Take Profit
        tp_frame = ctk.CTkFrame(params_grid, fg_color="transparent")
        tp_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(tp_frame, text="Take Profit (pips):", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY, width=120).pack(side="left")
        self.tp_entry = ctk.CTkEntry(tp_frame, width=80, height=28, fg_color=BG_DARK, border_color=BORDER_COLOR)
        self.tp_entry.insert(0, "100")
        self.tp_entry.pack(side="left", padx=5)
        
        # Lot Size
        lot_frame = ctk.CTkFrame(params_grid, fg_color="transparent")
        lot_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(lot_frame, text="Lot Size:", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY, width=120).pack(side="left")
        self.lot_entry = ctk.CTkEntry(lot_frame, width=80, height=28, fg_color=BG_DARK, border_color=BORDER_COLOR)
        self.lot_entry.insert(0, "0.1")
        self.lot_entry.pack(side="left", padx=5)
        
        # Initial Capital
        cap_frame = ctk.CTkFrame(params_grid, fg_color="transparent")
        cap_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(cap_frame, text="Initial Capital ($):", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY, width=120).pack(side="left")
        self.capital_entry = ctk.CTkEntry(cap_frame, width=80, height=28, fg_color=BG_DARK, border_color=BORDER_COLOR)
        self.capital_entry.insert(0, "10000")
        self.capital_entry.pack(side="left", padx=5)
        
    def _load_strategy_file(self):
        """Load strategy code from a file."""
        filename = filedialog.askopenfilename(filetypes=[("Python Files", "*.py"), ("Text Files", "*.txt"), ("All Files", "*.*")])
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                if hasattr(self, 'code_editor'):
                    self.code_editor.delete("1.0", "end")
                    self.code_editor.insert("1.0", content)
                # Switch to code mode
                self.mode_var.set("code")
                self._switch_mode()
                messagebox.showinfo("Success", "Strategy loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def _get_available_strategies(self):
        """Get list of strategy names from /strategies/ folder."""
        import os
        strategies = []
        strategies_folder = "strategies"
        
        if not os.path.exists(strategies_folder):
            return strategies
        
        for filename in os.listdir(strategies_folder):
            if filename.endswith(".py") and not filename.startswith("_"):
                # Extract strategy name from file
                name = filename[:-3].replace("_", " ").title()
                strategies.append(name)
        
        return sorted(strategies) if strategies else ["Custom Code"]
    
    def _on_strategy_selected(self, strategy_name):
        """Called when a strategy is selected from dropdown."""
        import os
        strategies_folder = "strategies"
        
        # Convert name back to filename
        filename = strategy_name.lower().replace(" ", "_") + ".py"
        filepath = os.path.join(strategies_folder, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.code_editor.delete("1.0", "end")
                self.code_editor.insert("1.0", content)
                logger.info(f"Loaded strategy: {strategy_name}")
            except Exception as e:
                logger.error(f"Failed to load strategy: {e}")
    
    def _refresh_strategies(self):
        """Refresh the strategy dropdown."""
        strategy_names = self._get_available_strategies()
        self.strategy_combo.configure(values=strategy_names if strategy_names else ["No strategies"])
        if strategy_names:
            self.strategy_combo.set(strategy_names[0])
        messagebox.showinfo("Refreshed", f"Found {len(strategy_names)} strategies")

    def _save_strategy_file(self):
        """Save current strategy code to a file."""
        filename = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Files", "*.py"), ("Text Files", "*.txt")])
        if filename:
            try:
                content = self.code_editor.get("1.0", "end-1c")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Success", "Strategy saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

    def _save_to_live(self):
        """Save current strategy to /strategies/ folder for live trading."""
        import os
        import re
        from tkinter import simpledialog
        
        # Get strategy name from user
        strategy_name = simpledialog.askstring(
            "Save to Live Trading",
            "Enter a name for your strategy:\n(letters, numbers, underscores only)",
            initialvalue="my_strategy"
        )
        
        if not strategy_name:
            return
        
        # Sanitize name
        strategy_name = re.sub(r'[^a-zA-Z0-9_]', '_', strategy_name.lower())
        
        # Get current code
        code = self.code_editor.get("1.0", "end-1c")
        
        # Create class name
        class_name = ''.join(word.capitalize() for word in strategy_name.split('_')) + "Strategy"
        
        # Generate BaseStrategy compatible code
        strategy_code = f'''# --- START OF FILE {strategy_name}.py ---
"""
{strategy_name.replace('_', ' ').title()} Strategy
===============================================
Auto-generated from Backtest Editor.
Modify as needed for live trading.
"""

import pandas as pd
from typing import Optional, Dict
from src.core.base_strategy import BaseStrategy, StrategySignal, ExitSignal


# ═══════════════════════════════════════════════════════════════════════
# USER'S BACKTEST LOGIC (from editor)
# ═══════════════════════════════════════════════════════════════════════
{code}


class {class_name}(BaseStrategy):
    """Strategy generated from backtest."""
    
    # Strategy Identity
    name = "{strategy_name.replace('_', ' ').title()}"
    version = "1.0.0"
    description = "Strategy generated from backtest editor"
    author = "TheQuanta"
    strategy_type = "custom"
    
    # Supported instruments
    timeframes = ["M5", "M15", "M30", "H1", "H4"]
    pairs = ["*"]  # All pairs
    
    # Parameters
    default_params = {{
        'sl_pips': {self.sl_entry.get()},
        'tp_pips': {self.tp_entry.get()},
        'risk_percent': 1.0,
        'min_rr': 1.5,
        'max_trades_per_day': 5,
    }}
    
    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY INTERFACE
    # ═══════════════════════════════════════════════════════════════════════
    def should_enter(self, data: Dict, pair: str, timeframe: str) -> Optional[StrategySignal]:
        """Check for entry signal using backtest logic."""
        df = data.get('ohlcv')
        if df is None or len(df) < 50:
            return None
        
        # Calculate indicators
        close = df['close']
        indicators = {{
            'rsi': self._calculate_rsi(close, 14),
            'macd': self._calculate_macd(close),
            'sma_20': close.rolling(20).mean().iloc[-1] if len(close) >= 20 else 0,
            'sma_50': close.rolling(50).mean().iloc[-1] if len(close) >= 50 else 0,
            'ema_20': close.ewm(span=20).mean().iloc[-1],
        }}
        
        # Call the user's generate_signal function
        try:
            signal = generate_signal(df, indicators)
        except NameError:
            # generate_signal not defined, return None
            return None
        except Exception:
            signal = "HOLD"
        
        if signal == "BUY":
            return self.create_buy_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=close.iloc[-1],
                confidence=0.75,
                reasons=["Backtest strategy BUY signal"]
            )
        elif signal == "SELL":
            return self.create_sell_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=close.iloc[-1],
                confidence=0.75,
                reasons=["Backtest strategy SELL signal"]
            )
        
        return None
    
    def should_exit(self, position: Dict, data: Dict) -> Optional[ExitSignal]:
        """Exit logic - use SL/TP from parameters."""
        return None  # Let SL/TP handle exits
    
    def _calculate_rsi(self, series, period=14):
        if len(series) < period + 1:
            return 50
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.001)
        return (100 - (100 / (1 + rs))).iloc[-1]
    
    def _calculate_macd(self, series):
        if len(series) < 26:
            return 0
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        return (exp1 - exp2).iloc[-1]


# --- END OF FILE {strategy_name}.py ---
'''
        
        # Save to strategies folder
        strategies_folder = "strategies"
        os.makedirs(strategies_folder, exist_ok=True)
        
        filepath = os.path.join(strategies_folder, f"{strategy_name}.py")
        
        # Check if exists
        if os.path.exists(filepath):
            if not messagebox.askyesno("Overwrite?", f"Strategy '{strategy_name}' already exists. Overwrite?"):
                return
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(strategy_code)
            
            messagebox.showinfo(
                "🚀 Strategy Saved!",
                f"Strategy saved to:\\n{filepath}\\n\\n"
                "Go to 🎯 Strategies tab and click Reload to see it.\\n"
                "Then click Deploy to use for live trading!"
            )
            logger.info(f"Strategy saved to live: {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save strategy: {e}")
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code block by the specified number of spaces."""
        indent = " " * spaces
        lines = code.split("\\n")
        # Comment out the original code and add as reference
        indented = []
        for line in lines:
            indented.append(f"{indent}# {line}" if line.strip() else f"{indent}#")
        return "\\n".join(indented)

    def _create_chart_panel(self):
        """Create the Chart panel (center)."""
        self.chart_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        self.chart_frame.grid(row=0, column=1, padx=5, pady=(10, 5), sticky="nsew")
        
        # Header with controls
        header = ctk.CTkFrame(self.chart_frame, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(12, 8))
        
        ctk.CTkLabel(header, text="📈 Chart", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT_PRIMARY).pack(side="left")
        
        # Controls row
        controls = ctk.CTkFrame(self.chart_frame, fg_color=BG_DARK, corner_radius=8)
        controls.pack(fill="x", padx=12, pady=(0, 8))
        
        # Pair selector
        ctk.CTkLabel(controls, text="Pair:", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(side="left", padx=(10, 5))
        self.pair_combo = ctk.CTkComboBox(
            controls, values=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"],
            width=100, height=28, fg_color=BG_CARD, border_color=BORDER_COLOR,
            button_color=ACCENT_TEAL, button_hover_color=ACCENT_BLUE
        )
        self.pair_combo.set("EURUSD")
        self.pair_combo.pack(side="left", padx=5)
        
        # Timeframe selector
        ctk.CTkLabel(controls, text="TF:", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(side="left", padx=(15, 5))
        self.tf_combo = ctk.CTkComboBox(
            controls, values=["M5", "M15", "H1", "H4", "D1"],
            width=70, height=28, fg_color=BG_CARD, border_color=BORDER_COLOR,
            button_color=ACCENT_TEAL, button_hover_color=ACCENT_BLUE
        )
        self.tf_combo.set("H1")
        self.tf_combo.pack(side="left", padx=5)
        
        # Date Range
        ctk.CTkLabel(controls, text="From:", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(side="left", padx=(15, 5))
        self.start_date_entry = ctk.CTkEntry(controls, width=90, height=28, fg_color=BG_CARD, border_color=BORDER_COLOR)
        self.start_date_entry.insert(0, (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))
        self.start_date_entry.pack(side="left", padx=2)
        
        ctk.CTkLabel(controls, text="To:", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(side="left", padx=(10, 5))
        self.end_date_entry = ctk.CTkEntry(controls, width=90, height=28, fg_color=BG_CARD, border_color=BORDER_COLOR)
        self.end_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.end_date_entry.pack(side="left", padx=2)
        
        # Run Button
        self.run_btn = ctk.CTkButton(
            controls, text="▶ Run Backtest", width=120, height=32,
            fg_color=SUCCESS_GREEN, hover_color="#00a844",
            font=ctk.CTkFont(size=14, weight="bold"), command=self._run_backtest
        )
        self.run_btn.pack(side="right", padx=10, pady=5)
        
        # Indicator toggles
        indicators_frame = ctk.CTkFrame(self.chart_frame, fg_color="transparent")
        indicators_frame.pack(fill="x", padx=12, pady=(0, 5))
        
        ctk.CTkLabel(indicators_frame, text="Indicators:", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(side="left")
        
        self.sma_var = ctk.BooleanVar(value=True)
        self.ema_var = ctk.BooleanVar(value=False)
        self.bb_var = ctk.BooleanVar(value=False)
        
        ctk.CTkCheckBox(indicators_frame, text="SMA", variable=self.sma_var, font=ctk.CTkFont(size=11), 
                       fg_color=ACCENT_TEAL, hover_color=ACCENT_BLUE, width=60).pack(side="left", padx=5)
        ctk.CTkCheckBox(indicators_frame, text="EMA", variable=self.ema_var, font=ctk.CTkFont(size=11),
                       fg_color=ACCENT_TEAL, hover_color=ACCENT_BLUE, width=60).pack(side="left", padx=5)
        ctk.CTkCheckBox(indicators_frame, text="BB", variable=self.bb_var, font=ctk.CTkFont(size=11),
                       fg_color=ACCENT_TEAL, hover_color=ACCENT_BLUE, width=60).pack(side="left", padx=5)
        
        # Chart canvas container
        self.canvas_container = ctk.CTkFrame(self.chart_frame, fg_color=BG_DARK, corner_radius=8)
        self.canvas_container.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        
        # Initialize placeholder
        self._create_chart_placeholder()
    
    def _create_chart_placeholder(self):
        """Create placeholder for chart area."""
        for widget in self.canvas_container.winfo_children():
            widget.destroy()
        
        # Simple dark placeholder
        placeholder = ctk.CTkFrame(self.canvas_container, fg_color="transparent")
        placeholder.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(placeholder, text="📊", font=ctk.CTkFont(size=48), text_color=TEXT_SECONDARY).pack()
        ctk.CTkLabel(placeholder, text="Run backtest to view chart", font=ctk.CTkFont(size=16), text_color=TEXT_SECONDARY).pack(pady=10)
        
        if HAS_LIGHTWEIGHT_CHARTS:
            ctk.CTkLabel(placeholder, text="✓ lightweight-charts ready", font=ctk.CTkFont(size=12), text_color=SUCCESS_GREEN).pack()
        else:
            ctk.CTkLabel(placeholder, text="⚠ Install: pip install lightweight-charts", font=ctk.CTkFont(size=12), text_color=DANGER_RED).pack()
    
    def _render_chart(self, data, trades):
        """Open lightweight-charts in new window."""
        for widget in self.canvas_container.winfo_children():
            widget.destroy()
        
        # Show summary in chart area
        summary = ctk.CTkFrame(self.canvas_container, fg_color="transparent")
        summary.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(summary, text="✅ Backtest Complete!", font=ctk.CTkFont(size=20, weight="bold"), text_color=SUCCESS_GREEN).pack(pady=(0, 15))
        ctk.CTkLabel(summary, text=f"{len(trades)} trades generated", font=ctk.CTkFont(size=14), text_color=TEXT_SECONDARY).pack()
        ctk.CTkLabel(summary, text="📈 Chart opening in new window...", font=ctk.CTkFont(size=12), text_color=ACCENT_TEAL).pack(pady=(10, 5))
        
        # Button to reopen chart
        ctk.CTkButton(summary, text="📊 Open Chart", command=self._open_tv_chart, fg_color=ACCENT_BLUE, hover_color="#3a8aee").pack(pady=10)
        
        # Auto-open chart
        if HAS_LIGHTWEIGHT_CHARTS:
            threading.Thread(target=self._create_tv_chart, daemon=True).start()
    
    def _open_tv_chart(self):
        """Open TradingView chart in new window."""
        if not HAS_LIGHTWEIGHT_CHARTS:
            messagebox.showwarning("Not Available", "Install: pip install lightweight-charts")
            return
        if not hasattr(self, 'backtest_data') or self.backtest_data is None:
            messagebox.showinfo("No Data", "Run a backtest first.")
            return
        threading.Thread(target=self._create_tv_chart, daemon=True).start()
    
    def _create_tv_chart(self):
        """Create TradingView chart in separate window."""
        try:
            data = self.backtest_data.copy()
            trades = self.trades
            pair = self.pair_combo.get() if hasattr(self, 'pair_combo') else "EURUSD"
            
            # Prepare data
            chart_df = data[['open', 'high', 'low', 'close']].copy().reset_index()
            chart_df.columns = ['time', 'open', 'high', 'low', 'close']
            
            if 'volume' in data.columns:
                chart_df['volume'] = data['volume'].values
            elif 'tick_volume' in data.columns:
                chart_df['volume'] = data['tick_volume'].values
            else:
                chart_df['volume'] = 1000
            
            # Create chart
            chart = Chart(title=f"{pair} Backtest")
            chart.watermark(pair, color='rgba(180, 180, 200, 0.3)')
            chart.set(chart_df)
            
            # Add SMA if enabled
            if self.sma_var.get():
                sma20 = data['close'].rolling(20).mean()
                sma_df = pd.DataFrame({'time': data.index, 'SMA20': sma20}).dropna()
                sma_line = chart.create_line(name='SMA20', color='#4a9eff')
                sma_line.set(sma_df)
                
                sma50 = data['close'].rolling(50).mean()
                sma50_df = pd.DataFrame({'time': data.index, 'SMA50': sma50}).dropna()
                sma50_line = chart.create_line(name='SMA50', color='#ff9f43')
                sma50_line.set(sma50_df)
            
            # Add trade markers
            for trade in trades:
                try:
                    idx = trade['entry_idx']
                    if idx < len(data):
                        t = data.index[idx]
                        pos = 'below' if trade['type'] == 'BUY' else 'above'
                        shape = 'arrowUp' if trade['type'] == 'BUY' else 'arrowDown'
                        color = '#00c853' if trade['type'] == 'BUY' else '#ff4757'
                        chart.marker(time=t, position=pos, shape=shape, color=color, text=trade['type'])
                except:
                    pass
            
            chart.show(block=True)
        except Exception as e:
            logger.error(f"Chart error: {e}")
            self.after(0, lambda msg=str(e): messagebox.showerror("Chart Error", msg))
    
    def _create_empty_chart(self):
        """Legacy compatibility."""
        self._create_chart_placeholder()
        
        
    def _create_trade_log_panel(self):
        """Create the Trade Log panel (right)."""
        self.log_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        self.log_frame.grid(row=0, column=2, padx=(5, 10), pady=(10, 5), sticky="nsew")
        
        # Header
        header = ctk.CTkFrame(self.log_frame, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(12, 8))
        
        ctk.CTkLabel(header, text="📋 Trade Log", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT_PRIMARY).pack(side="left")
        
        # Export button
        self.export_btn = ctk.CTkButton(
            header, text="📤 Export", width=80, height=28,
            fg_color=BG_HOVER, hover_color=BORDER_COLOR,
            font=ctk.CTkFont(size=11), command=self._export_trades
        )
        self.export_btn.pack(side="right")
        
        # Trade count
        self.trade_count_label = ctk.CTkLabel(header, text="0 trades", font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY)
        self.trade_count_label.pack(side="right", padx=10)
        
        # Trade list (scrollable)
        self.trade_list_frame = ctk.CTkScrollableFrame(self.log_frame, fg_color=BG_DARK, corner_radius=8)
        self.trade_list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        
        # Table header
        header_frame = ctk.CTkFrame(self.trade_list_frame, fg_color=BG_HOVER, corner_radius=4)
        header_frame.pack(fill="x", pady=(0, 5))
        
        headers = ["Time", "Type", "Entry", "Exit", "PnL"]
        widths = [80, 50, 70, 70, 70]
        
        for i, (text, width) in enumerate(zip(headers, widths)):
            ctk.CTkLabel(
                header_frame, text=text, font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY, width=width
            ).pack(side="left", padx=5, pady=5)
            
        # Placeholder
        self.log_placeholder = ctk.CTkLabel(
            self.trade_list_frame, text="No trades yet\nRun backtest to see results",
            font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY
        )
        self.log_placeholder.pack(pady=40)
        
    def _create_metrics_panel(self):
        """Create the Performance Metrics panel (bottom)."""
        self.metrics_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        self.metrics_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=(5, 10), sticky="nsew")
        
        # Header
        header = ctk.CTkFrame(self.metrics_frame, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(12, 8))
        
        ctk.CTkLabel(header, text="📊 Performance Metrics", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT_PRIMARY).pack(side="left")
        
        # Metrics cards container
        cards_frame = ctk.CTkFrame(self.metrics_frame, fg_color="transparent")
        cards_frame.pack(fill="x", padx=12, pady=(0, 12))
        cards_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        
        # Create metric cards
        self.metric_cards = {}
        metrics = [
            ("net_pnl", "💰 Net PnL", "$0.00", SUCCESS_GREEN),
            ("pnl_pct", "📈 Return %", "0.00%", TEXT_SECONDARY),
            ("max_dd", "📉 Max Drawdown", "0.00%", DANGER_RED),
            ("sharpe", "📐 Sharpe Ratio", "0.00", ACCENT_BLUE),
            ("win_rate", "🎯 Win Rate", "0.0%", TEXT_SECONDARY),
            ("profit_factor", "⚖️ Profit Factor", "0.00", TEXT_SECONDARY),
        ]
        
        for i, (key, label, default, color) in enumerate(metrics):
            card = self._create_metric_card(cards_frame, label, default, color)
            card.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            self.metric_cards[key] = card
            
    def _create_metric_card(self, parent, title, value, color):
        """Create a metric display card."""
        card = ctk.CTkFrame(parent, fg_color=BG_DARK, corner_radius=8)
        
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).pack(pady=(10, 2))
        value_label = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=14, weight="bold"), text_color=color)
        value_label.pack(pady=(2, 10))
        
        # Store reference to value label
        card.value_label = value_label
        return card
        
    def _switch_mode(self):
        """Switch between Code and Visual modes."""
        mode = self.mode_var.get()
        if mode == "code":
            self.visual_container.pack_forget()
            self.editor_container.pack(fill="both", expand=True, padx=12, pady=8)
        else:
            self.editor_container.pack_forget()
            self.visual_container.pack(fill="both", expand=True, padx=12, pady=8)
            
    def _run_backtest(self):
        """Run the backtest simulation."""
        if self.is_running:
            return
            
        self.is_running = True
        self.run_btn.configure(text="⏳ Running...", state="disabled")
        
        # Run in thread to keep UI responsive
        thread = threading.Thread(target=self._execute_backtest, daemon=True)
        thread.start()
        
    def _execute_backtest(self):
        """Execute backtest logic (runs in thread)."""
        try:
            # Get parameters
            pair = self.pair_combo.get()
            tf = self.tf_combo.get()
            sl_pips = float(self.sl_entry.get())
            tp_pips = float(self.tp_entry.get())
            lot_size = float(self.lot_entry.get())
            capital = float(self.capital_entry.get())
            strategy = self.strategy_combo.get() if hasattr(self, 'strategy_combo') else "Multi-Indicator Bot"
            
            logger.info(f"Starting backtest: {pair} {tf}, SL={sl_pips}, TP={tp_pips}")
            
            # Set engine parameters
            self.engine.set_parameters(capital, lot_size, sl_pips, tp_pips)
            
            # Get date range (parse from UI entries)
            try:
                # Use correct widget names
                start_str = self.start_date_entry.get()
                end_str = self.end_date_entry.get()
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
            except Exception as e:
                logger.warning(f"Date parse error: {e}. Using default 30 days.")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365) # Default to 1 year if parse fails (though fallback shouldn't happen if UI is correct)
            
            # Ensure symbol is valid in MT5
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    symbol_info = mt5.symbol_info(pair)
                    if symbol_info is None:
                        # Try to enable it
                        mt5.symbol_select(pair, True)
            except:
                pass
            
            # Fetch historical data using engine
            data = self.engine.fetch_historical_data(pair, tf, start_date, end_date)
            
            # Prepare strategy/code
            strategy_code = None
            if self.mode_var.get() == "code":
                strategy = "Code Mode"
                if hasattr(self, 'code_editor'):
                    strategy_code = self.code_editor.get("1.0", "end-1c")
            
            # Run backtest using engine
            results = self.engine.run_backtest(data, strategy, strategy_code)
            
            # Update UI on main thread
            self.after(0, lambda: self._update_results(data, results))
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            self.after(0, lambda: messagebox.showerror("Backtest Error", str(e)))
        finally:
            self.is_running = False
            self.after(0, lambda: self.run_btn.configure(text="▶ Run Backtest", state="normal"))
            
    def _generate_mock_data(self, bars=500):
        """Generate realistic mock OHLCV data with trending and ranging phases."""
        dates = pd.date_range(end=datetime.now(), periods=bars, freq='H')
        
        # Create realistic price action with trends and consolidation
        np.random.seed(int(datetime.now().timestamp()) % 1000)  # Different each run
        
        # Start price
        base_price = 1.1000
        prices = [base_price]
        
        # Generate price with momentum and mean reversion
        trend = 0
        for i in range(1, bars):
            # Change trend occasionally
            if np.random.random() < 0.02:  # 2% chance to reverse trend
                trend = np.random.choice([-1, 0, 1]) * 0.0002
            
            # Random component + trend + mean reversion
            noise = np.random.randn() * 0.0008
            mean_reversion = (base_price - prices[-1]) * 0.01  # Pull back to mean
            
            new_price = prices[-1] + noise + trend + mean_reversion
            new_price = max(base_price * 0.95, min(base_price * 1.05, new_price))  # Clamp
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Generate OHLC from close prices
        spread = 0.0003  # Typical forex spread
        volatility = np.abs(np.random.randn(bars)) * 0.001 + 0.0002
        
        data = pd.DataFrame({
            'open': np.roll(prices, 1),  # Open = previous close
            'high': prices + volatility * np.abs(np.random.randn(bars)),
            'low': prices - volatility * np.abs(np.random.randn(bars)),
            'close': prices,
            'tick_volume': np.random.randint(500, 5000, bars),
            'volume': np.random.randint(1000, 10000, bars)
        }, index=dates)
        
        data['open'].iloc[0] = prices[0]  # Fix first open
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        logger.info(f"Generated {bars} bars of mock data. Price range: {data['low'].min():.5f} - {data['high'].max():.5f}")
        
        return data
        
    def _simulate_strategy(self, data, sl_pips, tp_pips, lot_size, capital):
        """Simulate the strategy using actual bot SignalGenerator."""
        trades = []
        equity = [capital]
        current_equity = capital
        
        # Use default pip multiplier for forex
        pip_value = 0.0001
        
        # If we have a real signal generator, use it
        use_real_signals = self.signal_generator is not None
        
        if use_real_signals:
            logger.info("Using real SignalGenerator for backtesting")
            # Calculate indicators using the real signal generator
            try:
                indicators = self.signal_generator.calculate_indicators(data.copy(), pair="BACKTEST")
            except Exception as e:
                logger.warning(f"Failed to use real SignalGenerator: {e}. Falling back to basic.")
                use_real_signals = False
        
        if not use_real_signals:
            # Fallback: Calculate basic indicators
            logger.info("Using built-in indicators for backtesting")
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            data['ema_9'] = data['close'].ewm(span=9).mean()
            data['ema_21'] = data['close'].ewm(span=21).mean()
            
            # Bollinger Bands
            data['bb_mid'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_mid'] + (data['bb_std'] * 2)
            data['bb_lower'] = data['bb_mid'] - (data['bb_std'] * 2)
            
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            data['macd'] = ema12 - ema26
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            # ADX (simplified)
            tr = pd.DataFrame()
            tr['hl'] = data['high'] - data['low']
            tr['hc'] = abs(data['high'] - data['close'].shift(1))
            tr['lc'] = abs(data['low'] - data['close'].shift(1))
            tr['tr'] = tr[['hl', 'hc', 'lc']].max(axis=1)
            data['atr'] = tr['tr'].rolling(window=14).mean()
        
        position = None
        
        for i in range(50, len(data)):
            row = data.iloc[i]
            close = row['close']
            
            # Generate signal
            signal = "HOLD"
            confidence = 0.0
            
            if use_real_signals:
                # Use real signal generator's logic
                try:
                    rsi = indicators.get('rsi')
                    macd_hist = indicators.get('macd_histogram')
                    adx = indicators.get('adx')
                    ema_short = indicators.get('ema_short')
                    ema_long = indicators.get('ema_long')
                    
                    # Get values at current index
                    if rsi is not None and not rsi.empty and i < len(rsi):
                        rsi_val = rsi.iloc[i] if i < len(rsi) else 50
                    else:
                        rsi_val = 50
                        
                    # Bot's multi-indicator logic
                    buy_signals = 0
                    sell_signals = 0
                    
                    # RSI conditions
                    if rsi_val < 35:
                        buy_signals += 1
                    elif rsi_val > 65:
                        sell_signals += 1
                    
                    # EMA trend
                    if ema_short is not None and ema_long is not None:
                        if i < len(ema_short) and i < len(ema_long):
                            if ema_short.iloc[i] > ema_long.iloc[i]:
                                buy_signals += 1
                            else:
                                sell_signals += 1
                    
                    # MACD histogram
                    if macd_hist is not None and i < len(macd_hist):
                        mh = macd_hist.iloc[i]
                        if mh > 0:
                            buy_signals += 0.5
                        elif mh < 0:
                            sell_signals += 0.5
                    
                    # Generate signal based on score
                    if buy_signals >= 2:
                        signal = "BUY"
                        confidence = buy_signals / 3.0
                    elif sell_signals >= 2:
                        signal = "SELL"
                        confidence = sell_signals / 3.0
                        
                except Exception as sig_e:
                    logger.debug(f"Signal generation error at index {i}: {sig_e}")
                    signal = "HOLD"
            else:
                # Use strategy based on dropdown selection
                strategy_name = self.strategy_combo.get() if hasattr(self, 'strategy_combo') else "Multi-Indicator Bot"
                
                rsi = row.get('rsi', 50)
                sma_20 = row.get('sma_20', close)
                sma_50 = row.get('sma_50', close)
                ema_9 = row.get('ema_9', close)
                ema_21 = row.get('ema_21', close)
                macd = row.get('macd', 0)
                macd_signal = row.get('macd_signal', 0)
                macd_hist = row.get('macd_hist', 0)
                bb_lower = row.get('bb_lower', close * 0.99)
                bb_upper = row.get('bb_upper', close * 1.01)
                bb_mid = row.get('bb_mid', close)
                
                buy_score = 0
                sell_score = 0
                
                if strategy_name == "RSI + SMA Confluence":
                    # RSI oversold + trend up
                    if rsi < 40 and close > sma_20:
                        buy_score = 2
                    elif rsi > 60 and close < sma_20:
                        sell_score = 2
                        
                elif strategy_name == "MACD Crossover":
                    # MACD crosses signal line
                    prev_macd = data['macd'].iloc[i-1] if i > 0 else 0
                    prev_signal = data['macd_signal'].iloc[i-1] if i > 0 else 0
                    
                    if macd > macd_signal and prev_macd <= prev_signal:
                        buy_score = 2  # Bullish crossover
                    elif macd < macd_signal and prev_macd >= prev_signal:
                        sell_score = 2  # Bearish crossover
                        
                elif strategy_name == "Bollinger Bounce":
                    # Price touches or crosses BB
                    if close <= bb_lower * 1.001:
                        buy_score = 2  # Bounce from lower
                    elif close >= bb_upper * 0.999:
                        sell_score = 2  # Bounce from upper
                        
                elif strategy_name == "EMA Trend Follow":
                    # Fast EMA crosses slow EMA
                    prev_ema9 = data['ema_9'].iloc[i-1] if i > 0 else ema_9
                    prev_ema21 = data['ema_21'].iloc[i-1] if i > 0 else ema_21
                    
                    if ema_9 > ema_21 and prev_ema9 <= prev_ema21:
                        buy_score = 2  # Golden cross
                    elif ema_9 < ema_21 and prev_ema9 >= prev_ema21:
                        sell_score = 2  # Death cross
                        
                else:  # Multi-Indicator Bot (default)
                    # Looser conditions for more trades
                    if rsi < 45:
                        buy_score += 1
                    elif rsi > 55:
                        sell_score += 1
                    
                    if close > sma_20:
                        buy_score += 0.5
                    elif close < sma_20:
                        sell_score += 0.5
                    
                    if macd_hist > 0:
                        buy_score += 0.5
                    elif macd_hist < 0:
                        sell_score += 0.5
                    
                    if close <= bb_lower * 1.005:
                        buy_score += 1
                    elif close >= bb_upper * 0.995:
                        sell_score += 1
                
                # Signal decision (lowered threshold)
                if buy_score >= 1.5:
                    signal = "BUY"
                    confidence = min(buy_score / 3.0, 1.0)
                elif sell_score >= 1.5:
                    signal = "SELL"
                    confidence = min(sell_score / 3.0, 1.0)
            
            # Position management
            if position is None:
                if signal == "BUY":
                    position = {
                        'type': 'BUY',
                        'entry_price': close,
                        'entry_time': data.index[i],
                        'entry_idx': i,
                        'sl': close - (sl_pips * pip_value),
                        'tp': close + (tp_pips * pip_value),
                        'confidence': confidence
                    }
                elif signal == "SELL":
                    position = {
                        'type': 'SELL',
                        'entry_price': close,
                        'entry_time': data.index[i],
                        'entry_idx': i,
                        'sl': close + (sl_pips * pip_value),
                        'tp': close - (tp_pips * pip_value),
                        'confidence': confidence
                    }
            else:
                # Check exit conditions
                exit_triggered = False
                exit_price = close
                exit_reason = ""
                
                if position['type'] == 'BUY':
                    if row['low'] <= position['sl']:
                        exit_price = position['sl']
                        exit_triggered = True
                        exit_reason = "SL"
                    elif row['high'] >= position['tp']:
                        exit_price = position['tp']
                        exit_triggered = True
                        exit_reason = "TP"
                else:  # SELL
                    if row['high'] >= position['sl']:
                        exit_price = position['sl']
                        exit_triggered = True
                        exit_reason = "SL"
                    elif row['low'] <= position['tp']:
                        exit_price = position['tp']
                        exit_triggered = True
                        exit_reason = "TP"
                        
                if exit_triggered:
                    # Calculate PnL
                    if position['type'] == 'BUY':
                        pnl = (exit_price - position['entry_price']) * lot_size * 100000
                    else:
                        pnl = (position['entry_price'] - exit_price) * lot_size * 100000
                        
                    current_equity += pnl
                    equity.append(current_equity)
                    
                    trades.append({
                        'time': data.index[i].strftime("%m/%d %H:%M"),
                        'type': position['type'],
                        'entry': position['entry_price'],
                        'exit': exit_price,
                        'pnl': pnl,
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'exit_reason': exit_reason,
                        'confidence': position.get('confidence', 0)
                    })
                    
                    position = None
                    
        # Calculate metrics
        equity_curve = np.array(equity)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak * 100
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        
        results = {
            'trades': trades,
            'equity': equity,
            'net_pnl': current_equity - capital,
            'pnl_pct': ((current_equity - capital) / capital) * 100,
            'max_drawdown': np.max(drawdown) if len(drawdown) > 0 else 0,
            'sharpe': self._calculate_sharpe(equity),
            'win_rate': (len(wins) / len(trades) * 100) if trades else 0,
            'profit_factor': (total_wins / total_losses) if total_losses > 0 else 0,
            'total_trades': len(trades),
            'avg_win': (total_wins / len(wins)) if wins else 0,
            'avg_loss': (total_losses / len(losses)) if losses else 0
        }
        
        return results
        
    def _calculate_sharpe(self, equity, risk_free_rate=0.02):
        """Calculate Sharpe ratio."""
        if len(equity) < 2:
            return 0.0
        returns = np.diff(equity) / equity[:-1]
        if np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) * 252 - risk_free_rate) / (np.std(returns) * np.sqrt(252))
        
    def _update_results(self, data, results):
        """Update UI with backtest results."""
        self.trades = results['trades']
        
        # Update metrics
        pnl_color = SUCCESS_GREEN if results['net_pnl'] >= 0 else DANGER_RED
        self.metric_cards['net_pnl'].value_label.configure(
            text=f"${results['net_pnl']:,.2f}", text_color=pnl_color
        )
        self.metric_cards['pnl_pct'].value_label.configure(
            text=f"{results['pnl_pct']:+.2f}%", text_color=pnl_color
        )
        self.metric_cards['max_dd'].value_label.configure(
            text=f"{results['max_drawdown']:.2f}%"
        )
        self.metric_cards['sharpe'].value_label.configure(
            text=f"{results['sharpe']:.2f}"
        )
        self.metric_cards['win_rate'].value_label.configure(
            text=f"{results['win_rate']:.1f}%"
        )
        self.metric_cards['profit_factor'].value_label.configure(
            text=f"{results['profit_factor']:.2f}"
        )
        
        # Store data for TradingView chart
        self.backtest_data = data
        
        # Update trade log
        self._update_trade_log(results['trades'])
        
        # Render chart in embedded WebView
        self._render_chart(data, results['trades'])
        
        
    def _update_trade_log(self, trades):
        """Update the trade log panel."""
        # Clear existing
        if hasattr(self, 'log_placeholder'):
            self.log_placeholder.destroy()
            
        for widget in self.trade_list_frame.winfo_children()[1:]:  # Keep header
            widget.destroy()
            
        # Update count
        self.trade_count_label.configure(text=f"{len(trades)} trades")
        
        # Add trade rows
        for trade in trades:
            row = ctk.CTkFrame(self.trade_list_frame, fg_color="transparent", height=28)
            row.pack(fill="x", pady=1)
            
            pnl_color = SUCCESS_GREEN if trade['pnl'] >= 0 else DANGER_RED
            type_color = SUCCESS_GREEN if trade['type'] == 'BUY' else DANGER_RED
            
            # Use entry_idx as time (or format from backtest_data if available)
            time_str = str(trade.get('entry_idx', '-'))
            if hasattr(self, 'backtest_data') and self.backtest_data is not None:
                try:
                    idx = trade.get('entry_idx', 0)
                    if idx < len(self.backtest_data):
                        time_str = self.backtest_data.index[idx].strftime('%m/%d %H:%M')
                except:
                    pass
            
            ctk.CTkLabel(row, text=time_str, font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY, width=80).pack(side="left", padx=5)
            ctk.CTkLabel(row, text=trade['type'], font=ctk.CTkFont(size=11), text_color=type_color, width=50).pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"{trade['entry']:.5f}", font=ctk.CTkFont(size=11), text_color=TEXT_PRIMARY, width=70).pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"{trade['exit']:.5f}", font=ctk.CTkFont(size=11), text_color=TEXT_PRIMARY, width=70).pack(side="left", padx=5)
            ctk.CTkLabel(row, text=f"${trade['pnl']:+.2f}", font=ctk.CTkFont(size=11), text_color=pnl_color, width=70).pack(side="left", padx=5)
            
    def _export_trades(self):
        """Export trades to CSV."""
        if not self.trades:
            messagebox.showinfo("Export", "No trades to export. Run a backtest first.")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json")],
            title="Export Trade Log"
        )
        
        if filepath:
            try:
                if filepath.endswith('.json'):
                    with open(filepath, 'w') as f:
                        json.dump(self.trades, f, indent=2)
                else:
                    with open(filepath, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['time', 'type', 'entry', 'exit', 'pnl'])
                        writer.writeheader()
                        for trade in self.trades:
                            writer.writerow({
                                'time': trade['time'],
                                'type': trade['type'],
                                'entry': trade['entry'],
                                'exit': trade['exit'],
                                'pnl': trade['pnl']
                            })
                messagebox.showinfo("Export", f"Trades exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))
    
    def _open_tv_chart(self):
        """Deprecated - chart is now embedded directly."""
        pass


