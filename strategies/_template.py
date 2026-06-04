# --- START OF FILE _template.py ---
"""
Strategy Template
=================
Copy this file and rename it to create your own strategy.

Example: my_strategy.py

IMPORTANT:
1. Rename the class to match your strategy
2. Change the 'name' attribute to your strategy name
3. Implement should_enter() with your entry logic
4. Implement should_exit() with your exit logic
"""

import pandas as pd
from typing import Optional, Dict
from src.core.base_strategy import BaseStrategy, StrategySignal, ExitSignal


class MyCustomStrategy(BaseStrategy):  # <-- Rename this class
    """Your strategy description here."""
    
    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY IDENTITY (Change these!)
    # ═══════════════════════════════════════════════════════════════════════
    name = "My Custom Strategy"      # <-- Your strategy name
    version = "1.0.0"
    description = "Describe what your strategy does"
    author = "Your Name"
    strategy_type = "trend"          # trend, reversal, breakout, scalp
    
    # Which timeframes and pairs to trade
    timeframes = ["H1", "H4"]        # <-- Add your timeframes
    pairs = ["EURUSD", "GBPUSD"]     # <-- Add your pairs, or ["*"] for all
    
    # Your parameters (easy to adjust)
    default_params = {
        # Add your custom parameters here
        'my_param_1': 20,
        'my_param_2': 50,
        'risk_percent': 1.0,
        'min_rr': 1.5,
        'max_trades_per_day': 5,
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # ENTRY LOGIC (Required - Implement this!)
    # ═══════════════════════════════════════════════════════════════════════
    def should_enter(self, data: Dict, pair: str, timeframe: str) -> Optional[StrategySignal]:
        """
        Your entry logic here.
        
        Args:
            data: Contains 'ohlcv' DataFrame with columns: open, high, low, close, volume
            pair: The currency pair (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "H1")
            
        Returns:
            StrategySignal if entry conditions met, or None
            
        Available helper methods:
            self.ema(series, period)      - Exponential Moving Average
            self.sma(series, period)      - Simple Moving Average
            self.rsi(series, period)      - Relative Strength Index
            self.macd(series, fast, slow, signal) - MACD
            self.bollinger_bands(series, period, std) - Bollinger Bands
            
        To create signals:
            return self.create_buy_signal(pair, timeframe, confidence=0.8, reasons=[...])
            return self.create_sell_signal(pair, timeframe, confidence=0.8, reasons=[...])
        """
        # Get OHLCV data
        df = data.get('ohlcv')
        if df is None or len(df) < 50:
            return None
        
        close = df['close']
        
        # ───────────────────────────────────────────────────────────────────
        # EXAMPLE: Simple EMA strategy (replace with your logic)
        # ───────────────────────────────────────────────────────────────────
        ema_short = self.ema(close, 10)
        ema_long = self.ema(close, 20)
        
        # Buy when short EMA crosses above long EMA
        if ema_short.iloc[-2] <= ema_long.iloc[-2] and ema_short.iloc[-1] > ema_long.iloc[-1]:
            return self.create_buy_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=close.iloc[-1],
                confidence=0.75,  # 0.0 - 1.0
                reasons=[
                    "EMA crossover detected",
                    "Short EMA crossed above Long EMA",
                    # Add more reasons...
                ]
            )
        
        # Sell when short EMA crosses below long EMA
        if ema_short.iloc[-2] >= ema_long.iloc[-2] and ema_short.iloc[-1] < ema_long.iloc[-1]:
            return self.create_sell_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=close.iloc[-1],
                confidence=0.75,
                reasons=[
                    "EMA crossover detected",
                    "Short EMA crossed below Long EMA",
                ]
            )
        
        return None  # No signal
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXIT LOGIC (Required - Implement this!)
    # ═══════════════════════════════════════════════════════════════════════
    def should_exit(self, position: Dict, data: Dict) -> Optional[ExitSignal]:
        """
        Your exit logic here.
        
        Args:
            position: Contains:
                - 'ticket': Position ticket number
                - 'symbol': Symbol name
                - 'direction': 'BUY' or 'SELL'
                - 'entry_price': Entry price
                - 'current_price': Current price
                - 'pnl_pips': Current profit/loss in pips
                - 'sl': Stop loss
                - 'tp': Take profit
            data: Same as should_enter
            
        Returns:
            ExitSignal if exit conditions met, or None (let SL/TP handle)
        """
        df = data.get('ohlcv')
        if df is None:
            return None
        
        direction = position.get('direction', '')
        pnl_pips = position.get('pnl_pips', 0)
        
        # ───────────────────────────────────────────────────────────────────
        # EXAMPLE: Exit on opposite signal (replace with your logic)
        # ───────────────────────────────────────────────────────────────────
        close = df['close']
        ema_short = self.ema(close, 10)
        ema_long = self.ema(close, 20)
        
        # Exit long on bearish crossover
        if direction == 'BUY':
            if ema_short.iloc[-2] >= ema_long.iloc[-2] and ema_short.iloc[-1] < ema_long.iloc[-1]:
                return ExitSignal(
                    should_exit=True,
                    exit_type='SIGNAL_EXIT',
                    reasons=["Opposite signal - bearish crossover"]
                )
        
        # Exit short on bullish crossover
        if direction == 'SELL':
            if ema_short.iloc[-2] <= ema_long.iloc[-2] and ema_short.iloc[-1] > ema_long.iloc[-1]:
                return ExitSignal(
                    should_exit=True,
                    exit_type='SIGNAL_EXIT',
                    reasons=["Opposite signal - bullish crossover"]
                )
        
        return None  # Let SL/TP handle the exit


# --- END OF FILE _template.py ---
