# --- START OF FILE breakout_strategy.py ---
"""
Breakout Strategy
=================
Detects and trades breakouts from consolidation zones.

Entry:
- BUY: Price breaks above resistance with volume confirmation
- SELL: Price breaks below support with volume confirmation

Features:
- Dynamic support/resistance detection
- Volatility squeeze detection (low ATR before breakout)
- Volume confirmation
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from src.core.base_strategy import BaseStrategy, StrategySignal, ExitSignal


class BreakoutStrategy(BaseStrategy):
    """Breakout trading with dynamic S/R levels."""
    
    # Strategy Identity
    name = "Breakout Trader"
    version = "1.0.0"
    description = "Trades breakouts from consolidation with volume confirmation"
    author = "QuantaTSB"
    strategy_type = "breakout"
    
    # Supported instruments
    timeframes = ["H1", "H4", "D1"]
    pairs = ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD", "*"]
    
    # Parameters
    default_params = {
        'lookback': 20,           # Bars to find S/R
        'atr_period': 14,
        'atr_squeeze_pct': 0.7,   # ATR < 70% of avg = squeeze
        'breakout_buffer': 0.2,   # ATR multiplier for breakout confirmation
        'volume_increase': 1.5,   # Volume must be 1.5x average
        'hold_bars_min': 3,       # Min bars to hold
        'risk_percent': 1.5,
        'min_rr': 2.0,
        'max_trades_per_day': 3,
    }
    
    def should_enter(self, data: Dict, pair: str, timeframe: str) -> Optional[StrategySignal]:
        """Check for breakout entry."""
        df = data.get('ohlcv')
        if df is None or len(df) < self.params['lookback'] + 10:
            return None
        
        lookback = self.params['lookback']
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df else np.ones(len(df))
        
        # Current values
        current_close = close[-1]
        current_high = high[-1]
        current_low = low[-1]
        current_volume = volume[-1]
        
        # Calculate S/R levels (high/low of lookback period, excluding last bar)
        resistance = np.max(high[-lookback-1:-1])
        support = np.min(low[-lookback-1:-1])
        
        # ATR for buffer
        atr = self._calculate_atr_np(high, low, close, self.params['atr_period'])
        breakout_buffer = atr * self.params['breakout_buffer']
        
        # Check for volatility squeeze (low ATR)
        atr_history = []
        for i in range(-20, -1):
            atr_history.append(self._calculate_atr_np(
                high[:i], low[:i], close[:i], self.params['atr_period']
            ))
        avg_atr = np.mean(atr_history) if atr_history else atr
        is_squeeze = atr < avg_atr * self.params['atr_squeeze_pct']
        
        # Volume confirmation
        avg_volume = np.mean(volume[-20:-1])
        volume_confirm = current_volume > avg_volume * self.params['volume_increase']
        
        # BULLISH BREAKOUT
        if current_close > resistance + breakout_buffer:
            confidence = 0.65
            reasons = [f"🔥 Breakout above resistance {resistance:.5f}"]
            
            if is_squeeze:
                confidence += 0.15
                reasons.append("📊 Volatility squeeze before breakout")
            
            if volume_confirm:
                confidence += 0.10
                reasons.append("📈 Volume confirms breakout")
            
            # Strong close
            if current_close > current_high * 0.95:
                confidence += 0.05
                reasons.append("💪 Strong close near session high")
            
            return self.create_buy_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=current_close,
                confidence=min(0.95, confidence),
                reasons=reasons
            )
        
        # BEARISH BREAKOUT
        if current_close < support - breakout_buffer:
            confidence = 0.65
            reasons = [f"🔥 Breakdown below support {support:.5f}"]
            
            if is_squeeze:
                confidence += 0.15
                reasons.append("📊 Volatility squeeze before breakdown")
            
            if volume_confirm:
                confidence += 0.10
                reasons.append("📉 Volume confirms breakdown")
            
            if current_close < current_low * 1.05:
                confidence += 0.05
                reasons.append("💪 Strong close near session low")
            
            return self.create_sell_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=current_close,
                confidence=min(0.95, confidence),
                reasons=reasons
            )
        
        return None
    
    def _calculate_atr_np(self, high, low, close, period: int) -> float:
        """Calculate ATR using numpy."""
        if len(high) < period + 1:
            return 0.001
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        return np.mean(tr[-period:])
    
    def should_exit(self, position: Dict, data: Dict) -> Optional[ExitSignal]:
        """Check for exit - let SL/TP handle most exits."""
        # This strategy relies primarily on SL/TP
        # Can add re-entry into consolidation as exit
        df = data.get('ohlcv')
        if df is None:
            return None
        
        direction = position.get('direction', '')
        entry_price = position.get('entry_price', 0)
        current_price = position.get('current_price', 0)
        
        # Exit if price returns to consolidation zone
        lookback = self.params['lookback']
        high = df['high'].values
        low = df['low'].values
        
        resistance = np.max(high[-lookback-1:-1])
        support = np.min(low[-lookback-1:-1])
        
        if direction == 'BUY' and current_price < resistance:
            # Price fell back below breakout level
            pnl_pips = position.get('pnl_pips', 0)
            if pnl_pips < 0:
                return ExitSignal(
                    should_exit=True,
                    exit_type='SIGNAL_EXIT',
                    reasons=["⚠️ Breakout failed - price returned to range"]
                )
        
        if direction == 'SELL' and current_price > support:
            pnl_pips = position.get('pnl_pips', 0)
            if pnl_pips < 0:
                return ExitSignal(
                    should_exit=True,
                    exit_type='SIGNAL_EXIT',
                    reasons=["⚠️ Breakdown failed - price returned to range"]
                )
        
        return None
    
    def calculate_stops(self, signal: StrategySignal, data: Dict) -> StrategySignal:
        """Custom stop calculation for breakouts - wider stops."""
        df = data.get('ohlcv')
        if df is None:
            return signal
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        atr = self._calculate_atr_np(high, low, close, self.params['atr_period'])
        pip_value = 0.0001 if 'JPY' not in signal.pair else 0.01
        
        entry = signal.entry_price if signal.entry_price > 0 else close[-1]
        
        # Breakout stops: 2x ATR for SL, 4x ATR for TP (2:1 R:R)
        sl_distance = atr * 2.0
        
        if signal.direction == "BUY":
            signal.stop_loss = entry - sl_distance
            signal.take_profit = entry + (sl_distance * 2)
            signal.tp1 = entry + (sl_distance * 1.0)
            signal.tp2 = entry + (sl_distance * 2.0)
            signal.tp3 = entry + (sl_distance * 3.0)
        else:
            signal.stop_loss = entry + sl_distance
            signal.take_profit = entry - (sl_distance * 2)
            signal.tp1 = entry - (sl_distance * 1.0)
            signal.tp2 = entry - (sl_distance * 2.0)
            signal.tp3 = entry - (sl_distance * 3.0)
        
        signal.risk_pips = sl_distance / pip_value
        signal.risk_reward = 2.0
        
        return signal


# --- END OF FILE breakout_strategy.py ---
