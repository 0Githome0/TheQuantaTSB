# --- START OF FILE ema_crossover.py ---
"""
EMA Crossover Strategy
======================
Classic EMA crossover strategy with RSI confirmation.

Entry:
- BUY: Fast EMA crosses above Slow EMA + RSI > 50
- SELL: Fast EMA crosses below Slow EMA + RSI < 50

Exit:
- Opposite crossover or RSI extremes
"""

import pandas as pd
from typing import Optional, Dict
from src.core.base_strategy import BaseStrategy, StrategySignal, ExitSignal


class EMACrossoverStrategy(BaseStrategy):
    """EMA Crossover with RSI Confirmation."""
    
    # Strategy Identity
    name = "EMA Crossover"
    version = "1.0.0"
    description = "Classic EMA crossover with RSI filter"
    author = "QuantaTSB"
    strategy_type = "trend"
    
    # Supported instruments
    timeframes = ["M15", "M30", "H1", "H4"]
    pairs = ["*"]  # All pairs
    
    # Parameters
    default_params = {
        'fast_ema': 8,
        'slow_ema': 21,
        'rsi_period': 14,
        'rsi_buy_level': 50,
        'rsi_sell_level': 50,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'lookback_bars': 100,
        'risk_percent': 1.0,
        'min_rr': 1.5,
        'max_trades_per_day': 5,
    }
    
    def should_enter(self, data: Dict, pair: str, timeframe: str) -> Optional[StrategySignal]:
        """Check for EMA crossover entry."""
        df = data.get('ohlcv')
        if df is None or len(df) < self.params['slow_ema'] + 5:
            return None
        
        # Calculate indicators
        close = df['close']
        fast_ema = self.ema(close, self.params['fast_ema'])
        slow_ema = self.ema(close, self.params['slow_ema'])
        rsi = self.rsi(close, self.params['rsi_period'])
        
        # Current and previous values
        fast_now = fast_ema.iloc[-1]
        fast_prev = fast_ema.iloc[-2]
        slow_now = slow_ema.iloc[-1]
        slow_prev = slow_ema.iloc[-2]
        rsi_now = rsi.iloc[-1]
        
        # Check for crossover
        bullish_cross = fast_prev <= slow_prev and fast_now > slow_now
        bearish_cross = fast_prev >= slow_prev and fast_now < slow_now
        
        # BUY Signal
        if bullish_cross and rsi_now > self.params['rsi_buy_level']:
            confidence = 0.70
            
            # Boost confidence if strong RSI
            if rsi_now > 55 and rsi_now < self.params['rsi_overbought']:
                confidence += 0.10
            
            # Boost for trend alignment
            if close.iloc[-1] > slow_now:
                confidence += 0.10
            
            return self.create_buy_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=close.iloc[-1],
                confidence=min(0.95, confidence),
                reasons=[
                    f"✅ EMA{self.params['fast_ema']} crossed above EMA{self.params['slow_ema']}",
                    f"📈 RSI={rsi_now:.1f} confirms bullish momentum",
                    f"🔥 Trend continuation signal"
                ]
            )
        
        # SELL Signal
        if bearish_cross and rsi_now < self.params['rsi_sell_level']:
            confidence = 0.70
            
            if rsi_now < 45 and rsi_now > self.params['rsi_oversold']:
                confidence += 0.10
            
            if close.iloc[-1] < slow_now:
                confidence += 0.10
            
            return self.create_sell_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=close.iloc[-1],
                confidence=min(0.95, confidence),
                reasons=[
                    f"✅ EMA{self.params['fast_ema']} crossed below EMA{self.params['slow_ema']}",
                    f"📉 RSI={rsi_now:.1f} confirms bearish momentum",
                    f"❄️ Trend continuation signal"
                ]
            )
        
        return None
    
    def should_exit(self, position: Dict, data: Dict) -> Optional[ExitSignal]:
        """Check for exit conditions."""
        df = data.get('ohlcv')
        if df is None or len(df) < self.params['slow_ema']:
            return None
        
        close = df['close']
        rsi = self.rsi(close, self.params['rsi_period'])
        rsi_now = rsi.iloc[-1]
        
        direction = position.get('direction', '')
        
        # Exit long if RSI overbought
        if direction == 'BUY' and rsi_now > self.params['rsi_overbought']:
            return ExitSignal(
                should_exit=True,
                exit_type='SIGNAL_EXIT',
                exit_price=close.iloc[-1],
                reasons=["📊 RSI overbought - taking profit"]
            )
        
        # Exit short if RSI oversold
        if direction == 'SELL' and rsi_now < self.params['rsi_oversold']:
            return ExitSignal(
                should_exit=True,
                exit_type='SIGNAL_EXIT',
                exit_price=close.iloc[-1],
                reasons=["📊 RSI oversold - taking profit"]
            )
        
        # Check for opposite crossover
        fast_ema = self.ema(close, self.params['fast_ema'])
        slow_ema = self.ema(close, self.params['slow_ema'])
        
        fast_now = fast_ema.iloc[-1]
        fast_prev = fast_ema.iloc[-2]
        slow_now = slow_ema.iloc[-1]
        slow_prev = slow_ema.iloc[-2]
        
        if direction == 'BUY' and fast_prev >= slow_prev and fast_now < slow_now:
            return ExitSignal(
                should_exit=True,
                exit_type='SIGNAL_EXIT',
                exit_price=close.iloc[-1],
                reasons=["⚠️ Bearish EMA crossover - signal invalidated"]
            )
        
        if direction == 'SELL' and fast_prev <= slow_prev and fast_now > slow_now:
            return ExitSignal(
                should_exit=True,
                exit_type='SIGNAL_EXIT',
                exit_price=close.iloc[-1],
                reasons=["⚠️ Bullish EMA crossover - signal invalidated"]
            )
        
        return None


# --- END OF FILE ema_crossover.py ---
