# --- START OF FILE scalper_strategy.py ---
"""
Professional Scalper Strategy
============================
High-frequency scalping strategy for quick profits.

Entry:
- Multiple indicator confluence (RSI + MACD + Bollinger)
- Tight entry at key levels
- Quick profit targets

Features:
- Fast EMA trend filter
- RSI momentum confirmation
- Bollinger Band mean reversion
- Tight stops (10-20 pips)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from src.core.base_strategy import BaseStrategy, StrategySignal, ExitSignal


class ScalperStrategy(BaseStrategy):
    """High-frequency scalping with multi-indicator confluence."""
    
    # Strategy Identity
    name = "Pro Scalper"
    version = "1.0.0"
    description = "High-frequency scalping with tight stops and quick profits"
    author = "QuantaTSB"
    strategy_type = "scalp"
    
    # Supported instruments
    timeframes = ["M5", "M15"]
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    
    # Parameters
    default_params = {
        'trend_ema': 50,
        'fast_ema': 8,
        'rsi_period': 7,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'bb_period': 20,
        'bb_std': 2.0,
        'macd_fast': 8,
        'macd_slow': 17,
        'macd_signal': 9,
        'sl_pips': 15,
        'tp_pips': 20,
        'risk_percent': 0.5,
        'min_rr': 1.3,
        'max_trades_per_day': 10,
    }
    
    def should_enter(self, data: Dict, pair: str, timeframe: str) -> Optional[StrategySignal]:
        """Check for scalp entry with multi-indicator confluence."""
        df = data.get('ohlcv')
        if df is None or len(df) < self.params['trend_ema'] + 5:
            return None
        
        close = df['close']
        current_close = close.iloc[-1]
        
        # Trend filter
        trend_ema = self.ema(close, self.params['trend_ema'])
        is_uptrend = current_close > trend_ema.iloc[-1]
        is_downtrend = current_close < trend_ema.iloc[-1]
        
        # RSI
        rsi = self.rsi(close, self.params['rsi_period'])
        rsi_now = rsi.iloc[-1]
        
        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = self.bollinger_bands(
            close, 
            self.params['bb_period'], 
            self.params['bb_std']
        )
        
        # MACD
        macd_line, signal_line, histogram = self.macd(
            close,
            self.params['macd_fast'],
            self.params['macd_slow'],
            self.params['macd_signal']
        )
        
        macd_now = macd_line.iloc[-1]
        macd_prev = macd_line.iloc[-2]
        signal_now = signal_line.iloc[-1]
        signal_prev = signal_line.iloc[-2]
        
        # MACD crossovers
        macd_bullish_cross = macd_prev <= signal_prev and macd_now > signal_now
        macd_bearish_cross = macd_prev >= signal_prev and macd_now < signal_now
        
        # ═══════════════════════════════════════════════════════════════════
        # BULLISH SCALP ENTRY
        # ═══════════════════════════════════════════════════════════════════
        bullish_signals = 0
        reasons = []
        
        # 1. In uptrend
        if is_uptrend:
            bullish_signals += 1
            reasons.append(f"📈 Price above EMA{self.params['trend_ema']}")
        
        # 2. RSI oversold bounce
        if rsi_now < 40 and rsi_now > self.params['rsi_oversold']:
            bullish_signals += 1
            reasons.append(f"📊 RSI={rsi_now:.1f} bouncing from oversold")
        
        # 3. Price near lower Bollinger Band
        if current_close <= lower_bb.iloc[-1] * 1.002:  # Within 0.2%
            bullish_signals += 1
            reasons.append("📉 Price at lower Bollinger Band")
        
        # 4. MACD bullish cross
        if macd_bullish_cross:
            bullish_signals += 1
            reasons.append("✅ MACD bullish crossover")
        
        # Need 3+ confirmations for entry
        if bullish_signals >= 3 and is_uptrend:
            confidence = 0.60 + (bullish_signals * 0.10)
            
            return self.create_buy_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=current_close,
                confidence=min(0.95, confidence),
                reasons=reasons
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # BEARISH SCALP ENTRY
        # ═══════════════════════════════════════════════════════════════════
        bearish_signals = 0
        reasons = []
        
        # 1. In downtrend
        if is_downtrend:
            bearish_signals += 1
            reasons.append(f"📉 Price below EMA{self.params['trend_ema']}")
        
        # 2. RSI overbought rejection
        if rsi_now > 60 and rsi_now < self.params['rsi_overbought']:
            bearish_signals += 1
            reasons.append(f"📊 RSI={rsi_now:.1f} rejecting from overbought")
        
        # 3. Price near upper Bollinger Band
        if current_close >= upper_bb.iloc[-1] * 0.998:
            bearish_signals += 1
            reasons.append("📈 Price at upper Bollinger Band")
        
        # 4. MACD bearish cross
        if macd_bearish_cross:
            bearish_signals += 1
            reasons.append("✅ MACD bearish crossover")
        
        if bearish_signals >= 3 and is_downtrend:
            confidence = 0.60 + (bearish_signals * 0.10)
            
            return self.create_sell_signal(
                pair=pair,
                timeframe=timeframe,
                entry_price=current_close,
                confidence=min(0.95, confidence),
                reasons=reasons
            )
        
        return None
    
    def should_exit(self, position: Dict, data: Dict) -> Optional[ExitSignal]:
        """Quick exit on momentum reversal."""
        df = data.get('ohlcv')
        if df is None:
            return None
        
        close = df['close']
        direction = position.get('direction', '')
        pnl_pips = position.get('pnl_pips', 0)
        
        # RSI extreme exit
        rsi = self.rsi(close, self.params['rsi_period'])
        rsi_now = rsi.iloc[-1]
        
        if direction == 'BUY':
            # Exit if RSI hits overbought and we have profit
            if rsi_now > self.params['rsi_overbought'] and pnl_pips > 5:
                return ExitSignal(
                    should_exit=True,
                    exit_type='SIGNAL_EXIT',
                    exit_price=close.iloc[-1],
                    reasons=["🎯 RSI overbought - securing profit"]
                )
        
        if direction == 'SELL':
            if rsi_now < self.params['rsi_oversold'] and pnl_pips > 5:
                return ExitSignal(
                    should_exit=True,
                    exit_type='SIGNAL_EXIT',
                    exit_price=close.iloc[-1],
                    reasons=["🎯 RSI oversold - securing profit"]
                )
        
        return None
    
    def calculate_stops(self, signal: StrategySignal, data: Dict) -> StrategySignal:
        """Tight scalping stops."""
        pip_value = 0.0001 if 'JPY' not in signal.pair else 0.01
        
        entry = signal.entry_price if signal.entry_price > 0 else 0
        if entry == 0:
            df = data.get('ohlcv')
            if df is not None:
                entry = df['close'].iloc[-1]
        
        signal.entry_price = entry
        
        sl_distance = self.params['sl_pips'] * pip_value
        tp_distance = self.params['tp_pips'] * pip_value
        
        if signal.direction == "BUY":
            signal.stop_loss = entry - sl_distance
            signal.take_profit = entry + tp_distance
            signal.tp1 = entry + (sl_distance * 1.0)
            signal.tp2 = entry + (sl_distance * 1.5)
            signal.tp3 = entry + (sl_distance * 2.0)
        else:
            signal.stop_loss = entry + sl_distance
            signal.take_profit = entry - tp_distance
            signal.tp1 = entry - (sl_distance * 1.0)
            signal.tp2 = entry - (sl_distance * 1.5)
            signal.tp3 = entry - (sl_distance * 2.0)
        
        signal.risk_pips = self.params['sl_pips']
        signal.risk_reward = self.params['tp_pips'] / self.params['sl_pips']
        
        return signal


# --- END OF FILE scalper_strategy.py ---
