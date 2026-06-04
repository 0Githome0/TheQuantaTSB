# --- START OF FILE base_strategy.py ---
"""
Base Strategy Module
================================================================================
Abstract base class for all trading strategies. Custom strategies extend this
class to define their own entry/exit logic that can be used in both backtesting
and live trading.

Usage:
    1. Create a new file in /strategies folder
    2. Extend BaseStrategy class
    3. Implement should_enter() and should_exit()
    4. Strategy Manager will auto-load and register it
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import logging
import numpy as np
import pandas as pd

log = logging.getLogger('Strategy')


class SignalDirection(Enum):
    """Signal direction enum."""
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


class SignalGrade(Enum):
    """Signal quality grades."""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass
class StrategySignal:
    """
    Standardized signal output from strategies.
    All strategies must return this format.
    """
    # Core signal
    direction: str = "WAIT"      # BUY, SELL, WAIT
    pair: str = ""
    timeframe: str = ""
    
    # Quality
    grade: str = "C"             # A+, A, B, C, D
    confidence: float = 0.0      # 0.0 - 1.0
    score: float = 0.0           # 0 - 100
    
    # Prices
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Multiple TPs
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    
    # Position sizing
    risk_pips: float = 0.0
    risk_reward: float = 0.0
    suggested_lot_size: float = 0.01
    
    # Reasoning
    entry_reasons: List[str] = field(default_factory=list)
    
    # Metadata
    strategy_name: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility."""
        return {
            'direction': self.direction,
            'pair': self.pair,
            'timeframe': self.timeframe,
            'grade': self.grade,
            'confidence': self.confidence,
            'grade_score': self.score / 100,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'tp1': self.tp1,
            'tp2': self.tp2,
            'tp3': self.tp3,
            'risk_pips': self.risk_pips,
            'risk_reward': self.risk_reward,
            'position_size': self.suggested_lot_size,
            'entry_reasons': self.entry_reasons,
            'strategy': self.strategy_name,
            'timestamp': self.timestamp
        }


@dataclass
class ExitSignal:
    """Exit signal from strategy."""
    should_exit: bool = False
    exit_type: str = ""          # TP_HIT, SL_HIT, SIGNAL_EXIT, MANUAL
    exit_price: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    total_signals: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_rr: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    def update(self, won: bool, pnl_pips: float, rr: float):
        """Update metrics with new trade."""
        self.total_signals += 1
        if won:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.win_rate = self.win_count / self.total_signals * 100 if self.total_signals > 0 else 0


class BaseStrategy(ABC):
    """
    Abstract Base Strategy Class
    ============================
    
    All custom strategies must extend this class and implement:
    - should_enter(): Analyzes data and returns entry signal
    - should_exit(): Monitors position and returns exit signal
    
    Optional overrides:
    - calculate_stops(): Custom SL/TP logic
    - calculate_position_size(): Custom lot sizing
    - on_signal_generated(): Callback after signal
    
    Example:
    --------
    class MyStrategy(BaseStrategy):
        name = "My Awesome Strategy"
        version = "1.0.0"
        
        def should_enter(self, data: Dict, pair: str, timeframe: str) -> Optional[StrategySignal]:
            close = data['close']
            ema20 = data['indicators']['ema20']
            
            if close > ema20:
                return self.create_buy_signal(pair, timeframe, confidence=0.8, reasons=["Price above EMA20"])
            return None
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY IDENTITY (Override in subclass)
    # ═══════════════════════════════════════════════════════════════════════════
    name: str = "Base Strategy"
    version: str = "1.0.0"
    description: str = "Abstract base strategy - do not use directly"
    author: str = "Unknown"
    
    # Strategy characteristics
    timeframes: List[str] = ["H1", "H4"]           # Supported timeframes
    pairs: List[str] = ["EURUSD", "GBPUSD"]        # Supported pairs
    strategy_type: str = "trend"                    # trend, reversal, breakout, scalp
    
    # Default parameters (override in subclass)
    default_params: Dict = {
        'lookback_bars': 100,
        'risk_percent': 1.0,
        'min_rr': 1.5,
        'max_trades_per_day': 5,
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, params: Dict = None):
        """
        Initialize strategy with parameters.
        
        Args:
            params: Custom parameters to override defaults
        """
        self.params = {**self.default_params, **(params or {})}
        self.metrics = StrategyMetrics()
        self.signals_today = 0
        self.last_signal_time: Optional[datetime] = None
        self.is_active = True
        
        log.info(f"Strategy initialized: {self.name} v{self.version}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS (Must implement in subclass)
    # ═══════════════════════════════════════════════════════════════════════════
    @abstractmethod
    def should_enter(self, 
                     data: Dict, 
                     pair: str, 
                     timeframe: str) -> Optional[StrategySignal]:
        """
        Analyze market data and determine if entry conditions are met.
        
        Args:
            data: Dictionary containing:
                - 'ohlcv': DataFrame with OHLCV data
                - 'indicators': Dict of calculated indicators
                - 'tick': Current tick data
                - 'account': Account info
            pair: Currency pair (e.g., "EURUSD")  
            timeframe: Timeframe (e.g., "H1")
            
        Returns:
            StrategySignal if entry conditions met, None otherwise
            
        Example:
            def should_enter(self, data, pair, timeframe):
                df = data['ohlcv']
                close = df['close'].iloc[-1]
                ema50 = df['ema50'].iloc[-1]
                rsi = df['rsi'].iloc[-1]
                
                if close > ema50 and rsi > 50 and rsi < 70:
                    return self.create_buy_signal(
                        pair=pair,
                        timeframe=timeframe,
                        confidence=0.75,
                        reasons=["Price above EMA50", "RSI bullish"]
                    )
                return None
        """
        pass
    
    @abstractmethod
    def should_exit(self, 
                    position: Dict, 
                    data: Dict) -> Optional[ExitSignal]:
        """
        Analyze open position and determine if exit conditions are met.
        
        Args:
            position: Position data including:
                - 'ticket': Position ticket
                - 'symbol': Symbol
                - 'direction': BUY/SELL
                - 'entry_price': Entry price
                - 'current_price': Current price
                - 'pnl_pips': Current P&L in pips
                - 'sl': Stop loss
                - 'tp': Take profit
            data: Same as should_enter
            
        Returns:
            ExitSignal if exit conditions met, None otherwise
            
        Example:
            def should_exit(self, position, data):
                df = data['ohlcv']
                rsi = df['rsi'].iloc[-1]
                
                # Exit long if RSI overbought
                if position['direction'] == 'BUY' and rsi > 80:
                    return ExitSignal(
                        should_exit=True,
                        exit_type='SIGNAL_EXIT',
                        reasons=["RSI overbought - taking profit"]
                    )
                return None
        """
        pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS (Use in your strategy)
    # ═══════════════════════════════════════════════════════════════════════════
    def create_buy_signal(self,
                          pair: str,
                          timeframe: str,
                          entry_price: float = 0,
                          confidence: float = 0.5,
                          reasons: List[str] = None) -> StrategySignal:
        """Helper to create a BUY signal."""
        return StrategySignal(
            direction="BUY",
            pair=pair,
            timeframe=timeframe,
            confidence=confidence,
            grade=self._confidence_to_grade(confidence),
            score=confidence * 100,
            entry_price=entry_price,
            entry_reasons=reasons or [],
            strategy_name=self.name,
            timestamp=datetime.now().isoformat()
        )
    
    def create_sell_signal(self,
                           pair: str,
                           timeframe: str,
                           entry_price: float = 0,
                           confidence: float = 0.5,
                           reasons: List[str] = None) -> StrategySignal:
        """Helper to create a SELL signal."""
        return StrategySignal(
            direction="SELL",
            pair=pair,
            timeframe=timeframe,
            confidence=confidence,
            grade=self._confidence_to_grade(confidence),
            score=confidence * 100,
            entry_price=entry_price,
            entry_reasons=reasons or [],
            strategy_name=self.name,
            timestamp=datetime.now().isoformat()
        )
    
    def _confidence_to_grade(self, confidence: float) -> str:
        """Convert confidence score to letter grade."""
        if confidence >= 0.90: return "A+"
        if confidence >= 0.80: return "A"
        if confidence >= 0.70: return "B"
        if confidence >= 0.60: return "C"
        return "D"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STOP LOSS / TAKE PROFIT CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    def calculate_stops(self, 
                        signal: StrategySignal, 
                        data: Dict) -> StrategySignal:
        """
        Calculate SL/TP levels. Override for custom logic.
        
        Default: ATR-based stops with 1.5:1 R:R
        """
        df = data.get('ohlcv')
        if df is None or len(df) < 14:
            return signal
        
        # Calculate ATR
        atr = self._calculate_atr(df, period=14)
        pip_value = 0.0001 if 'JPY' not in signal.pair else 0.01
        
        # Get current price
        if signal.entry_price == 0:
            signal.entry_price = df['close'].iloc[-1]
        
        # Default SL: 1.5x ATR, TP: 2.25x ATR (1.5:1 R:R)
        sl_distance = atr * 1.5
        tp_distance = atr * 2.25
        
        if signal.direction == "BUY":
            signal.stop_loss = signal.entry_price - sl_distance
            signal.take_profit = signal.entry_price + tp_distance
            signal.tp1 = signal.entry_price + (sl_distance * 1.0)  # 1:1
            signal.tp2 = signal.entry_price + (sl_distance * 2.0)  # 2:1
            signal.tp3 = signal.entry_price + (sl_distance * 3.0)  # 3:1
        else:
            signal.stop_loss = signal.entry_price + sl_distance
            signal.take_profit = signal.entry_price - tp_distance
            signal.tp1 = signal.entry_price - (sl_distance * 1.0)
            signal.tp2 = signal.entry_price - (sl_distance * 2.0)
            signal.tp3 = signal.entry_price - (sl_distance * 3.0)
        
        signal.risk_pips = sl_distance / pip_value
        signal.risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
        
        return signal
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period:
            return 0.001  # Default
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        return np.mean(tr[-period:])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POSITION SIZING
    # ═══════════════════════════════════════════════════════════════════════════
    def calculate_position_size(self, 
                                signal: StrategySignal, 
                                account_balance: float,
                                account_currency: str = "USD") -> float:
        """
        Calculate position size based on risk. Override for custom logic.
        
        Default: Risk X% of account per trade
        """
        risk_percent = self.params.get('risk_percent', 1.0)
        risk_amount = account_balance * (risk_percent / 100)
        
        if signal.risk_pips <= 0:
            return 0.01
        
        # Approximate pip value (simplified)
        pip_value_per_lot = 10  # $10 per pip for 1 lot on major pairs
        
        lot_size = risk_amount / (signal.risk_pips * pip_value_per_lot)
        lot_size = round(max(0.01, min(2.0, lot_size)), 2)
        
        return lot_size
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INDICATOR HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series: pd.Series, 
             fast: int = 12, 
             slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series: pd.Series, 
                        period: int = 20, 
                        std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATION & CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════════
    def validate_signal(self, signal: StrategySignal) -> bool:
        """Validate signal before execution. Override for custom validation."""
        # Check daily limit
        if self.signals_today >= self.params.get('max_trades_per_day', 5):
            log.warning(f"{self.name}: Daily signal limit reached")
            return False
        
        # Check minimum R:R
        min_rr = self.params.get('min_rr', 1.5)
        if signal.risk_reward < min_rr:
            log.warning(f"{self.name}: R:R {signal.risk_reward:.2f} below minimum {min_rr}")
            return False
        
        return True
    
    def on_signal_generated(self, signal: StrategySignal) -> None:
        """Callback when signal is generated. Override for custom logic."""
        self.signals_today += 1
        self.last_signal_time = datetime.now()
        self.metrics.total_signals += 1
        log.info(f"{self.name}: Signal generated - {signal.direction} {signal.pair}")
    
    def on_trade_closed(self, won: bool, pnl_pips: float, rr: float) -> None:
        """Callback when trade closes. Override for custom logic."""
        self.metrics.update(won, pnl_pips, rr)
    
    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at start of new trading day)."""
        self.signals_today = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INFO METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    def get_info(self) -> Dict:
        """Get strategy information."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'type': self.strategy_type,
            'timeframes': self.timeframes,
            'pairs': self.pairs,
            'params': self.params,
            'is_active': self.is_active
        }
    
    def get_metrics(self) -> Dict:
        """Get strategy performance metrics."""
        return {
            'total_signals': self.metrics.total_signals,
            'wins': self.metrics.win_count,
            'losses': self.metrics.loss_count,
            'win_rate': round(self.metrics.win_rate, 1),
            'profit_factor': self.metrics.profit_factor,
        }
    
    def __repr__(self) -> str:
        return f"<{self.name} v{self.version} ({self.strategy_type})>"


# --- END OF FILE base_strategy.py ---
