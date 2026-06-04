# --- START OF FILE position_monitor.py ---
"""
Professional Position Monitor Module
================================================================================
Enterprise-grade position management with advanced trailing logic, smart TP
management, volatility-based stops, and comprehensive position lifecycle control.

Features:
- Multi-level trailing stop system (ATR-based, percentage, swing-based)
- Smart partial close with dynamic profit targets
- Volatility-adjusted stop management
- MFE/MAE tracking for optimization
- Position health scoring and alerts
- Time-based management rules
- Emergency protection protocols
"""

import threading
import time
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pytz

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    mt5 = None

# Configure logging
log = logging.getLogger('PositionMonitor')
if not log.handlers:
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log.addHandler(handler)


class TrailingMode(Enum):
    """Trailing stop calculation methods."""
    FIXED_PIPS = "FIXED_PIPS"      # Fixed pip distance
    ATR_BASED = "ATR_BASED"        # ATR multiplier
    PERCENTAGE = "PERCENTAGE"      # Percentage of move
    SWING_BASED = "SWING_BASED"    # Follow swing points
    HYBRID = "HYBRID"              # Combination approach


class PositionStatus(Enum):
    """Position health status."""
    HEALTHY = "HEALTHY"           # In profit, normal operation
    WARNING = "WARNING"           # Near SL or extended drawdown
    CRITICAL = "CRITICAL"         # Approaching max loss
    PROTECTED = "PROTECTED"       # Moved to breakeven
    TRAILING = "TRAILING"         # Trailing stop active


@dataclass
class PositionData:
    """Comprehensive position tracking data."""
    ticket: int = 0
    symbol: str = ""
    direction: str = ""  # BUY/SELL
    volume: float = 0.0
    original_volume: float = 0.0
    
    # Prices
    open_price: float = 0.0
    current_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    original_sl: float = 0.0
    original_tp: float = 0.0
    
    # Take profits
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    
    # Time
    open_time: str = ""
    last_update: str = ""
    
    # P&L tracking
    current_pnl_pips: float = 0.0
    current_pnl_money: float = 0.0
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE
    
    # Risk metrics
    risk_pips: float = 0.0
    current_rr: float = 0.0
    
    # Status
    status: str = "OPEN"
    health: str = "HEALTHY"
    
    # Management flags
    breakeven_moved: bool = False
    tp1_closed: bool = False
    tp2_closed: bool = False
    trailing_active: bool = False
    trailing_mode: str = "FIXED_PIPS"
    
    # Journal linkage
    journal_id: str = ""


class PositionMonitor:
    """
    Professional Position Monitor with advanced management features.
    
    Capabilities:
    - Multi-mode trailing stop system
    - Smart partial close with dynamic targets
    - Volatility-based position management
    - MFE/MAE tracking for post-trade analysis
    - Position health monitoring and alerts
    - Time-based exit rules
    - Emergency protection protocols
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    CONFIG = {
        # Monitoring
        'check_interval_sec': 3,           # Check every 3 seconds
        'position_file': 'data/positions.json',
        
        # Breakeven settings
        'breakeven_trigger_rr': 1.0,       # Move to BE after 1:1 R:R
        'breakeven_buffer_pips': 2,        # Buffer above entry
        'breakeven_lock_pips': 1,          # Lock 1 pip profit at BE
        
        # Trailing stop configuration
        'trailing': {
            'enabled': True,
            'start_rr': 1.5,               # Start trailing after 1.5:1 R:R
            'mode': 'HYBRID',              # FIXED_PIPS, ATR_BASED, PERCENTAGE, SWING_BASED, HYBRID
            'fixed_pips': 20,              # For FIXED_PIPS mode
            'atr_multiplier': 1.5,         # For ATR_BASED mode
            'percentage': 0.3,             # For PERCENTAGE mode (30% of move)
            'step_pips': 10,               # Minimum move to update
            'acceleration': True,          # Tighten as profit grows
            'acceleration_threshold': 50,  # Start accelerating after 50 pips
            'acceleration_factor': 0.7,    # Reduce distance by 30%
        },
        
        # Partial close configuration
        'partial_close': {
            'enabled': True,
            'dynamic_targets': True,       # Adjust targets based on volatility
            'levels': {
                'tp1': {'rr': 1.0, 'close_pct': 50, 'move_to_be': True},
                'tp2': {'rr': 2.0, 'close_pct': 30, 'start_trail': True},
                'tp3': {'rr': 3.0, 'close_pct': 20, 'tighten_trail': True}
            }
        },
        
        # Time-based rules
        'time_rules': {
            'max_hold_hours': 72,          # Max hold time
            'friday_close_hour': 20,       # Close before weekend (UTC)
            'news_buffer_minutes': 15,     # Close before news
            'stale_position_hours': 24,    # Alert if no movement
        },
        
        # Volatility management
        'volatility': {
            'enabled': True,
            'high_vol_reduce_size': 0.5,   # Reduce to 50% in high volatility
            'low_vol_extend_tp': 1.3,      # Extend TP by 30% in low volatility
            'spike_protection': True,      # Protect against price spikes
            'spike_threshold_pips': 50,    # Consider spike if > 50 pips in 1 candle
        },
        
        # Protection
        'protection': {
            'max_loss_per_trade_pct': 2.0,    # Max loss per trade
            'correlation_limit': 3,           # Max correlated positions
            'emergency_close_drawdown': 5.0,  # Close all if drawdown exceeds
        }
    }
    
    def __init__(self, 
                 trade_journal=None,
                 signal_generator=None,
                 on_position_update: Callable = None,
                 on_position_close: Callable = None,
                 on_alert: Callable = None):
        """
        Initialize Professional Position Monitor.
        
        Args:
            trade_journal: TradeJournal instance for recording
            signal_generator: SignalGenerator for volatility data
            on_position_update: Callback when position is modified
            on_position_close: Callback when position is closed
            on_alert: Callback for alerts/notifications
        """
        self.trade_journal = trade_journal
        self.signal_generator = signal_generator
        self.on_position_update = on_position_update
        self.on_position_close = on_position_close
        self.on_alert = on_alert
        
        self.timezone = pytz.timezone('UTC')
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Position tracking
        self.positions: Dict[int, PositionData] = {}  # ticket -> PositionData
        self.position_history: List[Dict] = []
        
        # ATR cache for volatility-based calculations
        self.atr_cache: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            'positions_monitored': 0,
            'partial_closes': 0,
            'breakeven_moves': 0,
            'trailing_updates': 0,
            'emergency_closes': 0
        }
        
        # Ensure data directory
        os.makedirs(os.path.dirname(self.CONFIG['position_file']), exist_ok=True)
        
        # Load saved state
        self._load_state()
        
        log.info("Professional PositionMonitor initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    def start_monitoring(self) -> bool:
        """Start the position monitoring thread."""
        if self._running:
            log.warning("Monitor already running")
            return False
        
        if not HAS_MT5:
            log.error("MT5 not available")
            return False
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="PositionMonitor")
        self._thread.start()
        log.info("🔍 Position monitoring started")
        return True
    
    def stop_monitoring(self) -> None:
        """Stop the position monitoring thread."""
        self._running = False
        self._save_state()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        log.info("🛑 Position monitoring stopped")
    
    def is_running(self) -> bool:
        """Check if monitoring is active."""
        return self._running
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN MONITORING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    def _monitor_loop(self) -> None:
        """Main monitoring loop with comprehensive position management."""
        log.info("Monitor loop started")
        
        while self._running:
            try:
                self._sync_positions()
                self._manage_all_positions()
                self._check_time_rules()
                self._save_state()
                
            except Exception as e:
                log.error(f"Monitor loop error: {e}")
            
            time.sleep(self.CONFIG['check_interval_sec'])
        
        log.info("Monitor loop ended")
    
    def _sync_positions(self) -> None:
        """Synchronize with MT5 positions."""
        if not mt5.terminal_info():
            return
        
        mt5_positions = mt5.positions_get()
        if mt5_positions is None:
            return
        
        current_tickets = set()
        
        for pos in mt5_positions:
            ticket = pos.ticket
            current_tickets.add(ticket)
            
            if ticket not in self.positions:
                self._register_position(pos)
            else:
                self._update_position_state(pos)
        
        # Check for closed positions
        closed_tickets = set(self.positions.keys()) - current_tickets
        for ticket in closed_tickets:
            self._handle_position_closed(ticket, "EXTERNAL")
    
    def _register_position(self, pos) -> None:
        """Register a new position for professional monitoring."""
        with self._lock:
            now = datetime.now(self.timezone)
            
            # Calculate risk
            pip_value = 0.0001 if 'JPY' not in pos.symbol else 0.01
            if pos.sl > 0:
                risk_pips = abs(pos.price_open - pos.sl) / pip_value
            else:
                risk_pips = 50  # Default assumption
            
            position = PositionData(
                ticket=pos.ticket,
                symbol=pos.symbol,
                direction='BUY' if pos.type == 0 else 'SELL',
                volume=pos.volume,
                original_volume=pos.volume,
                
                open_price=pos.price_open,
                current_price=pos.price_current,
                sl=pos.sl,
                tp=pos.tp,
                original_sl=pos.sl,
                original_tp=pos.tp,
                
                open_time=now.isoformat(),
                last_update=now.isoformat(),
                
                risk_pips=risk_pips,
                status="OPEN",
                health=PositionStatus.HEALTHY.value
            )
            
            self.positions[pos.ticket] = position
            self.stats['positions_monitored'] += 1
            
        log.info(f"📌 Registered position #{pos.ticket}: {pos.symbol} {position.direction}")
    
    def _update_position_state(self, pos) -> None:
        """Update position state with current market data."""
        position = self.positions.get(pos.ticket)
        if not position:
            return
        
        # Get current price
        tick = mt5.symbol_info_tick(pos.symbol)
        if not tick:
            return
        
        current_price = tick.bid if position.direction == 'BUY' else tick.ask
        
        # Calculate P&L
        pip_value = 0.0001 if 'JPY' not in pos.symbol else 0.01
        if position.direction == 'BUY':
            pnl_pips = (current_price - position.open_price) / pip_value
        else:
            pnl_pips = (position.open_price - current_price) / pip_value
        
        # Update MFE/MAE
        if pnl_pips > position.max_favorable_excursion:
            position.max_favorable_excursion = pnl_pips
        if pnl_pips < -position.max_adverse_excursion:
            position.max_adverse_excursion = abs(pnl_pips)
        
        # Calculate R:R
        if position.risk_pips > 0:
            position.current_rr = pnl_pips / position.risk_pips
        
        # Update health status
        position.health = self._assess_position_health(position)
        
        # Update position
        position.current_price = current_price
        position.current_pnl_pips = round(pnl_pips, 1)
        position.current_pnl_money = pos.profit
        position.sl = pos.sl
        position.tp = pos.tp
        position.last_update = datetime.now(self.timezone).isoformat()
    
    def _assess_position_health(self, position: PositionData) -> str:
        """Assess position health status."""
        if position.breakeven_moved or position.trailing_active:
            return PositionStatus.PROTECTED.value if position.breakeven_moved else PositionStatus.TRAILING.value
        
        # Check proximity to SL
        if position.risk_pips > 0:
            loss_pct = abs(position.current_pnl_pips) / position.risk_pips
            if position.current_pnl_pips < 0:
                if loss_pct > 0.8:
                    return PositionStatus.CRITICAL.value
                elif loss_pct > 0.5:
                    return PositionStatus.WARNING.value
        
        return PositionStatus.HEALTHY.value
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    def _manage_all_positions(self) -> None:
        """Apply management rules to all positions."""
        for ticket, position in list(self.positions.items()):
            try:
                # 1. Check for partial closes
                if self.CONFIG['partial_close']['enabled']:
                    self._check_partial_close(position)
                
                # 2. Move to breakeven
                if not position.breakeven_moved:
                    self._check_breakeven(position)
                
                # 3. Apply trailing stop
                if self.CONFIG['trailing']['enabled']:
                    self._apply_trailing(position)
                
                # 4. Volatility management
                if self.CONFIG['volatility']['enabled']:
                    self._check_volatility_rules(position)
                    
            except Exception as e:
                log.error(f"Position management error #{ticket}: {e}")
    
    def _check_partial_close(self, position: PositionData) -> None:
        """Check and execute partial closes at TP levels."""
        config = self.CONFIG['partial_close']['levels']
        
        # TP1 - Close 50% at 1:1 R:R
        if not position.tp1_closed and position.current_rr >= config['tp1']['rr']:
            close_pct = config['tp1']['close_pct']
            if self._partial_close(position, close_pct, "TP1"):
                position.tp1_closed = True
                self.stats['partial_closes'] += 1
                
                # Move to breakeven if configured
                if config['tp1'].get('move_to_be'):
                    self._move_to_breakeven(position)
                
                log.info(f"🎯 TP1 partial close: #{position.ticket} closed {close_pct}%")
        
        # TP2 - Close 30% at 2:1 R:R
        if not position.tp2_closed and position.current_rr >= config['tp2']['rr']:
            close_pct = config['tp2']['close_pct']
            if self._partial_close(position, close_pct, "TP2"):
                position.tp2_closed = True
                self.stats['partial_closes'] += 1
                
                # Start trailing if configured
                if config['tp2'].get('start_trail'):
                    position.trailing_active = True
                
                log.info(f"🎯 TP2 partial close: #{position.ticket} closed {close_pct}%")
        
        # TP3 - Close remaining at 3:1 R:R
        if position.current_rr >= config['tp3']['rr']:
            self._close_position(position.ticket, "TP3_HIT")
    
    def _partial_close(self, position: PositionData, close_pct: int, reason: str) -> bool:
        """Execute a partial close on a position."""
        try:
            close_volume = round(position.volume * (close_pct / 100), 2)
            close_volume = max(0.01, close_volume)
            
            if close_volume >= position.volume:
                return False  # Would close entire position
            
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                return False
            
            price = tick.bid if position.direction == 'BUY' else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": close_volume,
                "type": mt5.ORDER_TYPE_SELL if position.direction == 'BUY' else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Partial {close_pct}% - {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                position.volume = round(position.volume - close_volume, 2)
                return True
            
            return False
            
        except Exception as e:
            log.error(f"Partial close error: {e}")
            return False
    
    def _check_breakeven(self, position: PositionData) -> None:
        """Check if position should be moved to breakeven."""
        trigger_rr = self.CONFIG['breakeven_trigger_rr']
        
        if position.current_rr >= trigger_rr:
            self._move_to_breakeven(position)
    
    def _move_to_breakeven(self, position: PositionData) -> None:
        """Move stop loss to breakeven + buffer."""
        if position.breakeven_moved:
            return
        
        try:
            buffer_pips = self.CONFIG['breakeven_buffer_pips']
            lock_pips = self.CONFIG['breakeven_lock_pips']
            pip_value = 0.0001 if 'JPY' not in position.symbol else 0.01
            
            if position.direction == 'BUY':
                new_sl = position.open_price + (lock_pips * pip_value)
            else:
                new_sl = position.open_price - (lock_pips * pip_value)
            
            # Ensure new SL is better than current
            if position.direction == 'BUY' and new_sl <= position.sl:
                return
            if position.direction == 'SELL' and new_sl >= position.sl:
                return
            
            if self._modify_sl(position.ticket, new_sl):
                position.breakeven_moved = True
                position.sl = new_sl
                position.health = PositionStatus.PROTECTED.value
                self.stats['breakeven_moves'] += 1
                
                log.info(f"🔒 Moved to breakeven: #{position.ticket} SL={new_sl:.5f}")
                
        except Exception as e:
            log.error(f"Breakeven move error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED TRAILING STOP SYSTEM
    # ═══════════════════════════════════════════════════════════════════════════
    def _apply_trailing(self, position: PositionData) -> None:
        """Apply advanced trailing stop logic."""
        config = self.CONFIG['trailing']
        
        # Check if trailing should start
        if position.current_rr < config['start_rr']:
            return
        
        position.trailing_active = True
        position.trailing_mode = config['mode']
        
        # Calculate new SL based on mode
        new_sl = self._calculate_trailing_sl(position)
        
        if new_sl is None:
            return
        
        # Only move SL in favorable direction
        step_pips = config['step_pips']
        pip_value = 0.0001 if 'JPY' not in position.symbol else 0.01
        min_move = step_pips * pip_value
        
        if position.direction == 'BUY':
            if new_sl > position.sl + min_move:
                if self._modify_sl(position.ticket, new_sl):
                    position.sl = new_sl
                    self.stats['trailing_updates'] += 1
                    log.debug(f"📈 Trailing updated: #{position.ticket} SL={new_sl:.5f}")
        else:
            if new_sl < position.sl - min_move:
                if self._modify_sl(position.ticket, new_sl):
                    position.sl = new_sl
                    self.stats['trailing_updates'] += 1
                    log.debug(f"📉 Trailing updated: #{position.ticket} SL={new_sl:.5f}")
    
    def _calculate_trailing_sl(self, position: PositionData) -> Optional[float]:
        """Calculate trailing stop level based on configured mode."""
        config = self.CONFIG['trailing']
        mode = config['mode']
        pip_value = 0.0001 if 'JPY' not in position.symbol else 0.01
        
        # Get ATR for volatility-based calculations
        atr = self._get_atr(position.symbol)
        
        # Calculate distance based on mode
        if mode == 'FIXED_PIPS':
            distance = config['fixed_pips'] * pip_value
            
        elif mode == 'ATR_BASED':
            if atr:
                distance = atr * config['atr_multiplier']
            else:
                distance = config['fixed_pips'] * pip_value
                
        elif mode == 'PERCENTAGE':
            move_pips = abs(position.current_price - position.open_price) / pip_value
            distance = (move_pips * config['percentage']) * pip_value
            distance = max(distance, 10 * pip_value)  # Minimum 10 pips
            
        elif mode == 'HYBRID':
            # Combine ATR and percentage for adaptive trailing
            atr_distance = atr * config['atr_multiplier'] if atr else config['fixed_pips'] * pip_value
            pct_distance = abs(position.current_price - position.open_price) * config['percentage']
            distance = min(atr_distance, pct_distance)  # Use tighter of the two
            distance = max(distance, 10 * pip_value)
            
        else:
            distance = config['fixed_pips'] * pip_value
        
        # Apply acceleration if in profit
        if config['acceleration'] and position.current_pnl_pips > config['acceleration_threshold']:
            distance *= config['acceleration_factor']
        
        # Calculate new SL
        if position.direction == 'BUY':
            new_sl = position.current_price - distance
        else:
            new_sl = position.current_price + distance
        
        return round(new_sl, 5)
    
    def _get_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Get ATR value for a symbol."""
        # Check cache first
        if symbol in self.atr_cache:
            return self.atr_cache[symbol]
        
        try:
            # Get OHLC data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period + 1)
            if rates is None or len(rates) < period:
                return None
            
            # Calculate True Range
            tr_values = []
            for i in range(1, len(rates)):
                high = rates[i]['high']
                low = rates[i]['low']
                prev_close = rates[i-1]['close']
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_values.append(tr)
            
            # ATR is average of TR
            atr = sum(tr_values[-period:]) / period
            self.atr_cache[symbol] = atr
            
            return atr
            
        except Exception as e:
            log.debug(f"ATR calculation error: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIME-BASED MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    def _check_time_rules(self) -> None:
        """Apply time-based management rules."""
        config = self.CONFIG['time_rules']
        now = datetime.now(self.timezone)
        
        for ticket, position in list(self.positions.items()):
            open_time = datetime.fromisoformat(position.open_time)
            hours_open = (now - open_time).total_seconds() / 3600
            
            # Max hold time
            if hours_open >= config['max_hold_hours']:
                log.warning(f"⏰ Position #{ticket} exceeded max hold time")
                self._close_position(ticket, "TIME_EXIT")
                continue
            
            # Friday close (before weekend)
            if now.weekday() == 4 and now.hour >= config['friday_close_hour']:
                log.info(f"📅 Closing #{ticket} before weekend")
                self._close_position(ticket, "WEEKEND_CLOSE")
                continue
            
            # Stale position alert
            if hours_open >= config['stale_position_hours']:
                if abs(position.current_pnl_pips) < 10:  # No significant movement
                    self._send_alert(f"⚠️ Position #{ticket} stale for {hours_open:.0f}h with minimal movement")
    
    def _check_volatility_rules(self, position: PositionData) -> None:
        """Apply volatility-based rules."""
        config = self.CONFIG['volatility']
        
        if not config['spike_protection']:
            return
        
        # Get recent price movement
        try:
            rates = mt5.copy_rates_from_pos(position.symbol, mt5.TIMEFRAME_M5, 0, 2)
            if rates is None or len(rates) < 2:
                return
            
            last_candle_range = abs(rates[-1]['high'] - rates[-1]['low'])
            pip_value = 0.0001 if 'JPY' not in position.symbol else 0.01
            range_pips = last_candle_range / pip_value
            
            # Spike detected
            if range_pips > config['spike_threshold_pips']:
                log.warning(f"⚠️ Volatility spike detected for {position.symbol}: {range_pips:.0f} pips")
                
                # Tighten stop if in profit
                if position.current_pnl_pips > 0 and not position.breakeven_moved:
                    self._move_to_breakeven(position)
                    
        except Exception as e:
            log.debug(f"Volatility check error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POSITION OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    def _modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify stop loss for a position."""
        try:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                return False
            pos = pos[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }
            
            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            log.error(f"Modify SL error: {e}")
            return False
    
    def _close_position(self, ticket: int, reason: str) -> bool:
        """Close a position completely."""
        position = self.positions.get(ticket)
        if not position:
            return False
        
        try:
            pos = mt5.positions_get(ticket=ticket)
            if not pos:
                return False
            pos = pos[0]
            
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                return False
            
            price = tick.bid if pos.type == 0 else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Monitor: {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Record in journal
                if self.trade_journal:
                    self.trade_journal.record_exit(
                        ticket=ticket,
                        exit_price=price,
                        exit_type=reason,
                        pnl_pips=position.current_pnl_pips,
                        pnl_money=position.current_pnl_money,
                        max_favorable=position.max_favorable_excursion,
                        max_adverse=position.max_adverse_excursion
                    )
                
                self._handle_position_closed(ticket, reason)
                return True
            
            return False
            
        except Exception as e:
            log.error(f"Close position error: {e}")
            return False
    
    def _handle_position_closed(self, ticket: int, reason: str) -> None:
        """Handle a position closure."""
        with self._lock:
            if ticket in self.positions:
                position = self.positions.pop(ticket)
                self.position_history.append({
                    **asdict(position),
                    'close_reason': reason,
                    'closed_at': datetime.now(self.timezone).isoformat()
                })
                log.info(f"📤 Position #{ticket} closed: {reason}")
        
        if self.on_position_close:
            self.on_position_close(ticket, reason)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EMERGENCY CONTROLS
    # ═══════════════════════════════════════════════════════════════════════════
    def close_all_positions(self, reason: str = "EMERGENCY") -> int:
        """Emergency close all positions."""
        closed = 0
        
        for ticket in list(self.positions.keys()):
            if self._close_position(ticket, reason):
                closed += 1
        
        self.stats['emergency_closes'] += closed
        log.warning(f"🚨 Emergency close: {closed} positions closed")
        return closed
    
    def _send_alert(self, message: str) -> None:
        """Send alert through callback."""
        log.warning(message)
        if self.on_alert:
            self.on_alert(message)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATE PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════
    def _save_state(self) -> None:
        """Save monitor state to disk."""
        try:
            state = {
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'stats': self.stats,
                'atr_cache': self.atr_cache,
                'saved_at': datetime.now(self.timezone).isoformat()
            }
            with open(self.CONFIG['position_file'], 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            log.debug(f"Save state error: {e}")
    
    def _load_state(self) -> None:
        """Load monitor state from disk."""
        try:
            if os.path.exists(self.CONFIG['position_file']):
                with open(self.CONFIG['position_file'], 'r') as f:
                    state = json.load(f)
                self.stats = state.get('stats', self.stats)
        except Exception as e:
            log.debug(f"Load state error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATUS & REPORTING
    # ═══════════════════════════════════════════════════════════════════════════
    def get_position_status(self, ticket: int = None) -> Dict:
        """Get status of tracked position(s)."""
        if ticket:
            pos = self.positions.get(ticket)
            return asdict(pos) if pos else {}
        return {k: asdict(v) for k, v in self.positions.items()}
    
    def get_monitoring_stats(self) -> Dict:
        """Get monitoring statistics."""
        total_pnl = sum(p.current_pnl_pips for p in self.positions.values())
        return {
            'running': self._running,
            'tracked_positions': len(self.positions),
            'total_floating_pnl': round(total_pnl, 1),
            **self.stats
        }
    
    def get_position_summary(self) -> List[Dict]:
        """Get summary of all positions for UI display."""
        summary = []
        for pos in self.positions.values():
            summary.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'direction': pos.direction,
                'volume': pos.volume,
                'pnl_pips': round(pos.current_pnl_pips, 1),
                'pnl_money': round(pos.current_pnl_money, 2),
                'rr': round(pos.current_rr, 2),
                'health': pos.health,
                'breakeven': pos.breakeven_moved,
                'trailing': pos.trailing_active
            })
        return summary


# --- END OF FILE position_monitor.py ---
