# --- START OF FILE guardian.py ---
"""
Professional Guardian Module
================================================================================
Enterprise-grade trading protection system with multi-level monitoring,
intelligent failure detection, notification system, and automated recovery.

Features:
- Multi-tier heartbeat monitoring with escalation
- Intelligent failure detection and classification
- Multi-channel notification system (logs, callbacks, file-based)
- Automated position protection and emergency protocols
- State persistence and crash recovery
- Performance-based circuit breakers
- Connection health monitoring
"""

import threading
import time
import json
import os
import logging
import socket
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
log = logging.getLogger('Guardian')
if not log.handlers:
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log.addHandler(handler)
    
    # File handler for persistent logs
    try:
        os.makedirs('logs', exist_ok=True)
        fh = logging.FileHandler('logs/guardian.log', mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        log.addHandler(fh)
    except Exception:
        pass


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class FailureType(Enum):
    """Classification of failure types."""
    APP_UNRESPONSIVE = "APP_UNRESPONSIVE"
    MT5_DISCONNECTED = "MT5_DISCONNECTED"
    NETWORK_FAILURE = "NETWORK_FAILURE"
    DRAWDOWN_EXCEEDED = "DRAWDOWN_EXCEEDED"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    POSITION_ANOMALY = "POSITION_ANOMALY"
    PRICE_SPIKE = "PRICE_SPIKE"
    UNKNOWN = "UNKNOWN"


class ProtectionAction(Enum):
    """Actions that can be taken by the Guardian."""
    ALERT_ONLY = "ALERT_ONLY"
    TIGHTEN_STOPS = "TIGHTEN_STOPS"
    MOVE_TO_BREAKEVEN = "MOVE_TO_BREAKEVEN"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSE_ALL = "CLOSE_ALL"
    PAUSE_TRADING = "PAUSE_TRADING"


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: str  # HEALTHY, DEGRADED, FAILED
    last_check: str
    details: str = ""
    response_time_ms: float = 0.0


@dataclass
class ProtectionEvent:
    """Record of a protection event."""
    timestamp: str
    failure_type: str
    alert_level: str
    action_taken: str
    details: Dict = field(default_factory=dict)
    positions_affected: int = 0
    resolved: bool = False


@dataclass
class GuardianState:
    """Complete guardian state for persistence."""
    is_active: bool = False
    app_alive: bool = True
    mt5_connected: bool = False
    trading_paused: bool = False
    pause_until: str = ""
    
    # Counters
    total_alerts: int = 0
    total_interventions: int = 0
    positions_protected: int = 0
    emergency_closes: int = 0
    
    # Current session
    session_start: str = ""
    last_heartbeat: str = ""
    last_health_check: str = ""
    
    # Recent events
    recent_events: List[Dict] = field(default_factory=list)
    
    # Health status
    health_status: Dict = field(default_factory=dict)


class Guardian:
    """
    Professional Trading Guardian with comprehensive protection.
    
    Capabilities:
    - Multi-tier heartbeat monitoring with escalation
    - MT5 connection health monitoring
    - Network connectivity checks
    - Drawdown and loss-based circuit breakers
    - Multi-channel notification system
    - Automated emergency protocols
    - State persistence and recovery
    - Performance analytics
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    CONFIG = {
        # Core settings
        'check_interval_sec': 5,
        'state_file': 'data/guardian_state.json',
        'alert_file': 'data/guardian_alerts.json',
        
        # Heartbeat configuration
        'heartbeat': {
            'warning_timeout_sec': 30,    # Warn if no heartbeat for 30s
            'critical_timeout_sec': 60,   # Critical if no heartbeat for 60s
            'emergency_timeout_sec': 120, # Emergency action after 120s
        },
        
        # MT5 connection monitoring
        'mt5': {
            'check_interval_sec': 10,
            'reconnect_attempts': 3,
            'reconnect_delay_sec': 5,
        },
        
        # Network monitoring
        'network': {
            'enabled': True,
            'test_hosts': ['8.8.8.8', '1.1.1.1'],
            'timeout_sec': 5,
        },
        
        # Protection thresholds
        'protection': {
            'emergency_sl_pips': 30,      # Emergency SL distance
            'max_daily_loss_pct': 5.0,    # Daily loss limit
            'max_consecutive_losses': 5,   # Consecutive loss limit
            'max_drawdown_pct': 10.0,     # Max drawdown limit
            'spike_threshold_pct': 2.0,   # Price spike threshold
        },
        
        # Actions
        'actions': {
            'on_app_crash': 'CLOSE_ALL',       # CLOSE_ALL, TIGHTEN_STOPS, ALERT_ONLY
            'on_mt5_disconnect': 'ALERT_ONLY', # Wait for reconnection
            'on_network_failure': 'TIGHTEN_STOPS',
            'on_drawdown_exceeded': 'CLOSE_ALL',
            'on_consecutive_losses': 'PAUSE_TRADING',
        },
        
        # Circuit breakers
        'circuit_breakers': {
            'enabled': True,
            'pause_duration_minutes': 60,  # Pause trading for 1 hour
            'daily_reset': True,           # Reset at start of day
        },
        
        # Notifications
        'notifications': {
            'log_all': True,
            'file_alerts': True,
            'callback_enabled': True,
        }
    }
    
    def __init__(self, 
                 trade_journal=None, 
                 signal_generator=None,
                 on_alert: Callable = None,
                 on_emergency: Callable = None):
        """
        Initialize Professional Guardian.
        
        Args:
            trade_journal: TradeJournal instance
            signal_generator: SignalGenerator for drawdown tracking
            on_alert: Callback for alerts
            on_emergency: Callback for emergency actions
        """
        self.trade_journal = trade_journal
        self.signal_generator = signal_generator
        self.on_alert = on_alert
        self.on_emergency = on_emergency
        
        self.timezone = pytz.timezone('UTC')
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # State
        self.state = GuardianState()
        self.last_heartbeat: Optional[datetime] = None
        self.heartbeat_level = 'NORMAL'  # NORMAL, WARNING, CRITICAL
        
        # Protected positions cache
        self.protected_positions: Dict[int, Dict] = {}
        
        # Event history
        self.events: List[ProtectionEvent] = []
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Ensure directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load saved state
        self._load_state()
        
        log.info("🛡️ Professional Guardian initialized")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    def start_guardian(self) -> bool:
        """Start the guardian watchdog."""
        if self._running:
            log.warning("Guardian already running")
            return False
        
        self._running = True
        self.state.is_active = True
        self.state.session_start = datetime.now(self.timezone).isoformat()
        self.last_heartbeat = datetime.now(self.timezone)
        
        self._thread = threading.Thread(target=self._guardian_loop, daemon=True, name="Guardian")
        self._thread.start()
        
        log.info("🛡️ Guardian ACTIVATED - Protecting your trades")
        self._send_alert(AlertLevel.INFO, "Guardian activated", {"action": "START"})
        
        return True
    
    def stop_guardian(self) -> None:
        """Stop the guardian watchdog."""
        self._running = False
        self.state.is_active = False
        self._save_state()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        
        log.info("🛡️ Guardian DEACTIVATED")
    
    def is_running(self) -> bool:
        """Check if guardian is active."""
        return self._running
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEARTBEAT SYSTEM
    # ═══════════════════════════════════════════════════════════════════════════
    def send_heartbeat(self) -> None:
        """
        Called by main application to signal it's alive.
        Should be called every 5-10 seconds.
        """
        with self._lock:
            self.last_heartbeat = datetime.now(self.timezone)
            self.state.last_heartbeat = self.last_heartbeat.isoformat()
            self.state.app_alive = True
            self.heartbeat_level = 'NORMAL'
    
    def _check_heartbeat(self) -> Tuple[bool, str]:
        """
        Check heartbeat status with escalation.
        
        Returns:
            Tuple[is_alive, status_level]
        """
        if self.last_heartbeat is None:
            return True, 'UNKNOWN'
        
        config = self.CONFIG['heartbeat']
        elapsed = (datetime.now(self.timezone) - self.last_heartbeat).total_seconds()
        
        if elapsed >= config['emergency_timeout_sec']:
            return False, 'EMERGENCY'
        elif elapsed >= config['critical_timeout_sec']:
            return False, 'CRITICAL'
        elif elapsed >= config['warning_timeout_sec']:
            return True, 'WARNING'
        
        return True, 'NORMAL'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN GUARDIAN LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    def _guardian_loop(self) -> None:
        """Main guardian monitoring loop."""
        log.info("Guardian loop started")
        
        while self._running:
            try:
                # 1. Heartbeat monitoring
                self._monitor_heartbeat()
                
                # 2. MT5 connection monitoring
                self._monitor_mt5_connection()
                
                # 3. Network monitoring
                if self.CONFIG['network']['enabled']:
                    self._monitor_network()
                
                # 4. Position monitoring
                self._monitor_positions()
                
                # 5. Performance monitoring
                self._monitor_performance()
                
                # 6. Update health status
                self._update_health_status()
                
                # 7. Persist state
                self._save_state()
                
            except Exception as e:
                log.error(f"Guardian loop error: {e}")
            
            time.sleep(self.CONFIG['check_interval_sec'])
        
        log.info("Guardian loop ended")
    
    def _monitor_heartbeat(self) -> None:
        """Monitor application heartbeat with escalation."""
        is_alive, level = self._check_heartbeat()
        
        if self.heartbeat_level != level:
            self.heartbeat_level = level
            
            if level == 'WARNING':
                self._send_alert(
                    AlertLevel.WARNING,
                    "Application heartbeat delayed",
                    {"timeout": "warning", "action": "monitoring"}
                )
            elif level == 'CRITICAL':
                self._send_alert(
                    AlertLevel.CRITICAL,
                    "Application unresponsive - preparing intervention",
                    {"timeout": "critical", "action": "preparing"}
                )
            elif level == 'EMERGENCY':
                self.state.app_alive = False
                self._handle_failure(FailureType.APP_UNRESPONSIVE)
    
    def _monitor_mt5_connection(self) -> None:
        """Monitor MT5 connection health."""
        if not HAS_MT5:
            return
        
        start_time = time.time()
        
        try:
            info = mt5.terminal_info()
            response_time = (time.time() - start_time) * 1000
            
            is_connected = info is not None and info.connected
            
            self.health_checks['mt5'] = HealthCheck(
                component='MT5',
                status='HEALTHY' if is_connected else 'FAILED',
                last_check=datetime.now(self.timezone).isoformat(),
                details=f"Connected: {is_connected}",
                response_time_ms=response_time
            )
            
            if not is_connected and self.state.mt5_connected:
                self.state.mt5_connected = False
                self._handle_failure(FailureType.MT5_DISCONNECTED)
            elif is_connected:
                self.state.mt5_connected = True
                
        except Exception as e:
            self.health_checks['mt5'] = HealthCheck(
                component='MT5',
                status='FAILED',
                last_check=datetime.now(self.timezone).isoformat(),
                details=str(e)
            )
            self.state.mt5_connected = False
    
    def _monitor_network(self) -> None:
        """Monitor network connectivity."""
        config = self.CONFIG['network']
        
        for host in config['test_hosts']:
            try:
                start_time = time.time()
                socket.setdefaulttimeout(config['timeout_sec'])
                socket.create_connection((host, 53), timeout=config['timeout_sec'])
                response_time = (time.time() - start_time) * 1000
                
                self.health_checks['network'] = HealthCheck(
                    component='Network',
                    status='HEALTHY',
                    last_check=datetime.now(self.timezone).isoformat(),
                    details=f"Connected to {host}",
                    response_time_ms=response_time
                )
                return  # Success - no need to check other hosts
                
            except (socket.timeout, socket.error):
                continue
        
        # All hosts failed
        self.health_checks['network'] = HealthCheck(
            component='Network',
            status='FAILED',
            last_check=datetime.now(self.timezone).isoformat(),
            details="All network checks failed"
        )
        self._handle_failure(FailureType.NETWORK_FAILURE)
    
    def _monitor_positions(self) -> None:
        """Monitor positions for anomalies."""
        if not HAS_MT5 or not mt5.terminal_info():
            return
        
        positions = mt5.positions_get()
        if not positions:
            return
        
        current_tickets = set()
        
        for pos in positions:
            current_tickets.add(pos.ticket)
            
            # Track position
            if pos.ticket not in self.protected_positions:
                self.protected_positions[pos.ticket] = {
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'sl': pos.sl,
                    'protected_since': datetime.now(self.timezone).isoformat()
                }
            
            # Check for price spikes
            self._check_price_spike(pos)
        
        # Clean up closed positions
        closed = set(self.protected_positions.keys()) - current_tickets
        for ticket in closed:
            del self.protected_positions[ticket]
    
    def _check_price_spike(self, pos) -> None:
        """Check for abnormal price spikes."""
        try:
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                return
            
            current_price = tick.bid if pos.type == 0 else tick.ask
            move_pct = abs(current_price - pos.price_open) / pos.price_open * 100
            
            threshold = self.CONFIG['protection']['spike_threshold_pct']
            
            if move_pct > threshold:
                self._send_alert(
                    AlertLevel.WARNING,
                    f"Price spike detected: {pos.symbol} moved {move_pct:.2f}%",
                    {"symbol": pos.symbol, "move_pct": move_pct}
                )
                
        except Exception as e:
            log.debug(f"Price spike check error: {e}")
    
    def _monitor_performance(self) -> None:
        """Monitor trading performance for circuit breakers."""
        if not self.CONFIG['circuit_breakers']['enabled']:
            return
        
        config = self.CONFIG['protection']
        
        # Check daily P&L (if signal_generator tracks it)
        if self.signal_generator and hasattr(self.signal_generator, 'daily_pnl'):
            daily_pnl = getattr(self.signal_generator, 'daily_pnl', 0)
            
            if daily_pnl <= -config['max_daily_loss_pct']:
                self._handle_failure(FailureType.DRAWDOWN_EXCEEDED)
        
        # Check consecutive losses
        if self.signal_generator and hasattr(self.signal_generator, 'consecutive_losses'):
            losses = getattr(self.signal_generator, 'consecutive_losses', 0)
            
            if losses >= config['max_consecutive_losses']:
                self._handle_failure(FailureType.CONSECUTIVE_LOSSES)
    
    def _update_health_status(self) -> None:
        """Update overall health status."""
        self.state.last_health_check = datetime.now(self.timezone).isoformat()
        self.state.health_status = {
            k: asdict(v) for k, v in self.health_checks.items()
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FAILURE HANDLING
    # ═══════════════════════════════════════════════════════════════════════════
    def _handle_failure(self, failure_type: FailureType) -> None:
        """Handle a detected failure with appropriate action."""
        log.error(f"🚨 FAILURE DETECTED: {failure_type.value}")
        
        # Determine action
        actions = self.CONFIG['actions']
        action_map = {
            FailureType.APP_UNRESPONSIVE: actions['on_app_crash'],
            FailureType.MT5_DISCONNECTED: actions['on_mt5_disconnect'],
            FailureType.NETWORK_FAILURE: actions['on_network_failure'],
            FailureType.DRAWDOWN_EXCEEDED: actions['on_drawdown_exceeded'],
            FailureType.CONSECUTIVE_LOSSES: actions['on_consecutive_losses'],
        }
        
        action = action_map.get(failure_type, 'ALERT_ONLY')
        
        # Execute action
        positions_affected = 0
        
        if action == 'CLOSE_ALL':
            positions_affected = self._emergency_close_all(failure_type.value)
            self.state.emergency_closes += 1
        elif action == 'TIGHTEN_STOPS':
            positions_affected = self._apply_emergency_stops()
        elif action == 'MOVE_TO_BREAKEVEN':
            positions_affected = self._move_all_to_breakeven()
        elif action == 'PAUSE_TRADING':
            self._pause_trading()
        
        # Record event
        event = ProtectionEvent(
            timestamp=datetime.now(self.timezone).isoformat(),
            failure_type=failure_type.value,
            alert_level=AlertLevel.CRITICAL.value,
            action_taken=action,
            positions_affected=positions_affected
        )
        self.events.append(event)
        self.state.recent_events = [asdict(e) for e in self.events[-20:]]
        self.state.total_interventions += 1
        
        # Send alert
        self._send_alert(
            AlertLevel.CRITICAL,
            f"Protection triggered: {failure_type.value}",
            {"action": action, "positions_affected": positions_affected}
        )
        
        # Callback
        if self.on_emergency:
            self.on_emergency(failure_type.value, action, positions_affected)
    
    def _emergency_close_all(self, reason: str) -> int:
        """Emergency close all positions."""
        if not HAS_MT5:
            return 0
        
        closed = 0
        
        try:
            positions = mt5.positions_get()
            if not positions:
                return 0
            
            for pos in positions:
                if self._close_position(pos, reason):
                    closed += 1
            
            log.warning(f"🚨 Emergency closed {closed} positions: {reason}")
            
        except Exception as e:
            log.error(f"Emergency close error: {e}")
        
        self.state.positions_protected += closed
        return closed
    
    def _close_position(self, pos, reason: str) -> bool:
        """Close a single position."""
        try:
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                return False
            
            price = tick.bid if pos.type == 0 else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": price,
                "deviation": 50,  # Higher deviation for emergency
                "magic": 999999,
                "comment": f"GUARDIAN: {reason}"[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Record in journal
                if self.trade_journal:
                    self.trade_journal.record_exit(
                        ticket=pos.ticket,
                        exit_price=price,
                        exit_type=f"GUARDIAN_{reason}",
                        pnl_money=pos.profit
                    )
                return True
            
            return False
            
        except Exception as e:
            log.error(f"Close position error: {e}")
            return False
    
    def _apply_emergency_stops(self) -> int:
        """Apply emergency stop losses to all positions."""
        if not HAS_MT5:
            return 0
        
        modified = 0
        emergency_pips = self.CONFIG['protection']['emergency_sl_pips']
        
        try:
            positions = mt5.positions_get()
            if not positions:
                return 0
            
            for pos in positions:
                pip_value = 0.0001 if 'JPY' not in pos.symbol else 0.01
                
                if pos.type == 0:  # BUY
                    emergency_sl = pos.price_open - (emergency_pips * pip_value)
                    if pos.sl == 0 or emergency_sl > pos.sl:
                        if self._modify_sl(pos.ticket, emergency_sl, pos.tp):
                            modified += 1
                else:  # SELL
                    emergency_sl = pos.price_open + (emergency_pips * pip_value)
                    if pos.sl == 0 or emergency_sl < pos.sl:
                        if self._modify_sl(pos.ticket, emergency_sl, pos.tp):
                            modified += 1
            
            log.info(f"Applied emergency SL to {modified} positions")
            
        except Exception as e:
            log.error(f"Emergency SL error: {e}")
        
        self.state.positions_protected += modified
        return modified
    
    def _move_all_to_breakeven(self) -> int:
        """Move all profitable positions to breakeven."""
        if not HAS_MT5:
            return 0
        
        modified = 0
        
        try:
            positions = mt5.positions_get()
            if not positions:
                return 0
            
            for pos in positions:
                if pos.profit > 0:  # Only if in profit
                    pip_value = 0.0001 if 'JPY' not in pos.symbol else 0.01
                    buffer = 2 * pip_value  # 2 pip buffer
                    
                    if pos.type == 0:  # BUY
                        new_sl = pos.price_open + buffer
                        if new_sl > pos.sl:
                            if self._modify_sl(pos.ticket, new_sl, pos.tp):
                                modified += 1
                    else:  # SELL
                        new_sl = pos.price_open - buffer
                        if new_sl < pos.sl:
                            if self._modify_sl(pos.ticket, new_sl, pos.tp):
                                modified += 1
            
            log.info(f"Moved {modified} positions to breakeven")
            
        except Exception as e:
            log.error(f"Move to breakeven error: {e}")
        
        return modified
    
    def _modify_sl(self, ticket: int, new_sl: float, tp: float) -> bool:
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
                "tp": tp,
            }
            
            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            log.error(f"Modify SL error: {e}")
            return False
    
    def _pause_trading(self) -> None:
        """Pause trading via circuit breaker."""
        duration = self.CONFIG['circuit_breakers']['pause_duration_minutes']
        pause_until = datetime.now(self.timezone) + timedelta(minutes=duration)
        
        self.state.trading_paused = True
        self.state.pause_until = pause_until.isoformat()
        
        log.warning(f"⏸️ Trading PAUSED until {pause_until.isoformat()}")
        
        # Signal to signal generator
        if self.signal_generator and hasattr(self.signal_generator, 'trading_paused'):
            self.signal_generator.trading_paused = True
            self.signal_generator.pause_until = pause_until
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        if not self.state.trading_paused:
            return True, "OK"
        
        pause_until = datetime.fromisoformat(self.state.pause_until)
        if datetime.now(self.timezone) >= pause_until:
            self.state.trading_paused = False
            self.state.pause_until = ""
            return True, "Pause expired"
        
        remaining = (pause_until - datetime.now(self.timezone)).total_seconds() / 60
        return False, f"Trading paused for {remaining:.0f} more minutes"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ALERTS & NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    def _send_alert(self, level: AlertLevel, message: str, details: Dict = None) -> None:
        """Send an alert through configured channels."""
        now = datetime.now(self.timezone)
        
        alert = {
            'timestamp': now.isoformat(),
            'level': level.value,
            'message': message,
            'details': details or {}
        }
        
        self.state.total_alerts += 1
        
        # Log
        if self.CONFIG['notifications']['log_all']:
            log_msg = f"[{level.value}] {message}"
            if level == AlertLevel.INFO:
                log.info(log_msg)
            elif level == AlertLevel.WARNING:
                log.warning(log_msg)
            else:
                log.error(log_msg)
        
        # File
        if self.CONFIG['notifications']['file_alerts']:
            self._save_alert(alert)
        
        # Callback
        if self.CONFIG['notifications']['callback_enabled'] and self.on_alert:
            self.on_alert(level.value, message, details)
    
    def _save_alert(self, alert: Dict) -> None:
        """Save alert to file."""
        try:
            alerts = []
            if os.path.exists(self.CONFIG['alert_file']):
                with open(self.CONFIG['alert_file'], 'r') as f:
                    alerts = json.load(f)
            
            alerts.append(alert)
            alerts = alerts[-100:]  # Keep last 100
            
            with open(self.CONFIG['alert_file'], 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            log.debug(f"Save alert error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATE PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════
    def _save_state(self) -> None:
        """Save guardian state."""
        try:
            with open(self.CONFIG['state_file'], 'w') as f:
                json.dump(asdict(self.state), f, indent=2, default=str)
        except Exception as e:
            log.debug(f"Save state error: {e}")
    
    def _load_state(self) -> None:
        """Load guardian state."""
        try:
            if os.path.exists(self.CONFIG['state_file']):
                with open(self.CONFIG['state_file'], 'r') as f:
                    data = json.load(f)
                
                # Check if previous session crashed
                if data.get('is_active') and not data.get('app_alive', True):
                    log.warning("⚠️ Previous session crashed - running recovery")
                    self._handle_recovery()
                
                # Restore counters
                self.state.total_alerts = data.get('total_alerts', 0)
                self.state.total_interventions = data.get('total_interventions', 0)
                self.state.positions_protected = data.get('positions_protected', 0)
                self.state.emergency_closes = data.get('emergency_closes', 0)
                
        except Exception as e:
            log.debug(f"Load state error: {e}")
    
    def _handle_recovery(self) -> None:
        """Handle recovery after crash."""
        log.info("🔄 Running crash recovery...")
        
        if HAS_MT5 and mt5.terminal_info():
            # Check for orphaned positions
            positions = mt5.positions_get()
            if positions:
                log.info(f"Found {len(positions)} positions from previous session")
                # Don't close - just re-register for monitoring
                for pos in positions:
                    self._monitor_positions()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATUS & REPORTING
    # ═══════════════════════════════════════════════════════════════════════════
    def get_status(self) -> Dict:
        """Get comprehensive guardian status."""
        trading_allowed, pause_reason = self.is_trading_allowed()
        
        return {
            'running': self._running,
            'app_alive': self.state.app_alive,
            'mt5_connected': self.state.mt5_connected,
            'heartbeat_level': self.heartbeat_level,
            'trading_allowed': trading_allowed,
            'pause_reason': pause_reason if not trading_allowed else None,
            'protected_positions': len(self.protected_positions),
            'total_alerts': self.state.total_alerts,
            'total_interventions': self.state.total_interventions,
            'emergency_closes': self.state.emergency_closes,
            'health': self.state.health_status,
            'session_start': self.state.session_start
        }
    
    def get_recent_events(self, count: int = 10) -> List[Dict]:
        """Get recent protection events."""
        return self.state.recent_events[-count:]
    
    def manual_close_all(self, reason: str = "MANUAL_CLOSE") -> int:
        """Manually trigger close all positions."""
        return self._emergency_close_all(reason)


# --- END OF FILE guardian.py ---
