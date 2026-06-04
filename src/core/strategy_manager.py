# --- START OF FILE strategy_manager.py ---
"""
Strategy Manager Module
================================================================================
Manages strategy loading, registration, and execution. Dynamically loads
strategies from the /strategies folder and integrates them with the trading bot.

Features:
- Dynamic strategy loading from files
- Strategy registry with enable/disable
- Multi-strategy signal aggregation
- Performance tracking per strategy
- Hot-reload support
"""

import os
import sys
import importlib
import importlib.util
import logging
from typing import Dict, List, Optional, Type, Any
from datetime import datetime
from pathlib import Path
import json

from src.core.base_strategy import BaseStrategy, StrategySignal, ExitSignal

log = logging.getLogger('StrategyManager')


class StrategyManager:
    """
    Strategy Manager - Loads, registers, and executes trading strategies.
    
    Usage:
        manager = StrategyManager()
        manager.load_strategies_from_folder("strategies/")
        
        # Get signals from all active strategies
        signals = manager.get_signals(data, pair, timeframe)
        
        # Execute best signal
        if signals:
            best_signal = signals[0]  # Highest confidence
            execution_manager.execute_auto_trade(best_signal.to_dict())
    """
    
    STRATEGIES_FOLDER = "strategies"
    CONFIG_FILE = "data/strategy_config.json"
    
    def __init__(self, strategies_folder: str = None):
        """
        Initialize Strategy Manager.
        
        Args:
            strategies_folder: Path to folder containing strategy files
        """
        self.strategies_folder = strategies_folder or self.STRATEGIES_FOLDER
        self.registered_strategies: Dict[str, BaseStrategy] = {}
        self.strategy_config: Dict[str, Dict] = {}
        self.strategy_stats: Dict[str, Dict] = {}
        
        # Ensure folders exist
        os.makedirs(self.strategies_folder, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Load config
        self._load_config()
        
        log.info(f"StrategyManager initialized. Strategies folder: {self.strategies_folder}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    def load_strategies_from_folder(self) -> int:
        """
        Load all strategy files from the strategies folder.
        
        Returns:
            Number of strategies loaded
        """
        loaded = 0
        folder_path = Path(self.strategies_folder)
        
        if not folder_path.exists():
            log.warning(f"Strategies folder not found: {self.strategies_folder}")
            return 0
        
        # Find all Python files in strategies folder
        for file_path in folder_path.glob("*.py"):
            if file_path.name.startswith("_"):
                continue  # Skip __init__.py etc.
            
            try:
                strategy = self._load_strategy_from_file(file_path)
                if strategy:
                    self.register_strategy(strategy)
                    loaded += 1
                    log.info(f"✅ Loaded strategy: {strategy.name}")
            except Exception as e:
                log.error(f"❌ Failed to load {file_path.name}: {e}")
        
        log.info(f"Loaded {loaded} strategies from {self.strategies_folder}")
        return loaded
    
    def _load_strategy_from_file(self, file_path: Path) -> Optional[BaseStrategy]:
        """
        Dynamically load a strategy class from a Python file.
        
        Args:
            file_path: Path to the strategy file
            
        Returns:
            Instantiated strategy object
        """
        try:
            # Create module spec
            module_name = f"strategy_{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if spec is None or spec.loader is None:
                return None
            
            # Load module
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find strategy class (subclass of BaseStrategy)
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, BaseStrategy) and 
                    obj is not BaseStrategy):
                    
                    # Get custom params from config if available
                    params = self.strategy_config.get(obj.name, {}).get('params', {})
                    return obj(params=params)
            
            return None
            
        except Exception as e:
            log.error(f"Error loading strategy from {file_path}: {e}")
            return None
    
    def reload_strategy(self, strategy_name: str) -> bool:
        """
        Reload a specific strategy (hot-reload).
        
        Args:
            strategy_name: Name of strategy to reload
            
        Returns:
            True if successful
        """
        try:
            for file_path in Path(self.strategies_folder).glob("*.py"):
                strategy = self._load_strategy_from_file(file_path)
                if strategy and strategy.name == strategy_name:
                    self.registered_strategies[strategy_name] = strategy
                    log.info(f"🔄 Reloaded strategy: {strategy_name}")
                    return True
            return False
        except Exception as e:
            log.error(f"Reload error: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY REGISTRATION
    # ═══════════════════════════════════════════════════════════════════════════
    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Register a strategy instance."""
        name = strategy.name
        self.registered_strategies[name] = strategy
        
        # Initialize stats
        if name not in self.strategy_stats:
            self.strategy_stats[name] = {
                'signals_generated': 0,
                'trades_executed': 0,
                'wins': 0,
                'losses': 0,
                'total_pips': 0.0,
                'registered_at': datetime.now().isoformat()
            }
        
        # Check if disabled in config
        if name in self.strategy_config:
            strategy.is_active = self.strategy_config[name].get('enabled', True)
    
    def unregister_strategy(self, strategy_name: str) -> bool:
        """Unregister a strategy."""
        if strategy_name in self.registered_strategies:
            del self.registered_strategies[strategy_name]
            log.info(f"Unregistered strategy: {strategy_name}")
            return True
        return False
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a strategy."""
        if strategy_name in self.registered_strategies:
            self.registered_strategies[strategy_name].is_active = True
            self._update_config(strategy_name, {'enabled': True})
            log.info(f"✅ Enabled strategy: {strategy_name}")
            return True
        return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a strategy."""
        if strategy_name in self.registered_strategies:
            self.registered_strategies[strategy_name].is_active = False
            self._update_config(strategy_name, {'enabled': False})
            log.warning(f"⏸️ Disabled strategy: {strategy_name}")
            return True
        return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════════════════════════
    def get_signals(self, 
                    data: Dict, 
                    pair: str, 
                    timeframe: str) -> List[StrategySignal]:
        """
        Get signals from all active strategies.
        
        Args:
            data: Market data dictionary
            pair: Currency pair
            timeframe: Timeframe
            
        Returns:
            List of signals sorted by confidence (highest first)
        """
        signals = []
        
        for name, strategy in self.registered_strategies.items():
            if not strategy.is_active:
                continue
            
            # Check if strategy supports this pair/timeframe
            if pair not in strategy.pairs and '*' not in strategy.pairs:
                continue
            if timeframe not in strategy.timeframes and '*' not in strategy.timeframes:
                continue
            
            try:
                signal = strategy.should_enter(data, pair, timeframe)
                
                if signal and signal.direction in ['BUY', 'SELL']:
                    # Calculate stops if not set
                    if signal.stop_loss == 0 or signal.take_profit == 0:
                        signal = strategy.calculate_stops(signal, data)
                    
                    # Validate signal
                    if strategy.validate_signal(signal):
                        signals.append(signal)
                        strategy.on_signal_generated(signal)
                        self.strategy_stats[name]['signals_generated'] += 1
                        log.info(f"📊 Signal from {name}: {signal.direction} {pair} "
                                f"Grade={signal.grade} Conf={signal.confidence:.0%}")
                        
            except Exception as e:
                log.error(f"Error getting signal from {name}: {e}")
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        return signals
    
    def get_best_signal(self, 
                        data: Dict, 
                        pair: str, 
                        timeframe: str,
                        min_grade: str = "B") -> Optional[StrategySignal]:
        """
        Get the best signal from all strategies.
        
        Args:
            data: Market data
            pair: Currency pair
            timeframe: Timeframe
            min_grade: Minimum acceptable grade
            
        Returns:
            Best signal or None
        """
        signals = self.get_signals(data, pair, timeframe)
        
        grade_order = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
        min_grade_value = grade_order.get(min_grade, 3)
        
        for signal in signals:
            signal_grade_value = grade_order.get(signal.grade, 0)
            if signal_grade_value >= min_grade_value:
                return signal
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXIT SIGNAL CHECKING
    # ═══════════════════════════════════════════════════════════════════════════
    def check_exit_signals(self, 
                           position: Dict, 
                           data: Dict) -> Optional[ExitSignal]:
        """
        Check if any strategy wants to exit a position.
        
        Args:
            position: Position data
            data: Market data
            
        Returns:
            ExitSignal if exit recommended
        """
        for name, strategy in self.registered_strategies.items():
            if not strategy.is_active:
                continue
            
            try:
                exit_signal = strategy.should_exit(position, data)
                if exit_signal and exit_signal.should_exit:
                    log.info(f"🚪 Exit signal from {name}: {exit_signal.exit_type}")
                    return exit_signal
            except Exception as e:
                log.error(f"Error checking exit from {name}: {e}")
        
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRADE RESULT TRACKING
    # ═══════════════════════════════════════════════════════════════════════════
    def record_trade_result(self, 
                            strategy_name: str, 
                            won: bool, 
                            pnl_pips: float,
                            rr: float = 0) -> None:
        """Record trade result for a strategy."""
        if strategy_name not in self.strategy_stats:
            return
        
        stats = self.strategy_stats[strategy_name]
        stats['trades_executed'] += 1
        stats['total_pips'] += pnl_pips
        
        if won:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        
        # Update strategy's internal metrics
        if strategy_name in self.registered_strategies:
            self.registered_strategies[strategy_name].on_trade_closed(won, pnl_pips, rr)
        
        self._save_config()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    def _load_config(self) -> None:
        """Load strategy configuration."""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self.strategy_config = data.get('strategies', {})
                    self.strategy_stats = data.get('stats', {})
        except Exception as e:
            log.debug(f"Config load error: {e}")
    
    def _save_config(self) -> None:
        """Save strategy configuration."""
        try:
            data = {
                'strategies': self.strategy_config,
                'stats': self.strategy_stats,
                'saved_at': datetime.now().isoformat()
            }
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            log.debug(f"Config save error: {e}")
    
    def _update_config(self, strategy_name: str, updates: Dict) -> None:
        """Update config for a strategy."""
        if strategy_name not in self.strategy_config:
            self.strategy_config[strategy_name] = {}
        self.strategy_config[strategy_name].update(updates)
        self._save_config()
    
    def set_strategy_params(self, strategy_name: str, params: Dict) -> bool:
        """Set custom parameters for a strategy."""
        if strategy_name in self.registered_strategies:
            self.registered_strategies[strategy_name].params.update(params)
            self._update_config(strategy_name, {'params': params})
            return True
        return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATUS & REPORTING
    # ═══════════════════════════════════════════════════════════════════════════
    def get_registered_strategies(self) -> List[Dict]:
        """Get list of all registered strategies."""
        return [
            {
                **strategy.get_info(),
                'stats': self.strategy_stats.get(name, {}),
                'enabled': strategy.is_active
            }
            for name, strategy in self.registered_strategies.items()
        ]
    
    def get_active_strategies(self) -> List[str]:
        """Get names of active strategies."""
        return [name for name, s in self.registered_strategies.items() if s.is_active]
    
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict]:
        """Get info for a specific strategy."""
        if strategy_name in self.registered_strategies:
            strategy = self.registered_strategies[strategy_name]
            return {
                **strategy.get_info(),
                'stats': self.strategy_stats.get(strategy_name, {}),
                'metrics': strategy.get_metrics()
            }
        return None
    
    def get_strategy_ranking(self) -> List[Dict]:
        """Get strategies ranked by performance."""
        rankings = []
        
        for name, stats in self.strategy_stats.items():
            trades = stats.get('trades_executed', 0)
            wins = stats.get('wins', 0)
            win_rate = (wins / trades * 100) if trades > 0 else 0
            
            rankings.append({
                'name': name,
                'trades': trades,
                'wins': wins,
                'losses': stats.get('losses', 0),
                'win_rate': round(win_rate, 1),
                'total_pips': round(stats.get('total_pips', 0), 1),
                'active': name in self.registered_strategies and \
                         self.registered_strategies[name].is_active
            })
        
        # Sort by win rate then total pips
        rankings.sort(key=lambda x: (x['win_rate'], x['total_pips']), reverse=True)
        
        return rankings
    
    def reset_daily_counters(self) -> None:
        """Reset daily counters for all strategies."""
        for strategy in self.registered_strategies.values():
            strategy.reset_daily_counters()


# --- END OF FILE strategy_manager.py ---
