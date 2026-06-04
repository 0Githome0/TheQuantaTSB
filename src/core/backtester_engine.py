"""
Backtester Engine - Core backtesting logic for strategy simulation.

This module provides the BacktesterEngine class which handles:
- Historical data fetching/generation
- Strategy simulation with real trading logic
- Performance metrics calculation
- Trade logging
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pytz

logger = logging.getLogger("BacktesterEngine")


class BacktesterEngine:
    """
    Core backtesting engine for simulating trading strategies on historical data.
    
    Supports multiple strategy types and calculates comprehensive performance metrics.
    """
    
    # Available strategies
    STRATEGIES = {
        "RSI + SMA Confluence": "rsi_sma_confluence",
        "MACD Crossover": "macd_crossover",
        "Bollinger Bounce": "bollinger_bounce",
        "EMA Trend Follow": "ema_trend_follow",
        "Multi-Indicator Bot": "multi_indicator",
    }
    
    def __init__(self, data_manager=None, signal_generator=None):
        """
        Initialize the backtester engine.
        
        Args:
            data_manager: Optional DataManager for fetching real historical data
            signal_generator: Optional SignalGenerator for advanced signal logic
        """
        self.data_manager = data_manager
        self.signal_generator = signal_generator
        self.initial_capital = 10000.0
        self.lot_size = 0.1
        self.stop_loss_pips = 50
        self.take_profit_pips = 100
        
    def set_parameters(self, initial_capital: float = 10000.0, lot_size: float = 0.1,
                       stop_loss_pips: int = 50, take_profit_pips: int = 100):
        """Set backtesting parameters."""
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
    
    def fetch_historical_data(self, pair: str, timeframe: str, 
                              start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from MT5 for backtesting.
        
        Args:
            pair: Trading pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "H1", "M15")
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            DataFrame with OHLCV data indexed by datetime
        """
        # Try MT5 directly first
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                logger.warning("MT5 not initialized, trying to initialize...")
                if not mt5.initialize():
                    raise Exception("Could not initialize MT5")
            
            # Convert timeframe string to MT5 constant
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1,
            }
            mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Ensure timezone aware dates
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=pytz.UTC)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=pytz.UTC)
            
            # Fetch data from MT5
            logger.info(f"Attempting to fetch {pair} {timeframe} from {start_date} to {end_date}")
            
            # Check connection
            terminal_info = mt5.terminal_info()
            if terminal_info:
                logger.info(f"MT5 Connected: {terminal_info.connected}, Trade Allowed: {terminal_info.trade_allowed}")
            else:
                logger.error("Failed to get terminal info")
                
            # Check symbol
            symbol_info = mt5.symbol_info(pair)
            if symbol_info is None:
                logger.warning(f"Symbol {pair} not found. Searching for broker-specific variant...")
                # Search for similar symbols (match start)
                symbols = mt5.symbols_get()
                found_variant = False
                if symbols:
                    for s in symbols:
                        if pair in s.name and "USD" in s.name: # Simple safety check
                            logger.info(f"Auto-switching symbol from {pair} to {s.name}")
                            pair = s.name
                            found_variant = True
                            break
                
                if not found_variant:
                    logger.error(f"Failed to find symbol {pair} or variants.")
                else:
                    mt5.symbol_select(pair, True)
                    symbol_info = mt5.symbol_info(pair)

            if symbol_info is not None:
                if not symbol_info.visible:
                    mt5.symbol_select(pair, True)
            
            # Fetch rates using the (potentially corrected) pair
            rates = mt5.copy_rates_range(pair, mt5_tf, start_date, end_date)
            
            if rates is None:
                error = mt5.last_error()
                logger.error(f"mt5.copy_rates_range returned None for {pair}. Error code: {error}")
            elif len(rates) == 0:
                logger.warning(f"mt5.copy_rates_range returned empty list for {pair}.")
            else:
                logger.info(f"Successfully fetched {len(rates)} bars from MT5 for {pair}")
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df.set_index('time', inplace=True)
                df.columns = [c.lower() for c in df.columns]
                return df
                
        except ImportError:
            logger.warning("⚠️ MetaTrader5 not installed - backtest will use SIMULATED data (for testing only)")
        except Exception as e:
            logger.error(f"MT5 data fetch failed with exception: {e}", exc_info=True)
            logger.warning("⚠️ MT5 data unavailable - backtest will use SIMULATED data (for testing only)")
        
        # Fallback to data_manager if available
        if self.data_manager:
            try:
                data = self.data_manager.get_data(pair, timeframe)
                if data is not None and len(data) > 0:
                    if isinstance(data.index, pd.DatetimeIndex):
                        mask = (data.index >= start_date) & (data.index <= end_date)
                        return data[mask]
                    return data
            except Exception as e:
                logger.warning(f"Data manager failed: {e}")
        
        # Generate simulated data as last resort - FOR TESTING ONLY
        logger.warning("⚠️ USING SIMULATED DATA - Results are for TESTING PURPOSES ONLY!")
        logger.warning("⚠️ DO NOT use these backtest results to make real trading decisions without real MT5 data!")
        return self._generate_mock_data(pair, timeframe, start_date, end_date)
    
    def _generate_mock_data(self, pair: str, timeframe: str,
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate realistic mock OHLCV data for backtesting."""
        # Determine timeframe in hours
        tf_hours = {
            'M1': 1/60, 'M5': 5/60, 'M15': 0.25, 'M30': 0.5,
            'H1': 1, 'H4': 4, 'D1': 24, 'W1': 168
        }.get(timeframe, 1)
        
        # Calculate number of bars
        total_hours = (end_date - start_date).total_seconds() / 3600
        num_bars = int(total_hours / tf_hours)
        num_bars = min(max(num_bars, 100), 2000)  # Clamp between 100-2000
        
        # Generate datetime index
        dates = pd.date_range(start=start_date, end=end_date, periods=num_bars, tz=pytz.UTC)
        
        # Set base price based on pair
        base_prices = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2600, 'USDJPY': 149.50,
            'AUDUSD': 0.6500, 'USDCAD': 1.3600, 'XAUUSD': 2020.0,
        }
        base_price = base_prices.get(pair, 1.0)
        
        # Generate realistic price action with trends and ranges
        np.random.seed(42)  # Reproducible
        prices = [base_price]
        volatility = 0.0008  # Base volatility
        trend = 0
        
        for i in range(1, num_bars):
            # Switch between trending and ranging phases
            if i % 50 == 0:
                trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            
            # Price change
            change = np.random.normal(trend * 0.0001, volatility)
            
            # Add larger moves occasionally (news events)
            if np.random.random() < 0.02:
                change *= np.random.uniform(3, 8)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Generate OHLC from prices
        data = []
        for i in range(len(prices)):
            open_price = prices[i]
            
            # Generate intra-bar movement
            high_ext = abs(np.random.normal(0, volatility * 0.5))
            low_ext = abs(np.random.normal(0, volatility * 0.5))
            close_move = np.random.normal(0, volatility * 0.3)
            
            high = open_price * (1 + high_ext)
            low = open_price * (1 - low_ext)
            close = open_price * (1 + close_move)
            
            # Ensure OHLC consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume
            volume = int(np.random.uniform(500, 3000))
            
            data.append({
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close, 5),
                'volume': volume,
                'tick_volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def run_backtest(self, data: pd.DataFrame, strategy: str, strategy_code: str = None) -> Dict:
        """
        Run backtest simulation on historical data.
        
        Args:
            data: OHLCV DataFrame
            strategy: Strategy name from STRATEGIES
            strategy_code: Optional python code for custom strategy
            
        Returns:
            Dictionary with trades and performance metrics
        """
        # Determine strategy method
        if strategy == "Code Mode":
             strategy_method = "custom_code"
        else:
             strategy_method = self.STRATEGIES.get(strategy, "rsi_sma_confluence")
             
        trades = self._simulate_strategy(data, strategy_method, strategy_code)
        metrics = self._calculate_metrics(data, trades)
        
        return {
            'trades': trades,
            **metrics
        }
    
    def _simulate_strategy(self, data: pd.DataFrame, strategy: str, strategy_code: str = None) -> List[Dict]:
        """Run strategy simulation and generate trades."""
        trades = []
        position = None
        
        # Prepare custom function scope if code mode
        custom_scope = {}
        vectorized_signals = None
        if strategy == "custom_code" and strategy_code:
            try:
                # Prepare data with capitalized columns for compatibility
                exec_data = data.copy()
                exec_data['Open'] = exec_data['open']
                exec_data['High'] = exec_data['high']
                exec_data['Low'] = exec_data['low']
                exec_data['Close'] = exec_data['close']
                exec_data['Volume'] = exec_data['tick_volume']
                
                # Add safe globals
                exec(strategy_code, {'pd': pd, 'np': np, 'data': exec_data}, custom_scope)
                
                # Check for Vectorized Strategy (Class or Function)
                if 'generate_signals' in custom_scope and callable(custom_scope['generate_signals']):
                    try:
                        res = custom_scope['generate_signals'](exec_data.copy())
                        if isinstance(res, pd.DataFrame) and 'Signal' in res.columns:
                            vectorized_signals = res['Signal']
                            logger.info("Executed vectorized generate_signals function")
                    except Exception as e:
                        logger.warning(f"Failed to run generate_signals: {e}")
                
                # Check for Class based strategy
                if vectorized_signals is None:
                    for name, obj in custom_scope.items():
                        if isinstance(obj, type) and hasattr(obj, 'generate_signals'):
                            try:
                                logger.info(f"Instantiating strategy class {name}")
                                instance = obj()
                                res = instance.generate_signals(exec_data.copy())
                                if isinstance(res, pd.DataFrame) and 'Signal' in res.columns:
                                    vectorized_signals = res['Signal']
                                    logger.info(f"Executed vectorized strategy from class {name}")
                                    break
                            except Exception as e:
                                logger.warning(f"Failed to run class {name}: {e}")

                if vectorized_signals is None and 'generate_signal' not in custom_scope:
                    logger.error("Custom code must define 'generate_signal' or 'generate_signals'")
                    return []
            except Exception as e:
                logger.error(f"Error compiling custom strategy: {e}")
                return []
        
        # Calculate indicators
        close = data['close']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # SMAs
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        
        # EMAs
        ema9 = close.ewm(span=9).mean()
        ema21 = close.ewm(span=21).mean()
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        
        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + (bb_std * 2)
        bb_lower = bb_mid - (bb_std * 2)
        
        # Pip value (approximate)
        pip_value = 0.0001 if 'JPY' not in str(data.index.name) else 0.01
        
        for i in range(50, len(data) - 1):
            current_price = close.iloc[i]
            
            # Vectorized Signal Target (1/BUY=Long, -1/SELL=Short, 0=Flat)
            vect_target = None 
            if vectorized_signals is not None:
                try:
                    val = vectorized_signals.iloc[i]
                    if val in [1, 'BUY', 1.0]: vect_target = 1
                    elif val in [-1, 'SELL', -1.0]: vect_target = -1
                    elif val in [0, 0.0, 'FLAT']: vect_target = 0
                except Exception: pass

            # Check exit conditions first
            if position is not None:
                entry_price = position['entry']
                
                # Check Vectorized Exit (State Change)
                if vect_target is not None:
                    should_exit = False
                    if position['type'] == 'BUY' and vect_target != 1: should_exit = True
                    elif position['type'] == 'SELL' and vect_target != -1: should_exit = True
                    
                    if should_exit:
                        trades.append({
                            'type': position['type'],
                             'entry_idx': position['entry_idx'],
                             'exit_idx': i,
                             'entry': entry_price,
                             'exit': current_price,
                             'pnl': ((current_price - entry_price) if position['type'] == 'BUY' else (entry_price - current_price)) * self.lot_size * 100000,
                             'duration': i - position['entry_idx']
                        })
                        position = None
                        continue

                entry_price = position['entry']
                
                # Check SL/TP
                if position['type'] == 'BUY':
                    pnl_pips = (current_price - entry_price) / pip_value
                    if pnl_pips <= -self.stop_loss_pips or pnl_pips >= self.take_profit_pips:
                        trades.append({
                            'type': 'BUY',
                            'entry_idx': position['entry_idx'],
                            'exit_idx': i,
                            'entry': entry_price,
                            'exit': current_price,
                            'pnl': (current_price - entry_price) * self.lot_size * 100000,
                            'duration': i - position['entry_idx']
                        })
                        position = None
                else:  # SELL
                    pnl_pips = (entry_price - current_price) / pip_value
                    if pnl_pips <= -self.stop_loss_pips or pnl_pips >= self.take_profit_pips:
                        trades.append({
                            'type': 'SELL',
                            'entry_idx': position['entry_idx'],
                            'exit_idx': i,
                            'entry': entry_price,
                            'exit': current_price,
                            'pnl': (entry_price - current_price) * self.lot_size * 100000,
                            'duration': i - position['entry_idx']
                        })
                        position = None
            
            # Entry signals (only if no position)
            if position is None:
                 # Vectorized Entry
                if vect_target is not None:
                     signal = None
                     if vect_target == 1: signal = 'BUY'
                     elif vect_target == -1: signal = 'SELL'
                     
                     if signal:
                         position = {'type': signal, 'entry': current_price, 'entry_idx': i}
                         continue

                # Pack indicators for custom strategy efficiency
                indicators = {
                    'rsi': rsi.iloc[i], 'sma20': sma20.iloc[i], 'sma50': sma50.iloc[i],
                    'ema9': ema9.iloc[i], 'ema21': ema21.iloc[i],
                    'macd': macd.iloc[i], 'signal_line': signal_line.iloc[i],
                    'bb_upper': bb_upper.iloc[i], 'bb_lower': bb_lower.iloc[i]
                }
                
                if strategy == 'custom_code':
                    # Only call scalar generate_signal if vectorized signals weren't generated
                    # AND generate_signal exists in the custom scope
                    if vectorized_signals is None and 'generate_signal' in custom_scope:
                        try:
                            signal = custom_scope['generate_signal'](data, i, indicators)
                        except Exception as e:
                            if i % 100 == 0: logger.error(f"Runtime error in custom strategy at {i}: {e}")
                            signal = None
                    else:
                        signal = None  # Use vectorized signals or skip if function doesn't exist
                else:
                    signal = self._get_entry_signal(
                        strategy, i, close, rsi, sma20, sma50, 
                        ema9, ema21, macd, signal_line, bb_upper, bb_lower
                    )
                
                if signal:
                    position = {
                        'type': signal,
                        'entry': current_price,
                        'entry_idx': i
                    }
        
        # Close any open position at end
        if position is not None:
            trades.append({
                'type': position['type'],
                'entry_idx': position['entry_idx'],
                'exit_idx': len(data) - 1,
                'entry': position['entry'],
                'exit': close.iloc[-1],
                'pnl': (close.iloc[-1] - position['entry']) * self.lot_size * 100000 if position['type'] == 'BUY' 
                       else (position['entry'] - close.iloc[-1]) * self.lot_size * 100000,
                'duration': len(data) - 1 - position['entry_idx']
            })
        
        return trades
    
    def _get_entry_signal(self, strategy: str, i: int, close: pd.Series,
                          rsi: pd.Series, sma20: pd.Series, sma50: pd.Series,
                          ema9: pd.Series, ema21: pd.Series, macd: pd.Series,
                          signal_line: pd.Series, bb_upper: pd.Series, 
                          bb_lower: pd.Series) -> Optional[str]:
        """Get entry signal based on strategy."""
        
        if strategy == "rsi_sma_confluence":
            # BUY: RSI < 35 and price above SMA20 and SMA20 > SMA50
            if rsi.iloc[i] < 35 and close.iloc[i] > sma20.iloc[i] and sma20.iloc[i] > sma50.iloc[i]:
                return 'BUY'
            # SELL: RSI > 65 and price below SMA20 and SMA20 < SMA50
            if rsi.iloc[i] > 65 and close.iloc[i] < sma20.iloc[i] and sma20.iloc[i] < sma50.iloc[i]:
                return 'SELL'
                
        elif strategy == "macd_crossover":
            # BUY: MACD crosses above signal line
            if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
                return 'BUY'
            # SELL: MACD crosses below signal line
            if macd.iloc[i] < signal_line.iloc[i] and macd.iloc[i-1] >= signal_line.iloc[i-1]:
                return 'SELL'
                
        elif strategy == "bollinger_bounce":
            # BUY: Price touches lower BB and RSI < 40
            if close.iloc[i] <= bb_lower.iloc[i] and rsi.iloc[i] < 40:
                return 'BUY'
            # SELL: Price touches upper BB and RSI > 60
            if close.iloc[i] >= bb_upper.iloc[i] and rsi.iloc[i] > 60:
                return 'SELL'
                
        elif strategy == "ema_trend_follow":
            # BUY: EMA9 crosses above EMA21
            if ema9.iloc[i] > ema21.iloc[i] and ema9.iloc[i-1] <= ema21.iloc[i-1]:
                return 'BUY'
            # SELL: EMA9 crosses below EMA21
            if ema9.iloc[i] < ema21.iloc[i] and ema9.iloc[i-1] >= ema21.iloc[i-1]:
                return 'SELL'
                
        elif strategy == "multi_indicator":
            # Multi-indicator confluence
            bullish_signals = 0
            bearish_signals = 0
            
            if rsi.iloc[i] < 40: bullish_signals += 1
            if rsi.iloc[i] > 60: bearish_signals += 1
            if ema9.iloc[i] > ema21.iloc[i]: bullish_signals += 1
            if ema9.iloc[i] < ema21.iloc[i]: bearish_signals += 1
            if macd.iloc[i] > signal_line.iloc[i]: bullish_signals += 1
            if macd.iloc[i] < signal_line.iloc[i]: bearish_signals += 1
            if close.iloc[i] < bb_lower.iloc[i]: bullish_signals += 1
            if close.iloc[i] > bb_upper.iloc[i]: bearish_signals += 1
            
            if bullish_signals >= 3:
                return 'BUY'
            if bearish_signals >= 3:
                return 'SELL'
        
        return None
    
    def _calculate_metrics(self, data: pd.DataFrame, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'net_pnl': 0.0,
                'pnl_pct': 0.0,
                'max_drawdown': 0.0,
                'sharpe': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_duration': 0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
            }
        
        pnls = [t['pnl'] for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]
        
        net_pnl = sum(pnls)
        win_rate = (len(winning) / len(trades)) * 100 if trades else 0
        
        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Drawdown calculation
        equity_curve = [self.initial_capital]
        for pnl in pnls:
            equity_curve.append(equity_curve[-1] + pnl)
        
        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = ((peak - equity) / peak) * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio
        returns = pd.Series(pnls) / self.initial_capital
        sharpe = self._calculate_sharpe(returns)
        
        # Trade duration
        avg_duration = np.mean([t.get('duration', 0) for t in trades])
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'net_pnl': net_pnl,
            'pnl_pct': (net_pnl / self.initial_capital) * 100,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_duration': int(avg_duration),
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
        }
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        return (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0
