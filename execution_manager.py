# --- START OF FILE execution_manager.py ---

import math
import os
import MetaTrader5 as mt5
import time
import logging
from datetime import datetime, date
import pandas as pd
import pytz
# At the top of execution_manager.py
from typing import TYPE_CHECKING, Dict, Optional, List, Tuple, Any # <--- Make sure Any is imported

# Assume ProfitabilityEnhancer exists and has calculate_position_size
# If not, create a placeholder or implement basic logic here
try:
    from profitability_enhancer import ProfitabilityEnhancer
except ImportError:
    log_placeholder = logging.getLogger('ExecutionManager') # Need logger before potential error
    log_placeholder.warning("ProfitabilityEnhancer not found. Using placeholder logic for position sizing.")
    class ProfitabilityEnhancer: # Placeholder
        def calculate_position_size(self, *args, **kwargs) -> Optional[float]: return 0.01
        def update_adaptive_risk_on_trade_close(self, *args, **kwargs): pass
    # Placeholder class if the real one isn't available
    class ProfitabilityEnhancer:
        def calculate_position_size(self, pair: str, direction: str, entry_price: float, sl_price: float,
                                     account_info: mt5.AccountInfo, config: Dict) -> Optional[float]:
            # --- !! Basic Placeholder Logic !! ---
            # Replace with actual risk management calculation
            log.warning("Using placeholder position size calculation (0.01 lots). Implement ProfitabilityEnhancer.")
            min_volume = config.get('min_lot', 0.01)
            max_volume = config.get('max_lot', 1.0)
            # Ensure size is within broker limits (fetch from symbol_info if needed)
            return max(min_volume, min(0.01, max_volume)) # Fixed 0.01 lot for placeholder

# Import PerformanceTracker (assuming it's in the same directory orPYTHONPATH)
try:
    # Use TYPE_CHECKING to avoid runtime circular dependency if ExecutionManager might be imported by tracker
    if TYPE_CHECKING:
        from performance_tracker import PerformanceTracker
    else:
        from performance_tracker import PerformanceTracker
except ImportError:
     log_placeholder = logging.getLogger('ExecutionManager')
     log_placeholder.critical("FATAL: Could not import PerformanceTracker. ExecutionManager cannot function.")
     raise



# Configure logging for this module
log = logging.getLogger('ExecutionManager') # Use a specific logger name
log.setLevel(logging.INFO) # Set default level
log.propagate = False # Stop messages going to the root logger

# Ensure handlers are not added multiple times if the module is reloaded
if not log.handlers:
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')

    # File handler specific to execution manager logs
    try:
        fh = logging.FileHandler('logs/execution.log', mode='a') # Append mode
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    except Exception as e:
         print(f"Error setting up execution file logger: {e}") # Use print as logger might fail
log.setLevel(logging.INFO)
    # Optional: Stream handler for console output
    # sh = logging.StreamHandler()
    # sh.setLevel(logging.INFO)
    # sh.setFormatter(formatter)
    # log.addHandler(sh)


class ExecutionManager:
    """
    Handles actual trade execution via MT5, manages open positions,
    and applies basic execution rules.
    """

    DEFAULT_CONFIG = {
        "max_daily_trades": 10,
        "max_concurrent_trades": 3,
        "allowed_slippage_pips": 3,
        "order_magic_number": 12345,
        "risk_per_trade_pct": 1.0,
        "min_lot": 0.01,
        "max_lot": 1.0,
        "symbol_mapping": {},
        "close_on_opposite_signal": False
    }

    # --- MODIFIED __init__ ---
    def __init__(self,
             mt5_instance: Optional[Any], # Accepts Optional[Any] (can be None)
             tracker: 'PerformanceTracker',
             profit_enhancer: ProfitabilityEnhancer,
             config: Optional[Dict] = None):
        """
        Initializes the ExecutionManager. Allows mt5_instance to be None initially.

        Args:
            mt5_instance: An initialized MT5 instance or None.
            tracker: An instance of PerformanceTracker.
            profit_enhancer: An instance for calculating position size and risk factors.
            config (Optional[Dict]): Configuration dictionary overriding defaults.
        """
        # Store potentially None MT5 instance
        self.mt5 = mt5_instance

        # --- REMOVED strict MT5 check from __init__ ---
        # The check below caused the ValueError when None was passed.
        # It's moved to _is_ready() and used in operational methods.
        # if not mt5_instance or not hasattr(mt5_instance, 'initialize') or not mt5_instance.terminal_info():
        #     log.critical("ExecutionManager requires a valid, initialized MT5 instance.")
        #     raise ValueError("Invalid MT5 instance provided to ExecutionManager.")
        # --- END REMOVED CHECK ---

        # Validate Tracker and Enhancer instances
        if not tracker or not isinstance(tracker, object) or not hasattr(tracker, 'get_performance_metrics'):
            log.critical("ExecutionManager requires a valid PerformanceTracker instance.")
            raise ValueError("Invalid PerformanceTracker instance provided.")
        if not profit_enhancer or not isinstance(profit_enhancer, ProfitabilityEnhancer):
            log.critical("ExecutionManager requires a valid ProfitabilityEnhancer instance.")
            raise ValueError("Invalid ProfitabilityEnhancer instance provided.")

        # Store validated dependencies
        self.tracker = tracker
        self.profit_enhancer = profit_enhancer

        # Merge config correctly
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Internal state initialization
        self.open_positions: Dict[int, Dict] = {}
        self.daily_trade_count: int = 0
        self.last_reset_day: Optional[date] = None

        # Initialize order lock conditionally based on initial mt5 state
        self.order_lock = None
        if self.mt5 and hasattr(self.mt5, 'lock'):
            try:
                self.order_lock = self.mt5.lock()
            except Exception as e:
                log.error(f"Failed to initialize MT5 lock during init: {e}")

        # Log initialization status based on the potentially None MT5 instance
        mt5_status = "Available" if self.is_mt5_ready() else "Missing/Invalid"
        lock_status = "Initialized" if self.order_lock else "Not Initialized (MT5 missing or lock failed)"
        log.info(f"ExecutionManager initialized. Initial MT5 Status: {mt5_status}, Lock Status: {lock_status}")
        log.info(f"Config loaded: Magic={self.config['order_magic_number']}, BaseRisk={self.config['risk_per_trade_pct']}%, MaxConcurrent={self.config['max_concurrent_trades']}")
        self._reset_daily_counter_if_needed() # Initialize day check
    # --- END MODIFIED __init__ ---

    def is_mt5_ready(self) -> bool:
        """Checks if MT5 is connected and the order lock is initialized."""
        if not self.mt5 or not hasattr(self.mt5, 'terminal_info') or not self.mt5.terminal_info():
            # log.debug("MT5 check failed: Instance is None or terminal_info unavailable.")
            return False
        # Also check if the lock was successfully initialized IF mt5 is present
        if self.order_lock is None and self.mt5 and hasattr(self.mt5, 'lock'):
             log.warning("MT5 seems connected, but order_lock was not initialized. Attempting re-init.")
             try:
                 self.order_lock = self.mt5.lock()
                 if self.order_lock: log.info("Successfully initialized order_lock.")
                 else: log.error("Failed to re-initialize order_lock."); return False
             except Exception as e:
                 log.error(f"Exception during order_lock re-initialization: {e}"); return False
        elif self.order_lock is None and not (self.mt5 and hasattr(self.mt5, 'lock')):
             # This case means MT5 object doesn't support lock or MT5 is None, expected if not connected
             pass # Don't log error here, handled by first check

        return self.order_lock is not None # Use MT5's lock for thread safety on order operations

        log.info("ExecutionManager initialized.")
        log.info(f"Configuration: Max Daily Trades={self.config['max_daily_trades']}, Max Concurrent={self.config['max_concurrent_trades']}, Magic={self.config['order_magic_number']}")
        self._reset_daily_counter_if_needed() # Initialize day check

    def _get_mt5_symbol(self, pair: str) -> str:
        """Maps the signal pair to the MT5 symbol using config mapping."""
        return self.config.get('symbol_mapping', {}).get(pair, pair)

    def _reset_daily_counter_if_needed(self):
        """Resets the daily trade counter at the start of a new day (UTC)."""
        today = datetime.now(pytz.utc).date()
        if self.last_reset_day is None or self.last_reset_day != today:
            log.info(f"New day detected ({today}). Resetting daily trade counter from {self.daily_trade_count} to 0.")
            self.daily_trade_count = 0
            self.last_reset_day = today

    def _check_trade_allowed(self, pair: str) -> Tuple[bool, str]:
        """
        Checks if a new trade is allowed based on configured rules.

        Returns:
            Tuple[bool, str]: (allowed, reason)
        """
        self._reset_daily_counter_if_needed()

        # 1. Max Daily Trades
        if self.daily_trade_count >= self.config['max_daily_trades']:
            reason = f"Maximum daily trades ({self.config['max_daily_trades']}) reached."
            log.warning(f"Trade Denied for {pair}: {reason}")
            return False, reason

        # 2. Max Concurrent Trades
        if len(self.open_positions) >= self.config['max_concurrent_trades']:
            reason = f"Maximum concurrent trades ({self.config['max_concurrent_trades']}) reached."
            log.warning(f"Trade Denied for {pair}: {reason}")
            return False, reason

        # 3. Max Total Risk (Placeholder - Requires tracking risk per position)
        # current_total_risk = sum(pos.get('risk_pct', 0) for pos in self.open_positions.values())
        # if current_total_risk >= self.config.get('max_total_risk_pct', 5.0):
        #     reason = f"Maximum total open risk ({self.config.get('max_total_risk_pct')}%) reached."
        #     log.warning(f"Trade Denied for {pair}: {reason}")
        #     return False, reason
        # log.debug(f"Current total open risk (placeholder check): Allowed.")

        # 4. Check MT5 connection status
        if not self.mt5.terminal_info():
             reason = "MT5 connection lost."
             log.error(f"Trade Denied for {pair}: {reason}")
             return False, reason

        log.debug(f"Trade Check for {pair}: Allowed (Daily: {self.daily_trade_count}/{self.config['max_daily_trades']}, Concurrent: {len(self.open_positions)}/{self.config['max_concurrent_trades']})")
        return True, "Allowed"

    def _get_symbol_info(self, mt5_symbol: str) -> Optional[mt5.SymbolInfo]:
        """Safely gets symbol info from MT5, handling selection if needed."""
        info = self.mt5.symbol_info(mt5_symbol)
        if info is None:
            log.warning(f"Symbol '{mt5_symbol}' not found in MarketWatch. Attempting to select...")
            if self.mt5.symbol_select(mt5_symbol, True):
                time.sleep(0.5) # Allow time for update
                info = self.mt5.symbol_info(mt5_symbol)
                if info:
                    log.info(f"Successfully selected symbol '{mt5_symbol}'.")
                else:
                    log.error(f"Failed to get info for symbol '{mt5_symbol}' even after selecting.")
                    return None
            else:
                log.error(f"Failed to select symbol '{mt5_symbol}' (Error: {self.mt5.last_error()}).")
                return None
        # Check if symbol is tradable
        if not info.visible:
             log.warning(f"Symbol '{mt5_symbol}' is available but not visible/tradable in MarketWatch.")
             # Optionally return None here if non-tradable symbols should block orders
             # return None
        return info

    def _get_current_price(self, mt5_symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Gets the current bid and ask price for a symbol."""
        tick = self.mt5.symbol_info_tick(mt5_symbol)
        if tick:
            return tick.bid, tick.ask
        else:
            log.warning(f"Could not retrieve tick data for {mt5_symbol}. Last error: {self.mt5.last_error()}")
            return None, None

    def _calculate_mt5_volume(self, signal_details: Dict) -> Optional[float]:
        """Calculates the trade volume (lot size) using the ProfitabilityEnhancer."""
        pair = signal_details['pair']
        direction = signal_details['direction']
        entry_price = signal_details['entry_price'] # Should be near current price for market order
        sl_price = signal_details['stop_loss']
        mt5_symbol = self._get_mt5_symbol(pair)

        log.debug(f"Calculating position size for {pair} {direction} @{entry_price}, SL={sl_price}")

        # Get necessary account info
        account_info = self.mt5.account_info()
        if not account_info:
             log.error("Failed to retrieve account info. Cannot calculate position size.")
             return None

        # Get symbol info for contract size, limits etc.
        symbol_info = self._get_symbol_info(mt5_symbol)
        if not symbol_info:
            log.error(f"Failed to retrieve symbol info for {mt5_symbol}. Cannot accurately calculate position size.")
            return None

        # --- Delegate to ProfitabilityEnhancer ---
        # Prepare config subset relevant for sizing
        sizing_config = {
            "risk_per_trade_pct": self.config['risk_per_trade_pct'],
            "min_lot": self.config['min_lot'],
            "max_lot": self.config['max_lot'],
            "lot_step": symbol_info.volume_step,
            "contract_size": symbol_info.trade_contract_size,
            "account_currency": account_info.currency,
            # Pass symbol info directly if enhancer needs more details
            "symbol_info": symbol_info
        }

        try:
            calculated_volume = self.profit_enhancer.calculate_position_size(
                pair=pair,
                direction=direction,
                entry_price=entry_price, # Use signal entry as estimate
                sl_price=sl_price,
                account_info=account_info,
                config=sizing_config # Pass relevant config
            )
        except Exception as e:
             log.exception(f"Error calling calculate_position_size from ProfitabilityEnhancer: {e}")
             return None


        # --- Validate Calculated Volume ---
        if calculated_volume is None or calculated_volume <= 0:
            log.warning(f"Position size calculation returned invalid volume ({calculated_volume}) for {pair}.")
            return None

        # Enforce broker volume limits
        min_vol = symbol_info.volume_min
        max_vol = symbol_info.volume_max
        step_vol = symbol_info.volume_step

        if calculated_volume < min_vol:
            log.warning(f"Calculated volume {calculated_volume} is below minimum {min_vol} for {mt5_symbol}. Adjusting to minimum.")
            calculated_volume = min_vol
        elif calculated_volume > max_vol:
            log.warning(f"Calculated volume {calculated_volume} exceeds maximum {max_vol} for {mt5_symbol}. Adjusting to maximum.")
            calculated_volume = max_vol

        # Adjust volume to match step
        if step_vol > 0:
             calculated_volume = round(round(calculated_volume / step_vol) * step_vol, 8) # Round to nearest step, handle potential float issues

        # Final check after adjustments
        if calculated_volume < min_vol or calculated_volume <= 0:
             log.error(f"Final calculated volume ({calculated_volume}) is invalid after adjustments for {mt5_symbol}. Cannot place order.")
             return None

        log.info(f"Calculated volume for {pair} {direction}: {calculated_volume:.3f} lots")
        return calculated_volume


    def _build_market_order_request(self, signal_details: Dict, volume: float) -> Optional[Dict]:
        """Constructs the request dictionary for a market order."""
        pair = signal_details['pair']
        direction = signal_details['direction']
        sl_price = signal_details['stop_loss']
        tp_price = signal_details['take_profit']
        mt5_symbol = self._get_mt5_symbol(pair)

        symbol_info = self._get_symbol_info(mt5_symbol)
        if not symbol_info:
            log.error(f"Cannot build order request: Symbol info not found for {mt5_symbol}.")
            return None

        order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
        deviation = self.config['allowed_slippage_pips']
        magic = self.config['order_magic_number']
        comment = f"SignalBot {direction} {pair} Conf:{signal_details.get('confidence', 0):.2f}"[:31] # Max comment length

        # Get current price for market order execution
        bid, ask = self._get_current_price(mt5_symbol)
        if bid is None or ask is None:
            log.error(f"Cannot build order request: Failed to get current price for {mt5_symbol}.")
            return None
        price = ask if order_type == mt5.ORDER_TYPE_BUY else bid

        # Validate SL/TP placement relative to current price and symbol properties
        freeze_level = symbol_info.trade_freeze_level # Distance in points where SL/TP cannot be placed
        current_point = symbol_info.point

        if direction == 'BUY':
            if sl_price > price - freeze_level * current_point:
                log.warning(f"SL price {sl_price} for BUY is too close to current price {price} (freeze level). Adjusting slightly lower.")
                sl_price = price - freeze_level * current_point * 1.1 # Adjust slightly away
            if tp_price < price + freeze_level * current_point:
                 log.warning(f"TP price {tp_price} for BUY is too close to current price {price} (freeze level). Ignoring TP for now or adjust.")
                 # Decide: ignore TP, adjust, or reject order? Let's ignore for now if too close.
                 # tp_price = 0.0 # Set TP to 0.0 if invalid
                 tp_price = price + freeze_level * current_point * 1.1
        else: # SELL
            if sl_price < price + freeze_level * current_point:
                log.warning(f"SL price {sl_price} for SELL is too close to current price {price} (freeze level). Adjusting slightly higher.")
                sl_price = price + freeze_level * current_point * 1.1
            if tp_price > price - freeze_level * current_point:
                 log.warning(f"TP price {tp_price} for SELL is too close to current price {price} (freeze level). Ignoring TP for now or adjust.")
                 # tp_price = 0.0
                 tp_price = price - freeze_level * current_point * 1.1

        # Ensure SL/TP are not zero if they were originally intended
        sl_price = round(sl_price, symbol_info.digits)
        tp_price = round(tp_price, symbol_info.digits)
        sl_final = sl_price if sl_price != 0 else 0.0
        tp_final = tp_price if tp_price != 0 else 0.0


        request = {
            "action": mt5.TRADE_ACTION_DEAL, # Market execution
            "symbol": mt5_symbol,
            "volume": volume,
            "type": order_type,
            "price": price, # MT5 uses this for context, execution is at market
            "sl": sl_final,
            "tp": tp_final,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC, # Good Till Cancelled (standard for market orders)
            "type_filling": mt5.ORDER_FILLING_IOC, # Immediate Or Cancel is usually preferred for market
        }
        log.debug(f"Built order request: {request}")
        return request


    def place_order(self, signal_details: Dict) -> Optional[int]:
        """
        Attempts to place a market order based on the signal details.

        Args:
            signal_details (Dict): The signal dictionary from SignalGenerator.

        Returns:
            Optional[int]: The MT5 position ticket if the order resulted in a
                           position, otherwise None.
        """
        pair = signal_details.get('pair')
        if not pair:
             log.error("Cannot place order: Signal details missing 'pair'.")
             return None

        log.info(f"Attempting to place order for signal: {pair} {signal_details.get('direction')}...")

        # 1. Check if trade is allowed
        allowed, reason = self._check_trade_allowed(pair)
        if not allowed:
            log.warning(f"Order placement denied for {pair}: {reason}")
            return None

        # 2. Calculate Volume
        volume = self._calculate_mt5_volume(signal_details)
        if volume is None or volume <= 0:
            log.error(f"Order placement failed for {pair}: Invalid volume calculated ({volume}).")
            return None

        # 3. Build Order Request
        request = self._build_market_order_request(signal_details, volume)
        if request is None:
            log.error(f"Order placement failed for {pair}: Could not build valid order request.")
            return None

        # 4. Send Order (Thread Safe)
        result = None
        with self.order_lock: # Acquire lock before sending order
            log.info(f"Sending order request to MT5: {request}")
            try:
                result = self.mt5.order_send(request)
            except Exception as e:
                log.exception(f"Exception during mt5.order_send for {pair}: {e}")
                # Ensure MT5 connection is still valid after exception
                if not self.mt5.terminal_info(): log.error("MT5 connection lost during order send.")
                return None

        # 5. Process Result
        if result is None:
            log.error(f"Order send failed for {pair}. MT5 returned None. Last error: {self.mt5.last_error()}")
            return None

        log.info(f"MT5 Order Send Result: Code={result.retcode}, Comment={result.comment}, Order={result.order}, Deal={result.deal}, Price={result.price}, Volume={result.volume}")

        # Check return code for success
        # TRADE_RETCODE_DONE usually means the request was processed and might have resulted in a deal/position
        # Other codes indicate rejection or errors. See MT5 docs for all codes.
        if result.retcode != mt5.TRADE_RETCODE_DONE:
             log.error(f"Order placement failed for {pair}. RetCode: {result.retcode}, Comment: {result.comment}. Last MT5 Error: {self.mt5.last_error()}")
             # Check for specific requote or price change errors
             if result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_CHANGED]:
                 log.warning(f"Order rejected due to price change/requote for {pair}.")
             # Check for margin issues
             elif result.retcode == mt5.TRADE_RETCODE_NO_MONEY:
                  log.error(f"Order rejected for {pair}: Insufficient funds (No Money).")
             return None


        # --- Order Accepted - Find the Resulting Position ---
        # A successful market order should result in a deal and a position almost immediately.
        # We need the *position ticket* for monitoring.
        time.sleep(0.5) # Small delay to allow MT5 server to update position state
        position_ticket = None
        mt5_symbol = self._get_mt5_symbol(pair)
        magic = self.config['order_magic_number']
        found_position = None

        try:
            # Get all positions for the symbol and magic number
            positions = self.mt5.positions_get(symbol=mt5_symbol, magic=magic)
            if positions:
                 # Try to match based on the deal ticket from the result
                 deal_ticket = result.deal
                 for pos in positions:
                     # Check if this position originated from our deal ticket
                     # Note: MT5 position.deal doesn't exist; need history_deals_get
                     # Alternative: Match based on direction, volume, and time proximity
                     # Let's use direction, volume, and check if it's a recent position
                     entry_time_dt = datetime.fromtimestamp(pos.time, tz=pytz.utc)
                     now_utc = datetime.now(pytz.utc)
                     time_diff_seconds = (now_utc - entry_time_dt).total_seconds()

                     is_correct_direction = (pos.type == mt5.POSITION_TYPE_BUY and signal_details['direction'] == 'BUY') or \
                                            (pos.type == mt5.POSITION_TYPE_SELL and signal_details['direction'] == 'SELL')
                     is_correct_volume = math.isclose(pos.volume, volume, rel_tol=1e-5)

                     # Assume a position opened within the last ~10 seconds is the one we just placed
                     if is_correct_direction and is_correct_volume and time_diff_seconds < 10:
                         position_ticket = pos.ticket
                         found_position = pos
                         log.info(f"Found matching position for order {result.order}: Position Ticket={position_ticket}, Time={entry_time_dt}, Volume={pos.volume}")
                         break
                 if not found_position:
                      log.warning(f"Order {result.order} successful, but couldn't definitively match a new position for {pair} (Magic: {magic}). Check active positions manually.")
            else:
                 log.warning(f"Order {result.order} successful, but no positions found for {pair} with magic {magic}.")

        except Exception as e:
            log.exception(f"Exception while searching for position after order send for {pair}: {e}")


        # 6. Update Internal State if Position Found
        if position_ticket and found_position:
            self.daily_trade_count += 1
            self.open_positions[position_ticket] = {
                'pair': pair,
                'mt5_symbol': mt5_symbol,
                'direction': signal_details['direction'],
                'volume': found_position.volume,
                'entry_price': found_position.price_open,
                'sl': found_position.sl, # Get actual SL/TP set on position
                'tp': found_position.tp,
                'entry_time': datetime.fromtimestamp(found_position.time, tz=pytz.utc),
                'magic': found_position.magic,
                'position_ticket': position_ticket,
                'identifier': found_position.identifier, # Crucial for linking deals later
                'signal_confidence': signal_details.get('confidence'),
                'original_signal': signal_details # Store original signal info if needed
            }
            log.info(f"Successfully opened position for {pair}. Ticket: {position_ticket}, Volume: {found_position.volume}, Entry: {found_position.price_open}")
            return position_ticket
        else:
            # Order was successful but couldn't confirm position opening or link it.
            # This is potentially problematic. Might need manual intervention.
            log.error(f"Order {result.order} executed, but failed to identify and store the resulting position for {pair}.")
            # Increment daily count anyway as an order was attempted/accepted? debatable.
            # self.daily_trade_count += 1
            return None

    def _check_sl_tp_hit(self, position: Any, current_bid: float, current_ask: float) -> Optional[str]:
        """Checks if the SL or TP level has been breached by current market prices."""
        if current_bid is None or current_ask is None: return None # Cannot check without prices

        sl_price = position.sl
        tp_price = position.tp

        # Check BUY position
        if position.type == mt5.POSITION_TYPE_BUY:
            # SL hit if current BID drops to or below SL price
            if sl_price > 0 and current_bid <= sl_price:
                log.info(f"SL Hit Detected for BUY Position {position.ticket} ({position.symbol}): Bid ({current_bid}) <= SL ({sl_price})")
                return 'SL'
            # TP hit if current BID rises to or above TP price
            if tp_price > 0 and current_bid >= tp_price:
                log.info(f"TP Hit Detected for BUY Position {position.ticket} ({position.symbol}): Bid ({current_bid}) >= TP ({tp_price})")
                return 'TP'

        # Check SELL position
        elif position.type == mt5.POSITION_TYPE_SELL:
            # SL hit if current ASK rises to or above SL price
            if sl_price > 0 and current_ask >= sl_price:
                log.info(f"SL Hit Detected for SELL Position {position.ticket} ({position.symbol}): Ask ({current_ask}) >= SL ({sl_price})")
                return 'SL'
            # TP hit if current ASK drops to or below TP price
            if tp_price > 0 and current_ask <= tp_price:
                log.info(f"TP Hit Detected for SELL Position {position.ticket} ({position.symbol}): Ask ({current_ask}) <= TP ({tp_price})")
                return 'TP'

        return None # No hit detected

    def _close_position_request(self, position: Any, close_price: float, volume: float) -> Optional[Dict]:
        """Builds the request dictionary to close an existing position."""
        deviation = self.config['allowed_slippage_pips']
        magic = position.magic
        comment = f"Close Bot Pos {position.ticket}"[:31]
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY # Opposite type to close

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket, # Specify the position ticket to close
            "symbol": position.symbol,
            "volume": volume, # Close the full volume
            "type": order_type,
            "price": close_price, # Current market price for closure
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        log.debug(f"Built position close request: {request}")
        return request

    def monitor_and_close_positions(self):
        """
        Monitors open positions, checks for SL/TP hits, and attempts to close them.
        Also handles positions closed externally (e.g., by server SL/TP).
        """
        if not self.open_positions:
            # log.debug("No open positions to monitor.")
            return

        log.debug(f"Monitoring {len(self.open_positions)} open positions...")
        # Iterate over a copy of tickets as the dict might change during iteration
        position_tickets_to_check = list(self.open_positions.keys())

        for ticket in position_tickets_to_check:
            # Retrieve stored details first
            stored_details = self.open_positions.get(ticket)
            if not stored_details:
                log.warning(f"Position ticket {ticket} found in check list but not in open_positions dict. Skipping.")
                continue

            mt5_symbol = stored_details['mt5_symbol']
            position_identifier = stored_details['identifier'] # Use the persistent identifier

            # --- Check if Position Still Exists on Server ---
            position_info_list = self.mt5.positions_get(ticket=ticket)

            if position_info_list and len(position_info_list) > 0:
                # --- Position Found - Check for Manual Close ---
                position = position_info_list[0]
                current_bid, current_ask = self._get_current_price(mt5_symbol)
                if current_bid is None or current_ask is None:
                    log.warning(f"Cannot monitor position {ticket}: Failed to get current price for {mt5_symbol}.")
                    continue # Skip check if price unavailable

                close_reason = self._check_sl_tp_hit(position, current_bid, current_ask)

                # TODO: Implement check for opposite signal if config enabled
                # if not close_reason and self.config.get('close_on_opposite_signal'):
                #     # Fetch latest signal for this pair/timeframe
                #     # If opposite signal exists, set close_reason = 'Opposite Signal'
                #     pass

                if close_reason:
                    log.info(f"Close condition met for position {ticket} ({mt5_symbol}): {close_reason}. Attempting closure.")
                    close_price = current_bid if position.type == mt5.POSITION_TYPE_BUY else current_ask
                    close_volume = position.volume
                    close_request = self._close_position_request(position, close_price, close_volume)

                    if close_request:
                        close_result = None
                        with self.order_lock: # Lock for closing order
                             log.info(f"Sending close request to MT5: {close_request}")
                             try:
                                 close_result = self.mt5.order_send(close_request)
                             except Exception as e:
                                  log.exception(f"Exception during mt5.order_send for closing position {ticket}: {e}")
                                  if not self.mt5.terminal_info(): log.error("MT5 connection lost during position close.")
                                  continue # Try again next cycle

                        if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                            log.info(f"Successfully sent close order for position {ticket}. Result: {close_result.comment}")
                            # Closure is successful. MT5 deals will update history.
                            # Remove from open list NOW, rely on history check later if needed
                            # Or directly record based on close_result? Less accurate P/L maybe.
                            # Best approach: Remove now, let the 'external close' check handle recording later.
                            try:
                                 del self.open_positions[ticket]
                                 log.info(f"Removed internally closed position {ticket} from tracking.")
                            except KeyError:
                                 log.warning(f"Tried to remove already removed position {ticket} after internal close.")

                        elif close_result:
                            log.error(f"Failed to close position {ticket}. RetCode: {close_result.retcode}, Comment: {close_result.comment}. Last MT5 Error: {self.mt5.last_error()}")
                        else:
                             log.error(f"Close order send failed for position {ticket}. MT5 returned None. Last error: {self.mt5.last_error()}")
                    else:
                         log.error(f"Failed to build close request for position {ticket}.")

            else:
                # --- Position Not Found - Assume Closed Externally ---
                log.info(f"Position {ticket} ({mt5_symbol}) no longer found on server. Assuming closed externally (SL/TP hit, Manual, etc.). Checking history...")

                # Query history deals related to this position identifier
                # Need a time range - from entry time to now
                entry_time_dt = stored_details.get('entry_time')
                if not isinstance(entry_time_dt, datetime):
                     # Attempt conversion if stored as string
                     try: entry_time_dt = pd.to_datetime(stored_details.get('entry_time'), errors='coerce')
                     except: entry_time_dt = None

                if not entry_time_dt:
                     log.error(f"Cannot check history for position {ticket}: Invalid entry time stored.")
                     # Remove from tracking to prevent repeated checks? Risky if history check fails.
                     # del self.open_positions[ticket]
                     continue

                from_date = entry_time_dt - pd.Timedelta(minutes=1) # Check from slightly before entry
                to_date = datetime.now(pytz.utc) + pd.Timedelta(minutes=1)

                deals = self.mt5.history_deals_get(position=position_identifier) # Get deals by position ID

                if deals is None:
                    log.warning(f"Failed to retrieve history deals for position identifier {position_identifier}. Last Error: {self.mt5.last_error()}. Cannot record closure.")
                    # Keep in open_positions? Or remove? If we remove, it's lost. Let's keep it for now.
                    continue

                if not deals:
                    log.warning(f"Position {ticket} (ID: {position_identifier}) not found, and no historical deals found for it. State uncertain.")
                    # Remove from tracking as it's definitively gone and we have no history
                    try:
                        del self.open_positions[ticket]
                        log.info(f"Removed untraceable closed position {ticket} from tracking.")
                    except KeyError: pass # Already removed?
                    continue

                # --- Process Deals to Find Closure ---
                # Deals are returned newest first usually. Find entry and exit deals.
                entry_deal = None
                exit_deal = None
                commission = 0.0
                swap = 0.0
                profit_currency = 0.0
                exit_reason = "Unknown External" # Default reason

                for deal in deals:
                    # Check if deal belongs to our magic number and identifier
                    if deal.magic != stored_details['magic'] or deal.position_id != position_identifier: continue

                    # Identify entry deal (type DEAL_ENTRY_IN)
                    if deal.entry == mt5.DEAL_ENTRY_IN:
                        entry_deal = deal
                        commission += deal.commission # Accumulate commission/swap
                        swap += deal.swap
                        profit_currency += deal.profit # Add profit from this deal leg

                    # Identify exit deal (type DEAL_ENTRY_OUT)
                    elif deal.entry == mt5.DEAL_ENTRY_OUT:
                        exit_deal = deal
                        commission += deal.commission
                        swap += deal.swap
                        profit_currency += deal.profit # Add profit from closing deal

                    # Handle DEAL_ENTRY_INOUT (e.g., SL/TP hit, partial close)
                    elif deal.entry == mt5.DEAL_ENTRY_INOUT:
                        # If volume matches position volume, it's likely the full closure
                        if math.isclose(deal.volume, stored_details['volume'], rel_tol=1e-5):
                            exit_deal = deal # Treat INOUT as the exit if volume matches
                            commission += deal.commission
                            swap += deal.swap
                            profit_currency += deal.profit
                            # Try to determine reason based on comment or SL/TP price match
                            if 'sl' in deal.comment.lower(): exit_reason = 'SL (Server)'
                            elif 'tp' in deal.comment.lower(): exit_reason = 'TP (Server)'
                            elif math.isclose(deal.price, stored_details['sl'], rel_tol=1e-9): exit_reason = 'SL (Server Price Match)'
                            elif math.isclose(deal.price, stored_details['tp'], rel_tol=1e-9): exit_reason = 'TP (Server Price Match)'
                            else: exit_reason = 'INOUT Close'

                # --- Record Trade if Exit Found ---
                if exit_deal:
                    # Calculate pips if possible (requires pip value logic)
                    pip_size = self._get_symbol_info(mt5_symbol).point * (10 if 'JPY' not in mt5_symbol else 1000)
                    price_diff = exit_deal.price - stored_details['entry_price']
                    profit_pips = (price_diff / pip_size) if stored_details['direction'] == 'BUY' else (-price_diff / pip_size)

                    closed_trade_details = {
                        'trade_id': f"{position_identifier}_{exit_deal.ticket}", # Combine position ID and exit deal ticket
                        'pair': stored_details['pair'],
                        'direction': stored_details['direction'],
                        'entry_time': stored_details['entry_time'], # Use stored entry time
                        'exit_time': datetime.fromtimestamp(exit_deal.time, tz=pytz.utc),
                        'entry_price': stored_details['entry_price'],
                        'exit_price': exit_deal.price,
                        'stop_loss': stored_details['sl'], # Original SL
                        'take_profit': stored_details['tp'], # Original TP
                        'position_size': exit_deal.volume,
                        'profit_currency': profit_currency, # Sum of profit from all related deals
                        'profit_pips': round(profit_pips, 1),
                        'profit_percentage': (profit_currency / (stored_details.get('account_equity_on_entry', 10000))) * 100 if 'account_equity_on_entry' in stored_details else None, # Placeholder
                        'exit_reason': exit_reason,
                        'signal_confidence': stored_details.get('signal_confidence'),
                        # Add commission/swap if needed by tracker
                        # 'commission': commission,
                        # 'swap': swap,
                    }
                    # Record the trade
                    self.tracker.record_trade(closed_trade_details)
                    # Remove from open positions
                    try:
                        del self.open_positions[ticket]
                        log.info(f"Recorded externally closed position {ticket} (ID: {position_identifier}) from history. Reason: {exit_reason}")
                    except KeyError:
                        log.warning(f"Tried to remove already removed position {ticket} after external close.")

                else:
                    log.warning(f"Position {ticket} (ID: {position_identifier}) not found, deals found but no clear exit deal. Cannot record closure accurately.")
                    # Decide whether to remove from open_positions or keep checking. Let's remove it to avoid infinite loops.
                    try:
                        del self.open_positions[ticket]
                        log.warning(f"Removed position {ticket} from tracking due to ambiguous history.")
                    except KeyError: pass


    def get_open_positions(self) -> List[Dict]:
        """Returns a list of currently tracked open positions."""
        # Return a deep copy if modifications are expected outside this class
        return list(self.open_positions.values())

    def close_all_positions(self, reason: str = "Manual Close All") -> List[Tuple[int, bool, str]]:
        """
        Attempts to close all currently tracked open positions.

        Args:
            reason (str): The comment/reason for closing.

        Returns:
            List[Tuple[int, bool, str]]: List of (ticket, success_status, message) for each attempt.
        """
        log.warning(f"Attempting to close all ({len(self.open_positions)}) open positions. Reason: {reason}")
        results = []
        position_tickets_to_close = list(self.open_positions.keys())

        for ticket in position_tickets_to_close:
            stored_details = self.open_positions.get(ticket)
            if not stored_details:
                results.append((ticket, False, "Position not found in internal tracking"))
                continue

            mt5_symbol = stored_details['mt5_symbol']
            position_info_list = self.mt5.positions_get(ticket=ticket)

            if position_info_list and len(position_info_list) > 0:
                position = position_info_list[0]
                current_bid, current_ask = self._get_current_price(mt5_symbol)
                if current_bid is None or current_ask is None:
                    log.warning(f"Cannot close position {ticket}: Failed to get current price for {mt5_symbol}.")
                    results.append((ticket, False, "Failed to get current price"))
                    continue

                close_price = current_bid if position.type == mt5.POSITION_TYPE_BUY else current_ask
                close_volume = position.volume
                close_request = self._close_position_request(position, close_price, close_volume)
                # Override comment
                if close_request: close_request['comment'] = reason[:31]

                if close_request:
                    close_result = None
                    with self.order_lock: # Lock for closing order
                         log.info(f"Sending close request for {ticket}: {close_request}")
                         try:
                             close_result = self.mt5.order_send(close_request)
                         except Exception as e:
                              log.exception(f"Exception closing position {ticket}: {e}")
                              results.append((ticket, False, f"Exception: {e}"))
                              continue # Next position

                    if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                        log.info(f"Close order sent successfully for position {ticket}.")
                        results.append((ticket, True, close_result.comment))
                        # Remove immediately, rely on history check later if needed for recording
                        try: del self.open_positions[ticket]
                        except KeyError: pass
                    elif close_result:
                        log.error(f"Failed to close position {ticket}. RetCode: {close_result.retcode}, Comment: {close_result.comment}")
                        results.append((ticket, False, f"MT5 Error {close_result.retcode}: {close_result.comment}"))
                    else:
                        log.error(f"Close order send failed for position {ticket}. MT5 returned None.")
                        results.append((ticket, False, "MT5 returned None"))
                else:
                    log.error(f"Failed to build close request for position {ticket}.")
                    results.append((ticket, False, "Failed to build close request"))
            else:
                log.info(f"Position {ticket} already closed or not found on server during close all.")
                results.append((ticket, True, "Already closed or not found"))
                # Ensure removed from tracking
                try: del self.open_positions[ticket]
                except KeyError: pass

        return results


# --- END OF FILE execution_manager.py ---