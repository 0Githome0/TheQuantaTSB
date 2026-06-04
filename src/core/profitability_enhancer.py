# --- START OF FILE profitability_enhancer.py ---

import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import math
from typing import Dict, Optional, TYPE_CHECKING, Any, Tuple, List
from datetime import datetime, timedelta

# Import PerformanceTracker safely
if TYPE_CHECKING:
    from src.core.performance_tracker import PerformanceTracker

# Configure logging
log = logging.getLogger('ProfitabilityEnhancer')
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    import os
    os.makedirs('logs', exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
    try:
        fh = logging.FileHandler('logs/profitability.log', mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    except Exception as e: print(f"Error setting up profitability logger: {e}")
    # sh = logging.StreamHandler(); sh.setLevel(logging.INFO); sh.setFormatter(formatter); log.addHandler(sh) # Console logs

class ProfitabilityEnhancer:
    """
    Enhances trading profitability through performance analysis, adaptive risk management,
    and optimized position sizing.
    Features:
    - Strategy adaptation checks based on performance metrics.
    - Position sizing based on fixed or dynamic risk percentage.
    - Optional margin checks before confirming size.
    - Optional adaptive risk adjustment based on consecutive wins/losses.
    """

    # --- Default Configuration Structure ---
    # These structure expectations should be reflected in config.json
    DEFAULT_CONFIG: Dict[str, Any] = {
        "adaptation_check": {
            "enabled": True,
            "lookback_trades": 50,
            "min_trades_for_check": 20,
            "cooldown_minutes": 180,
            "thresholds": {
                "min_win_rate": 0.40,
                "min_profit_factor": 0.80,
                "min_expectancy_currency": 0.0,
                # "max_drawdown_pct": 0.25 # Example if drawdown is calculated elsewhere
            }
        },
        "dynamic_risk": { # Based on signal confidence
            "enabled": False,
            "min_risk_factor": 0.5,
            "max_risk_factor": 1.2,
            "confidence_threshold": 0.5
        },
        "adaptive_risk": { # Based on consecutive wins/losses
            "enabled": False,
            "consecutive_losses_trigger": 3, # Reduce risk after N losses
            "loss_risk_reduction_factor": 0.75, # Multiply current risk by this factor
            "consecutive_wins_trigger": 5, # Increase risk after N wins (Use with caution!)
            "win_risk_increase_factor": 1.10, # Multiply current risk by this factor
            "max_risk_increase_limit_pct": 2.0, # Absolute maximum risk % allowed after increases
            "reset_counter_on_opposite": True # Reset win/loss counter when a trade of the opposite result occurs
        },
        "margin_check": {
            "enabled": True,
            "buffer_pct": 10.0
        }
    }

    def __init__(self, tracker: 'PerformanceTracker', mt5_instance: Optional[Any]):
        """Initializes the ProfitabilityEnhancer."""
        self.mt5 = mt5_instance
        if not tracker or not hasattr(tracker, 'get_performance_metrics'):
            raise ValueError("Invalid PerformanceTracker instance provided.")
        self.tracker = tracker

        # Internal state for adaptive risk
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0
        self._current_adaptive_risk_factor: float = 1.0 # Starts at 1.0 (no adjustment)

        self.last_adaptation_suggestion_time: Optional[datetime] = None
        self.adaptation_metrics_history: Dict[str, Any] = {}
        log.info(f"ProfitabilityEnhancer initialized. MT5 Status: {'Available' if self.is_mt5_ready() else 'Missing/Invalid'}")

    def is_mt5_ready(self) -> bool:
        """Checks if the MT5 instance is valid and initialized."""
        return bool(self.mt5 and hasattr(self.mt5, 'terminal_info') and self.mt5.terminal_info())

    def _merge_config(self, user_config: Optional[Dict]) -> Dict:
        """Deep merges user config with defaults."""
        def merge_dicts(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    merge_dicts(base[key], value)
                else:
                    base[key] = value
            return base

        # Start with a deep copy of defaults
        merged = {k: v.copy() if isinstance(v, dict) else v for k, v in self.DEFAULT_CONFIG.items()}
        if user_config and isinstance(user_config, dict):
            enhancer_user_cfg = user_config.get("profitability_enhancer", {})
            merged = merge_dicts(merged, enhancer_user_cfg)
        return merged

    def should_adapt_strategy(self, config: Optional[Dict] = None, lookback_trades: Optional[int] = None) -> Tuple[bool, Dict]:
        """
        Analyzes recent performance to determine if strategy adaptation is needed.
        Uses settings from the 'adaptation_check' section of the config.
        
        Args:
            config: Optional configuration dictionary
            lookback_trades: Optional override for the number of trades to analyze
        """
        cfg = self._merge_config(config).get("adaptation_check", {})
        if not cfg.get("enabled", True):
            log.debug("Adaptation check is disabled in config.")
            return False, {"reason": "Disabled"}

        # Use provided lookback_trades if specified, otherwise use config value
        trades_to_check = lookback_trades if lookback_trades is not None else cfg.get("lookback_trades", 50)
        thresholds = cfg.get("thresholds", {})
        min_req_trades = cfg.get("min_trades_for_check", 20)
        cooldown_minutes = cfg.get("cooldown_minutes", 180)

        # Cooldown Check
        now = datetime.now()
        if self.last_adaptation_suggestion_time and (now < self.last_adaptation_suggestion_time + timedelta(minutes=cooldown_minutes)):
             log.debug(f"Adaptation check skipped: Cooldown active.")
             return False, {"reason": "Cooldown Active"}

        log.info(f"Checking performance for adaptation (last {trades_to_check} trades)...")
        metrics = self.tracker.get_performance_metrics(lookback_trades=trades_to_check)

        if not metrics or metrics.get('error'):
            err = metrics.get('error', 'Unknown metrics error')
            log.warning(f"Adaptation check skipped: Metrics retrieval failed. Error: {err}")
            return False, {"reason": "Metrics Error", "error_details": err}

        total_trades = metrics.get('total_trades', 0)
        if total_trades < min_req_trades:
            log.info(f"Adaptation check skipped: Insufficient trades ({total_trades} < {min_req_trades}).")
            return False, {"reason": "Insufficient Trades", "trades_found": total_trades, "min_required": min_req_trades}

        needs_adapt = False
        reasons = []
        details = {"metrics": metrics, "thresholds": thresholds, "reasons": reasons}

        # Check each threshold defined in the config
        for metric_key, min_threshold in thresholds.items():
            if metric_key.startswith("min_"):
                metric_name = metric_key[4:] # e.g., "win_rate"
                current_value = metrics.get(metric_name)
                if current_value is not None:
                    # Special handling for profit_factor (ignore if >= 0 and < threshold if expectancy is ok)
                    if metric_name == "profit_factor":
                        expectancy = metrics.get('expectancy_currency', -float('inf'))
                        min_expectancy = thresholds.get("min_expectancy_currency", 0.0)
                        if (0 <= current_value < min_threshold) and expectancy < min_expectancy:
                            reasons.append(f"{metric_name.replace('_',' ').title()} ({current_value:.2f}) < Threshold ({min_threshold:.2f}) & Expectancy Low")
                            needs_adapt = True
                        elif current_value < 0: # Always flag negative PF
                             reasons.append(f"{metric_name.replace('_',' ').title()} ({current_value:.2f}) is Negative")
                             needs_adapt = True
                    elif current_value < min_threshold:
                        fmt = ".2%" if "rate" in metric_name else ".2f" if "factor" in metric_name else ".4f"
                        reasons.append(f"{metric_name.replace('_',' ').title()} ({current_value:{fmt}}) < Threshold ({min_threshold:{fmt}})")
                        needs_adapt = True

            # Example for max threshold (like drawdown)
            # elif metric_key.startswith("max_"):
            #     metric_name = metric_key[4:]
            #     current_value = metrics.get(metric_name)
            #     if current_value is not None and current_value > min_threshold: # Note: min_threshold variable holds the max value here
            #         reasons.append(f"{metric_name} ({current_value:.2%}) > Threshold ({min_threshold:.2%})")
            #         needs_adapt = True

        if needs_adapt:
            log.warning(f"PERFORMANCE ALERT: Adaptation suggested. Reasons: {'; '.join(reasons)}")
            self.last_adaptation_suggestion_time = now
            self.adaptation_metrics_history = metrics
            return True, details
        else:
            log.info("Performance metrics within adaptation thresholds.")
            return False, {"reason": "Performance OK", "metrics": metrics}

    def _update_adaptive_risk_counters(self, is_win: bool, config: Dict):
        """Updates consecutive win/loss counters based on the last trade result."""
        adaptive_cfg = config.get("adaptive_risk", {})
        if not adaptive_cfg.get("enabled", False):
            return # Adaptive risk disabled

        reset_on_opposite = adaptive_cfg.get("reset_counter_on_opposite", True)

        if is_win:
            self._consecutive_wins += 1
            if reset_on_opposite:
                 if self._consecutive_losses > 0: log.debug(f"Adaptive Risk: Win ended loss streak ({self._consecutive_losses}). Resetting loss counter.")
                 self._consecutive_losses = 0
        else: # Loss or break-even
            self._consecutive_losses += 1
            if reset_on_opposite:
                if self._consecutive_wins > 0: log.debug(f"Adaptive Risk: Loss ended win streak ({self._consecutive_wins}). Resetting win counter.")
                self._consecutive_wins = 0

        log.debug(f"Adaptive Risk Counters: Wins={self._consecutive_wins}, Losses={self._consecutive_losses}")

    def _calculate_adaptive_risk_factor(self, base_risk_pct: float, config: Dict) -> float:
        """Calculates the risk adjustment factor based on consecutive wins/losses."""
        adaptive_cfg = config.get("adaptive_risk", {})
        if not adaptive_cfg.get("enabled", False):
            return 1.0 # No adjustment if disabled

        losses_trigger = adaptive_cfg.get("consecutive_losses_trigger", 3)
        loss_reduction_factor = adaptive_cfg.get("loss_risk_reduction_factor", 0.75)
        wins_trigger = adaptive_cfg.get("consecutive_wins_trigger", 5)
        win_increase_factor = adaptive_cfg.get("win_risk_increase_factor", 1.10)
        max_risk_limit_pct = adaptive_cfg.get("max_risk_increase_limit_pct", 2.0) / 100.0 # Convert pct to decimal

        current_factor = self._current_adaptive_risk_factor

        # Check for loss trigger
        if self._consecutive_losses >= losses_trigger:
            new_factor = current_factor * loss_reduction_factor
            log.warning(f"Adaptive Risk: {self._consecutive_losses} consecutive losses >= trigger ({losses_trigger}). Reducing risk factor: {current_factor:.3f} -> {new_factor:.3f}")
            # Apply a minimum factor limit? e.g., max(0.25, new_factor)
            current_factor = max(0.25, new_factor) # Example: Minimum 25% of base risk
            # Reset counter after applying reduction to avoid repeated reduction on next trade
            # self._consecutive_losses = 0 # Or maybe keep counting until a win? Configurable? Let's keep counting for now.

        # Check for win trigger (only if not reducing due to losses)
        elif self._consecutive_wins >= wins_trigger:
             # Check absolute risk limit before increasing
             potential_new_risk_pct = base_risk_pct * (current_factor * win_increase_factor)
             if potential_new_risk_pct <= max_risk_limit_pct:
                 new_factor = current_factor * win_increase_factor
                 log.info(f"Adaptive Risk: {self._consecutive_wins} consecutive wins >= trigger ({wins_trigger}). Increasing risk factor: {current_factor:.3f} -> {new_factor:.3f}")
                 current_factor = new_factor
                 # Reset counter after applying increase?
                 # self._consecutive_wins = 0 # Let's keep counting for now.
             else:
                 log.warning(f"Adaptive Risk: Win trigger hit, but increasing risk factor would exceed max limit ({max_risk_limit_pct:.2%}). Factor remains {current_factor:.3f}.")
                 # Optionally clamp to max limit?
                 # current_factor = max_risk_limit_pct / base_risk_pct

        self._current_adaptive_risk_factor = current_factor
        return current_factor

    def _calculate_dynamic_risk_pct(self, base_risk_pct: float, signal_confidence: Optional[float], config: Dict) -> float:
        """Adjusts risk percentage based on signal confidence if enabled."""
        # (Keep the implementation from the previous response, it's good)
        dynamic_risk_cfg = config.get("dynamic_risk", self.DEFAULT_DYNAMIC_RISK_CONFIG)
        if not dynamic_risk_cfg.get("enabled", False) or signal_confidence is None or not (0 <= signal_confidence <= 1):
            return base_risk_pct
        min_factor = dynamic_risk_cfg.get("min_risk_factor", 0.5)
        max_factor = dynamic_risk_cfg.get("max_risk_factor", 1.2)
        conf_thresh = dynamic_risk_cfg.get("confidence_threshold", 0.5)
        if signal_confidence <= conf_thresh:
            relative_conf = signal_confidence / conf_thresh if conf_thresh > 0 else 1.0
            risk_factor = min_factor + (1.0 - min_factor) * relative_conf
        else:
            relative_conf = (signal_confidence - conf_thresh) / (1.0 - conf_thresh) if conf_thresh < 1.0 else 1.0
            risk_factor = 1.0 + (max_factor - 1.0) * relative_conf
        adjusted_risk_pct = base_risk_pct * risk_factor
        log.debug(f"Dynamic Risk (Confidence): Base={base_risk_pct:.4%}, Conf={signal_confidence:.3f}, Factor={risk_factor:.3f} -> Adjusted={adjusted_risk_pct:.4%}")
        return adjusted_risk_pct

    def _check_margin(self, mt5_symbol: str, direction: str, volume: float, entry_price: float, account_info: mt5.AccountInfo, config: Dict) -> Tuple[bool, Optional[float], Optional[float]]:
        """Checks if there is sufficient free margin."""
        # (Keep the implementation from the previous response, it's good)
        margin_cfg = config.get("margin_check", self.DEFAULT_MARGIN_CHECK_CONFIG)
        if not margin_cfg.get("enabled", False): return True, None, None
        order_type = mt5.ORDER_TYPE_BUY if direction.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
        try:
            margin_required = self.mt5.order_calc_margin(order_type, mt5_symbol, volume, entry_price)
            if margin_required is None: log.error(f"Margin Check ({mt5_symbol}): order_calc_margin failed. Error: {self.mt5.last_error()}"); return False, None, None
            margin_free = account_info.margin_free
            buffer_pct = margin_cfg.get("buffer_pct", 10.0) / 100.0
            required_free_margin_with_buffer = margin_required * (1.0 + buffer_pct)
            log.debug(f"Margin Check ({mt5_symbol}): Vol={volume:.4f}, Required={margin_required:.2f}, Req+Buf={required_free_margin_with_buffer:.2f}, Free={margin_free:.2f}")
            if margin_free >= required_free_margin_with_buffer: return True, margin_required, margin_free
            else: log.warning(f"Margin Check ({mt5_symbol}): Insufficient free margin."); return False, margin_required, margin_free
        except Exception as e: log.exception(f"Margin Check ({mt5_symbol}): Exception: {e}"); return False, None, None

    # --- Main Public Method ---
    def calculate_position_size(self, pair: str, direction: str, entry_price: float, sl_price: float,
                                account_info: mt5.AccountInfo, config: Dict,
                                signal_confidence: Optional[float] = None) -> Optional[float]:
        """
        Calculates the final position size considering all risk factors and checks.

        Args:
            pair, direction, entry_price, sl_price: Trade details.
            account_info: MT5 AccountInfo object.
            config: Main application configuration dictionary.
            signal_confidence: Optional signal confidence score (0-1).

        Returns:
            Optional[float]: Calculated volume, or None if invalid/blocked.
        """
        # Merge default and user config for this unit
        enhancer_cfg = self._merge_config(config)
        # Get Execution Manager config part for base risk and lot info
        exec_cfg = config.get("execution_manager", {})
        log_prefix = f"Position Size ({exec_cfg.get('symbol_mapping', {}).get(pair, pair)})"

        # --- 1. Initial Checks & MT5 Readiness ---
        if not self.is_mt5_ready(): log.error(f"{log_prefix} Error: MT5 not ready."); return None
        if not all([pair, direction, account_info, exec_cfg]) or entry_price <= 0 or sl_price <= 0:
            log.error(f"{log_prefix} Error: Missing/invalid basic arguments."); return None

        # --- 2. Get Symbol & Lot Info ---
        mt5_symbol = exec_cfg.get('symbol_mapping', {}).get(pair, pair)
        symbol_info = self.mt5.symbol_info(mt5_symbol)
        if not symbol_info: # Attempt select if missing
            if self.mt5.symbol_select(mt5_symbol, True): time.sleep(0.5); symbol_info = self.mt5.symbol_info(mt5_symbol)
            if not symbol_info: log.error(f"{log_prefix} Error: Failed to get symbol info for '{mt5_symbol}'."); return None
            else: log.info(f"{log_prefix}: Selected missing symbol '{mt5_symbol}'.")

        min_lot, max_lot_broker, lot_step = symbol_info.volume_min, symbol_info.volume_max, symbol_info.volume_step
        max_lot_config = exec_cfg.get('max_lot', 1.0)
        max_lot = min(max_lot_broker, max_lot_config)
        if min_lot <= 0 or max_lot <= 0 or lot_step < 0 or min_lot > max_lot:
             log.error(f"{log_prefix} Error: Invalid lot parameters (min={min_lot}, max={max_lot}, step={lot_step})."); return None

        # --- 3. Determine Effective Risk Percentage ---
        base_risk_pct_config = exec_cfg.get('risk_per_trade_pct', self.DEFAULT_RISK_PER_TRADE_PCT)
        if base_risk_pct_config <= 0: log.error(f"{log_prefix} Error: Base risk % must be positive."); return None
        base_risk_pct = base_risk_pct_config / 100.0

        # Adjust risk based on signal confidence (Dynamic Risk)
        risk_after_confidence = self._calculate_dynamic_risk_pct(base_risk_pct, signal_confidence, enhancer_cfg)

        # Adjust risk based on recent performance (Adaptive Risk)
        adaptive_factor = self._calculate_adaptive_risk_factor(base_risk_pct, enhancer_cfg.get("adaptive_risk", {}))
        effective_risk_pct = risk_after_confidence * adaptive_factor
        effective_risk_pct = max(0.0001, effective_risk_pct) # Ensure risk is still positive (minimum 0.01%)

        log.debug(f"{log_prefix}: Risk Factors: Base={base_risk_pct:.4%}, ConfidenceAdj={risk_after_confidence/base_risk_pct:.3f}, AdaptiveAdj={adaptive_factor:.3f} -> EffectiveRisk={effective_risk_pct:.4%}")

        # --- 4. Validate SL & Calculate Risk Amount ---
        # (SL validation logic remains the same)
        if abs(entry_price - sl_price) < symbol_info.point * 0.5: log.error(f"{log_prefix} Error: SL too close to entry."); return None
        direction_upper = direction.upper(); order_type = mt5.ORDER_TYPE_BUY if direction_upper == 'BUY' else mt5.ORDER_TYPE_SELL
        if (direction_upper == 'BUY' and sl_price >= entry_price): log.error(f"{log_prefix} Error: Invalid SL >= entry for BUY."); return None
        if (direction_upper == 'SELL' and sl_price <= entry_price): log.error(f"{log_prefix} Error: Invalid SL <= entry for SELL."); return None

        equity = account_info.equity; account_currency = account_info.currency
        if not isinstance(equity, (int, float)) or equity <= 0: log.error(f"{log_prefix} Error: Invalid equity ({equity})."); return None
        risk_amount_account_currency = equity * effective_risk_pct
        log.debug(f"{log_prefix}: Equity={equity:.2f}{account_currency}, RiskAmt={risk_amount_account_currency:.2f}{account_currency}")

        # --- 5. Calculate Loss per Lot ---
        # (Calculation logic remains the same)
        try:
            loss_per_lot = self.mt5.order_calc_profit(order_type, mt5_symbol, 1.0, float(entry_price), float(sl_price))
            if loss_per_lot is None: log.error(f"{log_prefix} Error: order_calc_profit failed. MT5 Error: {self.mt5.last_error()}"); return None
            if loss_per_lot >= -1e-9: log.error(f"{log_prefix} Error: SL calc suggests profit/no loss ({loss_per_lot:.4f})."); return None
            loss_per_lot_abs = abs(loss_per_lot)
        except Exception as e: log.exception(f"{log_prefix} Error: Exception during order_calc_profit: {e}"); return None
        if loss_per_lot_abs < 1e-9: log.error(f"{log_prefix} Error: Calculated loss per lot negligible ({loss_per_lot_abs:.4f})."); return None
        log.debug(f"{log_prefix}: Loss for 1.0 lot = {loss_per_lot_abs:.4f} {account_currency}")

        # --- 6. Calculate Volume & Apply Constraints ---
        # (Calculation and constraint logic remains the same)
        calculated_volume = risk_amount_account_currency / loss_per_lot_abs
        clamped_volume = max(min_lot, min(calculated_volume, max_lot))
        final_volume = clamped_volume
        if lot_step > 1e-9:
            final_volume = math.floor((clamped_volume + 1e-9) / lot_step) * lot_step
        final_volume = max(final_volume, min_lot)
        final_volume = round(final_volume, 8)
        if final_volume <= 1e-9: log.error(f"{log_prefix} Error: Final volume zero/negative ({final_volume:.8f})."); return None
        log.debug(f"{log_prefix}: Volume after constraints = {final_volume:.8f} lots")

        # --- 7. Optional Margin Check ---
        margin_ok, margin_req, margin_free = self._check_margin(
            mt5_symbol, direction, final_volume, entry_price, account_info, enhancer_cfg
        )
        if not margin_ok:
             log.error(f"{log_prefix} Error: Order blocked due to insufficient margin for volume {final_volume:.4f}.")
             # Potential fallback: Could try calculating the maximum volume possible with available margin? (More complex)
             return None

        # --- 8. Return Final Volume ---
        log.info(f"{log_prefix}: Final calculated volume = {final_volume:.4f} lots (EffRisk={effective_risk_pct:.3%})")
        return final_volume

    def update_adaptive_risk_on_trade_close(self, trade_profit: float, config: Optional[Dict] = None):
        """
        Call this method after a trade closes to update consecutive win/loss counters.
        Needs the profit of the closed trade.
        """
        cfg = self._merge_config(config)
        self._update_adaptive_risk_counters(trade_profit > 0, cfg.get("adaptive_risk", {}))


# --- END OF FILE profitability_enhancer.py ---