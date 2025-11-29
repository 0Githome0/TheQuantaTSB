"""
Advanced Entry Strategies for Forex Trading
Implements Multi-Timeframe Confirmation, Breakout, and Pullback strategies
Integrates with Signal.py for comprehensive trading signals
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import os
from typing import Dict, Optional, List, Tuple, Union, Any
from datetime import datetime
import pytz
from enum import Enum
import warnings
import json

# Import the SignalGenerator from Signal.py
from Signal import SignalGenerator, TimeFrames

# Configure logging
log = logging.getLogger('EntryStrategies')
log.setLevel(logging.INFO)
log.propagate = False

if not log.handlers:
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        fh = logging.FileHandler(os.path.join(logs_dir, 'entry_strategies.log'), mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    except Exception as e:
        print(f"Error setting up entry strategies file logger: {e}")
    
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    log.addHandler(sh)

class EntrySignalType(Enum):
    """Types of entry signals"""
    MULTI_TIMEFRAME_BUY = "MTF_BUY"
    MULTI_TIMEFRAME_SELL = "MTF_SELL"
    BREAKOUT_BUY = "BREAKOUT_BUY"
    BREAKOUT_SELL = "BREAKOUT_SELL"
    PULLBACK_BUY = "PULLBACK_BUY"
    PULLBACK_SELL = "PULLBACK_SELL"
    NO_SIGNAL = "NO_SIGNAL"

class EntryQuality(Enum):
    """Quality levels for entry signals"""
    EXCELLENT = "EXCELLENT"  # 90%+ confidence
    GOOD = "GOOD"           # 75-89% confidence
    FAIR = "FAIR"           # 60-74% confidence
    POOR = "POOR"           # Below 60% confidence

class AdvancedEntryStrategies:
    """
    Advanced entry strategies that work with SignalGenerator to provide
    high-quality entry points for forex trading.
    """
    
    def __init__(self, signal_generator: SignalGenerator = None, timezone: str = 'Asia/Riyadh'):
        """
        Initialize the Advanced Entry Strategies.
        
        Args:
            signal_generator: Instance of SignalGenerator from Signal.py
            timezone: Timezone for timestamps
        """
        log.info("Initializing Advanced Entry Strategies...")
        
        # Initialize SignalGenerator if not provided
        self.signal_generator = signal_generator or SignalGenerator(timezone=timezone)
        
        try:
            self.timezone = pytz.timezone(timezone)
        except pytz.UnknownTimeZoneError:
            log.warning(f"Unknown timezone '{timezone}'. Defaulting to UTC.")
            self.timezone = pytz.utc
        
        # Entry strategy parameters
        self.MTF_CONFIRMATION_THRESHOLD = 0.75  # Multi-timeframe confirmation threshold
        self.BREAKOUT_VOLUME_MULTIPLIER = 1.5   # Volume spike for breakout confirmation
        self.PULLBACK_FIBONACCI_LEVELS = [0.382, 0.5, 0.618]  # Key Fibonacci levels
        self.TREND_STRENGTH_THRESHOLD = 0.7     # Minimum trend strength for pullback
        
        # Risk management parameters
        self.MAX_RISK_PERCENT = 2.0             # Maximum risk per trade
        self.MIN_REWARD_RISK_RATIO = 2.0        # Minimum reward:risk ratio
        self.BREAKOUT_SL_MULTIPLIER = 1.2       # Stop loss multiplier for breakouts
        self.PULLBACK_SL_MULTIPLIER = 0.8       # Stop loss multiplier for pullbacks
        
        log.info("Advanced Entry Strategies initialized successfully.")
    
    @classmethod
    def from_config(cls, config_path: str = 'entry_config.json'):
        """Create instance from configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            instance = cls(timezone=config.get('timezone', 'Asia/Riyadh'))
            
            # Load parameters from config
            instance.MTF_CONFIRMATION_THRESHOLD = config.get('mtf_threshold', 0.75)
            instance.BREAKOUT_VOLUME_MULTIPLIER = config.get('breakout_volume_mult', 1.5)
            instance.PULLBACK_FIBONACCI_LEVELS = config.get('fib_levels', [0.382, 0.5, 0.618])
            instance.TREND_STRENGTH_THRESHOLD = config.get('trend_strength_threshold', 0.7)
            instance.MAX_RISK_PERCENT = config.get('max_risk_percent', 2.0)
            instance.MIN_REWARD_RISK_RATIO = config.get('min_rr_ratio', 2.0)
            
            return instance
            
        except Exception as e:
            log.error(f"Error loading config: {str(e)}")
            return cls()  # Return default instance
    
    def analyze_entry_opportunity(self, 
                                pair: str, 
                                timeframes_data: Dict[TimeFrames, pd.DataFrame],
                                primary_timeframe: TimeFrames = TimeFrames.H1) -> Dict:
        """
        Comprehensive entry analysis using all three strategies.
        
        Args:
            pair: Currency pair symbol
            timeframes_data: Dictionary with timeframe data {TimeFrames.H1: df, TimeFrames.H4: df, etc.}
            primary_timeframe: Primary timeframe for entry signals
            
        Returns:
            Dict: Complete entry analysis with recommendations
        """
        try:
            log.info(f"Analyzing entry opportunity for {pair} on {primary_timeframe.name}")
            
            # Validate input data
            if primary_timeframe not in timeframes_data:
                log.error(f"Primary timeframe {primary_timeframe.name} not found in data")
                return self._create_empty_analysis(pair, primary_timeframe)
            
            primary_df = timeframes_data[primary_timeframe]
            if primary_df is None or primary_df.empty:
                log.error(f"No data available for {pair} on {primary_timeframe.name}")
                return self._create_empty_analysis(pair, primary_timeframe)
            
            # Get base signal from SignalGenerator
            base_signal = self.signal_generator.generate_enhanced_signal(
                primary_df, pair, primary_timeframe
            )
            
            # Analyze each strategy
            mtf_analysis = self._analyze_multi_timeframe_confirmation(
                pair, timeframes_data, primary_timeframe
            )
            
            breakout_analysis = self._analyze_breakout_opportunity(
                pair, primary_df, primary_timeframe
            )
            
            pullback_analysis = self._analyze_pullback_opportunity(
                pair, primary_df, primary_timeframe
            )
            
            # Combine analyses and determine best entry
            combined_analysis = self._combine_entry_analyses(
                base_signal, mtf_analysis, breakout_analysis, pullback_analysis
            )
            
            # Add comprehensive entry recommendation
            entry_recommendation = self._generate_entry_recommendation(
                combined_analysis, pair, primary_timeframe
            )
            
            # Add best strategy action and details
            best_strategy = combined_analysis.get('best_strategy')
            best_strategy_action = None
            best_strategy_details = None
            if best_strategy:
                if best_strategy == 'multi_timeframe':
                    best_strategy_action = mtf_analysis.get('signal_type').name if mtf_analysis.get('signal_type') else None
                    best_strategy_details = mtf_analysis.get('details')
                elif best_strategy == 'breakout':
                    best_strategy_action = breakout_analysis.get('signal_type').name if breakout_analysis.get('signal_type') else None
                    best_strategy_details = breakout_analysis.get('details')
                elif best_strategy == 'pullback':
                    best_strategy_action = pullback_analysis.get('signal_type').name if pullback_analysis.get('signal_type') else None
                    best_strategy_details = pullback_analysis.get('details')

            result = {
                'pair': pair,
                'timeframe': primary_timeframe.name,
                'timestamp': datetime.now(self.timezone),
                'base_signal': base_signal,
                'multi_timeframe': mtf_analysis,
                'breakout': breakout_analysis,
                'pullback': pullback_analysis,
                'combined_analysis': combined_analysis,
                'entry_recommendation': entry_recommendation,
                'action': best_strategy_action,  # e.g., 'Breakout Buy', 'Pullback Sell', etc.
                'action_details': best_strategy_details  # e.g., confidence factors, technical context
            }
            
            log.info(f"Entry analysis completed for {pair}. Best strategy: {entry_recommendation.get('best_strategy', 'None')}")
            return result
            
        except Exception as e:
            log.error(f"Error analyzing entry opportunity for {pair}: {str(e)}", exc_info=True)
            return self._create_empty_analysis(pair, primary_timeframe)
    
    # Dictionary of disabled timeframes for specific pairs
    DISABLED_TIMEFRAMES = {
        "EURUSDm": [TimeFrames.D1, TimeFrames.H4],
        "XAUUSDm": [TimeFrames.D1, TimeFrames.H4],
        "BTCUSDm": [TimeFrames.D1, TimeFrames.H4],
        "GBPUSDm": [TimeFrames.D1, TimeFrames.H4],
        "USDJPYm": [TimeFrames.D1, TimeFrames.H4]
    }
    
    def _analyze_multi_timeframe_confirmation(self, 
                                            pair: str, 
                                            timeframes_data: Dict[TimeFrames, pd.DataFrame],
                                            primary_timeframe: TimeFrames) -> Dict:
        """
        Multi-Timeframe Confirmation Strategy Analysis.
        Confirms signals across multiple timeframes for higher probability entries.
        """
        try:
            log.debug(f"Analyzing multi-timeframe confirmation for {pair}")
            
            # Define timeframe hierarchy (higher to lower)
            timeframe_hierarchy = [TimeFrames.D1, TimeFrames.H4, TimeFrames.H1, TimeFrames.M15]
            if hasattr(TimeFrames, 'M5'):  # Check if M5 exists in the enum
                timeframe_hierarchy.append(TimeFrames.M5)
            
            # Apply pair-specific timeframe exclusions
            if pair in self.DISABLED_TIMEFRAMES:
                disabled = self.DISABLED_TIMEFRAMES[pair]
                log.info(f"Disabling {[tf.name for tf in disabled]} timeframes for {pair}")
                timeframe_hierarchy = [tf for tf in timeframe_hierarchy if tf not in disabled]
                
            available_timeframes = []
            for tf in timeframe_hierarchy:
                if tf in timeframes_data:
                    df = timeframes_data[tf]
                    if df is not None and len(df) >= 50:  # Minimum data requirement
                        available_timeframes.append(tf)
                    else:
                        log.warning(f"Insufficient data for {pair} on {tf.name}, excluding from analysis")
            
            if len(available_timeframes) < 2:
                return {
                    'signal_type': EntrySignalType.NO_SIGNAL,
                    'confidence': 0.0,
                    'quality': EntryQuality.POOR,
                    'details': {'error': 'Insufficient timeframes for MTF analysis'}
                }
            
            # Analyze each timeframe
            timeframe_signals = {}
            trend_alignment = 0
            signal_strength = 0
            
            for tf in available_timeframes:
                df = timeframes_data[tf]
                if df is None or df.empty:
                    continue
                
                # Get signal for this timeframe
                signal = self.signal_generator.generate_enhanced_signal(df, pair, tf)
                timeframe_signals[tf.name] = signal
                
                if signal and signal.get('direction') in ['BUY', 'SELL']:
                    # Weight higher timeframes more heavily
                    weight = self._get_timeframe_weight(tf, primary_timeframe)
                    
                    if signal['direction'] == 'BUY':
                        trend_alignment += weight
                        signal_strength += signal.get('confidence', 0) * weight
                    elif signal['direction'] == 'SELL':
                        trend_alignment -= weight
                        signal_strength += signal.get('confidence', 0) * weight
            
            # Normalize signal strength
            total_weight = sum(self._get_timeframe_weight(tf, primary_timeframe) for tf in available_timeframes)
            if total_weight > 0:
                signal_strength /= total_weight
            
            # Determine MTF signal
            mtf_signal_type = EntrySignalType.NO_SIGNAL
            confidence = abs(trend_alignment) / total_weight if total_weight > 0 else 0
            
            if trend_alignment > 0 and confidence >= self.MTF_CONFIRMATION_THRESHOLD:
                mtf_signal_type = EntrySignalType.MULTI_TIMEFRAME_BUY
            elif trend_alignment < 0 and confidence >= self.MTF_CONFIRMATION_THRESHOLD:
                mtf_signal_type = EntrySignalType.MULTI_TIMEFRAME_SELL
            
            # Determine quality
            quality = self._determine_signal_quality(confidence)
            
            return {
                'signal_type': mtf_signal_type,
                'confidence': round(confidence, 3),
                'quality': quality,
                'signal_strength': round(signal_strength, 3),
                'trend_alignment': round(trend_alignment, 3),
                'timeframe_signals': timeframe_signals,
                'details': {
                    'available_timeframes': [tf.name for tf in available_timeframes],
                    'total_weight': total_weight
                }
            }
            
        except Exception as e:
            log.error(f"Error in multi-timeframe analysis for {pair}: {str(e)}", exc_info=True)
            return {
                'signal_type': EntrySignalType.NO_SIGNAL,
                'confidence': 0.0,
                'quality': EntryQuality.POOR,
                'details': {'error': str(e)}
            }
    
    def _analyze_breakout_opportunity(self, 
                                    pair: str, 
                                    df: pd.DataFrame, 
                                    timeframe: TimeFrames) -> Dict:
        """
        Breakout Strategy Analysis.
        Identifies high-probability breakout opportunities with volume confirmation.
        """
        try:
            log.debug(f"Analyzing breakout opportunity for {pair}")
            
            if df is None or len(df) < 50:
                return {
                    'signal_type': EntrySignalType.NO_SIGNAL,
                    'confidence': 0.0,
                    'quality': EntryQuality.POOR,
                    'details': {'error': 'Insufficient data for breakout analysis'}
                }
            
            # Calculate key levels and indicators
            df = df.copy()
            
            # Support and Resistance levels
            resistance_level = df['high'].rolling(window=20).max().iloc[-1]
            support_level = df['low'].rolling(window=20).min().iloc[-1]
            
            # Bollinger Bands for volatility context
            bb = ta.bbands(df['close'], length=20, std=2)
            if bb is not None and not bb.empty:
                df['bb_upper'] = bb[f'BBU_20_2.0']
                df['bb_lower'] = bb[f'BBL_20_2.0']
                df['bb_middle'] = bb[f'BBM_20_2.0']
            
            # Volume analysis
            df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            
            # ATR for volatility
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Current price and volume
            current_price = df['close'].iloc[-1]
            current_volume_ratio = df['volume_ratio'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            # Breakout detection
            breakout_signal = EntrySignalType.NO_SIGNAL
            confidence = 0.0
            breakout_details = {}
            
            # Check for resistance breakout (BUY signal)
            if current_price > resistance_level:
                price_above_resistance = (current_price - resistance_level) / current_atr
                volume_confirmation = current_volume_ratio >= self.BREAKOUT_VOLUME_MULTIPLIER
                
                # Additional confirmations
                bb_breakout = current_price > df['bb_upper'].iloc[-1] if 'bb_upper' in df.columns else False
                momentum_confirmation = df['close'].iloc[-1] > df['close'].iloc[-3]  # 3-bar momentum
                
                confidence_factors = {
                    'price_breakout': min(price_above_resistance / 0.5, 1.0),  # Normalize to 0.5 ATR
                    'volume_confirmation': 1.0 if volume_confirmation else 0.3,
                    'bb_breakout': 0.8 if bb_breakout else 0.4,
                    'momentum': 0.7 if momentum_confirmation else 0.2
                }
                
                confidence = sum(confidence_factors.values()) / len(confidence_factors)
                
                if confidence >= 0.6:  # Minimum threshold for breakout signal
                    breakout_signal = EntrySignalType.BREAKOUT_BUY
                
                breakout_details = {
                    'breakout_type': 'resistance',
                    'resistance_level': resistance_level,
                    'price_above_resistance': price_above_resistance,
                    'volume_ratio': current_volume_ratio,
                    'confidence_factors': confidence_factors
                }
            
            # Check for support breakout (SELL signal)
            elif current_price < support_level:
                price_below_support = (support_level - current_price) / current_atr
                volume_confirmation = current_volume_ratio >= self.BREAKOUT_VOLUME_MULTIPLIER
                
                # Additional confirmations
                bb_breakout = current_price < df['bb_lower'].iloc[-1] if 'bb_lower' in df.columns else False
                momentum_confirmation = df['close'].iloc[-1] < df['close'].iloc[-3]  # 3-bar momentum
                
                confidence_factors = {
                    'price_breakout': min(price_below_support / 0.5, 1.0),  # Normalize to 0.5 ATR
                    'volume_confirmation': 1.0 if volume_confirmation else 0.3,
                    'bb_breakout': 0.8 if bb_breakout else 0.4,
                    'momentum': 0.7 if momentum_confirmation else 0.2
                }
                
                confidence = sum(confidence_factors.values()) / len(confidence_factors)
                
                if confidence >= 0.6:  # Minimum threshold for breakout signal
                    breakout_signal = EntrySignalType.BREAKOUT_SELL
                
                breakout_details = {
                    'breakout_type': 'support',
                    'support_level': support_level,
                    'price_below_support': price_below_support,
                    'volume_ratio': current_volume_ratio,
                    'confidence_factors': confidence_factors
                }
            
            # Calculate stop loss and take profit for breakout
            sl_tp = self._calculate_breakout_sl_tp(
                df, breakout_signal, current_price, current_atr
            )
            
            quality = self._determine_signal_quality(confidence)
            
            return {
                'signal_type': breakout_signal,
                'confidence': round(confidence, 3),
                'quality': quality,
                'stop_loss': sl_tp.get('stop_loss'),
                'take_profit': sl_tp.get('take_profit'),
                'details': {
                    'resistance_level': resistance_level,
                    'support_level': support_level,
                    'current_price': current_price,
                    'current_atr': current_atr,
                    **breakout_details
                }
            }
            
        except Exception as e:
            log.error(f"Error in breakout analysis for {pair}: {str(e)}", exc_info=True)
            return {
                'signal_type': EntrySignalType.NO_SIGNAL,
                'confidence': 0.0,
                'quality': EntryQuality.POOR,
                'details': {'error': str(e)}
            }
    
    def _analyze_pullback_opportunity(self, 
                                    pair: str, 
                                    df: pd.DataFrame, 
                                    timeframe: TimeFrames) -> Dict:
        """
        Pullback Strategy Analysis (Trend Continuation).
        Identifies optimal pullback entries in established trends.
        """
        try:
            log.debug(f"Analyzing pullback opportunity for {pair}")
            
            if df is None or len(df) < 50:
                return {
                    'signal_type': EntrySignalType.NO_SIGNAL,
                    'confidence': 0.0,
                    'quality': EntryQuality.POOR,
                    'details': {'error': 'Insufficient data for pullback analysis'}
                }
            
            df = df.copy()
            
            # Trend identification
            trend_analysis = self._identify_trend_strength(df)
            
            if trend_analysis['strength'] < self.TREND_STRENGTH_THRESHOLD:
                return {
                    'signal_type': EntrySignalType.NO_SIGNAL,
                    'confidence': 0.0,
                    'quality': EntryQuality.POOR,
                    'details': {'error': 'Trend not strong enough for pullback strategy'}
                }
            
            # Calculate Fibonacci retracement levels
            fib_levels = self._calculate_fibonacci_levels(df, trend_analysis['direction'])
            
            # Current price analysis
            current_price = df['close'].iloc[-1]
            
            # RSI for momentum
            df['rsi'] = ta.rsi(df['close'], length=14)
            current_rsi = df['rsi'].iloc[-1]
            
            # MACD for trend confirmation
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                df['macd_histogram'] = macd['MACDh_12_26_9']
            
            # EMA for trend direction
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            
            pullback_signal = EntrySignalType.NO_SIGNAL
            confidence = 0.0
            pullback_details = {}
            
            # Analyze pullback opportunity based on trend direction
            if trend_analysis['direction'] == 'uptrend':
                # Look for pullback buying opportunity
                pullback_analysis = self._analyze_uptrend_pullback(
                    df, current_price, fib_levels, current_rsi
                )
                
                if pullback_analysis['is_valid']:
                    pullback_signal = EntrySignalType.PULLBACK_BUY
                    confidence = pullback_analysis['confidence']
                    pullback_details = pullback_analysis['details']
            
            elif trend_analysis['direction'] == 'downtrend':
                # Look for pullback selling opportunity
                pullback_analysis = self._analyze_downtrend_pullback(
                    df, current_price, fib_levels, current_rsi
                )
                
                if pullback_analysis['is_valid']:
                    pullback_signal = EntrySignalType.PULLBACK_SELL
                    confidence = pullback_analysis['confidence']
                    pullback_details = pullback_analysis['details']
            
            # Calculate stop loss and take profit for pullback
            sl_tp = self._calculate_pullback_sl_tp(
                df, pullback_signal, current_price, trend_analysis
            )
            
            quality = self._determine_signal_quality(confidence)
            
            return {
                'signal_type': pullback_signal,
                'confidence': round(confidence, 3),
                'quality': quality,
                'stop_loss': sl_tp.get('stop_loss'),
                'take_profit': sl_tp.get('take_profit'),
                'details': {
                    'trend_analysis': trend_analysis,
                    'fibonacci_levels': fib_levels,
                    'current_price': current_price,
                    'current_rsi': current_rsi,
                    **pullback_details
                }
            }
            
        except Exception as e:
            log.error(f"Error in pullback analysis for {pair}: {str(e)}", exc_info=True)
            return {
                'signal_type': EntrySignalType.NO_SIGNAL,
                'confidence': 0.0,
                'quality': EntryQuality.POOR,
                'details': {'error': str(e)}
            }
    
    def _identify_trend_strength(self, df: pd.DataFrame) -> Dict:
        """
        Identify trend direction and strength using multiple indicators.
        """
        try:
            # EMA trend
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            
            # ADX for trend strength
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx is not None and not adx.empty:
                df['adx'] = adx['ADX_14']
                df['di_plus'] = adx['DMP_14']
                df['di_minus'] = adx['DMN_14']
            
            # Current values
            current_price = df['close'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1] if not pd.isna(df['ema_21'].iloc[-1]) else current_price
            ema_50 = df['ema_50'].iloc[-1] if not pd.isna(df['ema_50'].iloc[-1]) else current_price
            ema_200 = df['ema_200'].iloc[-1] if not pd.isna(df['ema_200'].iloc[-1]) else current_price
            current_adx = df['adx'].iloc[-1] if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]) else 25
            
            # Trend direction scoring
            trend_score = 0
            
            # Price vs EMAs - safely compare values
            if current_price > ema_21: trend_score += 1
            if current_price > ema_50: trend_score += 1
            if current_price > ema_200: trend_score += 1
            if ema_21 > ema_50: trend_score += 1
            if ema_50 > ema_200: trend_score += 1
            
            # Determine direction
            if trend_score >= 4:
                direction = 'uptrend'
            elif trend_score <= 1:
                direction = 'downtrend'
            else:
                direction = 'sideways'
            
            # Trend strength (0-1 scale)
            strength = min(current_adx / 50.0, 1.0)  # Normalize ADX to 0-1
            
            return {
                'direction': direction,
                'strength': strength,
                'trend_score': trend_score,
                'adx': current_adx
            }
            
        except Exception as e:
            log.error(f"Error identifying trend strength: {str(e)}")
            return {
                'direction': 'sideways',
                'strength': 0.0,
                'trend_score': 0,
                'adx': 0
            }
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame, trend_direction: str) -> Dict:
        """
        Calculate Fibonacci retracement levels based on recent swing high/low.
        """
        try:
            if trend_direction == 'uptrend':
                # Find recent swing low and current high
                swing_low = df['low'].rolling(window=20).min().iloc[-20:].min()
                swing_high = df['high'].iloc[-10:].max()  # Recent high
            else:
                # Find recent swing high and current low
                swing_high = df['high'].rolling(window=20).max().iloc[-20:].max()
                swing_low = df['low'].iloc[-10:].min()  # Recent low
            
            # Calculate Fibonacci levels
            price_range = abs(swing_high - swing_low)
            
            if trend_direction == 'uptrend':
                fib_levels = {
                    'swing_low': swing_low,
                    'swing_high': swing_high,
                    'fib_23.6': swing_high - (price_range * 0.236),
                    'fib_38.2': swing_high - (price_range * 0.382),
                    'fib_50.0': swing_high - (price_range * 0.5),
                    'fib_61.8': swing_high - (price_range * 0.618),
                    'fib_78.6': swing_high - (price_range * 0.786)
                }
            else:
                fib_levels = {
                    'swing_high': swing_high,
                    'swing_low': swing_low,
                    'fib_23.6': swing_low + (price_range * 0.236),
                    'fib_38.2': swing_low + (price_range * 0.382),
                    'fib_50.0': swing_low + (price_range * 0.5),
                    'fib_61.8': swing_low + (price_range * 0.618),
                    'fib_78.6': swing_low + (price_range * 0.786)
                }
            
            return fib_levels
            
        except Exception as e:
            log.error(f"Error calculating Fibonacci levels: {str(e)}")
            return {}
    
    def _analyze_uptrend_pullback(self, df: pd.DataFrame, current_price: float, 
                                fib_levels: Dict, current_rsi: float) -> Dict:
        """
        Analyze pullback opportunity in uptrend.
        """
        try:
            is_valid = False
            confidence = 0.0
            details = {}
            
            # Check if price is near key Fibonacci levels
            fib_proximity_score = 0
            closest_fib = None
            min_distance = float('inf')
            
            for level_name in ['fib_38.2', 'fib_50.0', 'fib_61.8']:
                if level_name in fib_levels:
                    level_price = fib_levels[level_name]
                    distance = abs(current_price - level_price) / current_price
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_fib = level_name
                    
                    # Score proximity (closer = higher score)
                    if distance < 0.005:  # Within 0.5%
                        fib_proximity_score = 1.0
                    elif distance < 0.01:  # Within 1%
                        fib_proximity_score = 0.8
                    elif distance < 0.02:  # Within 2%
                        fib_proximity_score = 0.5
            
            # RSI oversold condition (for uptrend pullback)
            rsi_score = 0
            if current_rsi < 40:
                rsi_score = (40 - current_rsi) / 40  # Higher score for lower RSI
            
            # EMA support
            ema_support_score = 0
            if 'ema_21' in df.columns:
                ema_21 = df['ema_21'].iloc[-1]
                if current_price > ema_21 * 0.995:  # Within 0.5% of EMA21
                    ema_support_score = 0.8
                elif current_price > df['ema_50'].iloc[-1] if 'ema_50' in df.columns else 0:
                    ema_support_score = 0.5
            
            # MACD bullish divergence
            macd_score = 0
            if 'macd_histogram' in df.columns:
                recent_macd = df['macd_histogram'].iloc[-3:]
                if (recent_macd.iloc[-1] > recent_macd.iloc[-2] and 
                    recent_macd.iloc[-2] > recent_macd.iloc[-3]):
                    macd_score = 0.7
            
            # Volume confirmation (lower volume during pullback)
            volume_score = 0
            if 'volume_ratio' in df.columns:
                current_volume = df['volume_ratio'].iloc[-1]
                if current_volume < 1.0:  # Below average volume
                    volume_score = 0.6
            
            # Calculate overall confidence
            confidence_factors = {
                'fibonacci_proximity': fib_proximity_score,
                'rsi_oversold': rsi_score,
                'ema_support': ema_support_score,
                'macd_bullish': macd_score,
                'volume_confirmation': volume_score
            }
            
            confidence = sum(confidence_factors.values()) / len(confidence_factors)
            
            # Validate pullback
            if confidence >= 0.6 and fib_proximity_score >= 0.5:
                is_valid = True
            
            details = {
                'closest_fibonacci': closest_fib,
                'fibonacci_distance': min_distance,
                'confidence_factors': confidence_factors,
                'ema_21_support': df['ema_21'].iloc[-1] if 'ema_21' in df.columns else None
            }
            
            return {
                'is_valid': is_valid,
                'confidence': confidence,
                'details': details
            }
            
        except Exception as e:
            log.error(f"Error in uptrend pullback analysis: {str(e)}")
            return {
                'is_valid': False,
                'confidence': 0.0,
                'details': {'error': str(e)}
            }
    
    def _safe_compare(self, value1, value2, default=False):
        """
        Safely compare two values that might be None.
        """
        if value1 is None or value2 is None:
            return default
        return value1 > value2

    def _analyze_downtrend_pullback(self, df: pd.DataFrame, current_price: float, 
                                  fib_levels: Dict, current_rsi: float) -> Dict:
        """
        Analyze pullback opportunity in downtrend.
        """
        try:
            is_valid = False
            confidence = 0.0
            details = {}
            
            # Check if price is near key Fibonacci levels
            fib_proximity_score = 0
            closest_fib = None
            min_distance = float('inf')
            
            for level_name in ['fib_38.2', 'fib_50.0', 'fib_61.8']:
                if level_name in fib_levels:
                    level_price = fib_levels[level_name]
                    distance = abs(current_price - level_price) / current_price
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_fib = level_name
                    
                    # Score proximity (closer = higher score)
                    if distance < 0.005:  # Within 0.5%
                        fib_proximity_score = 1.0
                    elif distance < 0.01:  # Within 1%
                        fib_proximity_score = 0.8
                    elif distance < 0.02:  # Within 2%
                        fib_proximity_score = 0.5
            
            # RSI overbought condition (for downtrend pullback)
            rsi_score = 0
            if current_rsi > 60:
                rsi_score = (current_rsi - 60) / 40  # Higher score for higher RSI
            
            # EMA resistance
            ema_resistance_score = 0
            if 'ema_21' in df.columns:
                ema_21 = df['ema_21'].iloc[-1]
                if current_price < ema_21 * 1.005:  # Within 0.5% of EMA21
                    ema_resistance_score = 0.8
                elif 'ema_50' in df.columns and self._safe_compare(df['ema_50'].iloc[-1], current_price, True):
                    ema_resistance_score = 0.5
            
            # MACD bearish divergence
            macd_score = 0
            if 'macd_histogram' in df.columns:
                recent_macd = df['macd_histogram'].iloc[-3:]
                if (recent_macd.iloc[-1] < recent_macd.iloc[-2] and 
                    recent_macd.iloc[-2] < recent_macd.iloc[-3]):
                    macd_score = 0.7
            
            # Volume confirmation (lower volume during pullback)
            volume_score = 0
            if 'volume_ratio' in df.columns:
                current_volume = df['volume_ratio'].iloc[-1]
                if current_volume < 1.0:  # Below average volume
                    volume_score = 0.6
            
            # Calculate overall confidence
            confidence_factors = {
                'fibonacci_proximity': fib_proximity_score,
                'rsi_overbought': rsi_score,
                'ema_resistance': ema_resistance_score,
                'macd_bearish': macd_score,
                'volume_confirmation': volume_score
            }
            
            confidence = sum(confidence_factors.values()) / len(confidence_factors)
            
            # Validate pullback
            if confidence >= 0.6 and fib_proximity_score >= 0.5:
                is_valid = True
            
            details = {
                'closest_fibonacci': closest_fib,
                'fibonacci_distance': min_distance,
                'confidence_factors': confidence_factors,
                'ema_21_resistance': df['ema_21'].iloc[-1] if 'ema_21' in df.columns else None
            }
            
            return {
                'is_valid': is_valid,
                'confidence': confidence,
                'details': details
            }
            
        except Exception as e:
            log.error(f"Error in downtrend pullback analysis: {str(e)}")
            return {
                'is_valid': False,
                'confidence': 0.0,
                'details': {'error': str(e)}
            }
    
    def _calculate_breakout_sl_tp(self, df: pd.DataFrame, signal_type: EntrySignalType, 
                                current_price: float, current_atr: float) -> Dict:
        """
        Calculate stop loss and take profit for breakout strategies.
        """
        try:
            if signal_type == EntrySignalType.NO_SIGNAL:
                return {'stop_loss': None, 'take_profit': None}
            
            # ATR-based calculations
            sl_distance = current_atr * self.BREAKOUT_SL_MULTIPLIER
            tp_distance = sl_distance * self.MIN_REWARD_RISK_RATIO
            
            if signal_type == EntrySignalType.BREAKOUT_BUY:
                stop_loss = current_price - sl_distance
                take_profit = current_price + tp_distance
                
                # Additional support level for SL
                recent_support = df['low'].rolling(window=10).min().iloc[-1]
                if recent_support < stop_loss:
                    stop_loss = recent_support - (current_atr * 0.2)  # Buffer below support
            
            elif signal_type == EntrySignalType.BREAKOUT_SELL:
                stop_loss = current_price + sl_distance
                take_profit = current_price - tp_distance
                
                # Additional resistance level for SL
                recent_resistance = df['high'].rolling(window=10).max().iloc[-1]
                if recent_resistance > stop_loss:
                    stop_loss = recent_resistance + (current_atr * 0.2)  # Buffer above resistance
            
            return {
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'risk_reward_ratio': self.MIN_REWARD_RISK_RATIO,
                'atr_used': current_atr
            }
            
        except Exception as e:
            log.error(f"Error calculating breakout SL/TP: {str(e)}")
            return {'stop_loss': None, 'take_profit': None}
    
    def _calculate_pullback_sl_tp(self, df: pd.DataFrame, signal_type: EntrySignalType, 
                                current_price: float, trend_analysis: Dict) -> Dict:
        """
        Calculate stop loss and take profit for pullback strategies.
        """
        try:
            if signal_type == EntrySignalType.NO_SIGNAL:
                return {'stop_loss': None, 'take_profit': None}
            
            # ATR for volatility adjustment
            current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            
            # Pullback SL is typically tighter
            sl_distance = current_atr * self.PULLBACK_SL_MULTIPLIER
            tp_distance = sl_distance * self.MIN_REWARD_RISK_RATIO
            
            if signal_type == EntrySignalType.PULLBACK_BUY:
                # SL below recent swing low or EMA support
                swing_low = df['low'].rolling(window=5).min().iloc[-1]
                ema_support = df['ema_21'].iloc[-1] if 'ema_21' in df.columns else current_price
                
                stop_loss = min(swing_low, ema_support) - (current_atr * 0.3)
                take_profit = current_price + tp_distance
                
                # Adjust TP to next resistance if closer
                resistance = df['high'].rolling(window=20).max().iloc[-1]
                if resistance < take_profit and (resistance - current_price) > sl_distance:
                    take_profit = resistance * 0.99  # Just below resistance
            
            elif signal_type == EntrySignalType.PULLBACK_SELL:
                # SL above recent swing high or EMA resistance
                swing_high = df['high'].rolling(window=5).max().iloc[-1]
                ema_resistance = df['ema_21'].iloc[-1] if 'ema_21' in df.columns else current_price
                
                stop_loss = max(swing_high, ema_resistance) + (current_atr * 0.3)
                take_profit = current_price - tp_distance
                
                # Adjust TP to next support if closer
                support = df['low'].rolling(window=20).min().iloc[-1]
                if support > take_profit and (current_price - support) > sl_distance:
                    take_profit = support * 1.01  # Just above support
            
            return {
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'risk_reward_ratio': abs((take_profit - current_price) / (stop_loss - current_price)),
                'atr_used': current_atr
            }
            
        except Exception as e:
            log.error(f"Error calculating pullback SL/TP: {str(e)}")
            return {'stop_loss': None, 'take_profit': None}
    
    def _combine_entry_analyses(self, base_signal: Dict, mtf_analysis: Dict, 
                              breakout_analysis: Dict, pullback_analysis: Dict) -> Dict:
        """
        Combine all entry analyses to determine the best overall strategy.
        """
        try:
            strategies = {
                'multi_timeframe': mtf_analysis,
                'breakout': breakout_analysis,
                'pullback': pullback_analysis
            }
            
            # Filter out strategies with no signal
            valid_strategies = {
                name: analysis for name, analysis in strategies.items()
                if analysis.get('signal_type') != EntrySignalType.NO_SIGNAL
            }
            
            if not valid_strategies:
                return {
                    'best_strategy': None,
                    'combined_confidence': 0.0,
                    'combined_quality': EntryQuality.POOR,
                    'signal_alignment': 0,
                    'strategies_count': 0
                }
            
            # Calculate combined confidence and signal alignment
            buy_signals = []
            sell_signals = []
            total_confidence = 0
            
            for name, analysis in valid_strategies.items():
                confidence = analysis.get('confidence', 0)
                signal_type = analysis.get('signal_type')
                
                if signal_type in [EntrySignalType.MULTI_TIMEFRAME_BUY, 
                                 EntrySignalType.BREAKOUT_BUY, 
                                 EntrySignalType.PULLBACK_BUY]:
                    buy_signals.append((name, confidence))
                elif signal_type in [EntrySignalType.MULTI_TIMEFRAME_SELL, 
                                   EntrySignalType.BREAKOUT_SELL, 
                                   EntrySignalType.PULLBACK_SELL]:
                    sell_signals.append((name, confidence))
                
                total_confidence += confidence
            
            # Determine signal alignment
            signal_alignment = 0
            if buy_signals and not sell_signals:
                signal_alignment = len(buy_signals)
            elif sell_signals and not buy_signals:
                signal_alignment = -len(sell_signals)
            
            # Combined confidence (weighted average)
            combined_confidence = total_confidence / len(valid_strategies) if valid_strategies else 0
            
            # Boost confidence if signals align
            if abs(signal_alignment) > 1:
                alignment_boost = min(abs(signal_alignment) * 0.1, 0.2)
                combined_confidence = min(combined_confidence + alignment_boost, 1.0)
            
            # Determine best strategy (highest confidence with alignment preference)
            best_strategy = None
            best_confidence = 0
            
            for name, analysis in valid_strategies.items():
                confidence = analysis.get('confidence', 0)
                
                # Boost confidence if it aligns with other signals
                if ((buy_signals and any(s[0] == name for s in buy_signals)) or
                    (sell_signals and any(s[0] == name for s in sell_signals))):
                    if abs(signal_alignment) > 1:
                        confidence += 0.1  # Alignment bonus
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_strategy = name
            
            # Combined quality
            combined_quality = self._determine_signal_quality(combined_confidence)
            
            return {
                'best_strategy': best_strategy,
                'combined_confidence': round(combined_confidence, 3),
                'combined_quality': combined_quality,
                'signal_alignment': signal_alignment,
                'strategies_count': len(valid_strategies),
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'valid_strategies': list(valid_strategies.keys())
            }
            
        except Exception as e:
            log.error(f"Error combining entry analyses: {str(e)}")
            return {
                'best_strategy': None,
                'combined_confidence': 0.0,
                'combined_quality': EntryQuality.POOR,
                'signal_alignment': 0,
                'strategies_count': 0
            }
    
    def _generate_entry_recommendation(self, combined_analysis: Dict, 
                                     pair: str, timeframe: TimeFrames) -> Dict:
        """
        Generate comprehensive entry recommendation based on combined analysis.
        """
        try:
            best_strategy = combined_analysis.get('best_strategy')
            combined_confidence = combined_analysis.get('combined_confidence', 0)
            signal_alignment = combined_analysis.get('signal_alignment', 0)
            
            if not best_strategy or combined_confidence < 0.6:
                return {
                    'action': 'WAIT',
                    'reason': 'Insufficient signal strength or conflicting signals',
                    'confidence': combined_confidence,
                    'risk_level': 'HIGH'
                }
            
            # Determine direction
            direction = 'BUY' if signal_alignment > 0 else 'SELL'
            
            # Risk assessment
            risk_level = self._assess_risk_level(combined_confidence, abs(signal_alignment))
            
            # Position sizing recommendation
            position_size = self._calculate_position_size(combined_confidence, risk_level)
            
            # Entry timing
            entry_timing = self._determine_entry_timing(combined_confidence, best_strategy)
            
            # Market conditions assessment
            market_conditions = self._assess_market_conditions(combined_analysis)
            
            recommendation = {
                'action': direction,
                'best_strategy': best_strategy,
                'confidence': combined_confidence,
                'quality': combined_analysis.get('combined_quality', EntryQuality.FAIR),
                'risk_level': risk_level,
                'position_size_percent': position_size,
                'entry_timing': entry_timing,
                'market_conditions': market_conditions,
                'signal_alignment': signal_alignment,
                'strategies_agreeing': abs(signal_alignment),
                'reason': f"{best_strategy.replace('_', ' ').title()} strategy with {combined_confidence:.1%} confidence"
            }
            
            # Add warnings if needed
            warnings = []
            if combined_confidence < 0.75:
                warnings.append("Moderate confidence - consider smaller position size")
            if abs(signal_alignment) == 1:
                warnings.append("Only one strategy signaling - watch for confirmation")
            if market_conditions.get('volatility', 'NORMAL') == 'HIGH':
                warnings.append("High volatility detected - use tighter stops")
                
            if warnings:
                recommendation['warnings'] = warnings
            
            return recommendation
            
        except Exception as e:
            log.error(f"Error generating entry recommendation: {str(e)}")
            return {
                'action': 'WAIT',
                'reason': f'Error in analysis: {str(e)}',
                'confidence': 0.0,
                'risk_level': 'HIGH'
            }
    
    def _get_timeframe_weight(self, timeframe: TimeFrames, primary_timeframe: TimeFrames) -> float:
        """
        Get weight for timeframe in multi-timeframe analysis.
        Higher timeframes get more weight.
        """
        timeframe_weights = {
            TimeFrames.D1: 4.0,
            TimeFrames.H4: 3.0,
            TimeFrames.H1: 2.0,
            TimeFrames.M15: 1.0
        }
        
        # Primary timeframe gets slight boost
        weight = timeframe_weights.get(timeframe, 1.0)
        if timeframe == primary_timeframe:
            weight *= 1.2
            
        return weight
    
    def _determine_signal_quality(self, confidence: float) -> EntryQuality:
        """
        Determine signal quality based on confidence level.
        """
        if confidence >= 0.9:
            return EntryQuality.EXCELLENT
        elif confidence >= 0.75:
            return EntryQuality.GOOD
        elif confidence >= 0.6:
            return EntryQuality.FAIR
        else:
            return EntryQuality.POOR
    
    def _assess_risk_level(self, confidence: float, signal_alignment: int) -> str:
        """
        Assess risk level based on confidence and signal alignment.
        """
        if confidence >= 0.8 and signal_alignment >= 2:
            return 'LOW'
        elif confidence >= 0.7 and signal_alignment >= 1:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _calculate_position_size(self, confidence: float, risk_level: str) -> float:
        """
        Calculate recommended position size as percentage of account.
        """
        base_size = self.MAX_RISK_PERCENT
        
        # Adjust based on confidence
        confidence_multiplier = confidence
        
        # Adjust based on risk level
        risk_multipliers = {
            'LOW': 1.0,
            'MEDIUM': 0.7,
            'HIGH': 0.5
        }
        
        risk_multiplier = risk_multipliers.get(risk_level, 0.5)
        
        position_size = base_size * confidence_multiplier * risk_multiplier
        return round(min(position_size, self.MAX_RISK_PERCENT), 2)
    
    def _determine_entry_timing(self, confidence: float, strategy: str) -> str:
        """
        Determine optimal entry timing based on strategy and confidence.
        """
        if confidence >= 0.85:
            return 'IMMEDIATE'
        elif confidence >= 0.75:
            if strategy == 'breakout':
                return 'ON_CONFIRMATION'  # Wait for volume confirmation
            else:
                return 'NEXT_CANDLE'
        else:
            return 'WAIT_FOR_CONFIRMATION'
    
    def _assess_market_conditions(self, combined_analysis: Dict) -> Dict:
        """
        Assess current market conditions based on analysis.
        """
        try:
            # This would typically analyze volatility, trend strength, etc.
            # For now, return basic assessment
            strategies_count = combined_analysis.get('strategies_count', 0)
            signal_alignment = abs(combined_analysis.get('signal_alignment', 0))
            
            if strategies_count >= 2 and signal_alignment >= 2:
                trend_strength = 'STRONG'
            elif strategies_count >= 1 and signal_alignment >= 1:
                trend_strength = 'MODERATE'
            else:
                trend_strength = 'WEAK'
            
            return {
                'trend_strength': trend_strength,
                'volatility': 'NORMAL',  # Could be enhanced with ATR analysis
                'market_phase': 'TRENDING' if signal_alignment > 0 else 'RANGING'
            }
            
        except Exception as e:
            log.error(f"Error assessing market conditions: {str(e)}")
            return {
                'trend_strength': 'UNKNOWN',
                'volatility': 'UNKNOWN',
                'market_phase': 'UNKNOWN'
            }
    
    def _create_empty_analysis(self, pair: str, timeframe: TimeFrames) -> Dict:
        """
        Create empty analysis structure for error cases.
        """
        return {
            'pair': pair,
            'timeframe': timeframe.name,
            'timestamp': datetime.now(self.timezone),
            'base_signal': None,
            'multi_timeframe': {
                'signal_type': EntrySignalType.NO_SIGNAL,
                'confidence': 0.0,
                'quality': EntryQuality.POOR
            },
            'breakout': {
                'signal_type': EntrySignalType.NO_SIGNAL,
                'confidence': 0.0,
                'quality': EntryQuality.POOR
            },
            'pullback': {
                'signal_type': EntrySignalType.NO_SIGNAL,
                'confidence': 0.0,
                'quality': EntryQuality.POOR
            },
            'combined_analysis': {
                'best_strategy': None,
                'combined_confidence': 0.0,
                'combined_quality': EntryQuality.POOR,
                'signal_alignment': 0,
                'strategies_count': 0
            },
            'entry_recommendation': {
                'action': 'WAIT',
                'reason': 'Insufficient data or analysis error',
                'confidence': 0.0,
                'risk_level': 'HIGH'
            }
        }

# Example usage and testing functions
def test_entry_strategies():
    """
    Test function for the AdvancedEntryStrategies class.
    """
    try:
        # Initialize the entry strategies
        entry_strategies = AdvancedEntryStrategies()
        
        # Create sample data for testing
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        
        # Generate sample OHLCV data
        np.random.seed(42)
        base_price = 1.1000
        
        sample_data = {
            'timestamp': dates,
            'open': base_price + np.cumsum(np.random.randn(100) * 0.0001),
            'high': None,
            'low': None,
            'close': None,
            'tick_volume': np.random.randint(1000, 5000, 100)
        }
        
        sample_data['close'] = sample_data['open'] + np.random.randn(100) * 0.0005
        sample_data['high'] = np.maximum(sample_data['open'], sample_data['close']) + np.random.rand(100) * 0.0003
        sample_data['low'] = np.minimum(sample_data['open'], sample_data['close']) - np.random.rand(100) * 0.0003
        
        df = pd.DataFrame(sample_data)
        
        # Test with multiple timeframes
        timeframes_data = {
            TimeFrames.H1: df,
            TimeFrames.H4: df.iloc[::4].reset_index(drop=True),  # Every 4th row for H4
            TimeFrames.D1: df.iloc[::24].reset_index(drop=True)   # Every 24th row for D1
        }
        
        # Analyze entry opportunity
        result = entry_strategies.analyze_entry_opportunity(
            pair='EURUSDm',
            timeframes_data=timeframes_data,
            primary_timeframe=TimeFrames.H1
        )
        
        # Print results
        print("=== ENTRY STRATEGIES TEST RESULTS ===")
        print(f"Pair: {result['pair']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Best Strategy: {result['entry_recommendation'].get('best_strategy', 'None')}")
        print(f"Action: {result['entry_recommendation'].get('action', 'WAIT')}")
        print(f"Confidence: {result['entry_recommendation'].get('confidence', 0):.1%}")
        print(f"Risk Level: {result['entry_recommendation'].get('risk_level', 'HIGH')}")
        
        return result
        
    except Exception as e:
        log.error(f"Error in test function: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    # Run test when script is executed directly
    test_result = test_entry_strategies()
    if test_result:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
