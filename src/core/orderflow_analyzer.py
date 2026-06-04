"""
Order Flow Analyzer - PRODUCTION GRADE
══════════════════════════════════════════════════════════════════════════════
Professional-grade Order Flow analysis for real money trading.

Features:
- Advanced Delta (Tick-based estimation)
- CVD with Divergence Detection
- VWAP with Standard Deviation Bands
- Volume Profile with Value Area (VAH, POC, VAL)
- Footprint Analysis (Imbalance Ratios)
- Smart Money / Institutional Flow Detection
- Liquidity Pool Detection
- Order Block Detection (OB)
- Fair Value Gap Detection (FVG)
- Multi-Timeframe Flow Aggregation
- Weighted Confidence Scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FlowSignal(Enum):
    CONFIRM = "CONFIRM"
    OPPOSE = "OPPOSE"
    NEUTRAL = "NEUTRAL"
    STRONG_CONFIRM = "STRONG_CONFIRM"
    STRONG_OPPOSE = "STRONG_OPPOSE"


@dataclass
class ValueArea:
    """Volume Profile Value Area"""
    poc: float  # Point of Control
    vah: float  # Value Area High
    val: float  # Value Area Low
    total_volume: float
    profile: Dict[float, float]


@dataclass
class OrderBlock:
    """Institutional Order Block"""
    price_high: float
    price_low: float
    block_type: str  # 'BULLISH' or 'BEARISH'
    strength: float  # 0-1
    bar_index: int
    volume: float


@dataclass
class FairValueGap:
    """Fair Value Gap (Imbalance Zone)"""
    high: float
    low: float
    gap_type: str  # 'BULLISH' or 'BEARISH'
    filled: bool
    bar_index: int


class OrderFlowAnalyzer:
    """
    Production-Grade Order Flow Analyzer
    ════════════════════════════════════════════════════════════════════════
    
    For real money trading - comprehensive institutional flow analysis.
    """
    
    def __init__(self, 
                 lookback: int = 50,
                 value_area_pct: float = 0.70,  # 70% of volume
                 imbalance_threshold: float = 2.5,  # 250% vs opposite side
                 smart_money_vol_mult: float = 3.0,  # 3x average = institution
                 min_confirmation_score: int = 4):
        """
        Initialize Production Order Flow Analyzer.
        
        Args:
            lookback: Bars for calculations
            value_area_pct: Percentage of volume for Value Area (0.70 = 70%)
            imbalance_threshold: Multiplier for significant imbalance
            smart_money_vol_mult: Volume multiple for institutional detection
            min_confirmation_score: Minimum score for CONFIRM status
        """
        self.lookback = lookback
        self.value_area_pct = value_area_pct
        self.imbalance_threshold = imbalance_threshold
        self.smart_money_vol_mult = smart_money_vol_mult
        self.min_confirmation_score = min_confirmation_score
        
    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED DELTA CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_delta(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Advanced Delta using price action analysis.
        
        Method:
        1. Base delta from candle direction
        2. Weighted by wick analysis (rejection = opposite flow)
        3. Adjusted for candle strength (body vs range)
        
        Returns DataFrame with:
        - delta: Net buying/selling pressure
        - buy_volume: Estimated buy volume
        - sell_volume: Estimated sell volume
        - delta_pct: Delta as percentage of total volume
        """
        result = data.copy()
        
        volume = data.get('tick_volume', data.get('volume', pd.Series(1, index=data.index)))
        
        # Calculate candle metrics
        body = abs(data['close'] - data['open'])
        full_range = data['high'] - data['low']
        
        # Avoid division by zero
        full_range = full_range.replace(0, np.nan)
        
        # Body ratio (strength of move)
        body_ratio = body / full_range
        body_ratio = body_ratio.fillna(0.5)
        
        # Upper and lower wicks
        upper_wick = data['high'] - data[['close', 'open']].max(axis=1)
        lower_wick = data[['close', 'open']].min(axis=1) - data['low']
        
        # Wick ratios (rejection signals)
        upper_wick_ratio = upper_wick / full_range
        lower_wick_ratio = lower_wick / full_range
        upper_wick_ratio = upper_wick_ratio.fillna(0)
        lower_wick_ratio = lower_wick_ratio.fillna(0)
        
        # Directional bias
        is_bullish = data['close'] > data['open']
        
        # Base buy/sell calculation
        # If bullish: more buy volume, but long upper wick = selling at highs
        # If bearish: more sell volume, but long lower wick = buying at lows
        
        buy_pct = np.where(
            is_bullish,
            0.5 + (body_ratio * 0.3) - (upper_wick_ratio * 0.2) + (lower_wick_ratio * 0.1),
            0.5 - (body_ratio * 0.3) + (lower_wick_ratio * 0.2) - (upper_wick_ratio * 0.1)
        )
        
        # Clamp to valid range
        buy_pct = np.clip(buy_pct, 0.1, 0.9)
        sell_pct = 1 - buy_pct
        
        result['buy_volume'] = volume * buy_pct
        result['sell_volume'] = volume * sell_pct
        result['delta'] = result['buy_volume'] - result['sell_volume']
        result['delta_pct'] = (result['delta'] / volume.replace(0, np.nan)) * 100
        result['delta_pct'] = result['delta_pct'].fillna(0)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CVD WITH DIVERGENCE DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_cvd(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CVD with divergence detection.
        
        Returns DataFrame with:
        - cvd: Cumulative Volume Delta
        - cvd_ma: Smoothed CVD (trend)
        - cvd_trend: 'RISING', 'FALLING', 'FLAT'
        - cvd_divergence: 'BULLISH_DIV', 'BEARISH_DIV', 'NONE'
        """
        delta_df = self.calculate_delta(data)
        result = delta_df.copy()
        
        # Calculate CVD
        result['cvd'] = result['delta'].cumsum()
        
        # Smooth CVD for trend detection
        result['cvd_ma'] = result['cvd'].rolling(window=min(10, len(data))).mean()
        
        # CVD Trend
        cvd_change = result['cvd_ma'].diff(5)
        result['cvd_trend'] = np.where(
            cvd_change > 0, 'RISING',
            np.where(cvd_change < 0, 'FALLING', 'FLAT')
        )
        
        # Divergence Detection
        # Price making new highs but CVD not = Bearish Divergence
        # Price making new lows but CVD not = Bullish Divergence
        
        if len(data) >= 20:
            price_high_20 = data['close'].rolling(20).max()
            price_low_20 = data['close'].rolling(20).min()
            cvd_high_20 = result['cvd'].rolling(20).max()
            cvd_low_20 = result['cvd'].rolling(20).min()
            
            # Price at new high but CVD below its high = Bearish Divergence
            bearish_div = (data['close'] >= price_high_20 * 0.998) & \
                         (result['cvd'] < cvd_high_20 * 0.95)
            
            # Price at new low but CVD above its low = Bullish Divergence
            bullish_div = (data['close'] <= price_low_20 * 1.002) & \
                         (result['cvd'] > cvd_low_20 * 1.05)
            
            result['cvd_divergence'] = np.where(
                bearish_div, 'BEARISH_DIV',
                np.where(bullish_div, 'BULLISH_DIV', 'NONE')
            )
        else:
            result['cvd_divergence'] = 'NONE'
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VWAP WITH BANDS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_vwap(self, data: pd.DataFrame, 
                       bands: bool = True,
                       num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate VWAP with standard deviation bands.
        
        Returns DataFrame with:
        - vwap: Volume Weighted Average Price
        - vwap_upper: Upper band (+2 std)
        - vwap_lower: Lower band (-2 std)
        - price_vs_vwap: 'ABOVE', 'BELOW', 'AT_VWAP'
        - vwap_distance_pct: Distance from VWAP as percentage
        """
        result = data.copy()
        
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        volume = data.get('tick_volume', data.get('volume', pd.Series(1, index=data.index)))
        
        # Cumulative VWAP
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum().replace(0, np.nan)
        
        result['vwap'] = cumulative_tp_vol / cumulative_vol
        
        if bands:
            # Calculate variance for bands
            squared_diff = (typical_price - result['vwap']) ** 2
            cumulative_sq_diff_vol = (squared_diff * volume).cumsum()
            variance = cumulative_sq_diff_vol / cumulative_vol
            std_dev = np.sqrt(variance)
            
            result['vwap_upper'] = result['vwap'] + (std_dev * num_std)
            result['vwap_lower'] = result['vwap'] - (std_dev * num_std)
        
        # Price position relative to VWAP
        vwap_tolerance = (result['vwap_upper'] - result['vwap_lower']) * 0.05 if bands else result['vwap'] * 0.001
        
        result['price_vs_vwap'] = np.where(
            data['close'] > result['vwap'] + vwap_tolerance, 'ABOVE',
            np.where(data['close'] < result['vwap'] - vwap_tolerance, 'BELOW', 'AT_VWAP')
        )
        
        # Distance percentage
        result['vwap_distance_pct'] = ((data['close'] - result['vwap']) / result['vwap']) * 100
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VOLUME PROFILE WITH VALUE AREA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_volume_profile(self, data: pd.DataFrame, 
                                  bins: int = 30) -> ValueArea:
        """
        Calculate Volume Profile with Value Area.
        
        Value Area = Price range containing 70% of volume
        - POC: Point of Control (highest volume price)
        - VAH: Value Area High
        - VAL: Value Area Low
        
        Returns ValueArea dataclass
        """
        if len(data) < 2:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 0
            return ValueArea(
                poc=current_price,
                vah=current_price,
                val=current_price,
                total_volume=0,
                profile={}
            )
        
        volume = data.get('tick_volume', data.get('volume', pd.Series(1, index=data.index)))
        
        # Create price bins
        price_min = data['low'].min()
        price_max = data['high'].max()
        price_range = price_max - price_min
        
        if price_range == 0 or price_range < 0.00001:
            current_price = data['close'].iloc[-1]
            return ValueArea(
                poc=current_price,
                vah=current_price,
                val=current_price,
                total_volume=volume.sum(),
                profile={current_price: volume.sum()}
            )
        
        bin_size = price_range / bins
        
        # Build volume profile
        profile = {}
        for i in range(len(data)):
            # Distribute volume across the bar's range
            bar_low = data['low'].iloc[i]
            bar_high = data['high'].iloc[i]
            bar_vol = volume.iloc[i]
            
            # Find which bins this bar touches
            start_bin = int((bar_low - price_min) / bin_size)
            end_bin = int((bar_high - price_min) / bin_size)
            
            if start_bin == end_bin:
                bin_price = price_min + (start_bin + 0.5) * bin_size
                profile[bin_price] = profile.get(bin_price, 0) + bar_vol
            else:
                # Distribute volume proportionally
                num_bins = end_bin - start_bin + 1
                vol_per_bin = bar_vol / num_bins
                for b in range(start_bin, end_bin + 1):
                    if b >= 0 and b < bins:
                        bin_price = price_min + (b + 0.5) * bin_size
                        profile[bin_price] = profile.get(bin_price, 0) + vol_per_bin
        
        if not profile:
            current_price = data['close'].iloc[-1]
            return ValueArea(poc=current_price, vah=current_price, val=current_price, 
                           total_volume=volume.sum(), profile={})
        
        # Find POC
        poc = max(profile, key=profile.get)
        total_volume = sum(profile.values())
        
        # Calculate Value Area (70% of volume around POC)
        target_volume = total_volume * self.value_area_pct
        
        # Sort by price
        sorted_prices = sorted(profile.keys())
        poc_idx = sorted_prices.index(poc)
        
        # Expand from POC until we have 70% volume
        val_idx = poc_idx
        vah_idx = poc_idx
        current_vol = profile[poc]
        
        while current_vol < target_volume:
            # Check which side has more volume to add
            lower_vol = profile.get(sorted_prices[val_idx - 1], 0) if val_idx > 0 else 0
            upper_vol = profile.get(sorted_prices[vah_idx + 1], 0) if vah_idx < len(sorted_prices) - 1 else 0
            
            if lower_vol == 0 and upper_vol == 0:
                break
            
            if lower_vol >= upper_vol and val_idx > 0:
                val_idx -= 1
                current_vol += profile[sorted_prices[val_idx]]
            elif vah_idx < len(sorted_prices) - 1:
                vah_idx += 1
                current_vol += profile[sorted_prices[vah_idx]]
            elif val_idx > 0:
                val_idx -= 1
                current_vol += profile[sorted_prices[val_idx]]
            else:
                break
        
        return ValueArea(
            poc=poc,
            vah=sorted_prices[vah_idx],
            val=sorted_prices[val_idx],
            total_volume=total_volume,
            profile=profile
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SMART MONEY / INSTITUTIONAL FLOW DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_smart_money(self, data: pd.DataFrame) -> Dict:
        """
        Detect potential institutional activity.
        
        Signals:
        1. Volume spikes (>3x average)
        2. Large candle with volume absorption
        3. Reversal patterns with high volume
        
        Returns Dict with detection results
        """
        if len(data) < self.lookback:
            return {
                'detected': False,
                'direction': 'NEUTRAL',
                'confidence': 0,
                'details': []
            }
        
        volume = data.get('tick_volume', data.get('volume', pd.Series(1, index=data.index)))
        avg_volume = volume.rolling(self.lookback).mean()
        
        # Current metrics
        current_vol = volume.iloc[-1]
        vol_ratio = current_vol / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
        
        body = abs(data['close'].iloc[-1] - data['open'].iloc[-1])
        full_range = data['high'].iloc[-1] - data['low'].iloc[-1]
        body_ratio = body / full_range if full_range > 0 else 0.5
        
        # Recent candle analysis
        delta_df = self.calculate_delta(data)
        recent_delta = delta_df['delta'].iloc[-3:].sum() if len(delta_df) >= 3 else 0
        
        details = []
        score = 0
        
        # Volume spike detection
        if vol_ratio > self.smart_money_vol_mult:
            score += 3
            details.append(f"Volume spike: {vol_ratio:.1f}x average")
        elif vol_ratio > 2.0:
            score += 1
            details.append(f"Above-average volume: {vol_ratio:.1f}x")
        
        # Large body with volume (institutional move)
        if body_ratio > 0.7 and vol_ratio > 1.5:
            score += 2
            details.append("Strong directional move with volume")
        
        # Absorption (high volume, small body = accumulation/distribution)
        if body_ratio < 0.3 and vol_ratio > 2.0:
            score += 2
            details.append("Volume absorption detected")
        
        # Determine direction
        direction = 'NEUTRAL'
        if recent_delta > 0 and score >= 3:
            direction = 'BULLISH'
        elif recent_delta < 0 and score >= 3:
            direction = 'BEARISH'
        
        return {
            'detected': score >= 3,
            'direction': direction,
            'confidence': min(score / 7, 1.0),
            'volume_ratio': vol_ratio,
            'details': details
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ORDER BLOCK DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_order_blocks(self, data: pd.DataFrame, 
                            min_strength: float = 0.5) -> List[OrderBlock]:
        """
        Detect institutional Order Blocks.
        
        Order Block: Last opposing candle before strong move
        - Bullish OB: Last bearish candle before bullish impulse
        - Bearish OB: Last bullish candle before bearish impulse
        
        Returns list of OrderBlock dataclasses
        """
        if len(data) < 10:
            return []
        
        order_blocks = []
        volume = data.get('tick_volume', data.get('volume', pd.Series(1, index=data.index)))
        avg_volume = volume.rolling(20).mean()
        
        # Calculate ATR for move strength
        atr = (data['high'] - data['low']).rolling(14).mean()
        
        for i in range(5, len(data) - 3):
            # Check for strong move (3+ candles same direction, > 2 ATR move)
            next_3_close = data['close'].iloc[i+1:i+4]
            next_3_open = data['open'].iloc[i+1:i+4]
            
            # Bullish impulse check
            bullish_move = (next_3_close > next_3_open).all()
            total_move_up = next_3_close.iloc[-1] - data['close'].iloc[i]
            
            # Bearish impulse check
            bearish_move = (next_3_close < next_3_open).all()
            total_move_down = data['close'].iloc[i] - next_3_close.iloc[-1]
            
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else (data['high'].iloc[i] - data['low'].iloc[i])
            
            # Current candle is opposing (potential OB)
            is_bearish = data['close'].iloc[i] < data['open'].iloc[i]
            is_bullish = data['close'].iloc[i] > data['open'].iloc[i]
            
            # Bullish Order Block (bearish candle before bullish impulse)
            if is_bearish and bullish_move and total_move_up > current_atr * 1.5:
                strength = min(total_move_up / (current_atr * 3), 1.0)
                vol_strength = volume.iloc[i] / avg_volume.iloc[i] if avg_volume.iloc[i] > 0 else 1
                
                if strength >= min_strength:
                    order_blocks.append(OrderBlock(
                        price_high=data['high'].iloc[i],
                        price_low=data['low'].iloc[i],
                        block_type='BULLISH',
                        strength=strength * (0.5 + min(vol_strength, 1) * 0.5),
                        bar_index=i,
                        volume=volume.iloc[i]
                    ))
            
            # Bearish Order Block (bullish candle before bearish impulse)
            if is_bullish and bearish_move and total_move_down > current_atr * 1.5:
                strength = min(total_move_down / (current_atr * 3), 1.0)
                vol_strength = volume.iloc[i] / avg_volume.iloc[i] if avg_volume.iloc[i] > 0 else 1
                
                if strength >= min_strength:
                    order_blocks.append(OrderBlock(
                        price_high=data['high'].iloc[i],
                        price_low=data['low'].iloc[i],
                        block_type='BEARISH',
                        strength=strength * (0.5 + min(vol_strength, 1) * 0.5),
                        bar_index=i,
                        volume=volume.iloc[i]
                    ))
        
        # Return most recent and strongest blocks
        return sorted(order_blocks, key=lambda x: x.strength, reverse=True)[:5]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FAIR VALUE GAP DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_fair_value_gaps(self, data: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps (Imbalances).
        
        FVG: Gap between candle 1 high/low and candle 3 low/high
        (candle 2 doesn't overlap the gap)
        
        Returns list of FairValueGap dataclasses
        """
        if len(data) < 3:
            return []
        
        fvgs = []
        current_price = data['close'].iloc[-1]
        
        for i in range(len(data) - 2):
            candle1 = data.iloc[i]
            candle2 = data.iloc[i + 1]
            candle3 = data.iloc[i + 2]
            
            # Bullish FVG: Candle 1 high < Candle 3 low
            if candle1['high'] < candle3['low']:
                fvg = FairValueGap(
                    high=candle3['low'],
                    low=candle1['high'],
                    gap_type='BULLISH',
                    filled=current_price <= candle1['high'],
                    bar_index=i + 1
                )
                if not fvg.filled:  # Only include unfilled gaps
                    fvgs.append(fvg)
            
            # Bearish FVG: Candle 1 low > Candle 3 high
            if candle1['low'] > candle3['high']:
                fvg = FairValueGap(
                    high=candle1['low'],
                    low=candle3['high'],
                    gap_type='BEARISH',
                    filled=current_price >= candle1['low'],
                    bar_index=i + 1
                )
                if not fvg.filled:
                    fvgs.append(fvg)
        
        # Return most recent unfilled gaps
        return sorted(fvgs, key=lambda x: x.bar_index, reverse=True)[:5]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIQUIDITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_liquidity(self, data: pd.DataFrame) -> Dict:
        """
        Analyze liquidity zones (where stops likely exist).
        
        Key levels:
        - Recent swing highs/lows (stop hunts)
        - Round numbers
        - Previous day high/low
        
        Returns Dict with liquidity analysis
        """
        if len(data) < 20:
            return {
                'buy_side_liquidity': [],
                'sell_side_liquidity': [],
                'nearest_liquidity': None,
                'liquidity_grabbed': False
            }
        
        current_price = data['close'].iloc[-1]
        
        # Find swing points (potential liquidity)
        window = 5
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(data) - window):
            # Swing high
            if data['high'].iloc[i] == data['high'].iloc[i-window:i+window+1].max():
                swing_highs.append({
                    'price': data['high'].iloc[i],
                    'index': i,
                    'touched': data['high'].iloc[i:].max() >= data['high'].iloc[i] * 0.9999
                })
            
            # Swing low
            if data['low'].iloc[i] == data['low'].iloc[i-window:i+window+1].min():
                swing_lows.append({
                    'price': data['low'].iloc[i],
                    'index': i,
                    'touched': data['low'].iloc[i:].min() <= data['low'].iloc[i] * 1.0001
                })
        
        # Buy-side liquidity (above price, short stops)
        buy_side = [sh for sh in swing_highs if sh['price'] > current_price and not sh['touched']]
        
        # Sell-side liquidity (below price, long stops)
        sell_side = [sl for sl in swing_lows if sl['price'] < current_price and not sl['touched']]
        
        # Nearest liquidity
        nearest = None
        nearest_dist = float('inf')
        
        for liq in buy_side + sell_side:
            dist = abs(liq['price'] - current_price)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = liq
        
        # Check if liquidity was just grabbed (price spiked past swing then reversed)
        liquidity_grabbed = False
        if len(data) >= 3:
            recent_high = data['high'].iloc[-3:].max()
            recent_low = data['low'].iloc[-3:].min()
            for sh in swing_highs[-5:]:
                if recent_high > sh['price'] and data['close'].iloc[-1] < sh['price']:
                    liquidity_grabbed = True
                    break
            for sl in swing_lows[-5:]:
                if recent_low < sl['price'] and data['close'].iloc[-1] > sl['price']:
                    liquidity_grabbed = True
                    break
        
        return {
            'buy_side_liquidity': sorted([l['price'] for l in buy_side])[:3],
            'sell_side_liquidity': sorted([l['price'] for l in sell_side], reverse=True)[:3],
            'nearest_liquidity': nearest['price'] if nearest else None,
            'liquidity_grabbed': liquidity_grabbed
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ABSORPTION DETECTION (ENHANCED)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_absorption(self, data: pd.DataFrame) -> Dict:
        """
        Enhanced absorption detection with multi-bar confirmation.
        
        Absorption: Market absorbing large orders without price moving
        """
        if len(data) < self.lookback:
            return {
                'detected': False,
                'type': 'NONE',
                'strength': 0,
                'location': 'MIDDLE'
            }
        
        volume = data.get('tick_volume', data.get('volume', pd.Series(1, index=data.index)))
        avg_volume = volume.rolling(self.lookback).mean()
        
        # ATR for price movement reference
        atr = (data['high'] - data['low']).rolling(14).mean()
        
        # Check last 3 bars for absorption pattern
        recent_vol = volume.iloc[-3:].sum()
        recent_avg = avg_volume.iloc[-1] * 3
        vol_ratio = recent_vol / recent_avg if recent_avg > 0 else 1
        
        # Price movement over same period
        price_move = abs(data['close'].iloc[-1] - data['close'].iloc[-4])
        expected_move = atr.iloc[-1] * 1.5 if not pd.isna(atr.iloc[-1]) else price_move
        
        # Absorption: High volume but low price movement
        is_absorption = vol_ratio > 1.5 and price_move < expected_move * 0.3
        
        # Determine location (at high, low, or middle of range)
        recent_range_high = data['high'].iloc[-20:].max()
        recent_range_low = data['low'].iloc[-20:].min()
        current_price = data['close'].iloc[-1]
        
        range_position = (current_price - recent_range_low) / (recent_range_high - recent_range_low) if recent_range_high != recent_range_low else 0.5
        
        if range_position > 0.8:
            location = 'AT_HIGH'
            absorption_type = 'DISTRIBUTION' if is_absorption else 'NONE'
        elif range_position < 0.2:
            location = 'AT_LOW'
            absorption_type = 'ACCUMULATION' if is_absorption else 'NONE'
        else:
            location = 'MIDDLE'
            absorption_type = 'CONSOLIDATION' if is_absorption else 'NONE'
        
        return {
            'detected': is_absorption,
            'type': absorption_type,
            'strength': min(vol_ratio / 3, 1.0) if is_absorption else 0,
            'location': location,
            'volume_ratio': vol_ratio
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MOMENTUM SHIFT DETECTION (ENHANCED)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_momentum_shift(self, data: pd.DataFrame) -> Dict:
        """
        Enhanced momentum shift with confirmation scoring.
        """
        if len(data) < self.lookback:
            return {
                'shift': 'NONE',
                'strength': 0,
                'confirmation': 0,
                'details': []
            }
        
        cvd_df = self.calculate_cvd(data)
        delta_df = self.calculate_delta(data)
        
        details = []
        score = 0
        
        # CVD direction change
        cvd_5_back = cvd_df['cvd'].iloc[-6:-1].mean() if len(cvd_df) >= 6 else 0
        cvd_now = cvd_df['cvd'].iloc[-1]
        cvd_5_prior = cvd_df['cvd'].iloc[-11:-6].mean() if len(cvd_df) >= 11 else cvd_5_back
        
        # Check for direction reversal
        prior_trend = 'UP' if cvd_5_back > cvd_5_prior else 'DOWN'
        current_trend = 'UP' if cvd_now > cvd_5_back else 'DOWN'
        
        if prior_trend != current_trend:
            score += 2
            details.append(f"CVD trend reversal: {prior_trend} → {current_trend}")
        
        # Delta sign change
        recent_delta = delta_df['delta'].iloc[-3:].mean()
        prior_delta = delta_df['delta'].iloc[-6:-3].mean() if len(delta_df) >= 6 else 0
        
        if (recent_delta > 0 and prior_delta < 0) or (recent_delta < 0 and prior_delta > 0):
            score += 2
            details.append("Delta cross zero line")
        
        # Price action confirmation
        price_5_back = data['close'].iloc[-6:-1].mean()
        price_now = data['close'].iloc[-1]
        price_direction = 'UP' if price_now > price_5_back else 'DOWN'
        
        if current_trend == 'UP' and price_direction == 'UP':
            score += 1
            details.append("Price confirms bullish shift")
        elif current_trend == 'DOWN' and price_direction == 'DOWN':
            score += 1
            details.append("Price confirms bearish shift")
        
        # Determine shift
        shift = 'NONE'
        if score >= 3:
            shift = 'BULLISH_SHIFT' if current_trend == 'UP' else 'BEARISH_SHIFT'
        
        return {
            'shift': shift,
            'strength': min(score / 5, 1.0),
            'confirmation': score,
            'details': details
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION BIAS (ENHANCED)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_session_bias(self, current_time: datetime = None) -> Dict:
        """
        Enhanced session analysis with specific strategy recommendations.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        hour = current_time.hour
        minute = current_time.minute
        
        sessions = {
            'ASIAN': (0, 8, 'LOW', 'RANGE', [
                "Expect range-bound price action",
                "Focus on support/resistance levels",
                "Smaller position sizes recommended",
                "Good for mean-reversion strategies"
            ]),
            'LONDON_OPEN': (8, 10, 'HIGH', 'BREAKOUT', [
                "Major liquidity injection",
                "Watch for stop hunts then reversal",
                "Good for breakout strategies",
                "Key time for trend establishment"
            ]),
            'LONDON': (10, 13, 'MEDIUM_HIGH', 'TREND', [
                "Established trend continuation",
                "Follow the established direction",
                "Good for momentum strategies"
            ]),
            'NY_OVERLAP': (13, 16, 'VERY_HIGH', 'VOLATILE', [
                "HIGHEST volatility period",
                "Major moves likely",
                "Best for experienced traders",
                "Watch for news releases"
            ]),
            'NY': (16, 21, 'MEDIUM', 'CONTINUATION', [
                "Trend continuation likely",
                "Watch for profit-taking near close",
                "Good for following established trends"
            ]),
            'QUIET': (21, 24, 'LOW', 'QUIET', [
                "Low liquidity period",
                "Avoid trading or use tight stops",
                "Prepare for next session"
            ])
        }
        
        for session_name, (start, end, volatility, behavior, tips) in sessions.items():
            if start <= hour < end:
                return {
                    'session': session_name,
                    'volatility': volatility,
                    'behavior': behavior,
                    'tips': tips,
                    'hours_until_next': (end - hour) if hour < end else (24 - hour + end),
                    'is_optimal': behavior in ['BREAKOUT', 'VOLATILE', 'TREND']
                }
        
        # Fallback for hours 0-8 (second check for Asian)
        return {
            'session': 'ASIAN',
            'volatility': 'LOW',
            'behavior': 'RANGE',
            'tips': ["Expect range-bound price action"],
            'hours_until_next': 8 - hour if hour < 8 else 32 - hour,
            'is_optimal': False
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMPREHENSIVE SIGNAL CONFIRMATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_confirmation(self, signal_direction: str, data: pd.DataFrame) -> Dict:
        """
        PRODUCTION-GRADE signal confirmation with weighted scoring.
        
        Scoring System (max 20 points):
        - CVD trend alignment: +/- 3
        - Delta direction: +/- 2
        - VWAP position: +/- 2
        - Value Area position: +/- 2
        - Smart Money alignment: +/- 3
        - Order Block proximity: +/- 2
        - FVG proximity: +/- 1
        - Divergence: +/- 3
        - Absorption: +/- 1
        - Momentum shift: +/- 2
        
        Returns comprehensive confirmation dict
        """
        if len(data) < max(20, self.lookback):
            return {
                'status': FlowSignal.NEUTRAL.value,
                'confidence': 0.5,
                'score': 0,
                'max_score': 20,
                'reasons': ["Insufficient data for analysis"],
                'warnings': [],
                'metrics': {}
            }
        
        score = 0
        reasons = []
        warnings = []
        
        # 1. Calculate all metrics
        cvd_df = self.calculate_cvd(data)
        vwap_df = self.calculate_vwap(data)
        value_area = self.calculate_volume_profile(data)
        smart_money = self.detect_smart_money(data)
        order_blocks = self.detect_order_blocks(data)
        fvgs = self.detect_fair_value_gaps(data)
        absorption = self.detect_absorption(data)
        momentum = self.detect_momentum_shift(data)
        liquidity = self.analyze_liquidity(data)
        session = self.get_session_bias()
        
        current_price = data['close'].iloc[-1]
        
        # 2. CVD Trend (±3)
        cvd_trend = cvd_df['cvd_trend'].iloc[-1]
        if signal_direction == 'BUY':
            if cvd_trend == 'RISING':
                score += 3
                reasons.append("CVD rising - buying pressure")
            elif cvd_trend == 'FALLING':
                score -= 3
                warnings.append("CVD falling - selling pressure")
        elif signal_direction == 'SELL':
            if cvd_trend == 'FALLING':
                score += 3
                reasons.append("CVD falling - selling pressure")
            elif cvd_trend == 'RISING':
                score -= 3
                warnings.append("CVD rising - buying pressure")
        
        # 3. CVD Divergence (±3)
        divergence = cvd_df['cvd_divergence'].iloc[-1]
        if divergence == 'BULLISH_DIV' and signal_direction == 'BUY':
            score += 3
            reasons.append("Bullish divergence detected")
        elif divergence == 'BEARISH_DIV' and signal_direction == 'SELL':
            score += 3
            reasons.append("Bearish divergence detected")
        elif divergence == 'BULLISH_DIV' and signal_direction == 'SELL':
            score -= 3
            warnings.append("Bullish divergence opposes SELL")
        elif divergence == 'BEARISH_DIV' and signal_direction == 'BUY':
            score -= 3
            warnings.append("Bearish divergence opposes BUY")
        
        # 4. Delta Direction (±2)
        current_delta = cvd_df['delta'].iloc[-1]
        if (signal_direction == 'BUY' and current_delta > 0) or \
           (signal_direction == 'SELL' and current_delta < 0):
            score += 2
            reasons.append(f"Delta confirms {'buying' if current_delta > 0 else 'selling'}")
        else:
            score -= 2
            warnings.append(f"Delta shows {'selling' if current_delta < 0 else 'buying'}")
        
        # 5. VWAP Position (±2)
        vwap_pos = vwap_df['price_vs_vwap'].iloc[-1]
        if (signal_direction == 'BUY' and vwap_pos == 'ABOVE') or \
           (signal_direction == 'SELL' and vwap_pos == 'BELOW'):
            score += 2
            reasons.append(f"Price {'above' if vwap_pos == 'ABOVE' else 'below'} VWAP")
        elif vwap_pos != 'AT_VWAP':
            score -= 1
            warnings.append(f"Price {'below' if vwap_pos == 'BELOW' else 'above'} VWAP")
        
        # 6. Value Area Position (±2)
        if signal_direction == 'BUY':
            if current_price > value_area.vah:
                score += 2
                reasons.append("Breakout above Value Area High")
            elif current_price < value_area.val:
                score -= 1
                warnings.append("Price below Value Area Low")
            elif current_price > value_area.poc:
                score += 1
                reasons.append("Price above POC")
        elif signal_direction == 'SELL':
            if current_price < value_area.val:
                score += 2
                reasons.append("Breakdown below Value Area Low")
            elif current_price > value_area.vah:
                score -= 1
                warnings.append("Price above Value Area High")
            elif current_price < value_area.poc:
                score += 1
                reasons.append("Price below POC")
        
        # 7. Smart Money (±3)
        if smart_money['detected']:
            if (signal_direction == 'BUY' and smart_money['direction'] == 'BULLISH') or \
               (signal_direction == 'SELL' and smart_money['direction'] == 'BEARISH'):
                score += 3
                reasons.append(f"Smart Money: {smart_money['direction']}")
            elif smart_money['direction'] != 'NEUTRAL':
                score -= 2
                warnings.append(f"Smart Money: {smart_money['direction']}")
        
        # 8. Order Block Proximity (±2)
        for ob in order_blocks[:2]:
            if ob.price_low <= current_price <= ob.price_high:
                if (signal_direction == 'BUY' and ob.block_type == 'BULLISH') or \
                   (signal_direction == 'SELL' and ob.block_type == 'BEARISH'):
                    score += 2
                    reasons.append(f"At {ob.block_type} Order Block")
                else:
                    score -= 1
                    warnings.append(f"At opposing {ob.block_type} Order Block")
                break
        
        # 9. FVG Proximity (±1)
        for fvg in fvgs[:2]:
            if fvg.low <= current_price <= fvg.high:
                if (signal_direction == 'BUY' and fvg.gap_type == 'BULLISH') or \
                   (signal_direction == 'SELL' and fvg.gap_type == 'BEARISH'):
                    score += 1
                    reasons.append(f"At {fvg.gap_type} Fair Value Gap")
                break
        
        # 10. Absorption (±1)
        if absorption['detected']:
            if (signal_direction == 'BUY' and absorption['type'] == 'ACCUMULATION') or \
               (signal_direction == 'SELL' and absorption['type'] == 'DISTRIBUTION'):
                score += 1
                reasons.append(f"{absorption['type']} detected")
            elif absorption['type'] in ['ACCUMULATION', 'DISTRIBUTION']:
                score -= 1
                warnings.append(f"Opposing {absorption['type']} detected")
        
        # 11. Momentum Shift (±2)
        if momentum['shift'] != 'NONE':
            if (signal_direction == 'BUY' and momentum['shift'] == 'BULLISH_SHIFT') or \
               (signal_direction == 'SELL' and momentum['shift'] == 'BEARISH_SHIFT'):
                score += 2
                reasons.append(momentum['shift'].replace('_', ' '))
            else:
                score -= 2
                warnings.append(f"Opposing {momentum['shift'].replace('_', ' ')}")
        
        # 12. Liquidity Warning
        if liquidity['liquidity_grabbed']:
            score += 1
            reasons.append("Liquidity grab complete - reversal possible")
        
        # Determine status
        max_score = 20
        if score >= 8:
            status = FlowSignal.STRONG_CONFIRM
        elif score >= self.min_confirmation_score:
            status = FlowSignal.CONFIRM
        elif score <= -8:
            status = FlowSignal.STRONG_OPPOSE
        elif score <= -self.min_confirmation_score:
            status = FlowSignal.OPPOSE
        else:
            status = FlowSignal.NEUTRAL
        
        confidence = (score + 10) / 20  # Normalize to 0-1
        confidence = max(0, min(1, confidence))
        
        return {
            'status': status.value,
            'confidence': round(confidence, 2),
            'score': score,
            'max_score': max_score,
            'reasons': reasons[:5],  # Top 5 reasons
            'warnings': warnings[:3],  # Top 3 warnings
            'metrics': {
                'cvd_trend': cvd_trend,
                'divergence': divergence,
                'delta': round(current_delta, 2),
                'vwap_position': vwap_pos,
                'poc': round(value_area.poc, 5),
                'vah': round(value_area.vah, 5),
                'val': round(value_area.val, 5),
                'smart_money': smart_money['direction'] if smart_money['detected'] else 'NONE',
                'absorption': absorption['type'],
                'momentum_shift': momentum['shift'],
                'session': session['session']
            }
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIMAL ENTRY TIME
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_optimal_entry_time(self, data: pd.DataFrame, signal_direction: str) -> Dict:
        """
        Calculate optimal entry timing with confidence score.
        """
        confirmation = self.get_confirmation(signal_direction, data)
        session = self.get_session_bias()
        
        # Determine action
        status = confirmation['status']
        
        if status in ['STRONG_CONFIRM', 'CONFIRM']:
            action = 'ENTER_NOW'
            entry_time = datetime.now(timezone.utc)
            note = f"Order Flow confirms {signal_direction}. Execute trade."
        elif status == 'NEUTRAL':
            action = 'WAIT_CANDLE'
            entry_time = None
            note = "Mixed signals. Wait for next candle confirmation."
        else:
            action = 'DO_NOT_ENTER'
            entry_time = None
            note = f"Order Flow OPPOSES {signal_direction}. Avoid this trade."
        
        # Session adjustment
        if not session.get('is_optimal', True) and action == 'ENTER_NOW':
            action = 'WAIT_SESSION'
            note = f"Signal confirmed but {session['session']} is not optimal. Consider waiting."
        
        return {
            'action': action,
            'entry_time': entry_time,
            'confidence': confirmation['confidence'],
            'session': session['session'],
            'is_optimal_session': session.get('is_optimal', False),
            'note': note,
            'score': confirmation['score'],
            'max_score': confirmation['max_score']
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FULL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze(self, data: pd.DataFrame, signal_direction: str = None) -> Dict:
        """
        Production-grade full Order Flow analysis.
        """
        if len(data) < 10:
            return {'error': 'Insufficient data', 'min_required': self.lookback}
        
        # Run all analyses
        cvd_df = self.calculate_cvd(data)
        vwap_df = self.calculate_vwap(data)
        value_area = self.calculate_volume_profile(data)
        smart_money = self.detect_smart_money(data)
        order_blocks = self.detect_order_blocks(data)
        fvgs = self.detect_fair_value_gaps(data)
        absorption = self.detect_absorption(data)
        momentum = self.detect_momentum_shift(data)
        liquidity = self.analyze_liquidity(data)
        session = self.get_session_bias()
        
        analysis = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'bars_analyzed': len(data),
            'metrics': {
                'current_price': data['close'].iloc[-1],
                'delta': cvd_df['delta'].iloc[-1],
                'cvd': cvd_df['cvd'].iloc[-1],
                'cvd_trend': cvd_df['cvd_trend'].iloc[-1],
                'divergence': cvd_df['cvd_divergence'].iloc[-1],
                'vwap': vwap_df['vwap'].iloc[-1],
                'vwap_upper': vwap_df['vwap_upper'].iloc[-1],
                'vwap_lower': vwap_df['vwap_lower'].iloc[-1],
                'price_vs_vwap': vwap_df['price_vs_vwap'].iloc[-1]
            },
            'value_area': {
                'poc': value_area.poc,
                'vah': value_area.vah,
                'val': value_area.val,
                'total_volume': value_area.total_volume
            },
            'smart_money': smart_money,
            'order_blocks': [
                {'type': ob.block_type, 'high': ob.price_high, 'low': ob.price_low, 'strength': ob.strength}
                for ob in order_blocks[:3]
            ],
            'fair_value_gaps': [
                {'type': fvg.gap_type, 'high': fvg.high, 'low': fvg.low}
                for fvg in fvgs[:3]
            ],
            'absorption': absorption,
            'momentum': momentum,
            'liquidity': liquidity,
            'session': session
        }
        
        # Add signal validation if direction provided
        if signal_direction:
            analysis['confirmation'] = self.get_confirmation(signal_direction, data)
            analysis['entry'] = self.get_optimal_entry_time(data, signal_direction)
        
        return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_signal_with_orderflow(data: pd.DataFrame, signal_direction: str) -> Dict:
    """Quick validation for signal confirmation."""
    analyzer = OrderFlowAnalyzer()
    return analyzer.get_confirmation(signal_direction, data)
