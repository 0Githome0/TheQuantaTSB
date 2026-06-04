"""
ULTIMATE Strategy V12.8 - PRECISION 70%
=======================================
V12.7 hit 63.6% WR. Excellent progress.
Target: 70% Win Rate.

STRATEGY UPGRADES:
1. ADDED Money Flow Index (MFI) - Volume-weighted RSI
2. ADDED Strict Volume Filter (Vol > Avg Vol)
3. INCREASED Confirmations needed (13 -> 14)
4. STRICTER Trends (ADX > 30 for all)
5. TIGHTER TP (0.8 ATR) to secure wins

Author: TheQuanta
Version: 12.8.0 - Precision 70%
"""

import pandas as pd
import numpy as np

class UltimateStrategyV11:
    """
    Ultimate strategy V12.8 - Pushing for 70% Win Rate.
    More selective, better volume analysis, tighter targets.
    """
    
    def __init__(self):
        pass
    
    def generate_signals(self, df):
        data = df.copy()
        
        if 'Close' not in data.columns:
            data['Close'] = data.get('close', data.iloc[:, 3])
            data['High'] = data.get('high', data.iloc[:, 1])
            data['Low'] = data.get('low', data.iloc[:, 2])
            data['Open'] = data.get('open', data.iloc[:, 0])
        if 'Volume' not in data.columns:
            data['Volume'] = data.get('tick_volume', data.get('volume', 1))
            
        # ============================================================
        # DETECT ASSET & SET PARAMETERS
        # ============================================================
        avg_price = data['Close'].mean()
        
        # Default (Forex) - Tighter, faster
        atr_mult_sl = 1.0
        atr_mult_tp = 0.8   # Tighter TP for higher win rate
        be_trigger = 0.05   # Slightly relaxed BE to avoid noise
        trail_start = 0.3
        trail_dist = 0.15
        confirmations_needed = 14 # Increased for selectivity
        adx_threshold = 30  # Strict trend only
        asset_name = "FOREX"

        # Gold/Crypto (High Volatility)
        if avg_price > 500: 
            asset_name = "GOLD/CRYPTO"
            atr_mult_sl = 1.3
            atr_mult_tp = 1.0
            be_trigger = 0.15
            trail_start = 0.5
            trail_dist = 0.3
            confirmations_needed = 14
            adx_threshold = 30

        # ============================================================
        # INDICATORS
        # ============================================================
        
        data['EMA9'] = data['Close'].ewm(span=9).mean()
        data['EMA21'] = data['Close'].ewm(span=21).mean()
        data['EMA50'] = data['Close'].ewm(span=50).mean()
        data['EMA100'] = data['Close'].ewm(span=100).mean()
        
        # EMA9 slope
        data['EMA9_Slope'] = data['EMA9'] - data['EMA9'].shift(3)
        data['EMA9_Up'] = (data['EMA9_Slope'] > 0).astype(int)
        data['EMA9_Down'] = (data['EMA9_Slope'] < 0).astype(int)
        
        data['Bull_Stack'] = (
            (data['EMA9'] > data['EMA21']) &
            (data['EMA21'] > data['EMA50']) &
            (data['EMA50'] > data['EMA100'])
        ).astype(int)
        
        data['Bear_Stack'] = (
            (data['EMA9'] < data['EMA21']) &
            (data['EMA21'] < data['EMA50']) &
            (data['EMA50'] < data['EMA100'])
        ).astype(int)
        
        trend_bars = 12
        data['Strong_Bull'] = (data['Bull_Stack'].rolling(trend_bars).sum() >= int(trend_bars * 0.8)).astype(int)
        data['Strong_Bear'] = (data['Bear_Stack'].rolling(trend_bars).sum() >= int(trend_bars * 0.8)).astype(int)
        
        # ATR
        data['TR'] = pd.concat([
            data['High'] - data['Low'],
            abs(data['High'] - data['Close'].shift(1)),
            abs(data['Low'] - data['Close'].shift(1))
        ], axis=1).max(axis=1)
        data['ATR'] = data['TR'].rolling(14).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        rsi_low = data['RSI'].rolling(14).min()
        rsi_high = data['RSI'].rolling(14).max()
        data['StochRSI'] = (data['RSI'] - rsi_low) / (rsi_high - rsi_low + 1e-10) * 100
        data['StochRSI_K'] = data['StochRSI'].rolling(3).mean()
        data['StochRSI_D'] = data['StochRSI_K'].rolling(3).mean()
        
        # MFI (Money Flow Index)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        raw_money_flow = typical_price * data['Volume']
        
        pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=data.index)
        neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=data.index)
        
        pos_mf = pos_flow.rolling(14).sum()
        neg_mf = neg_flow.rolling(14).sum()
        
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        data['MFI'] = mfi
        
        # Volume Filter
        data['Vol_MA'] = data['Volume'].rolling(20).mean()
        data['High_Vol'] = (data['Volume'] > data['Vol_MA']).astype(int)
        
        # MACD
        data['EMA12'] = data['Close'].ewm(span=12).mean()
        data['EMA26'] = data['Close'].ewm(span=26).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger
        data['BB_Mid'] = data['Close'].rolling(20).mean()
        
        # Ichimoku
        data['Tenkan'] = (data['High'].rolling(9).max() + data['Low'].rolling(9).min()) / 2
        data['Kijun'] = (data['High'].rolling(26).max() + data['Low'].rolling(26).min()) / 2
        data['Senkou_A'] = ((data['Tenkan'] + data['Kijun']) / 2).shift(26)
        
        data['Above_Cloud'] = (data['Close'] > data['Senkou_A']).astype(int) # Simplified Cloud
        
        # ADX
        plus_dm = data['High'].diff()
        minus_dm = data['Low'].diff().abs() * -1
        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
        
        atr14 = data['TR'].rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        data['ADX'] = dx.rolling(14).mean()
        data['Plus_DI'] = plus_di
        data['Minus_DI'] = minus_di
        
        # Supertrend
        data['ST_Lower'] = ((data['High'] + data['Low']) / 2) - (3 * data['ATR'])
        data['ST_Bullish'] = (data['Close'] > data['ST_Lower'].shift(1)).astype(int)
        
        # OBV
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        data['OBV'] = obv
        data['OBV_EMA'] = pd.Series(obv).ewm(span=20).mean().values
        data['OBV_Rising'] = (data['OBV'] > data['OBV_EMA']).astype(int)
        
        # Candles
        data['Bullish_Candle'] = (data['Close'] > data['Open']).astype(int)
        data['Bearish_Candle'] = (data['Close'] < data['Open']).astype(int)
        
        body = abs(data['Close'] - data['Open'])
        range_ = data['High'] - data['Low']
        data['Strong_Body'] = (body > range_ * 0.5).astype(int)
        
        # Close in upper/lower half
        mid_range = (data['High'] + data['Low']) / 2
        data['Close_Upper_Half'] = (data['Close'] > mid_range).astype(int)
        data['Close_Lower_Half'] = (data['Close'] < mid_range).astype(int)
        
        # ============================================================
        # ENTRY CONDITIONS (21 conditions, need 14+)
        # ============================================================
        
        data['L1'] = data['Strong_Bull']                                    
        data['L2'] = (data['RSI'] > 45).astype(int) & (data['RSI'] < 55).astype(int)
        data['L3'] = (data['RSI'] > data['RSI'].shift(1)).astype(int)       
        data['L4'] = (data['StochRSI_K'] > data['StochRSI_D']).astype(int)  
        data['L5'] = (data['MACD'] > data['MACD_Signal']).astype(int)  
        data['L6'] = (data['MACD_Hist'] > data['MACD_Hist'].shift(1)).astype(int)  
        data['L7'] = (data['ADX'] > adx_threshold).astype(int)                         
        data['L8'] = (data['Plus_DI'] > data['Minus_DI']).astype(int)       
        data['L9'] = data['Above_Cloud']                                     
        data['L10'] = (data['Tenkan'] > data['Kijun']).astype(int)                                   
        data['L11'] = data['ST_Bullish']                                     
        data['L12'] = data['OBV_Rising']                                     
        data['L13'] = (data['Close'] > data['Close'].shift(10)).astype(int) # Momentum                             
        data['L14'] = (data['Low'].rolling(15).min() > data['Low'].rolling(15).min().shift(8)).astype(int) # Higher Low                                     
        data['L15'] = data['Bullish_Candle']                                 
        data['L16'] = (data['Close'] > data['BB_Mid']).astype(int)
        data['L17'] = data['Strong_Body']
        data['L18'] = data['EMA9_Up']
        data['L19'] = data['Close_Upper_Half']
        data['L20'] = data['High_Vol'] # NEW
        data['L21'] = (data['MFI'] < 80).astype(int) # NEW: Not Overbought
        
        data['Long_Score'] = sum([data[f'L{i}'] for i in range(1, 22)])
        
        data['Long_Signal'] = (
            (data['Long_Score'] >= confirmations_needed) & 
            (data['Strong_Bull'] == 1) &
            (data['ADX'] > adx_threshold) &
            (data['Bullish_Candle'] == 1) &
            (data['High_Vol'] == 1) & # Must have volume
            (data['RSI'] > 45) & (data['RSI'] < 55)
        ).astype(int)
        
        # SHORT
        data['S1'] = data['Strong_Bear']
        data['S2'] = (data['RSI'] > 45).astype(int) & (data['RSI'] < 55).astype(int)
        data['S3'] = (data['RSI'] < data['RSI'].shift(1)).astype(int)
        data['S4'] = (data['StochRSI_K'] < data['StochRSI_D']).astype(int)
        data['S5'] = (data['MACD'] < data['MACD_Signal']).astype(int)
        data['S6'] = (data['MACD_Hist'] < data['MACD_Hist'].shift(1)).astype(int)
        data['S7'] = (data['ADX'] > adx_threshold).astype(int)
        data['S8'] = (data['Plus_DI'] < data['Minus_DI']).astype(int)
        data['S9'] = (data['Above_Cloud'] == 0).astype(int)
        data['S10'] = (data['Tenkan'] < data['Kijun']).astype(int)
        data['S11'] = (data['ST_Bullish'] == 0).astype(int)
        data['S12'] = (data['OBV_Rising'] == 0).astype(int)
        data['S13'] = (data['Close'] < data['Close'].shift(10)).astype(int)
        data['S14'] = (data['High'].rolling(15).max() < data['High'].rolling(15).max().shift(8)).astype(int)
        data['S15'] = data['Bearish_Candle']
        data['S16'] = (data['Close'] < data['BB_Mid']).astype(int)
        data['S17'] = data['Strong_Body']
        data['S18'] = data['EMA9_Down']
        data['S19'] = data['Close_Lower_Half']
        data['S20'] = data['High_Vol']
        data['S21'] = (data['MFI'] > 20).astype(int) # Not Oversold
        
        data['Short_Score'] = sum([data[f'S{i}'] for i in range(1, 22)])
        
        data['Short_Signal'] = (
            (data['Short_Score'] >= confirmations_needed) & 
            (data['Strong_Bear'] == 1) &
            (data['ADX'] > adx_threshold) &
            (data['Bearish_Candle'] == 1) &
            (data['High_Vol'] == 1) & 
            (data['RSI'] > 45) & (data['RSI'] < 55)
        ).astype(int)
        
        # ============================================================
        # SIGNAL GENERATION
        # ============================================================
        
        position = 0
        entry = sl = tp = best = entry_atr = 0
        signals = []
        wins = losses = 0
        
        for i in range(len(data)):
            if i < 60:
                signals.append(0)
                continue
            
            c = data['Close'].iloc[i]
            h = data['High'].iloc[i]
            l = data['Low'].iloc[i]
            atr = data['ATR'].iloc[i]
            
            if pd.isna(atr) or atr == 0:
                signals.append(position)
                continue
            
            # Hybrid position management
            if position == 1:
                if h > best:
                    best = h
                    profit_atr = (best - entry) / entry_atr
                    
                    if profit_atr > trail_start:
                        trail = best - (entry_atr * trail_dist)
                        if trail > sl:
                            sl = trail
                    elif profit_atr > be_trigger:
                        if entry > sl:
                            sl = entry
                
                if h >= tp:
                    wins += 1
                    position = 0
                elif l <= sl:
                    if c >= entry:
                        wins += 1
                    else:
                        losses += 1
                    position = 0
                    
            elif position == -1:
                if l < best:
                    best = l
                    profit_atr = (entry - best) / entry_atr
                    
                    if profit_atr > trail_start:
                        trail = best + (entry_atr * trail_dist)
                        if trail < sl:
                            sl = trail
                    elif profit_atr > be_trigger:
                        if entry < sl:
                            sl = entry
                
                if l <= tp:
                    wins += 1
                    position = 0
                elif h >= sl:
                    if c <= entry:
                        wins += 1
                    else:
                        losses += 1
                    position = 0
            
            # Entry
            if position == 0:
                if data['Long_Signal'].iloc[i] == 1:
                    position = 1
                    entry = c
                    entry_atr = atr
                    sl = c - (atr * atr_mult_sl)
                    tp = c + (atr * atr_mult_tp)
                    best = h
                    signals.append(1)
                    continue
                    
                elif data['Short_Signal'].iloc[i] == 1:
                    position = -1
                    entry = c
                    entry_atr = atr
                    sl = c + (atr * atr_mult_sl)
                    tp = c - (atr * atr_mult_tp)
                    best = l
                    signals.append(-1)
                    continue
            
            signals.append(position)
        
        data['Signal'] = signals
        
        total = wins + losses
        wr = (wins / total * 100) if total > 0 else 0
        entries = sum(1 for i in range(1, len(signals)) 
                     if signals[i] != 0 and signals[i] != signals[i-1])
        print(f"[V12.8 {asset_name} PRECISION] Entries: {entries}, Wins: {wins}, Losses: {losses}, WR: {wr:.1f}%")
        
        return data
