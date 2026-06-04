
import pandas as pd
from datetime import datetime, timedelta
import logging
from src.core import insider_api

import yfinance as yf

# Configure logging
logger = logging.getLogger("InsiderLogic")

def detect_cluster(ticker: str, current_trade_date: str, window_days: int = 7) -> tuple[int, int]:
    """
    Check if multiple insiders traded within +/- window_days.
    Returns: (cluster_size, score_boost)
    """
    try:
        # Fetch broader history for this ticker
        stock = yf.Ticker(ticker)
        insider_df = stock.insider_transactions
        
        if insider_df is None or insider_df.empty:
            return 0, 0
            
        target_date = pd.to_datetime(current_trade_date)
        start_date = target_date - timedelta(days=window_days)
        end_date = target_date + timedelta(days=window_days)
        
        # Filter trades in window
        # Note: 'Start Date' is often the index or 'Date' col
        # resetting index to handle if date is index
        df = insider_df.reset_index()
        
        # Ensure 'Start Date' column exists or handle varied YF formats
        date_col = 'Start Date' if 'Start Date' in df.columns else 'Date'
        if date_col not in df.columns:
            return 0, 0
            
        df[date_col] = pd.to_datetime(df[date_col])
        cluster_trades = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
        
        # Count unique insiders
        unique_insiders = cluster_trades['Insider'].nunique() if 'Insider' in cluster_trades.columns else 0
        
        score_boost = 0
        if unique_insiders >= 3:
            score_boost = 20 # Strong cluster
        elif unique_insiders == 2:
            score_boost = 10 # Minor cluster
            
        return unique_insiders, score_boost
        
    except Exception as e:
        logger.warning(f"Cluster detection failed for {ticker}: {e}")
        return 0, 0

def check_technical_confluence(ticker: str) -> tuple[bool, str, int]:
    """
    Check for technical factors (Support/Resistance).
    Returns: (is_confluence, reason, score_boost)
    """
    try:
        stock = yf.Ticker(ticker)
        # Fast info often has 50/200 DMA
        info = stock.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if not current_price:
            return False, "", 0
            
        ma50 = info.get('fiftyDayAverage')
        ma200 = info.get('twoHundredDayAverage')
        
        score_boost = 0
        reason = ""
        
        # Check if near 200 SMA (Major Support/Res) within 5%
        if ma200:
            diff = abs(current_price - ma200) / ma200
            if diff < 0.05:
                # If Price > MA200 (Support) or Price < MA200 (Resistance)
                # Simply "Near Major Technical Level" is good enough confluence for now
                score_boost += 10
                reason = "Near 200-Day SMA"
                
        # Check 50 SMA
        if ma50 and score_boost == 0: # Prioritize 200
             diff = abs(current_price - ma50) / ma50
             if diff < 0.05:
                score_boost += 5
                reason = "Near 50-Day SMA"
                
        return score_boost > 0, reason, score_boost

    except Exception as e:
        logger.warning(f"Technical check failed for {ticker}: {e}")
        return False, "", 0

def calculate_confidence(trade_data: dict, context_data: dict) -> int:
    """
    Calculate a confidence score (0-100) based on insider role, value, and ownership.
    """
    score = 50 # Base score
    
    # 1. Role Base
    title = trade_data.get('insider_title', '').upper()
    if "CEO" in title or "CHIEF EXECUTIVE OFFICER" in title:
        score += 20
    elif "CFO" in title or "CHIEF FINANCIAL OFFICER" in title:
        score += 18
    elif "DIRECTOR" in title:
        score += 10
    elif "PRESIDENT" in title:
        score += 15
        
    # 2. Transaction Value
    value = trade_data.get('value', 0)
    if value > 1_000_000: # > $1M
        score += 15
    elif value > 250_000: # > $250k
        score += 10
    elif value > 50_000:
        score += 5
        
    # 3. Ownership Change Impact
    # If they increased their position significantly (e.g., > 10% increase)
    shares = trade_data.get('shares', 0)
    owned_after = trade_data.get('owned_after', 0)
    owned_before = owned_after - shares if trade_data['transaction_type'] == 'Buy' else owned_after + shares
    
    if owned_before > 0:
        percent_change = (shares / owned_before) * 100
        if percent_change > 50: score += 10
        elif percent_change > 20: score += 5
        
    # 4. Market Context Alignment
    # If buying and Analyst Recommendation is Buy/Strong Buy
    rec = context_data.get('recommendation', '').lower()
    if trade_data['transaction_type'] == 'Buy' and 'buy' in rec:
        score += 10
    elif trade_data['transaction_type'] == 'Sell' and 'sell' in rec:
        score += 10
        
    # 5. Cluster Detection (New)
    try:
        cluster_size, cluster_score = detect_cluster(trade_data['ticker'], trade_data['trade_date'])
        score += cluster_score
        context_data['cluster_info'] = f"{cluster_size} Insiders Traded" if cluster_size > 1 else "No Cluster"
    except Exception:
        pass

    # 6. Technical Confluence (New)
    try:
        is_conf, tech_reason, tech_score = check_technical_confluence(trade_data['ticker'])
        if trade_data['transaction_type'] == 'Buy':
            score += tech_score
        context_data['technical_confluence'] = tech_reason
    except Exception:
        pass

    # Cap at 100
    return min(100, max(0, score))

def analyze_trade_context(ticker: str, trade_date_str: str) -> dict:
    """
    Analyze if price has moved favourably since trade date.
    Also returns sector context.
    """
    context = insider_api.get_market_context(ticker)
    
    try:
        # Simple backtest check
        # trade_date_str format typically YYYY-MM-DD from API
        trade_date = datetime.strptime(trade_date_str, "%Y-%m-%d")
        
        # Get history
        hist = insider_api.get_stock_history(ticker, period="3mo")
        
        # Find closest available date in history
        if not hist.empty:
            # Filter for data after trade date
            post_trade = hist[hist.index >= pd.Timestamp(trade_date)]
            if not post_trade.empty:
                entry_price = post_trade.iloc[0]['Close']
                current_price = post_trade.iloc[-1]['Close']
                change_pct = ((current_price - entry_price) / entry_price) * 100
                context['price_change_since_trade'] = change_pct
            else:
                context['price_change_since_trade'] = 0.0
        else:
            context['price_change_since_trade'] = 0.0
            
    except Exception as e:
        logger.warning(f"Context analysis failed for {ticker}: {e}")
        context['price_change_since_trade'] = 0.0
        
    return context

def classify_trade_type(trade_data: dict) -> str:
    """
    Classify trade as 'Opportunistic' or 'Routine'.
    Routine: Small, periodic-looking (heuristic).
    Opportunistic: Large value, random timing.
    """
    value = trade_data.get('value', 0)
    
    # Heuristic: Very large trades are usually Opportunistic or conviction based
    if value > 500_000:
        return "Opportunistic"
        
    # Heuristic: Check if '10b5-1' is mentioned (planned trading) - info not always in basic API view 
    # but we can infer from round numbers or specific dates if we had full history.
    # For now, simple value-based heuristic.
    
    return "Routine"

def generate_ai_rating(trade_data: dict, context_data: dict, confidence: int) -> str:
    """
    Generate a natural language explanation of the trade.
    """
    role = trade_data.get('insider_title', 'Insider')
    action = trade_data.get('transaction_type', 'Trade')
    value_str = f"${trade_data.get('value', 0):,.0f}"
    sector = context_data.get('sector', 'Unknown Sector')
    trade_type = classify_trade_type(trade_data)
    
    rating_scale = result_score = round(confidence / 10) # 0-10 scale
    
    signal = "NEUTRAL"
    if action == "Buy":
        if confidence > 80: signal = "STRONG BUY"
        elif confidence > 60: signal = "BUY"
    elif action == "Sell":
        if confidence > 80: signal = "STRONG SELL"
        elif confidence > 60: signal = "SELL"
    
    cluster_info = context_data.get('cluster_info', 'No Cluster')
    tech_reason = context_data.get('technical_confluence', '')
    
    explanation = (
        f"{signal} Signal (Rating: {rating_scale}/10).\n"
        f"{role} {action} worth {value_str}.\n"
        f"Type: {trade_type} | {cluster_info}\n"
        f"Sector: {sector}"
    )
    
    if tech_reason:
        explanation += f"\nConfluence: Price is {tech_reason}."
    
    return explanation
