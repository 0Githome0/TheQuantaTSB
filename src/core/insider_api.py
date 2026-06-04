
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger("InsiderAPI")

# Constants
# Constants
BIG_COMPANIES = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "INTC"]

def fetch_recent_trades(limit: int = 50) -> list:
    """
    Fetch recent insider trades using yfinance for a list of major tech companies.
    """
    logger.info(f"Fetching insider trades for: {BIG_COMPANIES}")
    
    all_trades = []
    
    for ticker in BIG_COMPANIES:
        try:
            stock = yf.Ticker(ticker)
            # Fetch insider transactions
            insider_df = stock.insider_transactions
            
            if insider_df is None or insider_df.empty:
                continue
                
            # Process each trade
            # yfinance columns usually: ['Shares', 'Value', 'URL', 'Text', 'Insider', 'Position', 'Transaction', 'Start Date', 'Ownership']
            for index, row in insider_df.iterrows():
                # Filter for meaningful transactions (Buy/Sell) checking limits
                try:
                    # Parse date - usually Timestamp or string
                    trade_date = row.get('Start Date')
                    if pd.isna(trade_date): continue
                    
                    # Ensure date is string YYYY-MM-DD
                    if isinstance(trade_date, (pd.Timestamp, datetime)):
                        trade_date_str = trade_date.strftime('%Y-%m-%d')
                    else:
                        trade_date_str = str(trade_date).split(' ')[0]
                        
                    # Filter for recent (last 365 days)
                    trade_dt = pd.to_datetime(trade_date_str)
                    if datetime.now() - trade_dt > timedelta(days=365):
                        # logger.debug(f"Skipping old trade: {trade_date_str}")
                        continue

                    # Safe numeric extraction
                    shares = row.get('Shares', 0)
                    value = row.get('Value', 0)
                    
                    # Try to fix 0 value
                    if pd.isna(value) or value == 0:
                        try:
                            # Estimate based on shares * roughly current price (not perfect but better than 0)
                             current_price = stock.fast_info.last_price
                             if not current_price: current_price = 0
                             value = abs(shares) * current_price
                        except Exception:
                            pass
                        
                    # Determine Buy/Sell/Grant
                    trans_text = str(row.get('Text', '')).lower()
                    
                    trans_type = "Unknown"
                    if "sale" in trans_text or "sold" in trans_text or "disposed" in trans_text:
                        trans_type = "Sell"
                    elif "purchase" in trans_text or "bought" in trans_text or "acquired" in trans_text:
                        trans_type = "Buy"
                    elif "grant" in trans_text or "award" in trans_text:
                        trans_type = "Grant"
                    elif "gift" in trans_text:
                        trans_type = "Gift"
                    elif "exercise" in trans_text:
                        trans_type = "Option Exercise"
                        
                    # Handle negative shares as Sell if not already caught
                    if shares < 0 and trans_type == "Unknown":
                        trans_type = "Sell"
                        
                    shares = abs(shares)
                    value = abs(value)
                        
                    # If specific 'Transaction' column exists and populated
                    if 'Transaction' in row:
                        t_raw = str(row['Transaction']).lower()
                        if "sale" in t_raw: trans_type = "Sell"
                        elif "purchase" in t_raw: trans_type = "Buy"
                        
                    if trans_type == "Unknown": 
                        continue

                    # Skip small useless trades unless it's a Buy
                    if value < 10000 and trans_type != "Buy":
                        continue

                    trade_data = {
                        "ticker": ticker,
                        "company_name": ticker, 
                        "insider_name": str(row.get('Insider', 'Unknown')),
                        "insider_title": str(row.get('Position', 'Insider')),
                        "trade_date": trade_date_str,
                        "filed_date": trade_date_str,
                        "transaction_type": trans_type,
                        "price": float(value) / float(shares) if shares > 0 else 0.0,
                        "shares": float(shares),
                        "value": float(value),
                        "owned_after": 0.0 # yfinance 'Ownership' col is 'D'/'I' string, not numeric shares
                    }
                    all_trades.append(trade_data)
                    
                except Exception as inner_e:
                    logger.warning(f"Error parsing row for {ticker}: {inner_e}")
                    continue

        except Exception as e:
            logger.warning(f"Error fetching {ticker}: {e}")
            continue
            
    # Sort all trades by date descending
    all_trades.sort(key=lambda x: x['trade_date'], reverse=True)
    
    logger.info(f"Collected {len(all_trades)} trades via yfinance.")
    return all_trades[:limit]

def get_market_context(ticker: str) -> dict:
    """
    Fetch market context (Sector, Current Price, Industry) using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0.0),
            "market_cap": info.get("marketCap", 0),
            "recommendation": info.get("recommendationKey", "none")
        }
    except Exception as e:
        logger.error(f"Error fetching context for {ticker}: {e}")
        return {"sector": "Unknown", "current_price": 0.0, "recommendation": "none"}

def get_stock_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    Fetch historical stock data for context and backtesting.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        logger.error(f"Error fetching history for {ticker}: {e}")
        return pd.DataFrame()
