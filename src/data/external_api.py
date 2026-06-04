import yfinance as yf
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ExternalDataManager:
    """
    Manages data fetching from external APIs:
    - yfinance (News)
    - Financial Modeling Prep (Sentiment)
    - EODHD (Fundamentals, Commodities)
    - Alpha Vantage (Backup for Fundamentals/Commodities)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.api_keys = config.get("api_keys", {})
        self.fmp_key = self.api_keys.get("financial_modeling_prep")
        self.eodhd_key = self.api_keys.get("eodhd")
        self.av_key = self.api_keys.get("alpha_vantage")
        
        if not self.fmp_key or self.fmp_key == "YOUR_FMP_KEY":
            logger.warning("Financial Modeling Prep API key is missing or default.")
        if not self.eodhd_key:
            logger.warning("EODHD API key is missing.")

    def fetch_news_yfinance(self, ticker: str, limit: int = 5) -> List[Dict]:
        """
        Fetches news for a given ticker using yfinance.
        For forex pairs, uses related ETFs/indices that have news coverage.
        """
        try:
            # Map to yfinance-compatible ticker with news
            yf_ticker = self._map_to_yfinance_news(ticker)
            if not yf_ticker:
                logger.info(f"No yfinance news mapping for {ticker}, skipping.")
                return []
            
            logger.info(f"Fetching news for {yf_ticker} (mapped from {ticker}) via yfinance...")
            
            t = yf.Ticker(yf_ticker)
            news = t.news
            
            if not news:
                logger.warning(f"No news returned from yfinance for {yf_ticker}")
                return []
            
            formatted_news = []
            for item in news[:limit]:
                try:
                    # New yfinance format has nested 'content' structure
                    content = item.get('content', {})
                    
                    # Extract title - try multiple locations
                    title = content.get('title', '') or item.get('title', '')
                    
                    # Extract URL
                    canonical_url = content.get('canonicalUrl', {})
                    link = canonical_url.get('url', '') if isinstance(canonical_url, dict) else str(canonical_url)
                    if not link:
                        link = item.get('link', '')
                    
                    # Extract publish time
                    pub_date = content.get('pubDate', '')
                    if pub_date:
                        # pubDate in new format is already a string like "2024-12-04T10:00:00Z"
                        published = pub_date[:19].replace('T', ' ') if isinstance(pub_date, str) else 'N/A'
                    else:
                        pub_time = item.get('providerPublishTime', 0)
                        published = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M:%S') if pub_time else 'N/A'
                    
                    # Extract publisher/source
                    provider = content.get('provider', {})
                    publisher = provider.get('displayName', '') if isinstance(provider, dict) else str(provider)
                    if not publisher:
                        publisher = item.get('publisher', 'Unknown')
                    
                    if title:  # Only add if we have a title
                        formatted_news.append({
                            'title': title,
                            'url': link,
                            'published': published,
                            'source': publisher,
                            'summary': f"Related to {ticker}"
                        })
                        logger.debug(f"Added news: {title[:50]}...")
                except Exception as item_e:
                    logger.warning(f"Error parsing news item: {item_e}")
                    continue
            
            logger.info(f"Successfully fetched {len(formatted_news)} news items for {ticker}")
            return formatted_news
            
        except Exception as e:
            logger.error(f"Error fetching news from yfinance for {ticker}: {e}")
            return []

    def _map_to_yfinance_news(self, ticker: str) -> str:
        """
        Maps trading pairs to yfinance tickers that actually have news.
        Forex pairs don't have direct news, so we map to related assets.
        """
        # Crypto - works directly
        if "BTC" in ticker: return "BTC-USD"
        if "ETH" in ticker: return "ETH-USD"
        
        # Gold/Commodities - use ETFs that have news
        if "XAU" in ticker or "GOLD" in ticker: return "GLD"  # Gold ETF
        if "XAG" in ticker: return "SLV"  # Silver ETF
        
        # Forex - map to currency ETFs or major indices
        if "EUR" in ticker: return "FXE"  # Euro Currency ETF
        if "GBP" in ticker: return "FXB"  # British Pound ETF
        if "JPY" in ticker or "USDJPY" in ticker: return "FXY"  # Yen ETF
        if "AUD" in ticker: return "FXA"  # Australian Dollar ETF
        if "CAD" in ticker: return "FXC"  # Canadian Dollar ETF
        
        # Default - try as-is for stocks
        clean = ticker.replace('m', '')
        return clean if clean else None

    def fetch_sentiment_fmp(self, ticker: str) -> Dict:
        """
        Fetches sentiment data from Financial Modeling Prep.
        """
        if not self.fmp_key:
            return {}

        try:
            # Try trending sentiment endpoint
            url = f"https://financialmodelingprep.com/api/v4/social-sentiment/trending?apikey={self.fmp_key}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list):
                    return data[0] 
            else:
                logger.warning(f"FMP Sentiment API returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Error fetching sentiment from FMP for {ticker}: {e}")
        
        return {}

    def fetch_fundamentals_eodhd(self, ticker: str) -> Dict:
        """
        Fetches fundamental data from EODHD.
        """
        if not self.eodhd_key:
            return {}

        try:
            eod_ticker = self._map_to_eodhd(ticker)
            url = f"https://eodhd.com/api/fundamentals/{eod_ticker}?api_token={self.eodhd_key}&fmt=json"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("General", {})
            else:
                logger.warning(f"EODHD Fundamentals API returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Error fetching fundamentals from EODHD for {ticker}: {e}")
        
        return {}

    def fetch_commodities_eodhd(self, commodity_type: str) -> Dict:
        """
        Fetches commodity data from EODHD.
        """
        if not self.eodhd_key:
            return {}

        try:
            # Map commodity types to EODHD symbols
            ticker_map = {
                "WTI": "USO.US", # Using ETF as proxy if direct commodity not available easily
                "BRENT": "BNO.US",
                "GOLD": "GLD.US",
                "COPPER": "CPER.US"
            }
            
            eod_ticker = ticker_map.get(commodity_type, commodity_type)
            url = f"https://eodhd.com/api/real-time/{eod_ticker}?api_token={self.eodhd_key}&fmt=json"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data 
            else:
                logger.warning(f"EODHD Commodity API ({commodity_type}) returned {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching commodity {commodity_type} from EODHD: {e}")
        
        return {}

    def fetch_fundamentals_alpha(self, ticker: str) -> Dict:
        """
        Fetches fundamental data (Overview) from Alpha Vantage.
        """
        if not self.av_key:
            return {}

        try:
            av_ticker = self._map_to_alpha_vantage(ticker)
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={av_ticker}&apikey={self.av_key}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "Symbol" in data:
                    return data
            else:
                logger.warning(f"Alpha Vantage Fundamentals API returned {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching fundamentals from Alpha Vantage for {ticker}: {e}")
        
        return {}

    def fetch_commodities_alpha(self, commodity_type: str) -> Dict:
        """
        Fetches commodity data (e.g., WTI, BRENT, COPPER) from Alpha Vantage.
        """
        if not self.av_key:
            return {}

        try:
            url = f"https://www.alphavantage.co/query?function={commodity_type}&interval=monthly&apikey={self.av_key}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    return data["data"][0]
            else:
                logger.warning(f"Alpha Vantage Commodity API ({commodity_type}) returned {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching commodity {commodity_type} from Alpha Vantage: {e}")
        
        return {}

    def _map_to_yfinance(self, ticker: str) -> str:
        """Helper to map internal/MT5 symbols to yfinance symbols."""
        if "EURUSD" in ticker: return "EURUSD=X"
        if "GBPUSD" in ticker: return "GBPUSD=X"
        if "USDJPY" in ticker: return "JPY=X" 
        if "XAUUSD" in ticker: return "GC=F" 
        if "BTCUSD" in ticker: return "BTC-USD"
        return ticker.replace('m', '')

    def _map_to_fmp(self, ticker: str) -> str:
        """Helper to map to FMP symbols."""
        if "EURUSD" in ticker: return "EURUSD"
        if "XAUUSD" in ticker: return "XAUUSD"
        if "BTCUSD" in ticker: return "BTCUSD"
        return ticker.replace('m', '')

    def _map_to_eodhd(self, ticker: str) -> str:
        """Helper to map to EODHD symbols."""
        clean = ticker.replace('m', '')
        if "EURUSD" in clean: return "EURUSD.FOREX"
        if "GBPUSD" in clean: return "GBPUSD.FOREX"
        if "USDJPY" in clean: return "USDJPY.FOREX"
        if "XAUUSD" in clean: return "XAUUSD.FOREX"
        if "BTCUSD" in clean: return "BTC-USD.CC"
        return f"{clean}.US"

    def _map_to_alpha_vantage(self, ticker: str) -> str:
        """Helper to map to Alpha Vantage symbols."""
        if "EURUSD" in ticker: return "EURUSD"
        return ticker.replace('m', '')
