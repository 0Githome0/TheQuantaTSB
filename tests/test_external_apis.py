import asyncio
import logging
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.external_api import ExternalDataManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestExternalAPI")

def load_config():
    with open('config/config.json', 'r') as f:
        return json.load(f)

async def test_apis():
    config = load_config()
    edm = ExternalDataManager(config)
    
    ticker = "EURUSD"
    
    # Test yfinance
    logger.info(f"--- Testing yfinance for AAPL ---")
    news = edm.fetch_news_yfinance("AAPL")
    logger.info(f"Fetched {len(news)} news items.")
    if news:
        logger.info(f"Sample: {news[0]['title']}")

    # Test FMP
    logger.info(f"--- Testing FMP Sentiment for {ticker} ---")
    sentiment = edm.fetch_sentiment_fmp(ticker)
    logger.info(f"Sentiment: {sentiment}")

    # Test EODHD
    logger.info(f"--- Testing EODHD Fundamentals for AAPL.US ---")
    fund = edm.fetch_fundamentals_eodhd("AAPL.US")
    logger.info(f"Fundamentals keys: {list(fund.keys()) if fund else 'None'}")
    
    logger.info(f"--- Testing EODHD Commodities (WTI) ---")
    comm = edm.fetch_commodities_eodhd("WTI")
    logger.info(f"WTI Data: {comm}")

    # Test Alpha Vantage
    logger.info(f"--- Testing Alpha Vantage Fundamentals for AAPL ---")
    av_fund = edm.fetch_fundamentals_alpha("AAPL")
    logger.info(f"AV Fundamentals keys: {list(av_fund.keys())[:5] if av_fund else 'None'}")
    
    logger.info(f"--- Testing Alpha Vantage Commodities (WTI) ---")
    av_comm = edm.fetch_commodities_alpha("WTI")
    logger.info(f"AV WTI Data: {av_comm}")

if __name__ == "__main__":
    asyncio.run(test_apis())
