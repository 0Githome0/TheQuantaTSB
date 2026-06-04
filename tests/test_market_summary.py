import asyncio
import logging
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.advanced_analysis import AdvancedAnalysis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMarketSummary")

async def test_market_summary():
    logger.info("Initializing AdvancedAnalysis...")
    analyzer = AdvancedAnalysis()
    
    pairs = ["EURUSD", "XAUUSD", "BTCUSD"]
    
    logger.info(f"Generating market summary for {pairs}...")
    summary = await analyzer.generate_market_summary(pairs)
    
    logger.info("\n" + "="*50)
    logger.info("MARKET SUMMARY")
    logger.info("="*50)
    logger.info(f"Timestamp: {summary.get('timestamp')}")
    logger.info(f"Fear & Greed Index: {summary.get('fear_greed_index')} ({summary.get('fear_greed_label')})")
    logger.info(f"Overall Market Sentiment: {summary.get('overall_market_sentiment')}")
    logger.info(f"Economic Events: {summary.get('economic_events_count')}")
    logger.info(f"Commodities: {summary.get('commodities')}")
    
    logger.info("\n--- Pair Sentiments ---")
    for pair, sentiment in summary.get('pairs_sentiment', {}).items():
        logger.info(f"{pair}: {sentiment.get('overall_sentiment')} (score: {sentiment.get('sentiment_score')}, news: {sentiment.get('news_count')})")
    
    logger.info("="*50)

if __name__ == "__main__":
    asyncio.run(test_market_summary())
