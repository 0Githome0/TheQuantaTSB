import asyncio
import logging
from advanced_analysis import AdvancedAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_news.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('test_news')

async def test_news_fetch():
    """Test the news fetching functionality in advanced_analysis.py"""
    logger.info("Starting news fetch test")
    
    try:
        # Initialize AdvancedAnalysis module
        analysis = AdvancedAnalysis()
        
        # Test without pair filter first
        logger.info("Fetching general market news")
        news = await analysis.fetch_market_news(timeout=10)  # Reduce timeout to 10 seconds
        
        logger.info(f"Found {len(news)} general news items")
        
        # Print first 5 news items
        for i, item in enumerate(news[:5]):
            logger.info(f"  {i+1}. {item.get('title')} ({item.get('source')})")
            
            # Print URL
            url = item.get('url', 'No URL')
            logger.info(f"     URL: {url}")
            
            # Print sentiment if available
            sentiment = item.get('sentiment', {})
            if sentiment:
                sentiment_label = sentiment.get('sentiment_label', 'unknown')
                sentiment_score = sentiment.get('combined_score', 0.0)
                logger.info(f"     Sentiment: {sentiment_label.upper()} ({sentiment_score:+.2f})")
        
        # Test with pair filter - only test one pair to reduce run time
        pairs = ["EURUSD"]
        
        for pair in pairs:
            logger.info(f"Fetching news for {pair}")
            pair_news = await analysis.fetch_market_news(pair=pair, timeout=10)  # Reduce timeout
            
            logger.info(f"Found {len(pair_news)} news items for {pair}")
            
            # Print first 3 news items for this pair
            for i, item in enumerate(pair_news[:3]):
                logger.info(f"  {i+1}. {item.get('title')} ({item.get('source')})")
                
                # Print URL
                url = item.get('url', 'No URL')
                logger.info(f"     URL: {url}")
    
        # Clean up
        await analysis.close()
        logger.info("Test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Run the test with timeout protection
        asyncio.run(test_news_fetch())
        logger.info("Test completed")
    except KeyboardInterrupt:
        logger.info("Test was interrupted by the user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}") 