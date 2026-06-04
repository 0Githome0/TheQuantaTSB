import pandas as pd
import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime, timedelta
from src.core.entry import AdvancedEntryStrategies, TimeFrames
from src.core.signal import SignalGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_entry')

# Create sample data for multiple timeframes
def create_sample_data(timeframe, length=100):
    # Create sample OHLCV data
    now = datetime.now()
    dates = [now - timedelta(minutes=i * timeframe_minutes(timeframe)) for i in range(length)]
    dates.reverse()
    
    # Simple uptrend data
    data = {
        'time': dates,
        'open': [100 + i for i in range(length)],
        'high': [100 + i + 2 for i in range(length)],
        'low': [100 + i - 1 for i in range(length)],
        'close': [100 + i + 1 for i in range(length)],
        'tick_volume': [1000 for _ in range(length)],
        'spread': [2 for _ in range(length)],
        'real_volume': [10000 for _ in range(length)]
    }
    return pd.DataFrame(data)

def timeframe_minutes(tf):
    if tf == TimeFrames.M15:
        return 15
    elif tf == TimeFrames.H1:
        return 60
    elif tf == TimeFrames.H4:
        return 240
    elif tf == TimeFrames.D1:
        return 1440
    return 60  # Default to H1

# Test function
def test_entry_strategies():
    logger.info("Testing Entry Strategies...")
    
    # Create signal generator
    signal_generator = SignalGenerator()
    
    # Create entry strategies
    entry_strategies = AdvancedEntryStrategies(signal_generator=signal_generator)
    
    # Try loading from config if it exists
    try:
        entry_strategies = AdvancedEntryStrategies.from_config('config/entry_config.json')
        logger.info("Loaded entry strategies from config")
    except Exception as e:
        logger.warning(f"Could not load from config: {e}")
    
    # Create sample data for multiple timeframes
    timeframes_data = {
        TimeFrames.M15: create_sample_data(TimeFrames.M15, 100),
        TimeFrames.H1: create_sample_data(TimeFrames.H1, 100),
        TimeFrames.H4: create_sample_data(TimeFrames.H4, 50),
        TimeFrames.D1: create_sample_data(TimeFrames.D1, 30)
    }
    
    # Test with single timeframe (simulating current Main.py behavior)
    logger.info("\nTesting with single timeframe (current implementation):")
    single_tf_data = {TimeFrames.H1: timeframes_data[TimeFrames.H1]}
    single_result = entry_strategies.analyze_entry_opportunity(
        pair="EURUSD",
        timeframes_data=single_tf_data,
        primary_timeframe=TimeFrames.H1
    )
    
    logger.info(f"Single timeframe result:")
    logger.info(f"  Action: {single_result.get('action')}")
    logger.info(f"  Best Strategy: {single_result.get('entry_recommendation', {}).get('best_strategy')}")
    logger.info(f"  Confidence: {single_result.get('entry_recommendation', {}).get('confidence')}")
    
    # Test with multiple timeframes (proper implementation)
    logger.info("\nTesting with multiple timeframes (proper implementation):")
    multi_result = entry_strategies.analyze_entry_opportunity(
        pair="EURUSD",
        timeframes_data=timeframes_data,
        primary_timeframe=TimeFrames.H1
    )
    
    logger.info(f"Multiple timeframe result:")
    logger.info(f"  Action: {multi_result.get('action')}")
    logger.info(f"  Best Strategy: {multi_result.get('entry_recommendation', {}).get('best_strategy')}")
    logger.info(f"  Confidence: {multi_result.get('entry_recommendation', {}).get('confidence')}")
    
    # Compare results
    logger.info("\nComparison:")
    if single_result.get('action') != multi_result.get('action'):
        logger.info("Different actions between single and multi-timeframe analysis!")
    
    if (single_result.get('entry_recommendation', {}).get('best_strategy') != 
        multi_result.get('entry_recommendation', {}).get('best_strategy')):
        logger.info("Different best strategies between single and multi-timeframe analysis!")
    
    if (single_result.get('entry_recommendation', {}).get('confidence') != 
        multi_result.get('entry_recommendation', {}).get('confidence')):
        logger.info("Different confidence levels between single and multi-timeframe analysis!")

if __name__ == "__main__":
    test_entry_strategies()