import asyncio
import logging
import sys
import time
from advanced_analysis import AdvancedAnalysis

# Configure logging to send output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_calendar_test.log", mode='w'),  # Overwrite if exists
        logging.StreamHandler(sys.stdout)
    ]
)

async def test_advanced_calendar():
    """Test the AdvancedAnalysis economic calendar functionality"""
    print("=" * 80)
    print("Starting test of AdvancedAnalysis economic calendar")
    logging.info("Starting test of AdvancedAnalysis economic calendar")
    
    # Create an instance of AdvancedAnalysis
    analysis = AdvancedAnalysis()
    print("AdvancedAnalysis instance created")
    
    try:
        # Fetch the economic calendar
        print("Fetching economic calendar...")
        logging.info("Fetching economic calendar...")
        start_time = time.time()
        calendar_data = await analysis.fetch_economic_calendar(timeout=60)  # Use a longer timeout
        elapsed = time.time() - start_time
        print(f"Calendar fetch completed in {elapsed:.2f} seconds")
        logging.info(f"Calendar fetch completed in {elapsed:.2f} seconds")
        
        # Debug output of raw calendar data
        print(f"Calendar data type: {type(calendar_data)}")
        logging.info(f"Calendar data type: {type(calendar_data)}")
        
        if isinstance(calendar_data, dict):
            print(f"Calendar data keys: {calendar_data.keys()}")
            logging.info(f"Calendar data keys: {calendar_data.keys()}")
            
            # Check for error messages
            if calendar_data.get('error'):
                print(f"Error in calendar data: {calendar_data['error']}")
                logging.error(f"Error in calendar data: {calendar_data['error']}")
            
            # Process events if available
            events = calendar_data.get('events', [])
            print(f"Found {len(events)} economic events")
            logging.info(f"Found {len(events)} economic events")
            
            # Display event details
            for i, event in enumerate(events[:10]):  # Show up to 10 events
                print(f"Event {i+1}: {event.get('title')}")
                print(f"  Date: {event.get('date')}")
                print(f"  Country: {event.get('country')}")
                print(f"  Impact: {event.get('impact')} stars")
                print(f"  Time: {event.get('time')}")
                print(f"  Forecast: {event.get('forecast')}")
                print(f"  Previous: {event.get('previous')}")
                print(f"  Actual: {event.get('actual', 'N/A')}")
                print("---")
                
                logging.info(f"Event {i+1}: {event.get('title')}")
                logging.info(f"  Date: {event.get('date')}")
                logging.info(f"  Country: {event.get('country')}")
                logging.info(f"  Impact: {event.get('impact')} stars")
                logging.info(f"  Time: {event.get('time')}")
                logging.info(f"  Forecast: {event.get('forecast')}")
                logging.info(f"  Previous: {event.get('previous')}")
                logging.info(f"  Actual: {event.get('actual', 'N/A')}")
                logging.info("---")
        else:
            print(f"Calendar data is not a dictionary: {calendar_data}")
            logging.error(f"Calendar data is not a dictionary: {calendar_data}")
        
        # Test high impact events
        print("\nFetching high impact events...")
        logging.info("Fetching high impact events...")
        high_impact_events = await analysis.get_high_impact_events(days_ahead=3)
        print(f"Found {len(high_impact_events)} high impact events for the next 3 days")
        logging.info(f"Found {len(high_impact_events)} high impact events for the next 3 days")
        
        # Display high impact events
        for i, event in enumerate(high_impact_events[:5]):
            print(f"High Impact Event {i+1}: {event.get('title')}")
            print(f"  Date: {event.get('date')}")
            print(f"  Country: {event.get('country')}")
            print(f"  Impact: {event.get('impact')} stars")
            print("---")
            
            logging.info(f"High Impact Event {i+1}: {event.get('title')}")
            logging.info(f"  Date: {event.get('date')}")
            logging.info(f"  Country: {event.get('country')}")
            logging.info(f"  Impact: {event.get('impact')} stars")
            logging.info("---")
        
    except Exception as e:
        print(f"Exception during test: {e}")
        logging.error(f"Exception during test: {e}", exc_info=True)
    finally:
        # Clean up
        await analysis.close()
        print("Test completed")
        logging.info("Test completed")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_advanced_calendar()) 