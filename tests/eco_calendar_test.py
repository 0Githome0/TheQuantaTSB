import asyncio
import logging
from src.core.advanced_analysis import AdvancedAnalysis

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_calendar():
    """Test economic calendar functionality"""
    print("Testing economic calendar...")
    
    # Create an AdvancedAnalysis instance
    analysis = AdvancedAnalysis()
    
    try:
        # Fetch the economic calendar with extended timeout
        print("Fetching economic calendar...")
        calendar_data = await analysis.fetch_economic_calendar(timeout=60)
        
        # Check if the calendar data was returned correctly
        if isinstance(calendar_data, dict):
            print(f"Calendar data keys: {calendar_data.keys()}")
            
            if "error" in calendar_data and calendar_data["error"]:
                print(f"Error: {calendar_data['error']}")
            else:
                # Print the number of events found
                events = calendar_data.get("events", [])
                print(f"Found {len(events)} economic events")
                
                # Print some sample events
                for i, event in enumerate(events[:5]):  # Show first 5 events
                    print(f"Event {i+1}: {event.get('title')}")
                    print(f"  Country: {event.get('country')}")
                    print(f"  Date: {event.get('date')}")
                    print(f"  Time: {event.get('time')}")
                    print(f"  Impact: {event.get('impact')} stars")
                    print(f"  Forecast: {event.get('forecast', 'N/A')}")
                    print(f"  Previous: {event.get('previous', 'N/A')}")
                    print("---")
        else:
            print(f"Unexpected result type: {type(calendar_data)}")
    
    except Exception as e:
        print(f"Error testing calendar: {e}")
    finally:
        # Clean up
        await analysis.close()
        print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_calendar()) 