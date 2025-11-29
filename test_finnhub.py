import os
import asyncio
import finnhub
import json
from datetime import datetime, timedelta

# This script tests the Finnhub API integration for getting real financial news

async def test_finnhub_api():
    print("Testing Finnhub API for real financial news...")
    
    # Using hard-coded API key
    api_key = 'd08j89pr01qju5m6rfe0d08j89pr01qju5m6rfeg'
    print("Using API key directly...")
    
    try:
        # Initialize Finnhub client
        finnhub_client = finnhub.Client(api_key=api_key)
        
        # Test API with forex news
        print("Fetching forex news...")
        forex_news = finnhub_client.general_news('forex', min_id=0)
        
        if not forex_news:
            print("No forex news found. API might be working but returned no data.")
        else:
            print(f"SUCCESS! Received {len(forex_news)} forex news items.")
            print("\nSample news items:")
            
            # Display first 3 news items
            for i, item in enumerate(forex_news[:3]):
                headline = item.get('headline', 'No headline')
                source = item.get('source', 'Unknown source')
                url = item.get('url', 'No URL')
                
                # Convert timestamp to readable date if available
                date_str = "Unknown date"
                if 'datetime' in item:
                    date_str = datetime.fromtimestamp(item['datetime']).strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n{i+1}. {headline}")
                print(f"   Source: {source}")
                print(f"   Date: {date_str}")
                print(f"   URL: {url}")
        
        # Test API with crypto news
        print("\nFetching crypto news...")
        crypto_news = finnhub_client.general_news('crypto', min_id=0)
        
        if not crypto_news:
            print("No crypto news found. API might be working but returned no data.")
        else:
            print(f"SUCCESS! Received {len(crypto_news)} crypto news items.")
            print("\nSample crypto news:")
            
            # Display first 3 news items
            for i, item in enumerate(crypto_news[:3]):
                headline = item.get('headline', 'No headline')
                source = item.get('source', 'Unknown source')
                
                print(f"\n{i+1}. {headline}")
                print(f"   Source: {source}")
        
        print("\nFinnhub API test completed successfully!")
        
    except Exception as e:
        print(f"ERROR: Failed to connect to Finnhub API: {str(e)}")
        print("Make sure your API key is valid and correctly set.")

if __name__ == "__main__":
    # Run the async test function
    asyncio.run(test_finnhub_api()) 