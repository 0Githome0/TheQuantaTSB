# ANOXILAL News API

This API provides endpoints to fetch financial market news and perform sentiment analysis on news related to specific trading pairs.

## Installation

1. Ensure you have the required dependencies:

```bash
pip install flask aiohttp beautifulsoup4 nltk textblob
```

2. Make sure NLTK data is downloaded (required for sentiment analysis):

```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

## Running the API Server

Start the API server by running:

```bash
python api_server.py
```

By default, the server runs on `http://localhost:5000`.

## API Endpoints

### Status Check

Check if the API server is running:

```
GET /api/status
```

**Response:**
```json
{
  "status": "OK",
  "service": "ANOXILAL News API",
  "timestamp": 1686123456.789
}
```

### Fetch News

Fetch market news with optional filtering by trading pair:

```
GET /api/news?pair=EURUSD&timeout=20
```

**Parameters:**
- `pair` (optional): Trading pair to filter news for (e.g., "EURUSD")
- `timeout` (optional): Request timeout in seconds (default: 20)

**Response:**
```json
{
  "data": [
    {
      "title": "EUR/USD rebounds as economic data beats expectations",
      "url": "https://example.com/news/1",
      "published": "2023-06-07T09:45:00",
      "source": "Example News Source",
      "sentiment": {
        "vader_compound": 0.456,
        "textblob_polarity": 0.321,
        "textblob_subjectivity": 0.567,
        "combined_score": 0.389,
        "sentiment_label": "positive"
      }
    },
    ...
  ],
  "count": 15,
  "pair": "EURUSD",
  "status": "success"
}
```

### Sentiment Analysis

Analyze sentiment for a specific trading pair:

```
GET /api/sentiment?pair=EURUSD
```

**Parameters:**
- `pair` (required): Trading pair to analyze (e.g., "EURUSD")

**Response:**
```json
{
  "data": {
    "pair": "EURUSD",
    "sentiment_score": 0.235,
    "sentiment": "positive",
    "news_count": 18,
    "confidence": 0.623,
    "distribution": {
      "positive": 11,
      "neutral": 5,
      "negative": 2
    },
    "recent_news": [
      {
        "title": "EUR/USD rebounds as economic data beats expectations",
        "url": "https://example.com/news/1",
        "published": "2023-06-07T09:45:00",
        "source": "Example News Source",
        "sentiment": {
          "vader_compound": 0.456,
          "textblob_polarity": 0.321,
          "textblob_subjectivity": 0.567,
          "combined_score": 0.389,
          "sentiment_label": "positive"
        }
      },
      ...
    ]
  },
  "status": "success"
}
```

### Available News Sources

Get a list of available news sources:

```
GET /api/sources
```

**Response:**
```json
{
  "data": [
    {
      "name": "Reuters",
      "url": "https://www.reuters.com/markets/currencies"
    },
    ...
  ],
  "count": 7,
  "status": "success"
}
```

## Error Handling

All endpoints return a standardized error format:

```json
{
  "error": "Error message describing what went wrong",
  "status": "error"
}
```

## Using the Client Example

A Python client example is provided in `client_example.py` to demonstrate how to use the API:

```bash
# Check API status
python client_example.py --command status

# Fetch news for a specific pair
python client_example.py --command news --pair EURUSD

# Analyze sentiment for a specific pair
python client_example.py --command sentiment --pair EURUSD

# Get available news sources
python client_example.py --command sources

# Output in JSON format
python client_example.py --command news --pair EURUSD --output json
```

## Integration with Other Systems

The News API can be easily integrated with other systems by making HTTP requests to the endpoints. Example in Python:

```python
import requests

# Fetch news for EURUSD
response = requests.get("http://localhost:5000/api/news?pair=EURUSD")
news_data = response.json()

# Analyze sentiment for EURUSD
response = requests.get("http://localhost:5000/api/sentiment?pair=EURUSD")
sentiment_data = response.json()
```

## Limitations and Performance Considerations

- News fetching can take several seconds since it needs to fetch data from multiple external sources.
- The API implements caching to reduce load on the news sources and improve performance for repeated requests.
- Sentiment analysis is performed using simple NLP techniques (VADER and TextBlob) and may not be as accurate as specialized financial sentiment analysis tools.
- The API is designed for single-user usage and doesn't implement rate limiting or authentication.

## Dependencies

- Flask: Web framework
- aiohttp: Asynchronous HTTP client/server
- BeautifulSoup4: HTML parsing
- NLTK: Natural Language Processing toolkit for sentiment analysis
- TextBlob: Simplified text processing for sentiment analysis 