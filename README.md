# Sentiment Analysis and Trade Signal Generation System

This project implements a high-performance sentiment analysis and trade signal generation system that aggregates real-time data from Twitter, Reddit, news feeds, and financial data sources. The system analyzes market sentiment, correlates it with fund flows and market data, and generates actionable trade signals based on fear and greed indicators.

## Features

- Real-time sentiment analysis from multiple data sources
- Multi-source data ingestion engine
- Advanced NLP and machine learning models
- Real-time trade signal generation
- Performance analytics and backtesting
- Fear and greed index calculation

## Requirements

- Python 3.9+
- Virtual environment (recommended)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sentiment-analysis-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spacy model
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with the following variables:

- `GEMINI_API_KEY`: Your Gemini API key
- `TWITTER_BEARER_TOKEN`: Twitter API Bearer Token
- `REDDIT_CLIENT_ID`: Reddit API Client ID
- `NEWS_API_KEY`: News API key
- `POLYGON_API_KEY`: Polygon.io API key
- Other API keys as needed

## Usage

```bash
# Run the main application
python -m src.main
```

## Architecture

- `src/data_ingestion/`: Real-time data ingestion from multiple sources
- `src/sentiment_analysis/`: NLP and sentiment analysis engine
- `src/signal_generation/`: Trade signal generation system
- `src/models/`: ML models and predictive analytics
- `src/utils/`: Utility functions and helpers