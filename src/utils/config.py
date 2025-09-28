import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the sentiment analysis trading system"""
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Twitter API
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
    TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")
    
    # Reddit API
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "sentiment_analysis:1.0")
    
    # News API
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    
    # Financial Data APIs
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
    IEX_CLOUD_TOKEN = os.getenv("IEX_CLOUD_TOKEN", "")
    
    # Database
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Application settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Data processing settings
    MAX_TEXT_LENGTH = 10000  # Maximum length of text to process
    SENTIMENT_THRESHOLD = 0.1  # Threshold for significant sentiment
    SIGNAL_STRENGTH_THRESHOLD = 0.7  # Threshold for generating trade signals
    
    # Time windows (in minutes)
    TIME_WINDOWS = {
        "1m": 1,
        "5m": 5, 
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440
    }