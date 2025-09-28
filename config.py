import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

YFINANCE_CACHE_DIR = os.getenv('YFINANCE_CACHE_DIR', '.yfinance_cache')