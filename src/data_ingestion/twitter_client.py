import tweepy
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, AsyncGenerator
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("twitter_client", "logs/twitter_client.log")

class TwitterClient:
    """
    Twitter API client for real-time data ingestion
    """
    def __init__(self):
        # Initialize Twitter API v2 client
        self.client = tweepy.Client(
            bearer_token=Config.TWITTER_BEARER_TOKEN,
            consumer_key=Config.TWITTER_API_KEY,
            consumer_secret=Config.TWITTER_API_SECRET,
            access_token=Config.TWITTER_ACCESS_TOKEN,
            access_token_secret=Config.TWITTER_ACCESS_TOKEN_SECRET,
            wait_on_rate_limit=True
        )
        
        # Set up streaming rules for financial content
        self.stream_rules = [
            "crypto OR bitcoin OR ethereum OR defi",
            "stock OR market OR trading OR invest",
            "sentiment OR fear OR greed OR bullish OR bearish"
        ]
        
        # Store active stream
        self.stream = None

    def setup_streaming_rules(self):
        """
        Set up streaming rules for financial content
        """
        try:
            # Get existing rules
            rules = self.client.get_rules()
            
            # Delete existing rules
            if rules.data:
                rule_ids = [rule.id for rule in rules.data]
                self.client.delete_rules(rule_ids)
            
            # Add new rules for financial content
            financial_keywords = [
                "crypto", "bitcoin", "ethereum", "btc", "eth", 
                "defi", "nft", "web3", "altcoin",
                "stock", "market", "trading", "invest", 
                "bullish", "bearish", "pump", "dump",
                "$AAPL", "$TSLA", "$MSFT", "$GOOGL", "$AMZN",
                "$BTC", "$ETH", "blockchain", "fintech"
            ]
            
            # Create rules for each keyword
            for keyword in financial_keywords:
                self.client.add_rules(tweepy.StreamRule(keyword))
                
            logger.info("Twitter stream rules set up successfully")
        except Exception as e:
            logger.error(f"Error setting up streaming rules: {e}")

    async def stream_tweets(self) -> AsyncGenerator[Dict, None]:
        """
        Stream tweets in real-time with financial relevance
        """
        try:
            # Connect to Twitter API v2 stream
            for tweet in tweepy.Paginator(
                self.client.search_stream,
                expansions=["author_id", "geo.place_id"],
                tweet_fields=["created_at", "author_id", "public_metrics", "context_annotations", "lang"],
                user_fields=["username", "verified", "followers_count"],
                place_fields=["full_name", "country"],
                max_results=100
            ).flatten(limit=10000):
                
                if tweet is not None:
                    # Process the tweet data
                    tweet_data = {
                        "id": tweet.id,
                        "text": tweet.text,
                        "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                        "author_id": tweet.author_id,
                        "lang": tweet.lang,
                        "public_metrics": tweet.public_metrics if hasattr(tweet, 'public_metrics') else {},
                        "possibly_sensitive": getattr(tweet, 'possibly_sensitive', False)
                    }
                    
                    yield tweet_data
                    await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                    
        except Exception as e:
            logger.error(f"Error in Twitter stream: {e}")
            # Reconnect logic could be added here

    def search_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search for historical tweets based on query
        """
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=["created_at", "author_id", "public_metrics", "context_annotations", "lang"],
                user_fields=["username", "verified", "followers_count"]
            )
            
            if not tweets.data:
                return []
                
            results = []
            for tweet in tweets.data:
                tweet_data = {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                    "author_id": tweet.author_id,
                    "lang": tweet.lang,
                    "public_metrics": tweet.public_metrics if hasattr(tweet, 'public_metrics') else {},
                    "possibly_sensitive": getattr(tweet, 'possibly_sensitive', False)
                }
                results.append(tweet_data)
                
            return results
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        Get user information by user ID
        """
        try:
            user = self.client.get_user(
                id=user_id,
                user_fields=["username", "verified", "followers_count", "description"]
            )
            
            if user.data:
                return {
                    "id": user.data.id,
                    "username": user.data.username,
                    "verified": user.data.verified,
                    "followers_count": user.data.followers_count,
                    "description": user.data.description
                }
            return None
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None

    def filter_financial_tweets(self, tweets: List[Dict]) -> List[Dict]:
        """
        Filter tweets for financial relevance
        """
        financial_keywords = [
            'crypto', 'bitcoin', 'ethereum', 'btc', 'eth', 'defi',
            'stock', 'market', 'trading', 'invest', 'bull', 'bear',
            'pump', 'dump', 'moon', 'hodl', 'fomo', 'fud', 'diamond hands',
            'paper hands', 'bagholder', 'rekt', 'to the moon'
        ]
        
        financial_symbols = [
            '$BTC', '$ETH', '$BNB', '$XRP', '$ADA', '$DOGE', '$DOT', '$UNI',
            '$LTC', '$LINK', '$MATIC', '$SOL', '$AVAX', '$ATOM', '$SHIB',
            '$AAPL', '$TSLA', '$MSFT', '$GOOGL', '$AMZN', '$NFLX', '$META'
        ]
        
        filtered_tweets = []
        for tweet in tweets:
            text_lower = tweet['text'].lower()
            
            # Check for financial keywords
            has_financial_content = any(keyword in text_lower for keyword in financial_keywords)
            
            # Check for financial symbols
            has_financial_symbol = any(symbol in text_lower or symbol.replace('$', '') in text_lower 
                                     for symbol in financial_symbols)
            
            if has_financial_content or has_financial_symbol:
                # Add credibility score based on user verification and followers
                user_info = self.get_user_info(tweet['author_id'])
                if user_info:
                    credibility_score = 0.5  # base score
                    if user_info.get('verified', False):
                        credibility_score += 0.3
                    # Scale followers to 0-0.2 range (max 1M followers = +0.2)
                    followers = user_info.get('followers_count', 0)
                    credibility_score += min(0.2, followers / 5_000_000)
                    
                    tweet['credibility_score'] = credibility_score
                    tweet['user_info'] = user_info
                
                filtered_tweets.append(tweet)
        
        return filtered_tweets