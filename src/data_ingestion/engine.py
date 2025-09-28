import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, AsyncGenerator
from src.data_ingestion.twitter_client import TwitterClient
from src.data_ingestion.reddit_client import RedditClient
from src.data_ingestion.news_client import NewsClient
from src.data_ingestion.financial_data_client import FinancialDataClient
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("data_ingestion_engine", "logs/data_ingestion_engine.log")

class DataIngestionEngine:
    """
    Main data ingestion engine that aggregates real-time data from multiple sources
    """
    def __init__(self):
        self.twitter_client = TwitterClient()
        self.reddit_client = RedditClient()
        self.news_client = NewsClient()
        self.financial_client = FinancialDataClient()
        
        # Track the last processed timestamps
        self.last_twitter_time = datetime.utcnow()
        self.last_reddit_time = datetime.utcnow()
        self.last_news_time = datetime.utcnow()

    async def start_streaming(self):
        """
        Start streaming data from all sources concurrently
        """
        logger.info("Starting data ingestion engine...")
        
        # Create tasks for each data source
        tasks = [
            self.stream_twitter_data(),
            self.stream_reddit_data(),
            self.stream_news_data(),
            self.stream_financial_data()
        ]
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)

    async def stream_twitter_data(self):
        """
        Stream Twitter data continuously
        """
        logger.info("Starting Twitter data stream...")
        
        try:
            async for tweet_data in self.twitter_client.stream_tweets():
                # Process and store tweet data
                processed_tweet = await self.process_tweet(tweet_data)
                if processed_tweet:
                    # Here we would typically send to a queue for further processing
                    logger.debug(f"Processed tweet: {processed_tweet['id']}")
                    yield processed_tweet
        except Exception as e:
            logger.error(f"Error in Twitter stream: {e}")

    async def stream_reddit_data(self):
        """
        Stream Reddit data continuously
        """
        logger.info("Starting Reddit data stream...")
        
        try:
            # Stream posts from multiple subreddits
            async for post_data in self.reddit_client.stream_posts(limit=50):
                processed_post = await self.process_reddit_post(post_data)
                if processed_post:
                    logger.debug(f"Processed Reddit post: {processed_post['id']}")
                    yield processed_post
        except Exception as e:
            logger.error(f"Error in Reddit stream: {e}")

    async def stream_news_data(self):
        """
        Stream news data continuously
        """
        logger.info("Starting News data stream...")
        
        try:
            # Get latest news headlines
            headlines = await self.news_client.get_top_headlines()
            for article in headlines:
                processed_article = await self.process_news_article(article)
                if processed_article:
                    logger.debug(f"Processed news article: {processed_article['title'][:50]}...")
                    yield processed_article
        except Exception as e:
            logger.error(f"Error in News stream: {e}")

    async def stream_financial_data(self):
        """
        Stream financial market data continuously
        """
        logger.info("Starting Financial data stream...")
        
        try:
            # Get key financial indicators
            indicators = await self.financial_client.get_market_sentiment_indicators()
            if indicators:
                logger.debug("Processed market sentiment indicators")
                yield indicators
                
            # Get sector performance
            sectors = await self.financial_client.get_sector_performance()
            if sectors:
                logger.debug("Processed sector performance data")
                yield sectors
        except Exception as e:
            logger.error(f"Error in Financial data stream: {e}")

    async def process_tweet(self, tweet_data: Dict) -> Optional[Dict]:
        """
        Process a tweet and extract relevant financial information
        """
        try:
            # Add ingestion timestamp
            tweet_data["ingested_at"] = datetime.utcnow().isoformat()
            
            # Add source identifier
            tweet_data["source"] = "twitter"
            
            # Calculate sentiment-related metrics
            tweet_data["text_length"] = len(tweet_data.get("text", ""))
            tweet_data["hashtag_count"] = tweet_data.get("text", "").count('#')
            tweet_data["mention_count"] = tweet_data.get("text", "").count('@')
            
            # Add credibility score if not already present
            if "credibility_score" not in tweet_data:
                tweet_data["credibility_score"] = 0.5  # Default score
                
            return tweet_data
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            return None

    async def process_reddit_post(self, post_data: Dict) -> Optional[Dict]:
        """
        Process a Reddit post and extract relevant financial information
        """
        try:
            # Add ingestion timestamp
            post_data["ingested_at"] = datetime.utcnow().isoformat()
            
            # Add source identifier
            post_data["source"] = "reddit"
            
            # Calculate engagement metrics
            upvote_ratio = post_data.get("ups", 0) / max(1, post_data.get("ups", 0) + post_data.get("downs", 0))
            post_data["upvote_ratio"] = upvote_ratio
            
            # Add credibility score if not already present
            if "credibility_score" not in post_data:
                post_data["credibility_score"] = 0.5  # Default score
                
            return post_data
        except Exception as e:
            logger.error(f"Error processing Reddit post: {e}")
            return None

    async def process_news_article(self, article_data: Dict) -> Optional[Dict]:
        """
        Process a news article and extract relevant financial information
        """
        try:
            # Add ingestion timestamp
            article_data["ingested_at"] = datetime.utcnow().isoformat()
            
            # Add source identifier
            article_data["source"] = "news"
            
            # Add credibility score based on source
            source = article_data.get("source", "").lower()
            financial_sources = [
                "bloomberg", "reuters", "financial-times", "wall-street-journal",
                "cnbc", "marketwatch", "yahoo-finance", "business-insider"
            ]
            
            credibility_score = 0.5  # Base score
            if any(fin_source in source for fin_source in financial_sources):
                credibility_score = 0.8  # High credibility for financial sources
                
            article_data["credibility_score"] = credibility_score
            
            return article_data
        except Exception as e:
            logger.error(f"Error processing news article: {e}")
            return None

    async def aggregate_sentiment_data(self, time_window_minutes: int = 5) -> Dict:
        """
        Aggregate sentiment data across all sources for a specific time window
        """
        try:
            # This would typically aggregate data from a database or message queue
            # For now, we'll simulate the aggregation process
            
            aggregated_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "time_window_minutes": time_window_minutes,
                "twitter_data": [],
                "reddit_data": [],
                "news_data": [],
                "financial_data": {},
                "total_items": 0,
                "average_sentiment": 0.0,
                "sentiment_volatility": 0.0
            }
            
            return aggregated_data
        except Exception as e:
            logger.error(f"Error aggregating sentiment data: {e}")
            return {}

    async def get_data_batch(self, symbols: List[str], time_window: str = "1m") -> Dict:
        """
        Get a batch of data for specific symbols within a time window
        """
        try:
            # Get social media and news data
            # Twitter
            twitter_financial_keywords = [f"${symbol}" for symbol in symbols] + symbols
            twitter_data = []
            for keyword in twitter_financial_keywords[:3]:  # Limit API calls
                tweets = self.twitter_client.search_tweets(f"{keyword} lang:en", max_results=20)
                filtered_tweets = self.twitter_client.filter_financial_tweets(tweets)
                twitter_data.extend(filtered_tweets)
            
            # Reddit
            reddit_data = []
            for symbol in symbols[:2]:  # Limit API calls
                posts = self.reddit_client.search_posts(symbol, limit=10)
                filtered_posts = self.reddit_client.filter_financial_content(posts)
                reddit_data.extend(filtered_posts)
                
            # News
            news_data = []
            for symbol in symbols[:2]:  # Limit API calls
                articles = await self.news_client.search_news(symbol, page_size=10)
                filtered_articles = self.news_client.filter_financial_news(articles)
                news_data.extend(filtered_articles)
            
            # Financial data
            financial_data = await self.financial_client.get_multiple_symbols_data(symbols)
            
            # Combine all data
            batch_data = {
                "symbols": symbols,
                "time_window": time_window,
                "timestamp": datetime.utcnow().isoformat(),
                "twitter_data": twitter_data,
                "reddit_data": reddit_data,
                "news_data": news_data,
                "financial_data": financial_data
            }
            
            return batch_data
        except Exception as e:
            logger.error(f"Error getting data batch: {e}")
            return {}

    def calculate_ingestion_metrics(self) -> Dict:
        """
        Calculate metrics about the data ingestion process
        """
        try:
            # In a real implementation, this would track actual metrics
            # from the ingestion pipeline
            metrics = {
                "total_ingested": 0,
                "ingestion_rate_per_minute": 0,
                "avg_processing_time_ms": 0,
                "error_rate": 0,
                "top_sources": [],
                "data_quality_score": 0.0,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating ingestion metrics: {e}")
            return {}