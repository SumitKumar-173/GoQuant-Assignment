import praw
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, AsyncGenerator
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("reddit_client", "logs/reddit_client.log")

class RedditClient:
    """
    Reddit API client for real-time data ingestion
    """
    def __init__(self):
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id=Config.REDDIT_CLIENT_ID,
            client_secret=Config.REDDIT_CLIENT_SECRET,
            user_agent=Config.REDDIT_USER_AGENT
        )
        
        # Financial subreddits to monitor
        self.financial_subreddits = [
            "cryptocurrency", "CryptoCurrency", "Bitcoin", "Ethereum", 
            "wallstreetbets", "stocks", "investing", "Finance", 
            "economy", "personalfinance", "options", "pennystocks",
            "algotrading", "Daytrading", "Trading"
        ]
        
        self.subreddits = [self.reddit.subreddit(name) for name in self.financial_subreddits]

    async def stream_posts(self, limit: int = 100) -> AsyncGenerator[Dict, None]:
        """
        Stream new posts from financial subreddits
        """
        for subreddit in self.subreddits:
            try:
                for submission in subreddit.new(limit=limit):
                    post_data = {
                        "id": submission.id,
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "url": submission.url,
                        "created_utc": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                        "subreddit": str(submission.subreddit),
                        "author": str(submission.author) if submission.author else "[deleted]",
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "ups": submission.ups,
                        "downs": submission.downs,
                        "permalink": f"https://reddit.com{submission.permalink}",
                        "is_self": submission.is_self
                    }
                    
                    yield post_data
                    await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
            except Exception as e:
                logger.error(f"Error streaming posts from {subreddit.display_name}: {e}")

    async def stream_comments(self, subreddit_name: str = "cryptocurrency", limit: int = 100) -> AsyncGenerator[Dict, None]:
        """
        Stream new comments from a specific subreddit
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        try:
            for comment in subreddit.stream.comments(limit=limit):
                comment_data = {
                    "id": comment.id,
                    "body": comment.body,
                    "created_utc": datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                    "subreddit": str(comment.subreddit),
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "score": comment.score,
                    "ups": comment.ups,
                    "downs": comment.downs,
                    "permalink": f"https://reddit.com{comment.permalink}",
                    "parent_id": comment.parent_id,
                    "is_submitter": comment.is_submitter
                }
                
                yield comment_data
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
        except Exception as e:
            logger.error(f"Error streaming comments from {subreddit_name}: {e}")

    def search_posts(self, query: str, subreddit: str = "all", limit: int = 25) -> List[Dict]:
        """
        Search for posts based on query
        """
        try:
            search_results = self.reddit.subreddit(subreddit).search(query, limit=limit)
            posts = []
            
            for submission in search_results:
                post_data = {
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "created_utc": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                    "subreddit": str(submission.subreddit),
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "ups": submission.ups,
                    "downs": submission.downs,
                    "permalink": f"https://reddit.com{submission.permalink}",
                    "is_self": submission.is_self
                }
                posts.append(post_data)
                
            return posts
        except Exception as e:
            logger.error(f"Error searching posts: {e}")
            return []

    def get_trending_topics(self, subreddit_name: str = "cryptocurrency", time_filter: str = "day") -> List[Dict]:
        """
        Get trending topics from a subreddit
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot posts
            hot_posts = subreddit.hot(limit=50)
            trending_posts = []
            
            for submission in hot_posts:
                post_data = {
                    "id": submission.id,
                    "title": submission.title,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                    "subreddit": str(submission.subreddit),
                    "author": str(submission.author) if submission.author else "[deleted]"
                }
                trending_posts.append(post_data)
                
            return trending_posts
        except Exception as e:
            logger.error(f"Error getting trending topics from {subreddit_name}: {e}")
            return []

    def filter_financial_content(self, posts: List[Dict]) -> List[Dict]:
        """
        Filter posts for financial relevance
        """
        financial_keywords = [
            'crypto', 'bitcoin', 'ethereum', 'btc', 'eth', 'defi',
            'stock', 'market', 'trading', 'invest', 'bull', 'bear',
            'pump', 'dump', 'moon', 'hodl', 'fomo', 'fud', 'diamond hands',
            'paper hands', 'bagholder', 'rekt', 'to the moon', 'bag', 'p&l',
            'portfolio', 'dividend', 'yield', 'pe ratio', 'eps', 'ipo'
        ]
        
        filtered_posts = []
        for post in posts:
            title_lower = post['title'].lower()
            text_lower = post['selftext'].lower()
            
            # Check for financial keywords in title or text
            has_financial_content = (
                any(keyword in title_lower for keyword in financial_keywords) or
                any(keyword in text_lower for keyword in financial_keywords)
            )
            
            if has_financial_content:
                # Calculate engagement score
                engagement_score = (post['score'] * 0.7) + (post['num_comments'] * 0.3)
                
                # Calculate credibility score
                credibility_score = 0.5  # base score
                if post['author'] != '[deleted]':
                    # If we want to check author reputation, we'd need additional API calls
                    # For now, we'll just give a small boost for non-deleted authors
                    credibility_score += 0.1
                
                post['engagement_score'] = engagement_score
                post['credibility_score'] = credibility_score
                
                filtered_posts.append(post)
        
        return filtered_posts