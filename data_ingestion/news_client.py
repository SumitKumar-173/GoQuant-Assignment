import requests
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncGenerator
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("news_client", "logs/news_client.log")

class NewsClient:
    """
    News API client for real-time news aggregation
    """
    def __init__(self):
        self.news_api_key = Config.NEWS_API_KEY
        self.alpha_vantage_key = Config.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://newsapi.org/v2"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        
        # Financial news sources to prioritize
        self.financial_sources = [
            "bloomberg", "reuters", "financial-times", "wall-street-journal",
            "cnbc", "marketwatch", "yahoo-finance", "business-insider",
            "fortune", "economist", "forbes", "the-economist"
        ]

    async def get_top_headlines(self, category: str = "business", language: str = "en", 
                               country: str = "us", page_size: int = 100) -> List[Dict]:
        """
        Get top financial headlines
        """
        try:
            params = {
                "category": category,
                "language": language,
                "country": country,
                "pageSize": page_size,
                "apiKey": self.news_api_key
            }
            
            response = requests.get(f"{self.base_url}/top-headlines", params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                article_data = {
                    "source": article.get('source', {}).get('name'),
                    "author": article.get('author'),
                    "title": article.get('title'),
                    "description": article.get('description'),
                    "url": article.get('url'),
                    "url_to_image": article.get('urlToImage'),
                    "published_at": article.get('publishedAt'),
                    "content": article.get('content'),
                    "sentiment_score": 0.0,  # Will be calculated later
                    "relevance_score": 0.0  # Will be calculated based on keywords
                }
                articles.append(article_data)
                
            return articles
        except Exception as e:
            logger.error(f"Error getting top headlines: {e}")
            return []

    async def search_news(self, query: str, language: str = "en", 
                         sort_by: str = "publishedAt", page_size: int = 100) -> List[Dict]:
        """
        Search for news articles based on query
        """
        try:
            params = {
                "q": query,
                "language": language,
                "sortBy": sort_by,
                "pageSize": page_size,
                "apiKey": self.news_api_key
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                article_data = {
                    "source": article.get('source', {}).get('name'),
                    "author": article.get('author'),
                    "title": article.get('title'),
                    "description": article.get('description'),
                    "url": article.get('url'),
                    "url_to_image": article.get('urlToImage'),
                    "published_at": article.get('publishedAt'),
                    "content": article.get('content'),
                    "sentiment_score": 0.0,  # Will be calculated later
                    "relevance_score": 0.0  # Will be calculated based on keywords
                }
                articles.append(article_data)
                
            return articles
        except Exception as e:
            logger.error(f"Error searching news: {e}")
            return []

    async def get_alpha_vantage_news_sentiment(self, tickers: List[str] = None) -> List[Dict]:
        """
        Get news sentiment from Alpha Vantage API
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not configured")
            return []
        
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ",".join(tickers) if tickers else "SPY,QQQ,IWM",
                "topics": "earnings,financial_markets,fiscal_policy,monetary_policy,major_news,sectors_and_industries",
                "limit": 1000,
                "apikey": self.alpha_vantage_key
            }
            
            response = requests.get(self.alpha_vantage_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            if "feed" in data:
                for item in data["feed"]:
                    article_data = {
                        "title": item.get("title"),
                        "summary": item.get("summary"),
                        "url": item.get("url"),
                        "time_published": item.get("time_published"),
                        "banner_image": item.get("banner_image"),
                        "source": item.get("source"),
                        "category_within_source": item.get("category_within_source"),
                        "source_domain": item.get("source_domain"),
                        "sentiment_score_definition": item.get("sentiment_score_definition"),
                        "relevance_score": float(item.get("relevance_score", 0)),
                        "sentiment_score": float(item.get("overall_sentiment_score", 0)),
                        "sentiment_labels": item.get("ticker_sentiment", []),
                        "topics": item.get("topics", [])
                    }
                    articles.append(article_data)
                    
            return articles
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage news sentiment: {e}")
            return []

    def filter_financial_news(self, articles: List[Dict]) -> List[Dict]:
        """
        Filter news articles for financial relevance
        """
        financial_keywords = [
            'stock', 'market', 'trading', 'invest', 'economy', 'economic',
            'earnings', 'revenue', 'profit', 'loss', 'bull', 'bear',
            'Federal Reserve', 'FED', 'inflation', 'interest rate', 'GDP',
            'cryptocurrency', 'bitcoin', 'ethereum', 'defi', 'blockchain',
            'merger', 'acquisition', 'ipo', 'dividend', 'yield', 'pe ratio'
        ]
        
        filtered_articles = []
        for article in articles:
            title_lower = (article.get('title') or '').lower()
            desc_lower = (article.get('description') or '').lower()
            content_lower = (article.get('content') or '').lower()
            
            # Calculate relevance score based on financial keywords
            relevance_score = 0
            for keyword in financial_keywords:
                if keyword in title_lower:
                    relevance_score += 2  # Title matches worth more
                if keyword in desc_lower:
                    relevance_score += 1
                if keyword in content_lower:
                    relevance_score += 1
            
            # Check if from financial source
            source = article.get('source', '').lower()
            is_financial_source = any(fin_source in source for fin_source in self.financial_sources)
            if is_financial_source:
                relevance_score += 3  # Financial sources get extra weight
            
            # Only include articles with some financial relevance
            if relevance_score > 0:
                article['relevance_score'] = relevance_score
                # Normalize relevance score to 0-1 range
                article['relevance_score_normalized'] = min(1.0, relevance_score / 10.0)
                filtered_articles.append(article)
        
        return filtered_articles

    async def get_recent_financial_news(self, hours: int = 24) -> List[Dict]:
        """
        Get financial news from the last N hours
        """
        from datetime import datetime, timedelta
        
        # Calculate time range
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Get news for major financial topics
        queries = ["financial markets", "stock market", "economy", "trading", "investing", 
                  "cryptocurrency", "bitcoin", "Federal Reserve", "inflation"]
        
        all_articles = []
        for query in queries:
            try:
                articles = await self.search_news(query, page_size=50)
                # Filter for time range
                recent_articles = []
                for article in articles:
                    pub_time_str = article.get('published_at')
                    if pub_time_str:
                        pub_time = datetime.fromisoformat(pub_time_str.replace('Z', '+00:00'))
                        if pub_time.replace(tzinfo=None) > start_time:
                            recent_articles.append(article)
                
                all_articles.extend(recent_articles)
            except Exception as e:
                logger.error(f"Error getting recent news for query {query}: {e}")
        
        # Remove duplicates
        unique_articles = []
        seen_urls = set()
        for article in all_articles:
            url = article.get('url')
            if url and url not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(url)
        
        return unique_articles