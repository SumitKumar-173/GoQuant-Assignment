import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.sentiment_analysis.analyzer import SentimentAnalyzer
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("sentiment_processor", "logs/sentiment_processor.log")

class SentimentProcessor:
    """
    Advanced sentiment processing system with aggregation, trending, and entity recognition
    """
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.time_windows = Config.TIME_WINDOWS
        
        # Store historical sentiment data
        self.sentiment_history = defaultdict(lambda: deque(maxlen=1000))  # Store up to 1000 data points per symbol
        
        # Track trending topics
        self.trending_topics = defaultdict(lambda: {"count": 0, "last_seen": None, "sentiment": 0.0})
        
        # For clustering similar content
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Store entity mentions (tickers, company names)
        self.entity_mentions = defaultdict(lambda: deque(maxlen=500))

    async def process_text_batch(self, texts: List[Dict]) -> List[Dict]:
        """
        Process a batch of text data with sentiment analysis and metadata
        """
        results = []
        
        for item in texts:
            text = item.get('text', item.get('title', item.get('selftext', '')))
            metadata = {k: v for k, v in item.items() if k not in ['text', 'title', 'selftext']}
            
            # Perform sentiment analysis with context
            sentiment_result = await self.analyzer.analyze_sentiment_with_context(text, metadata)
            
            # Combine with original metadata
            processed_item = {
                **item,
                **sentiment_result,
                "processed_at": datetime.utcnow().isoformat()
            }
            results.append(processed_item)
        
        return results

    def aggregate_sentiment_by_timeframe(self, data: List[Dict], timeframe: str = "1m") -> Dict:
        """
        Aggregate sentiment data by specified timeframe
        """
        if not data:
            return {}
        
        # Convert timeframe string to minutes
        minutes = self.time_windows.get(timeframe, 1)
        time_threshold = datetime.utcnow() - timedelta(minutes=minutes)
        
        # Filter data within timeframe
        recent_data = [
            item for item in data 
            if 'timestamp' in item and 
            datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')) > time_threshold
        ]
        
        if not recent_data:
            return {
                "timeframe": timeframe,
                "count": 0,
                "average_sentiment": 0.0,
                "sentiment_volatility": 0.0
            }
        
        # Calculate aggregate metrics
        sentiments = [item.get('overall_sentiment', 0.0) for item in recent_data]
        average_sentiment = sum(sentiments) / len(sentiments)
        sentiment_volatility = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        # Calculate sentiment momentum
        recent_sentiments = sentiments[-10:] if len(sentiments) >= 10 else sentiments
        if len(recent_sentiments) > 1:
            sentiment_momentum = recent_sentiments[-1] - recent_sentiments[0]
        else:
            sentiment_momentum = 0.0
        
        return {
            "timeframe": timeframe,
            "count": len(recent_data),
            "average_sentiment": average_sentiment,
            "sentiment_volatility": sentiment_volatility,
            "sentiment_momentum": sentiment_momentum,
            "max_sentiment": max(sentiments),
            "min_sentiment": min(sentiments),
            "positive_ratio": len([s for s in sentiments if s > 0]) / len(sentiments),
            "negative_ratio": len([s for s in sentiments if s < 0]) / len(sentiments)
        }

    def detect_trending_topics(self, texts: List[str], min_mentions: int = 3) -> List[Dict]:
        """
        Detect trending topics based on frequency and sentiment
        """
        # Simple keyword extraction (in a real system, you'd use NER or topic modeling)
        all_words = []
        for text in texts:
            words = text.lower().split()
            # Filter for potentially relevant words (could be enhanced with NER)
            filtered_words = [w for w in words if len(w) > 3 and w.isalpha()]
            all_words.extend(filtered_words)
        
        # Count word frequency
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Filter for trending topics
        trending = []
        for word, count in word_counts.items():
            if count >= min_mentions:
                trending.append({
                    "topic": word,
                    "mention_count": count,
                    "last_updated": datetime.utcnow().isoformat()
                })
        
        # Sort by mention count
        trending.sort(key=lambda x: x['mention_count'], reverse=True)
        return trending[:20]  # Return top 20 trending topics

    def extract_financial_entities(self, text: str) -> List[str]:
        """
        Extract financial entities (tickers, company names, etc.) from text
        """
        import re
        
        # Pattern for stock tickers (e.g., $AAPL, AAPL, $BTC, BTC-USD, etc.)
        ticker_pattern = r'\$?([A-Z]{1,5})(?:-[A-Z]{3,4})?|(?<=\s)([A-Z]{1,5})(?=\s|$)'
        tickers = re.findall(ticker_pattern, text)
        # Flatten the tuple results and remove empty strings
        tickers = [item for sublist in tickers for item in sublist if item]
        tickers = [t.upper() for t in tickers if len(t) >= 2]
        
        # Additional patterns could be added for company names, etc.
        # This is a simplified version
        return list(set(tickers))  # Remove duplicates

    async def calculate_fear_greed_index(self, sentiment_data: List[Dict]) -> float:
        """
        Calculate a fear and greed index based on sentiment data
        """
        if not sentiment_data:
            return 50.0  # Neutral
        
        # Calculate various components
        avg_sentiment = np.mean([d.get('overall_sentiment', 0) for d in sentiment_data])
        
        # Convert to 0-100 scale (0 = extreme fear, 100 = extreme greed)
        # Sentiment ranges from -1 to 1, so transform to 0-100
        fear_greed_index = (avg_sentiment + 1) * 50
        
        # Add other factors that could be incorporated:
        # - Volatility of sentiment scores
        # - Volume of data points
        # - Momentum of sentiment changes
        # - Source diversity
        
        return max(0.0, min(100.0, fear_greed_index))

    def detect_sentiment_momentum(self, sentiment_history: List[float], lookback_window: int = 10) -> Dict:
        """
        Detect changes in sentiment momentum
        """
        if len(sentiment_history) < lookback_window:
            return {
                "momentum": 0.0,
                "acceleration": 0.0,
                "trend_direction": "neutral",
                "significance": 0.0
            }
        
        recent_sentiment = sentiment_history[-lookback_window:]
        earlier_sentiment = sentiment_history[-lookback_window*2:-lookback_window]
        
        # Calculate recent vs earlier averages
        recent_avg = np.mean(recent_sentiment)
        earlier_avg = np.mean(earlier_sentiment) if earlier_sentiment else 0.0
        
        # Calculate momentum (change in average sentiment)
        momentum = recent_avg - earlier_avg
        
        # Calculate acceleration (change in momentum)
        if len(sentiment_history) >= lookback_window * 3:
            earlier_earlier_sentiment = sentiment_history[-lookback_window*3:-lookback_window*2]
            earlier_earlier_avg = np.mean(earlier_earlier_sentiment) if earlier_earlier_sentiment else 0.0
            acceleration = (recent_avg - earlier_avg) - (earlier_avg - earlier_earlier_avg)
        else:
            acceleration = 0.0
        
        # Determine trend direction
        if momentum > 0.1:
            trend_direction = "increasing"
        elif momentum < -0.1:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Calculate significance (statistical significance of change)
        if len(earlier_sentiment) > 1 and len(recent_sentiment) > 1:
            pooled_std = np.sqrt(
                ((len(earlier_sentiment) - 1) * np.var(earlier_sentiment) + 
                 (len(recent_sentiment) - 1) * np.var(recent_sentiment)) /
                (len(earlier_sentiment) + len(recent_sentiment) - 2)
            )
            if pooled_std > 0:
                significance = abs(momentum) / pooled_std
            else:
                significance = 0.0
        else:
            significance = 0.0
        
        return {
            "momentum": momentum,
            "acceleration": acceleration,
            "trend_direction": trend_direction,
            "significance": significance
        }

    def cluster_similar_content(self, texts: List[str], n_clusters: int = 5) -> List[Dict]:
        """
        Cluster similar content using TF-IDF and K-means
        """
        if len(texts) < n_clusters:
            n_clusters = len(texts)
        
        if n_clusters <= 0:
            return []
        
        try:
            # Vectorize the texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group texts by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append({
                    "text": texts[i],
                    "index": i
                })
            
            # Calculate cluster statistics
            cluster_results = []
            for cluster_id, items in clusters.items():
                cluster_sentiments = [item.get('overall_sentiment', 0) for item in items if 'overall_sentiment' in item]
                avg_sentiment = np.mean(cluster_sentiments) if cluster_sentiments else 0.0
                
                cluster_results.append({
                    "cluster_id": cluster_id,
                    "item_count": len(items),
                    "average_sentiment": avg_sentiment,
                    "items": items[:5]  # Include only first 5 items for brevity
                })
            
            return cluster_results
        except Exception as e:
            logger.error(f"Error in content clustering: {e}")
            return []

    async def process_and_store_sentiment(self, data: List[Dict], symbol: str = None) -> Dict:
        """
        Process sentiment data and store history for trend analysis
        """
        # Process each item in the batch
        processed_data = await self.process_text_batch(data)
        
        # Store in history if symbol is provided
        if symbol:
            for item in processed_data:
                self.sentiment_history[symbol].append(item)
        
        # Aggregate by different timeframes
        aggregations = {}
        for timeframe in ['1m', '5m', '15m', '1h', '4h', '1d']:
            agg_result = self.aggregate_sentiment_by_timeframe(processed_data, timeframe)
            aggregations[timeframe] = agg_result
        
        # Calculate fear and greed index
        fear_greed_index = await self.calculate_fear_greed_index(processed_data)
        
        # Detect trending topics
        texts = [item.get('text', '') for item in processed_data]
        trending_topics = self.detect_trending_topics(texts)
        
        # Extract financial entities
        all_entities = []
        for item in processed_data:
            text = item.get('text', item.get('title', ''))
            entities = self.extract_financial_entities(text)
            all_entities.extend(entities)
            # Store entities by symbol
            for entity in entities:
                self.entity_mentions[entity].append(item)
        
        # Remove duplicates
        unique_entities = list(set(all_entities))
        
        return {
            "processed_count": len(processed_data),
            "aggregations": aggregations,
            "fear_greed_index": fear_greed_index,
            "trending_topics": trending_topics,
            "financial_entities": unique_entities,
            "processing_timestamp": datetime.utcnow().isoformat()
        }

    def get_historical_sentiment(self, symbol: str, timeframe_hours: int = 24) -> List[Dict]:
        """
        Get historical sentiment for a specific symbol
        """
        if symbol not in self.sentiment_history:
            return []
        
        time_threshold = datetime.utcnow() - timedelta(hours=timeframe_hours)
        historical_data = []
        
        for item in self.sentiment_history[symbol]:
            if 'processed_at' in item:
                try:
                    item_time = datetime.fromisoformat(item['processed_at'].replace('Z', '+00:00'))
                    if item_time > time_threshold:
                        historical_data.append(item)
                except:
                    continue
        
        return historical_data

    def calculate_correlation_with_market(self, symbol: str, market_data: Dict) -> Dict:
        """
        Calculate correlation between sentiment and market data
        """
        historical_sentiment = self.get_historical_sentiment(symbol, 24)  # Last 24 hours
        
        if not historical_sentiment or not market_data:
            return {"correlation": 0.0, "significance": 0.0, "sample_size": 0}
        
        # Align sentiment and market data by time
        sentiment_scores = []
        price_changes = []
        
        for sent_item in historical_sentiment:
            sent_time = datetime.fromisoformat(sent_item['processed_at'].replace('Z', '+00:00'))
            
            # Find closest matching market data point
            closest_price_change = None
            min_time_diff = timedelta(hours=1)  # Maximum accepted time difference
            
            # This is simplified - in a real system, you'd match with actual market timestamps
            if 'current_price' in market_data and 'history' in market_data:
                # Calculate price changes from market history
                history = market_data['history']
                if len(history) > 1:
                    # This is a simplified approach
                    # In practice, you'd align timestamps properly
                    if len(price_changes) < len(sentiment_scores):
                        price_changes.append(0.001)  # Placeholder
                        
            sentiment_scores.append(sent_item.get('overall_sentiment', 0.0))
        
        # Calculate correlation (simplified implementation)
        if len(sentiment_scores) > 1 and len(price_changes) == len(sentiment_scores):
            correlation = np.corrcoef(sentiment_scores, price_changes)[0, 1]
            return {
                "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                "significance": min(1.0, len(sentiment_scores) / 10.0),  # Simple significance measure
                "sample_size": len(sentiment_scores)
            }
        else:
            return {"correlation": 0.0, "significance": 0.0, "sample_size": len(sentiment_scores)}

    def detect_change_points(self, sentiment_series: List[float], min_magnitude: float = 0.2) -> List[Dict]:
        """
        Detect significant change points in sentiment time series
        """
        change_points = []
        
        if len(sentiment_series) < 3:
            return change_points
        
        for i in range(1, len(sentiment_series) - 1):
            prev_val = sentiment_series[i - 1]
            curr_val = sentiment_series[i]
            next_val = sentiment_series[i + 1]
            
            # Detect significant changes
            change_magnitude = abs(curr_val - prev_val)
            
            # Look for peaks (local maxima) and valleys (local minima)
            is_peak = (curr_val > prev_val and curr_val > next_val)
            is_valley = (curr_val < prev_val and curr_val < next_val)
            
            if (is_peak or is_valley) and change_magnitude >= min_magnitude:
                change_points.append({
                    "index": i,
                    "value": curr_val,
                    "change_magnitude": change_magnitude,
                    "type": "peak" if is_peak else "valley",
                    "timestamp": datetime.utcnow().isoformat()  # This would be actual timestamp in real implementation
                })
        
        return change_points