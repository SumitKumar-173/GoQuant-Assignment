import asyncio
import logging
import re
import string
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("sentiment_analyzer", "logs/sentiment_analyzer.log")

class SentimentAnalyzer:
    """
    Advanced sentiment analysis system for financial text data
    """
    def __init__(self):
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformer model for financial sentiment
        try:
            # Use a pre-trained financial sentiment model
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer
            )
            self.use_finbert = True
        except Exception as e:
            logger.warning(f"Could not load FinBERT model: {e}, using fallback methods")
            self.use_finbert = False
        
        # Initialize general sentiment analysis pipeline
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.use_transformer = True
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}, using fallback methods")
            self.use_transformer = False
        
        # Financial lexicon for domain-specific sentiment
        self.positive_financial_words = {
            'bull', 'bullish', 'up', 'gain', 'gains', 'gainful', 'gainfully', 'winner', 
            'winning', 'success', 'successful', 'prosper', 'prosperity', 'prosperous', 
            'surge', 'surging', 'rally', 'rallies', 'rocket', 'rocketing', 'moon', 
            'mooning', 'diamond', 'diamonds', 'diamond hands', 'hodl', 'hodler', 
            'profit', 'profits', 'profitable', 'bull market', 'appreciate', 
            'appreciation', 'rebound', 'rebounded', 'rebounding', 'upside', 'green',
            'green candle', 'call', 'calls', 'long', 'long position', 'buy', 'buying',
            'accumulation', 'accumulate', 'bull run', 'parabolic', 'lamb', 'lambbo',
            'gain', 'gains', 'growth', 'growing', 'strong', 'bullish', 'optimistic',
            'positive', 'upbeat', 'recovery', 'recovering', 'breakout', 'breaking out'
        }
        
        self.negative_financial_words = {
            'bear', 'bearish', 'down', 'loss', 'losses', 'lose', 'losing', 'loser', 
            'losing', 'failure', 'fail', 'failing', 'failed', 'crash', 'crashing', 
            'dump', 'dumping', 'red', 'red candle', 'fall', 'falling', 'fell', 
            'decrease', 'decreasing', 'decline', 'declining', 'drop', 'dropping', 
            'dip', 'dipping', 'plunge', 'plummet', 'panic', 'panicking', 'fear',
            'scared', 'scary', 'frightening', 'worried', 'worry', 'worrisome',
            'negative', 'downside', 'bad', 'terrible', 'awful', 'disappoint',
            'disappointing', 'disappointment', 'weak', 'weakness', 'pump and dump',
            'manipulation', 'fraud', 'scam', 'scammy', 'exit scam', 'rug pull',
            'debt', 'bankrupt', 'bankruptcy', 'short', 'short position', 'put', 'puts',
            'sell', 'selling', 'dumpster', 'bleeding', 'bleed', 'bleeder'
        }
        
        # Initialize Gemini API if available
        if Config.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                self.use_gemini = True
            except ImportError:
                logger.warning("Google Generative AI not installed, skipping Gemini integration")
                self.use_gemini = False
            except Exception as e:
                logger.warning(f"Could not initialize Gemini: {e}")
                self.use_gemini = False
        else:
            self.use_gemini = False

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags but keep the text
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def calculate_financial_sentiment_score(self, text: str) -> Dict:
        """
        Calculate financial-specific sentiment using custom lexicon
        """
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        positive_count = 0
        negative_count = 0
        
        for word in words:
            # Remove punctuation
            word = word.translate(str.maketrans('', '', string.punctuation))
            if word in self.positive_financial_words:
                positive_count += 1
            elif word in self.negative_financial_words:
                negative_count += 1
        
        # Calculate score between -1 and 1
        total_financial_words = positive_count + negative_count
        if total_financial_words == 0:
            return {"score": 0.0, "positive_count": 0, "negative_count": 0}
        
        score = (positive_count - negative_count) / total_financial_words
        return {
            "score": max(-1.0, min(1.0, score)),
            "positive_count": positive_count,
            "negative_count": negative_count
        }

    def analyze_with_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER
        """
        scores = self.vader_analyzer.polarity_scores(text)
        # VADER returns: neg, neu, pos, compound
        # Compound is from -1 (most negative) to 1 (most positive)
        return {
            "compound": scores['compound'],
            "positive": scores['pos'],
            "neutral": scores['neu'],
            "negative": scores['neg']
        }

    def analyze_with_transformer(self, text: str) -> Optional[Dict]:
        """
        Analyze sentiment using transformer model
        """
        if not self.use_transformer:
            return None
        
        try:
            result = self.sentiment_pipeline(text)
            # Map results to standard format
            label = result[0]['label']
            score = result[0]['score']
            
            if label == 'LABEL_0':  # Negative
                return {"sentiment": "negative", "score": -score}
            elif label == 'LABEL_1':  # Neutral
                return {"sentiment": "neutral", "score": 0.0}
            elif label == 'LABEL_2':  # Positive
                return {"sentiment": "positive", "score": score}
            else:
                return {"sentiment": label.lower(), "score": score}
        except Exception as e:
            logger.error(f"Error in transformer analysis: {e}")
            return None

    def analyze_with_finbert(self, text: str) -> Optional[Dict]:
        """
        Analyze sentiment using FinBERT model
        """
        if not self.use_finbert:
            return None
        
        try:
            result = self.finbert_pipeline(text)
            # FinBERT returns: positive, negative, neutral
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            if label == 'positive':
                return {"sentiment": "positive", "score": score}
            elif label == 'negative':
                return {"sentiment": "negative", "score": -score}
            elif label == 'neutral':
                return {"sentiment": "neutral", "score": 0.0}
            else:
                return {"sentiment": label, "score": score}
        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return None

    def analyze_with_gemini(self, text: str) -> Optional[Dict]:
        """
        Analyze sentiment using Gemini API
        """
        if not self.use_gemini:
            return None
        
        try:
            prompt = f"""
            Analyze the financial sentiment of the following text. 
            Provide a sentiment score between -1 (very negative) and 1 (very positive).
            Also identify any financial instruments mentioned.
            Text: {text}
            
            Respond in JSON format with:
            {{
              "sentiment_score": <number between -1 and 1>,
              "confidence": <number between 0 and 1>,
              "financial_instruments": [<list of mentioned financial instruments>],
              "sentiment_label": "positive|negative|neutral"
            }}
            """
            
            response = self.gemini_model.generate_content(prompt)
            import json
            # Parse the JSON response
            result = json.loads(response.text)
            return result
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {e}")
            return None

    async def analyze_sentiment(self, text: str, use_gemini: bool = True) -> Dict:
        """
        Comprehensive sentiment analysis using multiple methods
        """
        if not text or len(text.strip()) == 0:
            return {
                "text": text,
                "overall_sentiment": 0.0,
                "confidence": 0.0,
                "methods_used": [],
                "detailed_results": {}
            }
        
        results = {}
        
        # 1. TextBlob analysis
        try:
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity  # from -1 to 1
            results["textblob"] = textblob_score
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
        
        # 2. VADER analysis
        try:
            vader_results = self.analyze_with_vader(text)
            results["vader"] = vader_results['compound']
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
        
        # 3. Financial lexicon analysis
        try:
            financial_results = self.calculate_financial_sentiment_score(text)
            results["financial_lexicon"] = financial_results['score']
        except Exception as e:
            logger.error(f"Error in financial lexicon analysis: {e}")
        
        # 4. Transformer model analysis
        try:
            transformer_results = self.analyze_with_transformer(text)
            if transformer_results:
                results["transformer"] = transformer_results['score']
        except Exception as e:
            logger.error(f"Error in transformer analysis: {e}")
        
        # 5. FinBERT analysis
        try:
            finbert_results = self.analyze_with_finbert(text)
            if finbert_results:
                results["finbert"] = finbert_results['score']
        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
        
        # 6. Gemini analysis (if enabled)
        if use_gemini and self.use_gemini:
            try:
                gemini_results = self.analyze_with_gemini(text)
                if gemini_results:
                    results["gemini"] = gemini_results['sentiment_score']
            except Exception as e:
                logger.error(f"Error in Gemini analysis: {e}")
        
        # Combine all results
        valid_scores = [score for score in results.values() if isinstance(score, (int, float))]
        
        if not valid_scores:
            return {
                "text": text,
                "overall_sentiment": 0.0,
                "confidence": 0.0,
                "methods_used": list(results.keys()),
                "detailed_results": results
            }
        
        # Calculate weighted average - give higher weight to financial-specific models
        weights = {}
        for method in results.keys():
            if method in ['financial_lexicon', 'finbert', 'gemini']:
                weights[method] = 1.2  # Higher weight for financial models
            else:
                weights[method] = 1.0
        
        # Compute weighted average
        weighted_sum = 0
        total_weight = 0
        for method, score in results.items():
            if isinstance(score, (int, float)):
                weight = weights.get(method, 1.0)
                weighted_sum += score * weight
                total_weight += weight
        
        overall_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence based on consistency across methods
        if len(valid_scores) > 1:
            std_dev = np.std(valid_scores)
            confidence = max(0.0, 1.0 - std_dev)  # Lower std dev = higher confidence
        else:
            confidence = 0.5  # Default confidence if only one method worked
        
        return {
            "text": text,
            "overall_sentiment": max(-1.0, min(1.0, overall_sentiment)),
            "confidence": max(0.0, min(1.0, confidence)),
            "methods_used": list(results.keys()),
            "detailed_results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts
        """
        results = []
        for text in texts:
            result = await self.analyze_sentiment(text)
            results.append(result)
        return results

    def calculate_time_weighted_sentiment(self, sentiment_data: List[Dict]) -> float:
        """
        Calculate time-weighted sentiment from a series of sentiment scores
        More recent sentiment scores have higher weight
        """
        if not sentiment_data:
            return 0.0
        
        # Apply time decay function to each sentiment score
        now = datetime.utcnow()
        weighted_sum = 0.0
        total_weight = 0.0
        
        for item in sentiment_data:
            score = item.get('overall_sentiment', 0.0)
            timestamp_str = item.get('timestamp')
            
            if not timestamp_str:
                continue
                
            try:
                # Parse timestamp
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                # Calculate time difference in hours
                time_diff = (now - timestamp).total_seconds() / 3600.0
                
                # Apply exponential decay (more recent = higher weight)
                # Half life of 6 hours
                weight = np.exp(-time_diff / 6.0)
                
                weighted_sum += score * weight
                total_weight += weight
            except:
                # If timestamp parsing fails, use equal weight
                weighted_sum += score
                total_weight += 1
        
        if total_weight == 0:
            return 0.0
            
        return weighted_sum / total_weight

    def detect_sarcasm_and_irony(self, text: str) -> bool:
        """
        Basic sarcasm/irony detection (to be expanded in advanced version)
        """
        # Simple pattern matching for basic sarcasm indicators
        sarcasm_indicators = [
            "yeah right", "sure thing", "oh great", "fantastic", "wonderful", 
            "as if", "like that's", "not surprised", "oh really", "oh sure",
            "because that makes sense", "right", "totally", "obviously"
        ]
        
        text_lower = text.lower()
        for indicator in sarcasm_indicators:
            if indicator in text_lower:
                return True
        
        # Check for contrasting positive/negative sentiment in close proximity
        positive_indicators = ['great', 'amazing', 'fantastic', 'wonderful', 'perfect']
        negative_indicators = ['terrible', 'awful', 'horrible', 'disaster', 'horrendous']
        
        words = text_lower.split()
        for i, word in enumerate(words):
            if word in positive_indicators:
                # Check nearby words for contradiction
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                nearby_words = words[start:end]
                
                for neg_word in negative_indicators:
                    if neg_word in nearby_words:
                        return True
        
        return False

    async def analyze_sentiment_with_context(self, text: str, metadata: Dict = None) -> Dict:
        """
        Analyze sentiment while considering additional context
        """
        # Perform basic sentiment analysis
        result = await self.analyze_sentiment(text)
        
        # Apply context adjustments if metadata is provided
        if metadata:
            # Adjust for credibility of source
            credibility_score = metadata.get('credibility_score', 0.5)
            if credibility_score != 0.5:  # If different from default
                # Adjust sentiment based on source credibility
                result['overall_sentiment'] *= credibility_score
            
            # Adjust for presence of sarcasm/irony
            if self.detect_sarcasm_and_irony(text):
                # Reverse sentiment if sarcasm detected
                result['overall_sentiment'] *= -0.7
                result['sarcasm_detected'] = True
            else:
                result['sarcasm_detected'] = False
        
        return result