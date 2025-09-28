import pytest
import asyncio
from datetime import datetime
from src.engine import SentimentTradingEngine
from src.sentiment_analysis.analyzer import SentimentAnalyzer
from src.signal_generation.generator import SignalGenerator

class TestSentimentAnalysis:
    """Test class for sentiment analysis components"""
    
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_sentiment_analysis_basic(self):
        """Test basic sentiment analysis functionality"""
        text = "This stock is going to the moon! Bullish on this!"
        result = asyncio.run(self.analyzer.analyze_sentiment(text))
        
        assert 'overall_sentiment' in result
        assert 'confidence' in result
        assert isinstance(result['overall_sentiment'], float)
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['overall_sentiment'] > 0  # Should be positive
    
    def test_sentiment_analysis_negative(self):
        """Test sentiment analysis with negative text"""
        text = "This company is going bankrupt. Terrible investment."
        result = asyncio.run(self.analyzer.analyze_sentiment(text))
        
        assert isinstance(result['overall_sentiment'], float)
        assert result['overall_sentiment'] < 0  # Should be negative
    
    def test_sentiment_analysis_neutral(self):
        """Test sentiment analysis with neutral text"""
        text = "The stock price was $150 yesterday."
        result = asyncio.run(self.analyzer.analyze_sentiment(text))
        
        assert isinstance(result['overall_sentiment'], float)
        # Should be relatively close to 0
        assert -0.3 <= result['overall_sentiment'] <= 0.3

class TestSignalGeneration:
    """Test class for signal generation components"""
    
    def setup_method(self):
        self.generator = SignalGenerator()
    
    def test_signal_generation_basic(self):
        """Test basic signal generation"""
        sentiment_data = {
            'overall_sentiment': 0.6,
            'confidence': 0.8
        }
        
        signal = self.generator.generate_single_signal(sentiment_data)
        
        assert signal is not None
        assert signal['signal_type'] in ['BUY', 'SELL']
        assert 0.0 <= signal['strength'] <= 1.0
        assert signal['signal_type'] == 'BUY'  # Positive sentiment should be BUY
    
    def test_signal_generation_weak_sentiment(self):
        """Test signal generation with weak sentiment (should return None)"""
        sentiment_data = {
            'overall_sentiment': 0.05,  # Below threshold
            'confidence': 0.5
        }
        
        signal = self.generator.generate_single_signal(sentiment_data)
        
        assert signal is None  # Should be None due to weak sentiment
    
    def test_position_size_calculation(self):
        """Test position size calculation"""
        position_size = self.generator.calculate_position_size(0.8, 0.9)
        
        assert isinstance(position_size, float)
        assert 0.0 < position_size <= 0.05  # Max 5% of portfolio

class TestMainEngine:
    """Test class for main engine functionality"""
    
    def setup_method(self):
        self.engine = SentimentTradingEngine()
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly"""
        assert self.engine is not None
        assert not self.engine.is_running
        assert len(self.engine.active_symbols) == 0
    
    def test_add_remove_symbols(self):
        """Test adding and removing symbols"""
        # Add a symbol
        asyncio.run(self.engine.add_symbol("BTC-USD"))
        assert "BTC-USD" in self.engine.active_symbols
        
        # Remove the symbol
        asyncio.run(self.engine.remove_symbol("BTC-USD"))
        assert "BTC-USD" not in self.engine.active_symbols
    
    def test_metrics_initialization(self):
        """Test that metrics are properly initialized"""
        metrics = self.engine.metrics
        
        assert "data_points_processed" in metrics
        assert "sentiment_analyses_performed" in metrics
        assert "signals_generated" in metrics
        assert "average_processing_time" in metrics
        assert "last_update" in metrics

# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        self.engine = SentimentTradingEngine()
    
    def test_process_single_symbol(self):
        """Test end-to-end processing for a single symbol"""
        # Add a symbol
        asyncio.run(self.engine.add_symbol("AAPL"))
        
        # Process the symbol (this will attempt to get real data)
        result = asyncio.run(self.engine.process_symbol_data("AAPL"))
        
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "timestamp" in result
        # Note: This might result in an error if no data is available, which is expected in test environment
        # The important thing is that it returns a well-formed result
    
    def test_fear_greed_index(self):
        """Test fear and greed index calculation"""
        # This will return neutral if no symbols are being tracked
        fgi = asyncio.run(self.engine.get_fear_greed_index())
        
        assert isinstance(fgi, float)
        assert 0.0 <= fgi <= 100.0

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])