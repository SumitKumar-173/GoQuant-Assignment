from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class SentimentData(BaseModel):
    """
    Model for sentiment data
    """
    text: str
    overall_sentiment: float
    confidence: float
    methods_used: List[str]
    detailed_results: Dict
    timestamp: datetime
    source: Optional[str] = None
    processed_at: Optional[datetime] = None

class Signal(BaseModel):
    """
    Model for trading signals
    """
    symbol: str
    signal_type: str  # BUY or SELL
    strength: float
    confidence: float
    sentiment_score: float
    position_size: float
    timestamp: datetime
    expiration_time: datetime
    strategy: str
    volume_factor: Optional[float] = None
    momentum: Optional[float] = None

class CorrelationData(BaseModel):
    """
    Model for correlation data
    """
    correlation: float
    p_value: float
    sample_size: int
    r_squared: float
    time_lag_minutes: Optional[int] = None

class MarketData(BaseModel):
    """
    Model for market data
    """
    symbol: str
    current_price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[float] = None
    history: Optional[List[Dict]] = None

class AnalysisResult(BaseModel):
    """
    Model for analysis results
    """
    symbol: str
    timestamp: datetime
    sentiment_analysis: Dict
    signals: Dict
    processing_time_seconds: float
    data_sources_count: Dict

class PerformanceMetrics(BaseModel):
    """
    Model for performance metrics
    """
    data_points_processed: int
    sentiment_analyses_performed: int
    signals_generated: int
    average_processing_time: float
    last_update: datetime
    active_symbols_count: int
    fear_greed_index: float

class BacktestResult(BaseModel):
    """
    Model for backtest results
    """
    symbol: str
    days_backtested: int
    total_signals_generated: int
    correct_predictions: int
    total_predictions: int
    accuracy: float
    total_simulated_return: float
    average_return_per_signal: float
    sharpe_ratio: float
    backtest_timestamp: datetime