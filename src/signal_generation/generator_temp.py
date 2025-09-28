import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger('signal_generator', 'logs/signal_generator.log')

class SignalGenerator:
    """
    Trade signal generation system based on sentiment analysis and market data
    """
    def __init__(self):
        self.config = Config
        self.signals_history = []
        
        # Signal thresholds
        self.sentiment_threshold = Config.SENTIMENT_THRESHOLD  # Default 0.1
        self.signal_strength_threshold = Config.SIGNAL_STRENGTH_THRESHOLD  # Default 0.7
        
        # Market state tracking
        self.market_regime = "neutral"  # bull, bear, neutral, volatile
        self.last_signal_time = datetime.utcnow()

    def calculate_signal_strength(self, sentiment_score: float, volume_factor: float = 1.0, 
                                confidence: float = 1.0, momentum: float = 0.0) -> float:
        """
        Calculate signal strength based on multiple factors
        """
        # Base strength from sentiment
        base_strength = abs(sentiment_score)
        
        # Apply volume factor (higher volume = stronger signal)
        volume_adjusted = base_strength * volume_factor
        
        # Apply confidence factor
        confidence_adjusted = volume_adjusted * confidence
        
        # Apply momentum factor
        momentum_factor = 1 + abs(momentum) * 0.5  # Momentum amplifies signal
        final_strength = confidence_adjusted * momentum_factor
        
        # Cap at 1.0
        return min(1.0, final_strength)

    def generate_single_signal(self, sentiment_data: Dict, market_data: Dict = None, 
                              symbol: str = None) -> Optional[Dict]:
        """
        Generate a single trade signal based on sentiment and market data
        """
        sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
        confidence = sentiment_data.get('confidence', 0.5)
        
        # Only generate signals when sentiment is significant
        if abs(sentiment_score) < self.sentiment_threshold:
            return None
        
        # Determine signal type
        signal_type = "BUY" if sentiment_score > 0 else "SELL"
        
        # Calculate additional factors if market data is available
        volume_factor = 1.0
        momentum = 0.0
        price_correlation = 0.0
        
        if market_data and symbol:
            # Calculate volume factor based on trading volume
            if 'volume' in market_data and market_data['volume']:
                avg_volume = market_data.get('avg_volume', 1000000)  # Default to 1M if not available
                current_volume = market_data['volume']
                volume_factor = max(0.5, min(2.0, current_volume / avg_volume))
            
            # Calculate momentum if historical price data is available
            if 'history' in market_data and len(market_data['history']) > 1:
                recent_prices = [p['Close'] for p in market_data['history'][-5:]]  # Last 5 data points
                if len(recent_prices) >= 2:
                    momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Calculate signal strength
        signal_strength = self.calculate_signal_strength(
            sentiment_score, volume_factor, confidence, momentum
        )
        
        # Only generate strong signals
        if signal_strength < self.signal_strength_threshold:
            return None
        
        # Calculate position size based on signal confidence and strength
        position_size = self.calculate_position_size(signal_strength, confidence)
        
        signal = {
            "symbol": symbol or "N/A",
            "signal_type": signal_type,
            "strength": signal_strength,
            "confidence": confidence,
            "sentiment_score": sentiment_score,
            "volume_factor": volume_factor,
            "momentum": momentum,
            "price_correlation": price_correlation,
            "position_size": position_size,
            "timestamp": datetime.utcnow().isoformat(),
            "expiration_time": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),  # 30-min expiration
            "strategy": "sentiment_based"
        }
        
        return signal

    def calculate_position_size(self, signal_strength: float, confidence: float) -> float:
        """
        Calculate position size based on signal strength and confidence
        """
        # Base position size is proportional to signal strength and confidence
        base_size = (signal_strength * confidence) * 0.1  # Max 10% of portfolio
        
        # Apply risk management
        max_position_size = 0.05  # Max 5% of portfolio per signal
        position_size = min(base_size, max_position_size)
        
        # Ensure minimum position size for strong signals
        if signal_strength > 0.9 and confidence > 0.8:
            position_size = max(position_size, 0.02)  # At least 2% for very strong signals
        
        return round(position_size, 4)

    def generate_multi_factor_signal(self, sentiment_data: Dict, market_data: Dict = None, 
                                   technical_indicators: Dict = None, symbol: str = None) -> Optional[Dict]:
        """
        Generate a multi-factor signal combining sentiment with technical indicators
        """
        # Start with basic signal generation
        basic_signal = self.generate_single_signal(sentiment_data, market_data, symbol)
        if not basic_signal:
            return None
        
        # Incorporate technical indicators if available
        if technical_indicators:
            # Adjust signal based on technical confirmation
            rsi = technical_indicators.get('rsi')
            macd = technical_indicators.get('macd')
            moving_avg = technical_indicators.get('moving_avg_trend')
            
            technical_confirmation = 0.0
            
            # RSI confirmation
            if basic_signal['signal_type'] == 'BUY' and rsi and rsi < 70 and rsi > 30:
                technical_confirmation += 0.1
            elif basic_signal['signal_type'] == 'SELL' and rsi and rsi > 30 and rsi < 70:
                technical_confirmation += 0.1
            
            # MACD confirmation
            if basic_signal['signal_type'] == 'BUY' and macd and macd > 0:
                technical_confirmation += 0.1
            elif basic_signal['signal_type'] == 'SELL' and macd and macd < 0:
                technical_confirmation += 0.1
            
            # Moving average trend confirmation
            if basic_signal['signal_type'] == 'BUY' and moving_avg and moving_avg > 0:
                technical_confirmation += 0.05
            elif basic_signal['signal_type'] == 'SELL' and moving_avg and moving_avg < 0:
                technical_confirmation += 0.05
            
            # Apply technical confirmation to signal strength
            enhanced_strength = min(1.0, basic_signal['strength'] + technical_confirmation)
            basic_signal['strength'] = enhanced_strength
            basic_signal['technical_confirmation'] = technical_confirmation
            basic_signal['strategy'] = "multi_factor"
        
        return basic_signal

    def detect_market_regime(self, sentiment_data: List[Dict], market_data: Dict = None) -> str:
        """
        Detect current market regime based on sentiment and market data
        """
        if not sentiment_data:
            return "neutral"
        
        # Analyze sentiment volatility
        sentiments = [item.get('overall_sentiment', 0.0) for item in sentiment_data]
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        sentiment_volatility = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        # Classify regime based on characteristics
        if sentiment_volatility > 0.5:
            return "volatile"
        elif avg_sentiment > 0.3:
            return "bullish"
        elif avg_sentiment < -0.3:
            return "bearish"
        else:
            return "neutral"

    def generate_contrarian_signals(self, sentiment_data: Dict, market_regime: str = "neutral") -> Optional[Dict]:
        """
        Generate contrarian signals when market sentiment is extreme
        """
        sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
        confidence = sentiment_data.get('confidence', 0.5)
        
        # Only generate contrarian signals for extreme sentiment
        extreme_threshold = 0.8
        if abs(sentiment_score) < extreme_threshold:
            return None
        
        # In a bullish regime, extreme positive sentiment might be a sell signal (greed)
        # In a bearish regime, extreme negative sentiment might be a buy signal (fear)
        if market_regime == "bullish" and sentiment_score > extreme_threshold:
            # Too much bullish sentiment - potential sell signal
            contrarian_signal = {
                "symbol": sentiment_data.get('symbol', 'N/A'),
                "signal_type": "SELL",
                "strength": min(0.8, abs(sentiment_score) * confidence),
                "confidence": confidence,
                "sentiment_score": sentiment_score,
                "contrarian": True,
                "reason": "extreme_bullish_sentiment_in_bull_market",
                "timestamp": datetime.utcnow().isoformat(),
                "expiration_time": (datetime.utcnow() + timedelta(minutes=60)).isoformat(),
                "strategy": "contrarian"
            }
            return contrarian_signal
        elif market_regime == "bearish" and sentiment_score < -extreme_threshold:
            # Too much bearish sentiment - potential buy signal
            contrarian_signal = {
                "symbol": sentiment_data.get('symbol', 'N/A'),
                "signal_type": "BUY",
                "strength": min(0.8, abs(sentiment_score) * confidence),
                "confidence": confidence,
                "sentiment_score": sentiment_score,
                "contrarian": True,
                "reason": "extreme_bearish_sentiment_in_bear_market",
                "timestamp": datetime.utcnow().isoformat(),
                "expiration_time": (datetime.utcnow() + timedelta(minutes=60)).isoformat(),
                "strategy": "contrarian"
            }
            return contrarian_signal
        
        return None

    def generate_correlation_signals(self, sentiment_data: Dict, price_changes: List[float], 
                                   correlation: float, symbol: str = None) -> Optional[Dict]:
        """
        Generate signals based on sentiment-price correlation
        """
        if abs(correlation) < 0.3:  # Weak correlation
            return None
        
        sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
        
        if correlation > 0.5 and sentiment_score > 0.3:
            # Strong positive correlation with positive sentiment = buy signal
            return {
                "symbol": symbol or "N/A",
                "signal_type": "BUY",
                "strength": min(0.9, correlation * abs(sentiment_score)),
                "confidence": sentiment_data.get('confidence', 0.5),
                "sentiment_score": sentiment_score,
                "price_correlation": correlation,
                "strategy": "correlation_based",
                "timestamp": datetime.utcnow().isoformat(),
                "expiration_time": (datetime.utcnow() + timedelta(minutes=45)).isoformat()
            }
        elif correlation < -0.5 and sentiment_score < -0.3:
            # Strong negative correlation with negative sentiment = sell signal
            return {
                "symbol": symbol or "N/A", 
                "signal_type": "SELL",
                "strength": min(0.9, abs(correlation) * abs(sentiment_score)),
                "confidence": sentiment_data.get('confidence', 0.5),
                "sentiment_score": sentiment_score,
                "price_correlation": correlation,
                "strategy": "correlation_based",
                "timestamp": datetime.utcnow().isoformat(),
                "expiration_time": (datetime.utcnow() + timedelta(minutes=45)).isoformat()
            }
        
        return None

    def generate_fear_greed_signals(self, fear_greed_index: float, symbol: str = None) -> Optional[Dict]:
        """
        Generate signals based on fear and greed index
        """
        if fear_greed_index < 20:  # Extreme fear
            # Potential buying opportunity
            return {
                "symbol": symbol or "N/A",
                "signal_type": "BUY",
                "strength": min(0.8, (40 - fear_greed_index) / 40),  # Inverse relationship
                "confidence": 0.7,  # High confidence in mean reversion at extremes
                "fear_greed_index": fear_greed_index,
                "strategy": "fear_greed",
                "timestamp": datetime.utcnow().isoformat(),
                "expiration_time": (datetime.utcnow() + timedelta(minutes=120)).isoformat()  # Longer expiration for mean reversion
            }
        elif fear_greed_index > 80:  # Extreme greed
            # Potential selling opportunity
            return {
                "symbol": symbol or "N/A",
                "signal_type": "SELL", 
                "strength": min(0.8, (fear_greed_index - 60) / 40),  # Direct relationship
                "confidence": 0.7,  # High confidence in mean reversion at extremes
                "fear_greed_index": fear_greed_index,
                "strategy": "fear_greed",
                "timestamp": datetime.utcnow().isoformat(),
                "expiration_time": (datetime.utcnow() + timedelta(minutes=120)).isoformat()
            }
        
        return None

    async def generate_signals_batch(self, sentiment_data_batch: List[Dict], 
                                   market_data: Dict = None, symbols: List[str] = None) -> List[Dict]:
        """
        Generate signals for a batch of sentiment data
        """
        signals = []
        
        for i, sentiment_data in enumerate(sentiment_data_batch):
            symbol = symbols[i] if symbols and i < len(symbols) else None
            
            # Generate primary signal
            primary_signal = self.generate_single_signal(sentiment_data, market_data, symbol)
            if primary_signal:
                signals.append(primary_signal)
            
            # Generate contrarian signal
            contrarian_signal = self.generate_contrarian_signals(sentiment_data, self.market_regime)
            if contrarian_signal:
                signals.append(contrarian_signal)
        
        return signals

    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate a signal before execution
        """
        if not signal:
            return False
        
        # Check expiration
        expiration_time = datetime.fromisoformat(signal['expiration_time'].replace('Z', '+00:00'))
        if datetime.utcnow() > expiration_time:
            return False
        
        # Check minimum strength
        if signal.get('strength', 0) < self.signal_strength_threshold:
            return False
        
        # Check for basic validity
        if signal.get('signal_type') not in ['BUY', 'SELL']:
            return False
        
        return True

    def apply_risk_management(self, signals: List[Dict], existing_positions: Dict = None) -> List[Dict]:
        """
        Apply risk management to generated signals
        """
        filtered_signals = []
        
        for signal in signals:
            # Skip if signal is invalid
            if not self.validate_signal(signal):
                continue
            
            # Risk management: limit number of signals per time period
            time_threshold = datetime.utcnow() - timedelta(minutes=5)
            recent_signals = [
                s for s in self.signals_history 
                if datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00')) > time_threshold
            ]
            
            # Don't generate too many signals in a short time
            if len(recent_signals) > 10:  # Max 10 signals per 5 minutes
                logger.warning("Too many signals generated recently, skipping signal")
                continue
            
            # Check for position overlap if existing positions are provided
            if existing_positions and signal['symbol'] in existing_positions:
                # Don't generate opposing signals for existing positions
                current_position = existing_positions[signal['symbol']]
                if (current_position['side'] == 'LONG' and signal['signal_type'] == 'SELL') or                    (current_position['side'] == 'SHORT' and signal['signal_type'] == 'BUY'):
                    continue
            
            filtered_signals.append(signal)
        
        # Update signals history
        self.signals_history.extend(filtered_signals)
        # Keep only recent history (last 1000 signals)
        self.signals_history = self.signals_history[-1000:]
        
        return filtered_signals

    def calculate_portfolio_impact(self, signals: List[Dict], portfolio_value: float) -> Dict:
        """
        Calculate potential portfolio impact of signals
        """
        total_exposure = 0
        for signal in signals:
            position_size = signal.get('position_size', 0)
            total_exposure += position_size
        
        max_single_position = max([s.get('position_size', 0) for s in signals], default=0) if signals else 0
        
        return {
            "total_exposure_percentage": total_exposure,
            "max_single_position_percentage": max_single_position,
            "total_exposure_usd": total_exposure * portfolio_value,
            "max_single_position_usd": max_single_position * portfolio_value,
            "number_of_signals": len(signals),
            "portfolio_value": portfolio_value
        }

    async def generate_comprehensive_signals(self, processed_sentiment_data: Dict, 
                                           market_data: Dict = None) -> Dict:
        """
        Generate comprehensive signals combining all approaches
        """
        all_signals = []
        
        # Process sentiment by timeframe
        for timeframe, agg_data in processed_sentiment_data.get('aggregations', {}).items():
            if agg_data.get('count', 0) == 0:
                continue
                
            # Create sentiment data dict for signal generation
            sentiment_for_signal = {
                "overall_sentiment": agg_data.get('average_sentiment', 0.0),
                "confidence": min(1.0, agg_data.get('positive_ratio', 0.5) + agg_data.get('negative_ratio', 0.5)),
                "momentum": agg_data.get('sentiment_momentum', 0.0),
                "volatility": agg_data.get('sentiment_volatility', 0.0)
            }
            
            # Generate signals for this timeframe
            timeframe_signals = await self.generate_signals_batch(
                [sentiment_for_signal], 
                market_data, 
                [processed_sentiment_data.get('symbol', 'N/A')]
            )
            all_signals.extend(timeframe_signals)
        
        # Generate fear and greed based signals
        fear_greed_signals = self.generate_fear_greed_signals(
            processed_sentiment_data.get('fear_greed_index', 50),
            processed_sentiment_data.get('symbol', 'N/A')
        )
        if fear_greed_signals:
            all_signals.append(fear_greed_signals)
        
        # Apply risk management
        risk_managed_signals = self.apply_risk_management(all_signals)
        
        return {
            "signals": risk_managed_signals,
            "total_signals_generated": len(all_signals),
            "signals_after_risk_management": len(risk_managed_signals),
            "generation_timestamp": datetime.utcnow().isoformat(),
            "market_regime": self.market_regime
        }
