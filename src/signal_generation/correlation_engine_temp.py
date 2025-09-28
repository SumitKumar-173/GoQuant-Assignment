import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger('correlation_engine', 'logs/correlation_engine.log')

class CorrelationEngine:
    """
    Advanced correlation engine for analyzing relationships between sentiment and market data
    """
    def __init__(self):
        self.config = Config
        self.correlation_cache = {}
        self.lookback_windows = [5, 15, 30, 60, 120]  # minutes
        self.sentiment_market_data = {}  # Store paired sentiment and market data

    async def calculate_sentiment_price_correlation(self, sentiment_history: List[Dict], 
                                                   price_history: List[Dict], 
                                                   time_lag: int = 0) -> Dict:
        """
        Calculate correlation between sentiment and price with optional time lag
        """
        if not sentiment_history or not price_history:
            return {"correlation": 0.0, "p_value": 1.0, "sample_size": 0, "r_squared": 0.0}
        
        # Align sentiment and price data by timestamp
        aligned_sentiment = []
        aligned_prices = []
        
        # Convert to pandas for easier manipulation
        sent_df = pd.DataFrame(sentiment_history)
        price_df = pd.DataFrame(price_history)
        
        # Ensure datetime format
        sent_df['timestamp'] = pd.to_datetime(sent_df['timestamp'])
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        
        # Apply time lag to sentiment (if positive, sentiment leads price)
        if time_lag != 0:
            sent_df['timestamp'] = sent_df['timestamp'] + timedelta(minutes=time_lag)
        
        # Merge on timestamp (with tolerance for matching timestamps)
        merged_df = pd.merge_asof(
            sent_df.sort_values('timestamp'),
            price_df.sort_values('timestamp'),
            on='timestamp',
            tolerance=timedelta(minutes=5),  # Match if within 5 minutes
            direction='nearest'
        )
        
        if merged_df.empty:
            return {"correlation": 0.0, "p_value": 1.0, "sample_size": 0, "r_squared": 0.0}
        
        # Extract aligned values
        sentiment_values = merged_df['overall_sentiment'].dropna().values
        price_changes = merged_df['price_change'].dropna().values if 'price_change' in merged_df.columns else []
        
        # If no price changes available, calculate from closing prices
        if len(price_changes) == 0 and 'Close' in merged_df.columns:
            closing_prices = merged_df['Close'].dropna().values
            if len(closing_prices) > 1:
                price_changes = np.diff(closing_prices) / closing_prices[:-1]  # Calculate returns
                # Align with sentiment (remove first sentiment as it doesn't have prior price)
                if len(sentiment_values) > len(price_changes):
                    sentiment_values = sentiment_values[1:]
                elif len(price_changes) > len(sentiment_values):
                    price_changes = price_changes[:len(sentiment_values)]
            else:
                return {"correlation": 0.0, "p_value": 1.0, "sample_size": 0, "r_squared": 0.0}
        
        # Ensure same length
        min_len = min(len(sentiment_values), len(price_changes))
        sentiment_values = sentiment_values[:min_len]
        price_changes = price_changes[:min_len]
        
        if len(sentiment_values) < 5:  # Need at least 5 data points for reliable correlation
            return {"correlation": 0.0, "p_value": 1.0, "sample_size": len(sentiment_values), "r_squared": 0.0}
        
        # Calculate correlation
        correlation_matrix = np.corrcoef(sentiment_values, price_changes)
        correlation = correlation_matrix[0, 1]
        
        # Calculate R-squared
        if len(sentiment_values) > 2:
            # Use simple linear regression to get R-squared
            X = sentiment_values.reshape(-1, 1)
            y = price_changes
            reg = LinearRegression().fit(X, y)
            r_squared = reg.score(X, y)
        else:
            r_squared = 0.0
        
        # Approximate p-value calculation (simplified)
        n = len(sentiment_values)
        if abs(correlation) == 1.0:
            p_value = 0.0
        else:
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            # Simplified p-value approximation
            p_value = 2 * (1 - np.abs(t_stat) / np.sqrt(n - 2 + t_stat**2)) if n > 2 else 1.0
        
        return {
            "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "p_value": float(min(1.0, max(0.0, p_value))) if not np.isnan(p_value) else 1.0,
            "sample_size": len(sentiment_values),
            "r_squared": float(r_squared) if not np.isnan(r_squared) else 0.0,
            "time_lag_minutes": time_lag
        }

    async def calculate_cross_asset_correlation(self, sentiment_data: Dict, 
                                              other_assets_data: Dict) -> Dict:
        """
        Calculate correlation between sentiment for one asset and performance of other assets
        """
        correlations = {}
        
        for asset_symbol, asset_data in other_assets_data.items():
            if 'sentiment_history' in asset_data and 'price_history' in asset_data:
                correlation_result = await self.calculate_sentiment_price_correlation(
                    sentiment_data.get('sentiment_history', []),
                    asset_data['price_history']
                )
                correlations[asset_symbol] = correlation_result
        
        return correlations

    def detect_correlation_regimes(self, correlation_series: List[float], 
                                 regime_threshold: float = 0.3) -> List[Dict]:
        """
        Detect different correlation regimes (high, low, negative, positive)
        """
        regimes = []
        
        if len(correlation_series) < 2:
            return regimes
        
        # Calculate moving average to smooth correlation changes
        window_size = min(10, len(correlation_series))
        moving_avg = np.convolve(correlation_series, np.ones(window_size)/window_size, mode='valid')
        
        # Detect regime changes
        current_regime = None
        regime_start_idx = 0
        
        for i, corr in enumerate(moving_avg):
            if corr > regime_threshold:
                regime = "positive"
            elif corr < -regime_threshold:
                regime = "negative"
            else:
                regime = "neutral"
            
            if current_regime is None:
                current_regime = regime
            elif current_regime != regime:
                # Regime changed, record the previous regime
                regimes.append({
                    "regime": current_regime,
                    "start_idx": regime_start_idx,
                    "end_idx": i - 1,
                    "duration": i - regime_start_idx,
                    "avg_correlation": np.mean(correlation_series[regime_start_idx:i])
                })
                current_regime = regime
                regime_start_idx = i
        
        # Add the last regime
        if regime_start_idx < len(moving_avg):
            regimes.append({
                "regime": current_regime,
                "start_idx": regime_start_idx,
                "end_idx": len(moving_avg) - 1,
                "duration": len(moving_avg) - regime_start_idx,
                "avg_correlation": np.mean(correlation_series[regime_start_idx:])
            })
        
        return regimes

    async def calculate_fund_flow_correlation(self, sentiment_data: List[Dict], 
                                            fund_flow_data: List[Dict]) -> Dict:
        """
        Calculate correlation between sentiment and fund flows
        """
        if not sentiment_data or not fund_flow_data:
            return {"correlation": 0.0, "significance": 0.0}
        
        # Align sentiment and fund flow data by timestamp
        # This is a simplified implementation - in practice, you'd have more sophisticated alignment
        aligned_sentiment = [d['overall_sentiment'] for d in sentiment_data if 'overall_sentiment' in d]
        aligned_fund_flows = [d.get('net_flow', 0) for d in fund_flow_data if 'net_flow' in d]
        
        # Equalize lengths
        min_len = min(len(aligned_sentiment), len(aligned_fund_flows))
        aligned_sentiment = aligned_sentiment[:min_len]
        aligned_fund_flows = aligned_fund_flows[:min_len]
        
        if len(aligned_sentiment) < 3:
            return {"correlation": 0.0, "significance": 0.0}
        
        # Calculate correlation
        try:
            correlation_matrix = np.corrcoef(aligned_sentiment, aligned_fund_flows)
            correlation = correlation_matrix[0, 1]
            
            # Calculate significance based on sample size and correlation strength
            significance = abs(correlation) * min(1.0, len(aligned_sentiment) / 20.0)
            
            return {
                "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                "significance": float(significance) if not np.isnan(significance) else 0.0,
                "sample_size": len(aligned_sentiment)
            }
        except:
            return {"correlation": 0.0, "significance": 0.0}

    def predict_price_impact(self, sentiment_score: float, correlation: float, 
                           historical_volatility: float = 0.02) -> Dict:
        """
        Predict potential price impact based on sentiment and correlation
        """
        # Basic prediction model: expected return = correlation * sentiment_score
        expected_return = correlation * sentiment_score
        
        # Adjust for market volatility
        volatility_adjusted_return = expected_return * (1 + historical_volatility)
        
        # Calculate confidence in prediction
        prediction_confidence = abs(correlation) * 0.7 + (1 - historical_volatility) * 0.3
        
        return {
            "expected_return": float(expected_return),
            "volatility_adjusted_return": float(volatility_adjusted_return),
            "prediction_confidence": float(prediction_confidence),
            "risk_adjusted_impact": float(abs(volatility_adjusted_return) * prediction_confidence)
        }

    async def calculate_real_time_correlation(self, symbol: str, 
                                           sentiment_value: float, 
                                           price_value: float) -> float:
        """
        Calculate real-time correlation for a single data point against history
        """
        # Store new data point
        if symbol not in self.sentiment_market_data:
            self.sentiment_market_data[symbol] = {'sentiment': [], 'price': []}
        
        symbol_data = self.sentiment_market_data[symbol]
        symbol_data['sentiment'].append(sentiment_value)
        symbol_data['price'].append(price_value)
        
        # Keep only recent 100 data points
        if len(symbol_data['sentiment']) > 100:
            symbol_data['sentiment'] = symbol_data['sentiment'][-100:]
            symbol_data['price'] = symbol_data['price'][-100:]
        
        # Calculate correlation of recent data
        if len(symbol_data['sentiment']) >= 5:
            correlation_matrix = np.corrcoef(symbol_data['sentiment'], symbol_data['price'])
            correlation = correlation_matrix[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0

    async def generate_correlation_signals(self, correlation_data: Dict, 
                                        symbol: str) -> List[Dict]:
        """
        Generate trading signals based on correlation analysis
        """
        signals = []
        
        # Check for significant correlations
        correlation = correlation_data.get('correlation', 0.0)
        p_value = correlation_data.get('p_value', 1.0)
        r_squared = correlation_data.get('r_squared', 0.0)
        
        # Only generate signals if correlation is significant
        if abs(correlation) > 0.3 and p_value < 0.05 and r_squared > 0.1:
            # Calculate signal strength based on correlation and statistical significance
            signal_strength = min(1.0, abs(correlation) * (1 - p_value) * np.sqrt(r_squared))
            
            # Determine signal direction
            if correlation > 0:
                # Positive correlation: sentiment follows price, use for confirmation
                if correlation_data.get('sentiment_score', 0) > 0.3:
                    signal = {
                        "symbol": symbol,
                        "signal_type": "BUY",
                        "strength": signal_strength,
                        "confidence": signal_strength,
                        "correlation": correlation,
                        "p_value": p_value,
                        "r_squared": r_squared,
                        "strategy": "correlation_confirmation",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    signals.append(signal)
            else:
                # Negative correlation: sentiment leads price in opposite direction
                if correlation_data.get('sentiment_score', 0) < -0.3:
                    signal = {
                        "symbol": symbol,
                        "signal_type": "BUY",  # Countercyclical
                        "strength": signal_strength,
                        "confidence": signal_strength,
                        "correlation": correlation,
                        "p_value": p_value,
                        "r_squared": r_squared,
                        "strategy": "negative_correlation",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    signals.append(signal)
                elif correlation_data.get('sentiment_score', 0) > 0.3:
                    signal = {
                        "symbol": symbol,
                        "signal_type": "SELL",  # Countercyclical
                        "strength": signal_strength,
                        "confidence": signal_strength,
                        "correlation": correlation,
                        "p_value": p_value,
                        "r_squared": r_squared,
                        "strategy": "negative_correlation",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    signals.append(signal)
        
        return signals

    def detect_corrrelation_breakdowns(self, correlation_series: List[float], 
                                     threshold: float = 0.5) -> List[Dict]:
        """
        Detect when correlations are breaking down (becoming unreliable)
        """
        breakdowns = []
        
        if len(correlation_series) < 10:
            return breakdowns
        
        # Calculate rolling correlation of correlations to detect instability
        window_size = 5
        for i in range(window_size, len(correlation_series)):
            window_corr = correlation_series[i-window_size:i]
            if len(window_corr) > 1:
                volatility = np.std(window_corr)
                avg_corr = np.mean(window_corr)
                
                # If correlation is highly volatile or near zero, it may be breaking down
                if volatility > threshold or abs(avg_corr) < 0.2:
                    breakdowns.append({
                        "index": i,
                        "timestamp": datetime.utcnow().isoformat(),
                        "correlation_at_breakdown": correlation_series[i],
                        "volatility": volatility,
                        "average_recent_correlation": avg_corr
                    })
        
        return breakdowns

    async def analyze_sentiment_spillover(self, source_asset: str, target_assets: List[str], 
                                        correlation_matrix: Dict) -> Dict:
        """
        Analyze how sentiment in one asset affects others (spillover effects)
        """
        spillover_effects = {}
        
        for target_asset in target_assets:
            if source_asset in correlation_matrix and target_asset in correlation_matrix[source_asset]:
                correlation = correlation_matrix[source_asset][target_asset]
                spillover_effects[target_asset] = {
                    "correlation": correlation,
                    "spillover_strength": abs(correlation),
                    "direction": "positive" if correlation > 0 else "negative"
                }
        
        return {
            "source_asset": source_asset,
            "spillover_effects": spillover_effects,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    async def evaluate_correlation_predictiveness(self, historical_correlations: List[float],
                                               future_returns: List[float]) -> Dict:
        """
        Evaluate how predictive correlations have been historically
        """
        if len(historical_correlations) < 2 or len(future_returns) < 2:
            return {"predictiveness_score": 0.0, "max_correlation": 0.0, "avg_prediction_error": 1.0}
        
        # Calculate how often correlation correctly predicted direction
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(min(len(historical_correlations), len(future_returns))):
            correlation = historical_correlations[i]
            future_return = future_returns[i]
            
            # Check if correlation sign matches return sign (for positive correlations)
            if (correlation > 0 and future_return > 0) or (correlation < 0 and future_return < 0):
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        max_correlation = max(abs(c) for c in historical_correlations) if historical_correlations else 0.0
        avg_correlation = np.mean([abs(c) for c in historical_correlations]) if historical_correlations else 0.0
        
        return {
            "predictiveness_score": accuracy * avg_correlation,  # Balance accuracy with correlation strength
            "prediction_accuracy": accuracy,
            "max_correlation": max_correlation,
            "avg_correlation": avg_correlation,
            "sample_size": total_predictions
        }
