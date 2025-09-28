import yfinance as yf
import requests
import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncGenerator
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("financial_data_client", "logs/financial_data_client.log")

class FinancialDataClient:
    """
    Financial data client for integrating market data, prices, and fund flows
    """
    def __init__(self):
        self.alpha_vantage_key = Config.ALPHA_VANTAGE_API_KEY
        self.polygon_key = Config.POLYGON_API_KEY
        self.iex_cloud_token = Config.IEX_CLOUD_TOKEN
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.polygon_url = "https://api.polygon.io"

    async def get_stock_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> Dict:
        """
        Get stock data using yfinance
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return {}
                
            # Convert to dictionary format
            data = {
                "symbol": symbol,
                "history": hist.to_dict('records'),
                "current_price": hist['Close'].iloc[-1] if len(hist) > 0 else None,
                "volume": hist['Volume'].iloc[-1] if len(hist) > 0 else None,
                "change": None
            }
            
            if len(hist) >= 2:
                data["change"] = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                data["change_percent"] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            
            return data
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return {}

    async def get_crypto_data(self, symbol: str) -> Dict:
        """
        Get cryptocurrency data
        """
        try:
            # Using yfinance for crypto data as well (format: BTC-USD, ETH-USD, etc.)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return {}
                
            data = {
                "symbol": symbol,
                "history": hist.to_dict('records'),
                "current_price": hist['Close'].iloc[-1] if len(hist) > 0 else None,
                "volume": hist['Volume'].iloc[-1] if len(hist) > 0 else None,
                "change": None
            }
            
            if len(hist) >= 2:
                data["change"] = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                data["change_percent"] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            
            return data
        except Exception as e:
            logger.error(f"Error getting crypto data for {symbol}: {e}")
            return {}

    async def get_multiple_symbols_data(self, symbols: List[str], period: str = "1d", interval: str = "1m") -> Dict:
        """
        Get data for multiple symbols at once
        """
        data = {}
        for symbol in symbols:
            if symbol.upper().endswith(("=X", "-USD", "-USDT", "-BTC", "-ETH")):
                # Crypto symbol
                data[symbol] = await self.get_crypto_data(symbol)
            else:
                # Stock symbol
                data[symbol] = await self.get_stock_data(symbol, period, interval)
            await asyncio.sleep(0.1)  # Rate limiting
        return data

    async def get_fund_flow_data(self, symbol: str) -> Dict:
        """
        Get fund flow data for a symbol (using Alpha Vantage if available)
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not configured for fund flow data")
            return {"error": "Alpha Vantage API key not configured"}
        
        try:
            params = {
                "function": "ETF_PROFILE",
                "symbol": symbol,
                "apikey": self.alpha_vantage_key
            }
            
            response = requests.get(self.alpha_vantage_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data
        except Exception as e:
            logger.error(f"Error getting fund flow data for {symbol}: {e}")
            # Fallback: Calculate from price/volume data
            stock_data = await self.get_stock_data(symbol, "5d", "1d")
            if 'history' in stock_data and len(stock_data['history']) > 1:
                # Calculate money flow based on price and volume
                hist = pd.DataFrame(stock_data['history'])
                typical_price = (hist['High'] + hist['Low'] + hist['Close']) / 3
                money_flow = typical_price * hist['Volume']
                
                return {
                    "symbol": symbol,
                    "money_flow": money_flow.tolist(),
                    "avg_money_flow": money_flow.mean()
                }
            return {}

    async def get_market_sentiment_indicators(self) -> Dict:
        """
        Get market sentiment indicators like VIX, put/call ratios, etc.
        """
        try:
            # Get VIX data as a fear/greed indicator
            vix_data = await self.get_stock_data("^VIX", "1d", "1m")
            
            # Get major market indices
            indices = await self.get_multiple_symbols_data(["^GSPC", "^DJI", "^IXIC", "^RUT"])  # S&P 500, Dow, NASDAQ, Russell 2000
            
            return {
                "vix": vix_data,
                "indices": indices,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market sentiment indicators: {e}")
            return {}

    def calculate_correlation(self, sentiment_scores: List[float], price_changes: List[float]) -> float:
        """
        Calculate correlation between sentiment scores and price changes
        """
        if len(sentiment_scores) < 2 or len(price_changes) < 2:
            return 0.0
            
        try:
            import numpy as np
            # Calculate Pearson correlation coefficient
            correlation_matrix = np.corrcoef(sentiment_scores, price_changes)
            return float(correlation_matrix[0, 1])
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0

    async def get_economic_indicators(self) -> Dict:
        """
        Get key economic indicators that might affect market sentiment
        """
        try:
            # For demo purposes, we'll simulate some economic indicators
            # In a real implementation, we'd use an economic data API
            indicators = {
                "fed_rate": 5.50,  # Simulated Federal funds rate
                "unemployment_rate": 3.7,  # Simulated unemployment rate
                "inflation_rate": 3.2,  # Simulated inflation rate (CPI)
                "gdp_growth": 2.1,  # Simulated GDP growth
                "consumer_confidence": 105.4,  # Simulated consumer confidence index
                "pmi_manufacturing": 49.8,  # Simulated manufacturing PMI
                "pmi_services": 51.2,  # Simulated services PMI
                "retail_sales": 0.6,  # Monthly change in retail sales (%)
                "housing_starts": 1.425,  # Millions of units (annual rate)
                "timestamp": datetime.now().isoformat()
            }
            
            return indicators
        except Exception as e:
            logger.error(f"Error getting economic indicators: {e}")
            return {}

    async def get_sector_performance(self) -> Dict:
        """
        Get performance of different sectors
        """
        try:
            # Define sector ETFs
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV", 
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Real Estate": "XLRE",
                "Utilities": "XLU",
                "Communication": "XLC"
            }
            
            sector_data = {}
            for sector, etf in sector_etfs.items():
                data = await self.get_stock_data(etf, "1d", "5m")
                sector_data[sector] = data
                
            return sector_data
        except Exception as e:
            logger.error(f"Error getting sector performance: {e}")
            return {}