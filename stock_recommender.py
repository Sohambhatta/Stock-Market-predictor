"""
Stock recommendation system based on sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass


@dataclass
class StockRecommendation:
    """Data class for stock recommendations"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    sentiment_score: float  # -1.0 to 1.0
    current_price: Optional[float]
    target_price: Optional[float]
    reasoning: str
    news_count: int
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'


class StockRecommendationEngine:
    """Generate stock recommendations based on news sentiment"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        
        # Sentiment to action thresholds
        self.buy_threshold = 0.3
        self.sell_threshold = -0.3
        self.high_confidence_threshold = 0.6
        
        # Risk assessment parameters
        self.min_news_for_confidence = 3
        self.volatility_risk_threshold = 0.05
    
    def get_stock_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current stock price and basic metrics
        Uses Alpha Vantage API (free tier available)
        """
        if 'alpha_vantage' not in self.api_keys:
            # Return mock data for demonstration
            return {
                'price': 150.00 + np.random.uniform(-10, 10),
                'change': np.random.uniform(-5, 5),
                'change_percent': np.random.uniform(-3, 3),
                'volume': np.random.randint(1000000, 10000000),
                'source': 'mock_data'
            }
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_keys['alpha_vantage']
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': float(quote.get('10. change percent', '0%').rstrip('%')),
                    'volume': int(quote.get('06. volume', 0)),
                    'source': 'alpha_vantage'
                }
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
        
        return None
    
    def calculate_sentiment_score(self, predictions: List[float]) -> float:
        """
        Convert model predictions to sentiment score
        
        Args:
            predictions: List of prediction probabilities [negative, neutral, positive]
            
        Returns:
            Sentiment score from -1.0 (very negative) to 1.0 (very positive)
        """
        if not predictions:
            return 0.0
        
        # Average the predictions
        avg_predictions = np.mean(predictions, axis=0)
        
        # Convert to sentiment score: positive - negative
        sentiment_score = avg_predictions[2] - avg_predictions[0]
        
        return float(np.clip(sentiment_score, -1.0, 1.0))
    
    def assess_risk_level(self, symbol: str, news_count: int, sentiment_variance: float) -> str:
        """
        Assess risk level for a stock recommendation
        
        Args:
            symbol: Stock symbol
            news_count: Number of news articles
            sentiment_variance: Variance in sentiment scores
            
        Returns:
            Risk level: 'LOW', 'MEDIUM', 'HIGH'
        """
        risk_factors = 0
        
        # Risk factor 1: Low news count
        if news_count < self.min_news_for_confidence:
            risk_factors += 1
        
        # Risk factor 2: High sentiment variance (conflicting news)
        if sentiment_variance > 0.3:
            risk_factors += 1
        
        # Risk factor 3: Penny stocks or very high priced stocks
        price_data = self.get_stock_price(symbol)
        if price_data:
            price = price_data['price']
            if price < 5 or price > 1000:  # Penny stock or very expensive
                risk_factors += 1
        
        # Determine risk level
        if risk_factors >= 2:
            return 'HIGH'
        elif risk_factors == 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_recommendation(self, symbol: str, news_data: pd.DataFrame, 
                              sentiment_predictions: List[List[float]]) -> StockRecommendation:
        """
        Generate recommendation for a single stock
        
        Args:
            symbol: Stock symbol
            news_data: DataFrame with news data for this symbol
            sentiment_predictions: List of model predictions for each news item
            
        Returns:
            StockRecommendation object
        """
        # Calculate overall sentiment
        sentiment_score = self.calculate_sentiment_score(sentiment_predictions)
        
        # Calculate confidence based on consistency and volume
        prediction_array = np.array(sentiment_predictions)
        sentiment_variance = np.var(np.argmax(prediction_array, axis=1))
        news_count = len(news_data)
        
        # Base confidence on news volume and sentiment consistency
        volume_factor = min(news_count / 10.0, 1.0)  # Max at 10+ articles
        consistency_factor = 1.0 - (sentiment_variance / 2.0)  # Penalize inconsistency
        confidence = (volume_factor * consistency_factor) * abs(sentiment_score)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        # Determine action
        if sentiment_score > self.buy_threshold:
            action = 'BUY'
        elif sentiment_score < self.sell_threshold:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Get current price
        price_data = self.get_stock_price(symbol)
        current_price = price_data['price'] if price_data else None
        
        # Calculate target price (simple model)
        target_price = None
        if current_price and sentiment_score != 0:
            # Estimate price change based on sentiment
            price_change_estimate = sentiment_score * 0.1  # 10% max change
            target_price = current_price * (1 + price_change_estimate)
        
        # Assess risk
        risk_level = self.assess_risk_level(symbol, news_count, sentiment_variance)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            action, sentiment_score, news_count, confidence, risk_level
        )
        
        return StockRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            sentiment_score=sentiment_score,
            current_price=current_price,
            target_price=target_price,
            reasoning=reasoning,
            news_count=news_count,
            risk_level=risk_level
        )
    
    def _generate_reasoning(self, action: str, sentiment_score: float, 
                           news_count: int, confidence: float, risk_level: str) -> str:
        """Generate human-readable reasoning for recommendation"""
        reasoning = []
        
        # Sentiment reasoning
        if sentiment_score > 0.5:
            reasoning.append("Very positive news sentiment")
        elif sentiment_score > 0.1:
            reasoning.append("Positive news sentiment")
        elif sentiment_score < -0.5:
            reasoning.append("Very negative news sentiment")
        elif sentiment_score < -0.1:
            reasoning.append("Negative news sentiment")
        else:
            reasoning.append("Neutral news sentiment")
        
        # Volume reasoning
        if news_count >= 10:
            reasoning.append(f"High news volume ({news_count} articles)")
        elif news_count >= 5:
            reasoning.append(f"Moderate news volume ({news_count} articles)")
        else:
            reasoning.append(f"Limited news volume ({news_count} articles)")
        
        # Confidence reasoning
        if confidence > 0.7:
            reasoning.append("High confidence recommendation")
        elif confidence > 0.4:
            reasoning.append("Moderate confidence recommendation")
        else:
            reasoning.append("Low confidence recommendation")
        
        # Risk reasoning
        reasoning.append(f"{risk_level.lower()} risk level")
        
        return ". ".join(reasoning) + "."
    
    def generate_portfolio_recommendations(self, news_with_sentiment: pd.DataFrame, 
                                         top_n: int = 10) -> List[StockRecommendation]:
        """
        Generate recommendations for multiple stocks
        
        Args:
            news_with_sentiment: DataFrame with news data and sentiment predictions
            top_n: Number of top recommendations to return
            
        Returns:
            List of StockRecommendation objects sorted by confidence
        """
        recommendations = []
        
        # Group by stock symbol
        for symbol in news_with_sentiment['symbol'].unique():
            symbol_news = news_with_sentiment[news_with_sentiment['symbol'] == symbol]
            predictions = symbol_news['sentiment_predictions'].tolist()
            
            recommendation = self.generate_recommendation(symbol, symbol_news, predictions)
            recommendations.append(recommendation)
        
        # Sort by confidence (descending)
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations[:top_n]
    
    def format_recommendations_report(self, recommendations: List[StockRecommendation]) -> str:
        """Format recommendations as a readable report"""
        if not recommendations:
            return "No recommendations generated."
        
        report = ["="*80]
        report.append("STOCK RECOMMENDATIONS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        for i, rec in enumerate(recommendations, 1):
            report.append(f"\n{i}. {rec.symbol} - {rec.action}")
            report.append(f"   Confidence: {rec.confidence:.1%}")
            report.append(f"   Sentiment Score: {rec.sentiment_score:+.2f}")
            
            if rec.current_price:
                report.append(f"   Current Price: ${rec.current_price:.2f}")
                if rec.target_price:
                    change = ((rec.target_price - rec.current_price) / rec.current_price) * 100
                    report.append(f"   Target Price: ${rec.target_price:.2f} ({change:+.1f}%)")
            
            report.append(f"   Risk Level: {rec.risk_level}")
            report.append(f"   Reasoning: {rec.reasoning}")
        
        report.append("\n" + "="*80)
        report.append("DISCLAIMER: This is for educational purposes only.")
        report.append("Always do your own research before making investment decisions.")
        report.append("="*80)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    print("Testing Stock Recommendation Engine...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'TSLA', 'GOOGL'],
        'title': [
            'Apple reports strong quarterly earnings',
            'Apple stock rises on new product launch',
            'Tesla faces production challenges',
            'Google AI breakthrough drives optimism'
        ],
        'sentiment_predictions': [
            [0.1, 0.2, 0.7],  # Positive
            [0.15, 0.25, 0.6],  # Positive
            [0.6, 0.3, 0.1],  # Negative
            [0.05, 0.15, 0.8]  # Very positive
        ]
    })
    
    engine = StockRecommendationEngine()
    recommendations = engine.generate_portfolio_recommendations(sample_data)
    
    print(engine.format_recommendations_report(recommendations))
