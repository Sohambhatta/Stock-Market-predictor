"""
Live news data collection for stock sentiment analysis
"""

import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re
import time


class LiveNewsCollector:
    """Collect live financial news from various sources"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize news collector
        
        Args:
            api_keys: Dictionary of API keys for different news sources
                     e.g., {'newsapi': 'your_key', 'alpha_vantage': 'your_key'}
        """
        self.api_keys = api_keys or {}
        self.stock_keywords = [
            'stock', 'shares', 'trading', 'market', 'NYSE', 'NASDAQ', 
            'earnings', 'revenue', 'profit', 'loss', 'IPO', 'dividend',
            'bullish', 'bearish', 'rally', 'crash', 'volatility'
        ]
    
    def get_newsapi_headlines(self, query: str = "stocks OR trading OR market", 
                             days_back: int = 1, max_articles: int = 100) -> List[Dict]:
        """
        Get news from NewsAPI
        Requires free API key from https://newsapi.org/
        """
        if 'newsapi' not in self.api_keys:
            print("Warning: NewsAPI key not provided. Skipping NewsAPI.")
            return []
        
        url = "https://newsapi.org/v2/everything"
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'language': 'en',
            'domains': 'cnbc.com,marketwatch.com,bloomberg.com,reuters.com,wsj.com,yahoo.com,seeking alpha.com',
            'pageSize': min(max_articles, 100),
            'apiKey': self.api_keys['newsapi']
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for article in data.get('articles', []):
                if article['title'] and article['description']:
                    articles.append({
                        'title': article['title'],
                        'description': article['description'],
                        'content': article.get('content', ''),
                        'source': article['source']['name'],
                        'published_at': article['publishedAt'],
                        'url': article['url']
                    })
            
            print(f"Collected {len(articles)} articles from NewsAPI")
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from NewsAPI: {e}")
            return []
    
    def get_free_news_sources(self, max_articles: int = 50) -> List[Dict]:
        """
        Get news from free sources (RSS feeds, etc.)
        This is a fallback when API keys aren't available
        """
        articles = []
        
        # You can add RSS feed parsing here
        # For now, returning sample structure
        print("Note: Using free news sources. For better results, add API keys.")
        
        # Sample data structure - in real implementation, parse RSS feeds
        sample_articles = [
            {
                'title': 'Sample: Tech Stocks Rally on Strong Earnings',
                'description': 'Technology companies show strong quarterly results driving market optimism',
                'content': 'Major tech companies including AAPL, GOOGL, and MSFT reported better than expected earnings...',
                'source': 'Sample Finance News',
                'published_at': datetime.now().isoformat(),
                'url': 'https://example.com'
            }
        ]
        
        return sample_articles[:max_articles]
    
    def extract_stock_symbols(self, text: str) -> List[str]:
        """
        Extract stock symbols from text
        
        Args:
            text: Text to search for stock symbols
            
        Returns:
            List of found stock symbols
        """
        # Pattern to match stock symbols (1-5 uppercase letters)
        pattern = r'\b[A-Z]{1,5}\b'
        potential_symbols = re.findall(pattern, text)
        
        # Filter out common false positives
        false_positives = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW',
            'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'HAS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO',
            'USE', 'CEO', 'CFO', 'CTO', 'USA', 'USD', 'NEWS', 'SAID', 'WILL', 'FROM', 'WITH'
        }
        
        # Known major stock symbols (you can expand this)
        known_symbols = {
            'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD',
            'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'DIS', 'BA', 'JPM', 'GS', 'MS', 'BAC',
            'WFC', 'C', 'BRK', 'JNJ', 'PFE', 'UNH', 'CVX', 'XOM', 'KO', 'PEP', 'WMT', 'HD'
        }
        
        # Filter symbols
        filtered_symbols = []
        for symbol in potential_symbols:
            if symbol not in false_positives and (
                len(symbol) <= 4 or symbol in known_symbols
            ):
                filtered_symbols.append(symbol)
        
        return list(set(filtered_symbols))  # Remove duplicates
    
    def collect_live_news(self, max_articles: int = 100) -> pd.DataFrame:
        """
        Collect live news from all available sources
        
        Returns:
            DataFrame with news articles and extracted stock symbols
        """
        print("Collecting live financial news...")
        
        all_articles = []
        
        # Try NewsAPI first
        if 'newsapi' in self.api_keys:
            all_articles.extend(self.get_newsapi_headlines(max_articles=max_articles))
        
        # If no articles or no API key, use free sources
        if not all_articles:
            all_articles.extend(self.get_free_news_sources(max_articles=max_articles))
        
        if not all_articles:
            print("No articles collected. Check your API keys or internet connection.")
            return pd.DataFrame()
        
        # Process articles
        processed_articles = []
        for article in all_articles:
            # Combine title and description for analysis
            full_text = f"{article['title']} {article['description']} {article.get('content', '')}"
            
            # Extract stock symbols
            symbols = self.extract_stock_symbols(full_text)
            
            processed_articles.append({
                'title': article['title'],
                'description': article['description'],
                'full_text': full_text,
                'source': article['source'],
                'published_at': article['published_at'],
                'url': article['url'],
                'stock_symbols': symbols,
                'symbol_count': len(symbols)
            })
        
        df = pd.DataFrame(processed_articles)
        
        # Filter to only articles mentioning stock symbols
        df_with_stocks = df[df['symbol_count'] > 0].copy()
        
        print(f"Collected {len(df)} total articles")
        print(f"Found {len(df_with_stocks)} articles mentioning stock symbols")
        
        return df_with_stocks


# Configuration helper
def create_sample_config():
    """Create sample configuration file for API keys"""
    config = {
        'newsapi': 'your_newsapi_key_here',  # Get from https://newsapi.org/
        'alpha_vantage': 'your_alpha_vantage_key_here',  # Get from https://www.alphavantage.co/
        'polygon': 'your_polygon_key_here'  # Get from https://polygon.io/
    }
    
    return config


if __name__ == "__main__":
    # Example usage
    print("Testing Live News Collector...")
    
    # Initialize without API keys (will use free sources)
    collector = LiveNewsCollector()
    
    # Collect news
    news_df = collector.collect_live_news(max_articles=10)
    
    if not news_df.empty:
        print(f"\nSample collected data:")
        print(news_df[['title', 'stock_symbols', 'source']].head())
    else:
        print("No news data collected. Consider adding API keys for better results.")
