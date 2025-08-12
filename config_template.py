# Configuration file for API keys
# Copy this file to config.py and add your actual API keys

API_KEYS = {
    # NewsAPI - Get free key at https://newsapi.org/
    # Provides: Live financial news articles
    # Free tier: 1000 requests/month
    'newsapi': 'your_newsapi_key_here',
    
    # Alpha Vantage - Get free key at https://www.alphavantage.co/
    # Provides: Stock prices, company data
    # Free tier: 5 requests/minute, 500 requests/day
    'alpha_vantage': 'your_alpha_vantage_key_here',
    
    # Optional: Polygon.io - Get key at https://polygon.io/
    # Provides: Real-time stock data
    # Free tier: 5 requests/minute
    'polygon': 'your_polygon_key_here'
}

# Analysis settings
ANALYSIS_SETTINGS = {
    'max_articles': 50,  # Maximum number of news articles to analyze
    'top_recommendations': 10,  # Number of stock recommendations to generate
    'confidence_threshold': 0.6,  # Minimum confidence for high-confidence recommendations
    'risk_tolerance': 'MEDIUM'  # Risk tolerance: LOW, MEDIUM, HIGH
}

# Model settings
MODEL_SETTINGS = {
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 32,
    'epochs': 4,
    'learning_rate': 2e-5
}
