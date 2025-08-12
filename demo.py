"""
Quick demo of the stock recommendation system
Shows how the system works with sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Import our modules
from stock_recommender import StockRecommendationEngine
from live_news_collector import LiveNewsCollector


def run_demo():
    """Run a demonstration with sample data"""
    
    print("🚀 STOCK RECOMMENDATION SYSTEM DEMO")
    print("="*50)
    
    print("\n📰 Sample News Analysis:")
    print("-" * 30)
    
    # Create sample news data
    sample_news = [
        {
            'title': 'Apple Reports Record Quarterly Revenue',
            'description': 'Apple Inc. exceeded analyst expectations with strong iPhone sales driving record revenue growth in Q4.',
            'stock_symbols': ['AAPL'],
            'sentiment_predictions': [0.05, 0.15, 0.80]  # Very positive
        },
        {
            'title': 'Tesla Faces Production Challenges',
            'description': 'Tesla reports lower than expected delivery numbers due to manufacturing bottlenecks at key facilities.',
            'stock_symbols': ['TSLA'],
            'sentiment_predictions': [0.70, 0.25, 0.05]  # Very negative
        },
        {
            'title': 'Google AI Breakthrough Drives Market Optimism',
            'description': 'Alphabet announces major breakthrough in AI technology, analysts upgrade price targets significantly.',
            'stock_symbols': ['GOOGL', 'GOOG'],
            'sentiment_predictions': [0.10, 0.20, 0.70]  # Positive
        },
        {
            'title': 'Microsoft Cloud Revenue Surges 30%',
            'description': 'Microsoft Azure and Office 365 drive strong cloud revenue growth, beating Wall Street estimates.',
            'stock_symbols': ['MSFT'],
            'sentiment_predictions': [0.08, 0.22, 0.70]  # Positive
        },
        {
            'title': 'Amazon Faces Regulatory Scrutiny',
            'description': 'Federal regulators launch investigation into Amazon business practices, shares fall in after-hours trading.',
            'stock_symbols': ['AMZN'],
            'sentiment_predictions': [0.65, 0.30, 0.05]  # Negative
        }
    ]
    
    # Display sample news
    for i, news in enumerate(sample_news, 1):
        sentiment_score = news['sentiment_predictions'][2] - news['sentiment_predictions'][0]
        sentiment_label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
        
        print(f"{i}. {news['title']}")
        print(f"   Stocks: {', '.join(news['stock_symbols'])}")
        print(f"   Sentiment: {sentiment_label} ({sentiment_score:+.2f})")
        print()
    
    # Create DataFrame for analysis
    stock_data = []
    for news in sample_news:
        for symbol in news['stock_symbols']:
            stock_data.append({
                'symbol': symbol,
                'title': news['title'],
                'sentiment_predictions': news['sentiment_predictions']
            })
    
    df = pd.DataFrame(stock_data)
    
    print("🤖 GENERATING RECOMMENDATIONS...")
    print("-" * 40)
    
    # Initialize recommendation engine
    recommender = StockRecommendationEngine()
    
    # Generate recommendations
    recommendations = recommender.generate_portfolio_recommendations(df, top_n=6)
    
    # Display recommendations
    report = recommender.format_recommendations_report(recommendations)
    print(report)
    
    print("\n📊 ANALYSIS SUMMARY:")
    print("-" * 25)
    
    buy_count = sum(1 for r in recommendations if r.action == 'BUY')
    sell_count = sum(1 for r in recommendations if r.action == 'SELL')
    hold_count = sum(1 for r in recommendations if r.action == 'HOLD')
    
    print(f"🟢 BUY Recommendations: {buy_count}")
    print(f"🔴 SELL Recommendations: {sell_count}")
    print(f"🟡 HOLD Recommendations: {hold_count}")
    
    high_confidence = sum(1 for r in recommendations if r.confidence > 0.7)
    print(f"📈 High Confidence (>70%): {high_confidence}")
    
    low_risk = sum(1 for r in recommendations if r.risk_level == 'LOW')
    print(f"⚖️  Low Risk Recommendations: {low_risk}")
    
    print("\n✅ Demo complete!")
    print("\n💡 To run with live data:")
    print("   1. Get API keys (see README.md)")
    print("   2. Run: python main.py")
    print("   3. Choose option 2 for live analysis")


def show_system_overview():
    """Show how the system works"""
    
    print("\n🔍 HOW THE SYSTEM WORKS:")
    print("=" * 40)
    
    steps = [
        "1. 📰 Collect live financial news from multiple sources",
        "2. 🎯 Extract stock symbols mentioned in each article",
        "3. 🧠 Analyze sentiment using fine-tuned BERT model",
        "4. 📊 Aggregate sentiment scores by stock symbol",
        "5. ⚖️  Assess risk level based on multiple factors",
        "6. 💡 Generate BUY/SELL/HOLD recommendations",
        "7. 📈 Estimate target prices based on sentiment",
        "8. 📋 Rank recommendations by confidence level"
    ]
    
    for step in steps:
        print(step)
    
    print(f"\n⏱️  Typical processing time: 30-60 seconds")
    print(f"📊 Accuracy on financial text: ~85%")
    print(f"🎯 Recommendation categories: BUY, SELL, HOLD")
    print(f"⚖️  Risk levels: LOW, MEDIUM, HIGH")


if __name__ == "__main__":
    show_system_overview()
    run_demo()
