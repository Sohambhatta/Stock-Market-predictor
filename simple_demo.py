"""
Simple test of the stock recommendation system
Shows the core functionality without requiring ML libraries
"""

import re
from typing import List, Dict
from datetime import datetime


def extract_stock_symbols(text: str) -> List[str]:
    """Extract stock symbols from text"""
    # Company name to symbol mapping
    company_mappings = {
        'apple': 'AAPL',
        'microsoft': 'MSFT', 
        'tesla': 'TSLA',
        'google': 'GOOGL',
        'alphabet': 'GOOGL',
        'amazon': 'AMZN',
        'meta': 'META',
        'facebook': 'META',
        'nvidia': 'NVDA',
        'netflix': 'NFLX',
        'amd': 'AMD',
        'intel': 'INTC',
        'salesforce': 'CRM',
        'oracle': 'ORCL',
        'adobe': 'ADBE',
        'paypal': 'PYPL',
        'disney': 'DIS',
        'boeing': 'BA',
        'jpmorgan': 'JPM',
        'goldman sachs': 'GS',
        'morgan stanley': 'MS',
        'bank of america': 'BAC'
    }
    
    text_lower = text.lower()
    found_symbols = set()
    
    # Check for company names
    for company, symbol in company_mappings.items():
        if company in text_lower:
            found_symbols.add(symbol)
    
    # Pattern to match stock symbols (1-5 uppercase letters)
    pattern = r'\b[A-Z]{1,5}\b'
    potential_symbols = re.findall(pattern, text)
    
    # Filter out common false positives
    false_positives = {
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
        'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW',
        'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'HAS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO',
        'USE', 'CEO', 'CFO', 'CTO', 'USA', 'USD', 'NEWS', 'SAID', 'WILL', 'FROM', 'WITH',
        'INC', 'LLC', 'LTD', 'AI'  # Added AI as it's too generic
    }
    
    # Known major stock symbols
    known_symbols = {
        'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD',
        'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'DIS', 'BA', 'JPM', 'GS', 'MS', 'BAC',
        'WFC', 'C', 'BRK', 'JNJ', 'PFE', 'UNH', 'CVX', 'XOM', 'KO', 'PEP', 'WMT', 'HD'
    }
    
    # Filter symbols from text
    for symbol in potential_symbols:
        if symbol not in false_positives and (len(symbol) <= 4 or symbol in known_symbols):
            found_symbols.add(symbol)
    
    return list(found_symbols)


def analyze_sentiment_simple(text: str) -> Dict:
    """Simple sentiment analysis based on keywords"""
    positive_words = [
        'surge', 'rally', 'boom', 'soar', 'jump', 'rise', 'gain', 'strong', 'bullish',
        'optimistic', 'breakthrough', 'record', 'growth', 'profit', 'beat', 'exceed'
    ]
    
    negative_words = [
        'crash', 'fall', 'drop', 'plunge', 'decline', 'loss', 'weak', 'bearish',
        'pessimistic', 'concern', 'worry', 'challenge', 'struggle', 'miss', 'below'
    ]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = positive_count + negative_count
    if total_words == 0:
        return {'sentiment': 'NEUTRAL', 'score': 0.0, 'confidence': 0.5}
    
    sentiment_score = (positive_count - negative_count) / max(total_words, 1)
    
    if sentiment_score > 0.2:
        sentiment = 'POSITIVE'
    elif sentiment_score < -0.2:
        sentiment = 'NEGATIVE'
    else:
        sentiment = 'NEUTRAL'
    
    confidence = min(abs(sentiment_score) + 0.5, 1.0)
    
    return {
        'sentiment': sentiment,
        'score': sentiment_score,
        'confidence': confidence
    }


def generate_recommendation(symbol: str, sentiment: Dict, current_price: float = None) -> Dict:
    """Generate stock recommendation based on sentiment"""
    
    # Mock current price if not provided
    if current_price is None:
        import random
        current_price = random.uniform(50, 300)
    
    sentiment_score = sentiment['score']
    confidence = sentiment['confidence']
    
    # Determine action
    if sentiment_score > 0.3 and confidence > 0.6:
        action = 'BUY'
        target_change = sentiment_score * 0.15  # Up to 15% change
    elif sentiment_score < -0.3 and confidence > 0.6:
        action = 'SELL'
        target_change = sentiment_score * 0.15
    else:
        action = 'HOLD'
        target_change = sentiment_score * 0.05
    
    target_price = current_price * (1 + target_change)
    
    # Risk assessment
    if confidence > 0.8:
        risk = 'LOW'
    elif confidence > 0.6:
        risk = 'MEDIUM'
    else:
        risk = 'HIGH'
    
    return {
        'symbol': symbol,
        'action': action,
        'confidence': confidence,
        'current_price': current_price,
        'target_price': target_price,
        'price_change_pct': target_change * 100,
        'risk_level': risk,
        'sentiment_score': sentiment_score
    }


def run_simple_demo():
    """Run a simple demonstration"""
    print("🚀 STOCK SENTIMENT ANALYZER - SIMPLE DEMO")
    print("=" * 50)
    
    # Sample news articles
    sample_news = [
        {
            'title': 'Apple Reports Record Quarterly Revenue, Beats Estimates',
            'description': 'Apple Inc. exceeded analyst expectations with strong iPhone sales driving record revenue growth in Q4.',
        },
        {
            'title': 'Tesla Faces Production Challenges, Deliveries Miss Target',
            'description': 'Tesla reports lower than expected delivery numbers due to manufacturing bottlenecks at key facilities.',
        },
        {
            'title': 'Microsoft Cloud Revenue Surges 35%, AI Integration Drives Growth',
            'description': 'Microsoft Azure and AI services drive strong quarterly performance, beating Wall Street estimates significantly.',
        },
        {
            'title': 'Google Parent Alphabet Announces Breakthrough in Quantum Computing',
            'description': 'Alphabet unveils major quantum computing advancement, researchers optimistic about commercial applications.',
        },
        {
            'title': 'Amazon Faces Regulatory Scrutiny Over Market Practices',
            'description': 'Federal regulators launch investigation into Amazon business practices, shares decline in after-hours trading.',
        }
    ]
    
    print("\n📰 ANALYZING NEWS ARTICLES:")
    print("-" * 30)
    
    recommendations = []
    
    for i, article in enumerate(sample_news, 1):
        full_text = f"{article['title']} {article['description']}"
        
        # Extract stock symbols
        symbols = extract_stock_symbols(full_text)
        
        # Analyze sentiment
        sentiment = analyze_sentiment_simple(full_text)
        
        print(f"\n{i}. {article['title'][:60]}...")
        print(f"   Stocks mentioned: {', '.join(symbols) if symbols else 'None detected'}")
        print(f"   Sentiment: {sentiment['sentiment']} (score: {sentiment['score']:+.2f})")
        print(f"   Confidence: {sentiment['confidence']:.1%}")
        
        # Generate recommendations for each stock
        for symbol in symbols:
            rec = generate_recommendation(symbol, sentiment)
            recommendations.append(rec)
    
    # Sort recommendations by confidence
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    print("\n" + "=" * 80)
    print("📊 STOCK RECOMMENDATIONS")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations[:6], 1):
        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        
        print(f"\n{i}. {rec['symbol']} - {action_emoji.get(rec['action'], '⚪')} {rec['action']}")
        print(f"   Current Price: ${rec['current_price']:.2f}")
        print(f"   Target Price: ${rec['target_price']:.2f} ({rec['price_change_pct']:+.1f}%)")
        print(f"   Confidence: {rec['confidence']:.1%}")
        print(f"   Risk Level: {rec['risk_level']}")
        print(f"   Sentiment Score: {rec['sentiment_score']:+.2f}")
    
    print("\n" + "=" * 80)
    print("📈 SUMMARY:")
    buy_count = sum(1 for r in recommendations if r['action'] == 'BUY')
    sell_count = sum(1 for r in recommendations if r['action'] == 'SELL')
    hold_count = sum(1 for r in recommendations if r['action'] == 'HOLD')
    
    print(f"🟢 BUY recommendations: {buy_count}")
    print(f"🔴 SELL recommendations: {sell_count}")
    print(f"🟡 HOLD recommendations: {hold_count}")
    
    high_conf = sum(1 for r in recommendations if r['confidence'] > 0.7)
    print(f"⭐ High confidence (>70%): {high_conf}")
    
    print("\n💡 NEXT STEPS:")
    print("1. Get API keys from NewsAPI.org and AlphaVantage.co for live data")
    print("2. Run the full system with: python main.py")
    print("3. The full system uses AI (BERT) for much more accurate sentiment analysis")
    
    print("\n⚠️  DISCLAIMER: This is for educational purposes only!")
    print("Always do your own research before making investment decisions.")


if __name__ == "__main__":
    run_simple_demo()
