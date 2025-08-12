import yfinance as yf
import requests
import json
from datetime import datetime
import time

def get_realtime_yahoo_api(symbol):
    """Get real-time data from Yahoo Finance API directly"""
    try:
        # Yahoo Finance real-time API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        if 'chart' in data and data['chart']['result']:
            result = data['chart']['result'][0]
            meta = result['meta']
            
            # Get the most recent price from the chart data
            timestamps = result['timestamp']
            prices = result['indicators']['quote'][0]
            
            # Find the last non-null price
            last_price = None
            for i in range(len(prices['close']) - 1, -1, -1):
                if prices['close'][i] is not None:
                    last_price = prices['close'][i]
                    break
            
            return {
                'symbol': symbol,
                'price': last_price or meta.get('regularMarketPrice', 0),
                'previous_close': meta.get('previousClose', 0),
                'open': meta.get('regularMarketOpen', 0),
                'high': meta.get('regularMarketDayHigh', 0),
                'low': meta.get('regularMarketDayLow', 0),
                'volume': meta.get('regularMarketVolume', 0),
                'market_time': datetime.fromtimestamp(meta.get('regularMarketTime', time.time())),
                'source': 'Yahoo Real-time API'
            }
    except Exception as e:
        print(f"Yahoo API error for {symbol}: {e}")
        return None

def get_finnhub_data(symbol):
    """Get data from Finnhub (free tier - real-time US stocks)"""
    try:
        # Free Finnhub API key (limited requests)
        api_key = "demo"  # Replace with your free API key from finnhub.io
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'c' in data:  # 'c' is current price
            return {
                'symbol': symbol,
                'price': data['c'],
                'previous_close': data['pc'],
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'market_time': datetime.fromtimestamp(data.get('t', time.time())),
                'source': 'Finnhub Real-time'
            }
    except Exception as e:
        print(f"Finnhub error for {symbol}: {e}")
        return None

def get_polygon_data(symbol):
    """Get data from Polygon.io (free tier available)"""
    try:
        # Free tier API key (sign up at polygon.io)
        api_key = "demo"  # Replace with your free API key
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apikey={api_key}"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'results' in data and data['results']:
            result = data['results'][0]
            return {
                'symbol': symbol,
                'price': result['c'],  # close price
                'previous_close': result['c'],  # This is previous day close
                'open': result['o'],
                'high': result['h'],
                'low': result['l'],
                'volume': result['v'],
                'source': 'Polygon.io'
            }
    except Exception as e:
        print(f"Polygon error for {symbol}: {e}")
        return None

def get_alpha_vantage_data(symbol):
    """Get data from Alpha Vantage (free tier available)"""
    try:
        # Free API key (sign up at alphavantage.co)
        api_key = "demo"  # Replace with your free API key
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'Global Quote' in data:
            quote = data['Global Quote']
            return {
                'symbol': symbol,
                'price': float(quote['05. price']),
                'previous_close': float(quote['08. previous close']),
                'open': float(quote['02. open']),
                'high': float(quote['03. high']),
                'low': float(quote['04. low']),
                'volume': int(quote['06. volume']),
                'source': 'Alpha Vantage'
            }
    except Exception as e:
        print(f"Alpha Vantage error for {symbol}: {e}")
        return None

def get_most_recent_price(symbol):
    """Try multiple sources and return the most recent price"""
    sources = []
    
    # Try all data sources
    yahoo_rt = get_realtime_yahoo_api(symbol)
    if yahoo_rt:
        sources.append(yahoo_rt)
    
    finnhub = get_finnhub_data(symbol)
    if finnhub:
        sources.append(finnhub)
    
    # Fallback to yfinance
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1d")
        if not hist.empty:
            sources.append({
                'symbol': symbol,
                'price': float(hist['Close'].iloc[-1]),
                'previous_close': info.get('previousClose', 0),
                'open': float(hist['Open'].iloc[-1]),
                'high': float(hist['High'].iloc[-1]),
                'low': float(hist['Low'].iloc[-1]),
                'volume': int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0,
                'source': 'yfinance (fallback)'
            })
    except:
        pass
    
    if not sources:
        return None
    
    # Return the source with the most recent data
    # For now, prefer Yahoo real-time API
    for source in sources:
        if 'Yahoo Real-time' in source['source']:
            return source
    
    return sources[0]  # Return first available

def compare_all_sources(symbol):
    """Compare prices from all sources for debugging"""
    print(f"\n🔍 Real-time price comparison for {symbol.upper()}:")
    print("=" * 60)
    
    yahoo_rt = get_realtime_yahoo_api(symbol)
    finnhub = get_finnhub_data(symbol)
    
    # yfinance fallback
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        yf_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
    except:
        yf_price = 0
    
    sources = []
    if yahoo_rt:
        sources.append(yahoo_rt)
    if finnhub:
        sources.append(finnhub)
    
    if yf_price > 0:
        sources.append({
            'price': yf_price,
            'source': 'yfinance library'
        })
    
    for source in sources:
        print(f"📊 {source['source']}: ${source['price']:.2f}")
        if 'market_time' in source:
            print(f"   Last updated: {source['market_time']}")
    
    if len(sources) > 1:
        prices = [s['price'] for s in sources]
        max_diff = max(prices) - min(prices)
        print(f"\n💰 Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        print(f"📈 Max difference: ${max_diff:.2f}")
        
        if max_diff < 0.05:
            print("✅ All sources are very close - data is consistent")
        elif max_diff < 0.50:
            print("⚠️  Small differences - normal for different update times")
        else:
            print("❌ Large differences - one source may be stale")

if __name__ == "__main__":
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    for stock in test_stocks:
        compare_all_sources(stock)
        time.sleep(1)
    
    print(f"\n🚀 To get the most up-to-date prices, we'll use:")
    print("1. Yahoo Finance real-time API (primary)")
    print("2. Finnhub as backup (sign up for free at finnhub.io)")
    print("3. yfinance library as final fallback")
    
    print(f"\n💡 For production use:")
    print("• Get free API keys from finnhub.io or alphavantage.co")
    print("• These provide truly real-time data during market hours")
    print("• Yahoo Finance direct API is fastest and most reliable")
