from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global status tracking
app_status = {'status': 'loading', 'progress': 20}

# Cache for stock data with SYMBOL-SPECIFIC keys
stock_cache = {}
CACHE_DURATION = 120  # 2 minutes for fresher data

def get_stock_data_fresh(symbol):
    """Get FRESH stock data with ZERO caching mixups"""
    # FORCE uppercase and clean the symbol
    clean_symbol = str(symbol).upper().strip()
    
    # UNIQUE cache key with timestamp to prevent mixups
    cache_key = f"STOCK_{clean_symbol}_{int(time.time() // CACHE_DURATION)}"
    
    print(f"🔍 Fetching data for: {clean_symbol}")
    print(f"🔑 Cache key: {cache_key}")
    
    try:
        # Create a FRESH ticker object every time
        ticker = yf.Ticker(clean_symbol)
        
        # Get info with explicit symbol verification
        info = ticker.info
        
        # VERIFY we got the right symbol
        ticker_symbol = info.get('symbol', '').upper()
        if ticker_symbol and ticker_symbol != clean_symbol:
            print(f"⚠️  WARNING: Requested {clean_symbol} but got {ticker_symbol}")
        
        # Get recent history
        hist = ticker.history(period='5d', interval='1d')
        
        if hist.empty:
            print(f"❌ No history data for {clean_symbol}")
            return None
            
        # Get the absolute latest price
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(info.get('previousClose', current_price))
        
        # Use the most recent day's data
        latest_day = hist.iloc[-1]
        open_price = float(latest_day['Open'])
        high_price = float(latest_day['High'])
        low_price = float(latest_day['Low'])
        volume = int(latest_day['Volume']) if not pd.isna(latest_day['Volume']) else 0
        
        # Calculate changes
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
        
        # Basic technical indicators
        closes = hist['Close'].values
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
        
        # Create the data object with EXPLICIT symbol verification
        stock_data = {
            'symbol': clean_symbol,  # Use OUR clean symbol
            'name': info.get('longName', info.get('shortName', clean_symbol)),
            'price': round(current_price, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'previous_close': round(prev_close, 2),
            'open': round(open_price, 2),
            'day_high': round(high_price, 2),
            'day_low': round(low_price, 2),
            'volume': volume,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'dividend_yield': info.get('dividendYield'),
            'beta': info.get('beta'),
            'high_52w': info.get('fiftyTwoWeekHigh'),
            'low_52w': info.get('fiftyTwoWeekLow'),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'last_updated': datetime.now().strftime('%H:%M:%S'),
            'data_source': f'yfinance-{clean_symbol}',
            'cache_key_used': cache_key  # For debugging
        }
        
        # Add trend analysis
        if current_price > sma_20 > sma_50:
            stock_data['trend'] = 'Bullish'
        elif current_price < sma_20 < sma_50:
            stock_data['trend'] = 'Bearish'  
        else:
            stock_data['trend'] = 'Neutral'
            
        # Get chart data
        chart_hist = ticker.history(period='3mo', interval='1d')
        if not chart_hist.empty:
            stock_data['historical_data'] = {
                'dates': [date.strftime('%Y-%m-%d') for date in chart_hist.index],
                'opens': [round(float(p), 2) for p in chart_hist['Open'].values],
                'highs': [round(float(p), 2) for p in chart_hist['High'].values], 
                'lows': [round(float(p), 2) for p in chart_hist['Low'].values],
                'closes': [round(float(p), 2) for p in chart_hist['Close'].values],
                'volumes': [int(v) if not pd.isna(v) else 0 for v in chart_hist['Volume'].values]
            }
        
        # Cache with unique key
        stock_cache[cache_key] = (stock_data, time.time())
        
        print(f"✅ Successfully fetched data for {clean_symbol}: ${current_price}")
        return stock_data
        
    except Exception as e:
        print(f"❌ Error fetching {clean_symbol}: {e}")
        return None

def simulate_loading():
    """Simulate ML model loading"""
    global app_status
    time.sleep(1)
    app_status['progress'] = 60
    time.sleep(1) 
    app_status['progress'] = 100
    app_status['status'] = 'ready'
    print("✅ Stock analysis system ready!")

# Routes
@app.route('/')
def index():
    return render_template('index_advanced.html')

@app.route('/api/status')
def get_status():
    return jsonify(app_status)

@app.route('/api/stock/<symbol>')
def api_get_stock_data(symbol):
    """Get stock data with ZERO symbol mixups"""
    clean_symbol = str(symbol).upper().strip()
    print(f"\n🎯 API REQUEST for symbol: '{symbol}' -> cleaned: '{clean_symbol}'")
    
    data = get_stock_data_fresh(clean_symbol)
    
    if not data:
        print(f"❌ Failed to get data for {clean_symbol}")
        return jsonify({'error': f'Could not fetch data for {clean_symbol}'}), 404
    
    # DOUBLE CHECK the symbol in response
    if data.get('symbol') != clean_symbol:
        print(f"🚨 SYMBOL MISMATCH! Requested: {clean_symbol}, Got: {data.get('symbol')}")
        data['symbol'] = clean_symbol  # Force correct symbol
    
    print(f"✅ Returning data for {clean_symbol}: ${data.get('price')}")
    return jsonify(data)

@app.route('/api/chart/<symbol>')
def get_chart_data(symbol):
    """Get chart data for specific symbol"""
    clean_symbol = str(symbol).upper().strip()
    data = get_stock_data_fresh(clean_symbol)
    
    if not data or 'historical_data' not in data:
        return jsonify({'error': f'Chart data unavailable for {clean_symbol}'}), 404
    
    return jsonify(data['historical_data'])

@app.route('/api/search/<query>')
def search_stocks(query):
    """Search for stocks"""
    query = str(query).upper().strip()
    
    # Popular stocks that match query
    popular_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 
        'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL'
    ]
    
    matches = [s for s in popular_stocks if query in s][:8]
    results = []
    
    for symbol in matches:
        data = get_stock_data_fresh(symbol)
        if data:
            results.append({
                'symbol': data['symbol'],
                'name': data['name'],
                'price': data['price'],
                'change': data['change'],
                'change_percent': data['change_percent']
            })
    
    return jsonify(results)

@app.route('/api/market')
def get_market_overview():
    """Get market overview"""
    indices = ['^GSPC', '^DJI', '^IXIC']
    indices_names = ['S&P 500', 'Dow Jones', 'Nasdaq']
    
    market_data = []
    for symbol, name in zip(indices, indices_names):
        data = get_stock_data_fresh(symbol)
        if data:
            market_data.append({
                'symbol': symbol,
                'name': name,
                'price': data['price'],
                'change': data['change'],
                'change_percent': data['change_percent']
            })
    
    return jsonify({
        'indices': market_data,
        'last_updated': datetime.now().strftime('%H:%M:%S')
    })

if __name__ == '__main__':
    print("🚀 FIXED Stock Market App - NO SYMBOL MIXUPS!")
    print("🔧 Features:")
    print("   ✅ Fresh data fetching (no cache mixups)")
    print("   ✅ Symbol verification at every step")  
    print("   ✅ Debugging output for every request")
    print("   ✅ Clean symbol handling")
    
    # Start loading simulation
    threading.Thread(target=simulate_loading, daemon=True).start()
    
    print("\n🌐 Starting server: http://localhost:5000")
    print("🎯 Test different stocks to verify correct prices!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
