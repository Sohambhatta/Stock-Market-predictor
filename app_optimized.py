"""
Optimized Stock Market Analysis Web Application
Lightweight version with lazy loading and mobile optimization
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import threading
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables for lazy loading
_financial_analyzer = None
_news_collector = None
_recommender = None
_loading_status = {"status": "not_started", "progress": 0}

# Lightweight cache for frequently accessed data
_cache = {
    'popular_stocks': {},
    'last_cache_update': None
}

# Popular stocks data (static, loads instantly)
POPULAR_STOCKS = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
    'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology'},
    'MSFT': {'name': 'Microsoft Corp.', 'sector': 'Technology'},
    'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive'},
    'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'E-commerce'},
    'META': {'name': 'Meta Platforms Inc.', 'sector': 'Social Media'},
    'NVDA': {'name': 'NVIDIA Corp.', 'sector': 'Technology'},
    'NFLX': {'name': 'Netflix Inc.', 'sector': 'Entertainment'},
    'JPM': {'name': 'JPMorgan Chase', 'sector': 'Banking'},
    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare'}
}

def lazy_load_modules():
    """Load heavy modules in background"""
    global _financial_analyzer, _news_collector, _recommender, _loading_status
    
    try:
        _loading_status["status"] = "loading"
        _loading_status["progress"] = 10
        
        # Load modules one by one with progress updates
        print("Loading financial analysis module...")
        from advanced_financial_analysis import AdvancedFinancialAnalysis
        _financial_analyzer = AdvancedFinancialAnalysis()
        _loading_status["progress"] = 40
        
        print("Loading news collector...")
        from live_news_collector import LiveNewsCollector
        _news_collector = LiveNewsCollector()
        _loading_status["progress"] = 70
        
        print("Loading recommendation engine...")
        from stock_recommender import StockRecommendationEngine
        _recommender = StockRecommendationEngine()
        _loading_status["progress"] = 100
        
        _loading_status["status"] = "ready"
        print("All modules loaded successfully!")
        
    except Exception as e:
        _loading_status["status"] = "error"
        _loading_status["error"] = str(e)
        print(f"Error loading modules: {e}")

def get_basic_stock_data(symbol: str):
    """Get basic stock data without heavy dependencies"""
    try:
        import yfinance as yf
        stock = yf.Ticker(symbol)
        
        # Get minimal data for quick response
        info = stock.info
        hist = stock.history(period="5d")  # Just 5 days for quick load
        
        if hist.empty:
            return None
            
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100
        
        return {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'price': round(current_price, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    """Main page - loads instantly"""
    return render_template('index_optimized.html')

@app.route('/api/status')
def get_status():
    """Get loading status of heavy modules"""
    return jsonify(_loading_status)

@app.route('/api/quick-search/<query>')
def quick_search(query):
    """Quick search without heavy analysis - instant response"""
    query = query.upper()
    results = []
    
    # Search in popular stocks first
    for symbol, data in POPULAR_STOCKS.items():
        if query in symbol or query.lower() in data['name'].lower():
            basic_data = get_basic_stock_data(symbol)
            if basic_data and 'error' not in basic_data:
                results.append({
                    'symbol': symbol,
                    'name': data['name'],
                    'sector': data['sector'],
                    'price': basic_data['price'],
                    'change_percent': basic_data['change_percent']
                })
    
    # If exact match not found in popular stocks, try direct lookup
    if not results and len(query) <= 5:
        basic_data = get_basic_stock_data(query)
        if basic_data and 'error' not in basic_data:
            results.append(basic_data)
    
    return jsonify(results[:10])  # Limit to 10 results

@app.route('/api/basic-analyze/<symbol>')
def basic_analyze(symbol):
    """Basic analysis - fast response without heavy calculations"""
    basic_data = get_basic_stock_data(symbol)
    if not basic_data or 'error' in basic_data:
        return jsonify({'error': 'Stock not found'})
    
    # Add simple trend analysis
    try:
        import yfinance as yf
        stock = yf.Ticker(symbol)
        hist = stock.history(period="30d")
        
        if not hist.empty:
            # Simple moving averages
            ma_5 = hist['Close'].rolling(5).mean().iloc[-1]
            ma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            current_price = hist['Close'].iloc[-1]
            
            # Simple signals
            trend = "bullish" if current_price > ma_20 > ma_5 else "bearish"
            
            basic_data.update({
                'trend': trend,
                'ma_5': round(ma_5, 2),
                'ma_20': round(ma_20, 2),
                'volume_avg': int(hist['Volume'].mean()),
                'high_52w': round(hist['High'].max(), 2),
                'low_52w': round(hist['Low'].min(), 2)
            })
    
    except Exception as e:
        basic_data['analysis_error'] = str(e)
    
    return jsonify(basic_data)

@app.route('/api/advanced-analyze/<symbol>')
def advanced_analyze(symbol):
    """Advanced analysis - requires modules to be loaded"""
    if _loading_status["status"] != "ready":
        return jsonify({
            'error': 'Advanced analysis not ready yet',
            'status': _loading_status["status"],
            'progress': _loading_status["progress"]
        })
    
    try:
        # Use the loaded modules for advanced analysis
        analysis = _financial_analyzer.comprehensive_analysis(symbol)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/recommendations')
def get_recommendations():
    """Get stock recommendations"""
    if _loading_status["status"] != "ready":
        # Return basic recommendations while modules load
        basic_recs = []
        for symbol in ['AAPL', 'GOOGL', 'MSFT', 'TSLA']:
            basic_data = get_basic_stock_data(symbol)
            if basic_data and 'error' not in basic_data:
                action = "BUY" if basic_data['change_percent'] < -2 else "HOLD"
                basic_recs.append({
                    'symbol': symbol,
                    'action': action,
                    'confidence': 0.6,
                    'reason': 'Basic trend analysis',
                    'price': basic_data['price'],
                    'change_percent': basic_data['change_percent']
                })
        return jsonify(basic_recs)
    
    try:
        # Use advanced recommender when ready
        recommendations = _recommender.get_recommendations()
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/market-overview')
def market_overview():
    """Quick market overview"""
    overview = []
    
    # Get data for major indices and popular stocks
    symbols = ['SPY', 'QQQ', 'AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    for symbol in symbols:
        data = get_basic_stock_data(symbol)
        if data and 'error' not in data:
            overview.append(data)
    
    return jsonify({
        'market_data': overview,
        'last_updated': datetime.now().isoformat(),
        'status': 'Market data current as of ' + datetime.now().strftime('%H:%M EST')
    })

@app.route('/api/chart-data/<symbol>')
def get_chart_data(symbol):
    """Get chart data - optimized for mobile"""
    try:
        import yfinance as yf
        
        period = request.args.get('period', '1mo')
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return jsonify({'error': 'No data found'})
        
        # Reduce data points for mobile
        if len(hist) > 50:
            # Sample every nth point to reduce data
            step = len(hist) // 50
            hist = hist.iloc[::step]
        
        chart_data = {
            'dates': [date.strftime('%Y-%m-%d') for date in hist.index],
            'prices': hist['Close'].round(2).tolist(),
            'volumes': hist['Volume'].tolist(),
            'symbol': symbol
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Optimized Stock Market Analysis App...")
    print("📱 Mobile-friendly version with lazy loading")
    
    # Start background loading of heavy modules
    threading.Thread(target=lazy_load_modules, daemon=True).start()
    
    # Start the app immediately
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
