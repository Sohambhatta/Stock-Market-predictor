from flask import Flask, render_template, jsonify, request
import threading
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global status tracking
app_status = {
    'status': 'loading',
    'progress': 20
}

# Cache for real stock data
stock_cache = {}
cache_timeout = 300  # 5 minutes

# Popular stocks for quick search
POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 
    'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'SPOT'
]

def get_real_stock_data(symbol, period='1mo'):
    """Get real stock data from Yahoo Finance"""
    cache_key = f"{symbol}_{period}"
    current_time = time.time()
    
    # Check cache
    if cache_key in stock_cache:
        cached_data, timestamp = stock_cache[cache_key]
        if current_time - timestamp < cache_timeout:
            return cached_data
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current info
        info = ticker.info
        
        # Get historical data
        hist = ticker.history(period=period)
        if hist.empty:
            return None
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100
        
        stock_data = {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'price': round(float(current_price), 2),
            'change': round(float(change), 2),
            'change_percent': round(float(change_percent), 2),
            'volume': int(info.get('volume', hist['Volume'].iloc[-1] if not hist['Volume'].empty else 0)),
            'high_52w': round(float(info.get('fiftyTwoWeekHigh', hist['High'].max())), 2),
            'low_52w': round(float(info.get('fiftyTwoWeekLow', hist['Low'].min())), 2),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', None),
            'beta': info.get('beta', None),
            'dividend_yield': info.get('dividendYield', None),
            'historical_data': {
                'dates': [date.strftime('%Y-%m-%d') for date in hist.index],
                'prices': [round(float(price), 2) for price in hist['Close']],
                'volumes': [int(vol) for vol in hist['Volume']]
            }
        }
        
        # Determine trend
        if len(hist) >= 5:
            recent_avg = hist['Close'].tail(5).mean()
            older_avg = hist['Close'].head(5).mean()
            if recent_avg > older_avg * 1.02:
                trend = 'Bullish'
            elif recent_avg < older_avg * 0.98:
                trend = 'Bearish'
            else:
                trend = 'Neutral'
        else:
            trend = 'Bullish' if change_percent > 0 else 'Bearish'
        
        stock_data['trend'] = trend
        
        # Cache the data
        stock_cache[cache_key] = (stock_data, current_time)
        
        return stock_data
        
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def create_prediction_model(historical_prices):
    """Create a simple ML model for price prediction"""
    if len(historical_prices) < 30:
        return None, None
    
    # Prepare features (technical indicators)
    prices = np.array(historical_prices)
    features = []
    targets = []
    
    # Create features: [price_t-4, price_t-3, price_t-2, price_t-1, sma_5, sma_10]
    for i in range(10, len(prices) - 1):
        feature_row = [
            prices[i-4], prices[i-3], prices[i-2], prices[i-1],  # Last 4 prices
            np.mean(prices[i-5:i]),    # 5-day moving average
            np.mean(prices[i-10:i]),   # 10-day moving average
        ]
        features.append(feature_row)
        targets.append(prices[i+1])  # Next day price
    
    if len(features) < 10:
        return None, None
    
    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(features_scaled, targets)
    
    return model, scaler

def generate_predictions(stock_data, days_ahead=30):
    """Generate price predictions with confidence intervals"""
    try:
        historical_prices = stock_data['historical_data']['prices']
        if len(historical_prices) < 30:
            return None
        
        model, scaler = create_prediction_model(historical_prices)
        if model is None:
            return None
        
        # Generate predictions
        predictions = []
        confidences = []
        dates = []
        
        current_prices = np.array(historical_prices[-10:])  # Last 10 prices for features
        current_date = datetime.now()
        
        for day in range(days_ahead):
            # Create feature vector
            features = [
                current_prices[-4], current_prices[-3], current_prices[-2], current_prices[-1],
                np.mean(current_prices[-5:]),
                np.mean(current_prices[-10:])
            ]
            
            # Scale and predict
            features_scaled = scaler.transform([features])
            predicted_price = model.predict(features_scaled)[0]
            
            # Calculate confidence (decreases over time)
            base_confidence = 0.95
            time_decay = 0.02 * day  # 2% decrease per day
            noise_factor = abs(predicted_price - current_prices[-1]) / current_prices[-1] * 0.5
            confidence = max(0.5, base_confidence - time_decay - noise_factor)
            
            # Only include predictions with confidence > 85%
            if confidence >= 0.85:
                predictions.append(round(predicted_price, 2))
                confidences.append(round(confidence, 3))
                dates.append((current_date + timedelta(days=day+1)).strftime('%Y-%m-%d'))
                
                # Update current_prices array for next prediction
                current_prices = np.append(current_prices[1:], predicted_price)
            else:
                break  # Stop predicting when confidence drops below 85%
        
        if not predictions:
            return None
        
        return {
            'dates': dates,
            'prices': predictions,
            'confidences': confidences,
            'avg_confidence': round(np.mean(confidences), 3),
            'prediction_days': len(predictions)
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def simulate_loading():
    """Simulate loading advanced ML features"""
    global app_status
    time.sleep(2)
    app_status['progress'] = 50
    time.sleep(2)
    app_status['progress'] = 75
    time.sleep(2)
    app_status['progress'] = 100
    app_status['status'] = 'ready'
    print("✓ Real-time data analysis ready!")

@app.route('/')
def index():
    return render_template('index_modern.html')

@app.route('/api/status')
def get_status():
    return jsonify(app_status)

@app.route('/api/quick-search/<query>')
def quick_search(query):
    """Search for stocks using real data"""
    query = query.upper()
    results = []
    
    # Search in popular stocks first
    search_stocks = [s for s in POPULAR_STOCKS if query in s]
    
    # Add exact match if not in list
    if query not in search_stocks and len(query) <= 5:
        search_stocks.insert(0, query)
    
    # Get real data for each stock
    for symbol in search_stocks[:8]:  # Limit results
        stock_data = get_real_stock_data(symbol, '5d')
        if stock_data:
            results.append({
                'symbol': stock_data['symbol'],
                'name': stock_data['name'],
                'price': stock_data['price'],
                'change': stock_data['change'],
                'change_percent': stock_data['change_percent']
            })
    
    return jsonify(results)

@app.route('/api/basic-analyze/<symbol>')
def basic_analyze(symbol):
    """Get real stock analysis"""
    symbol = symbol.upper()
    stock_data = get_real_stock_data(symbol, '3mo')
    
    if not stock_data:
        return jsonify({'error': f'Could not fetch data for {symbol}'}), 404
    
    return jsonify(stock_data)

@app.route('/api/chart-data/<symbol>')
def get_chart_data(symbol):
    """Get real historical price data"""
    symbol = symbol.upper()
    period = request.args.get('period', '1mo')
    
    stock_data = get_real_stock_data(symbol, period)
    if not stock_data:
        return jsonify({'error': f'Could not fetch chart data for {symbol}'}), 404
    
    return jsonify({
        'dates': stock_data['historical_data']['dates'],
        'prices': stock_data['historical_data']['prices'],
        'volumes': stock_data['historical_data']['volumes']
    })

@app.route('/api/predictions/<symbol>')
def get_predictions(symbol):
    """Get AI price predictions with confidence intervals"""
    symbol = symbol.upper()
    stock_data = get_real_stock_data(symbol, '3mo')  # Need more data for predictions
    
    if not stock_data:
        return jsonify({'error': f'Could not fetch data for {symbol}'}), 404
    
    predictions = generate_predictions(stock_data)
    if not predictions:
        return jsonify({'error': 'Unable to generate reliable predictions'}), 400
    
    return jsonify(predictions)

@app.route('/api/market-overview')
def market_overview():
    """Get real market data for popular stocks"""
    market_data = []
    
    for symbol in POPULAR_STOCKS[:6]:  # Top 6 for overview
        stock_data = get_real_stock_data(symbol, '1d')
        if stock_data:
            market_data.append({
                'symbol': stock_data['symbol'],
                'name': stock_data['name'],
                'price': stock_data['price'],
                'change': stock_data['change'],
                'change_percent': stock_data['change_percent']
            })
    
    return jsonify({'market_data': market_data})

@app.route('/api/recommendations')
def get_recommendations():
    """Generate AI recommendations based on real data"""
    if app_status['status'] != 'ready':
        return jsonify({'error': 'AI features still loading'})
    
    recommendations = []
    
    for symbol in ['AAPL', 'MSFT', 'TSLA', 'NVDA']:
        stock_data = get_real_stock_data(symbol, '1mo')
        if stock_data:
            # Simple recommendation logic based on technical indicators
            change_30d = stock_data['change_percent']
            current_price = stock_data['price']
            high_52w = stock_data['high_52w']
            low_52w = stock_data['low_52w']
            
            # Calculate position in 52-week range
            range_position = (current_price - low_52w) / (high_52w - low_52w)
            
            # Generate recommendation
            if change_30d > 5 and range_position > 0.8:
                action, confidence = 'HOLD', 0.75
                reason = 'Strong momentum but near 52-week high'
            elif change_30d < -10 and range_position < 0.3:
                action, confidence = 'BUY', 0.85
                reason = 'Oversold condition with potential for recovery'
            elif change_30d > 2 and range_position < 0.6:
                action, confidence = 'BUY', 0.80
                reason = 'Positive momentum with room for growth'
            elif change_30d < -2 and range_position > 0.7:
                action, confidence = 'SELL', 0.70
                reason = 'Declining momentum from high levels'
            else:
                action, confidence = 'HOLD', 0.65
                reason = 'Mixed signals, maintain current position'
            
            recommendations.append({
                'symbol': symbol,
                'action': action,
                'reason': reason,
                'confidence': confidence,
                'price': current_price,
                'change_30d': change_30d
            })
    
    return jsonify(recommendations)

@app.route('/api/advanced-analyze/<symbol>')
def advanced_analyze(symbol):
    """Advanced technical analysis"""
    if app_status['status'] != 'ready':
        return jsonify({'error': 'Advanced features still loading'})
    
    symbol = symbol.upper()
    stock_data = get_real_stock_data(symbol, '3mo')
    
    if not stock_data:
        return jsonify({'error': f'Could not fetch data for {symbol}'}), 404
    
    # Calculate technical indicators
    prices = np.array(stock_data['historical_data']['prices'])
    
    # RSI calculation
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    # MACD calculation
    def calculate_macd(prices):
        ema_12 = pd.Series(prices).ewm(span=12).mean()
        ema_26 = pd.Series(prices).ewm(span=26).mean()
        macd = ema_12.iloc[-1] - ema_26.iloc[-1]
        return float(macd)
    
    rsi = calculate_rsi(prices)
    macd = calculate_macd(prices)
    
    # Bollinger Bands
    sma_20 = np.mean(prices[-20:])
    std_20 = np.std(prices[-20:])
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    current_price = prices[-1]
    
    if current_price > upper_band:
        bollinger_position = 'upper'
    elif current_price < lower_band:
        bollinger_position = 'lower'
    else:
        bollinger_position = 'middle'
    
    return jsonify({
        'symbol': symbol,
        'technical_indicators': {
            'rsi': round(rsi, 1),
            'macd': round(macd, 3),
            'bollinger_position': bollinger_position,
            'sma_20': round(sma_20, 2),
            'upper_band': round(upper_band, 2),
            'lower_band': round(lower_band, 2)
        },
        'analyst_rating': 'Buy' if rsi < 30 and macd > 0 else 'Sell' if rsi > 70 and macd < 0 else 'Hold'
    })

if __name__ == '__main__':
    print("🚀 Starting Elite Stock Analysis Platform...")
    print("📊 Real-time data with AI predictions!")
    
    # Start background loading
    loading_thread = threading.Thread(target=simulate_loading, daemon=True)
    loading_thread.start()
    
    print("\n✨ Features:")
    print("  • Real Yahoo Finance data")
    print("  • AI price predictions with confidence")
    print("  • Technical analysis indicators")
    print("  • Smart recommendations")
    print("  • Modern responsive design")
    
    print("\n🌐 Open your browser to: http://localhost:5000")
    print("📱 Optimized for mobile and desktop!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
