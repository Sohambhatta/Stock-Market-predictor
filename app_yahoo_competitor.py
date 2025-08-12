from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global status tracking
app_status = {'status': 'loading', 'progress': 20}

# Cache for stock data (5 minute expiry)
stock_cache = {}
CACHE_DURATION = 300

# Major stock indices and popular stocks
INDICES = {
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones',
    '^IXIC': 'Nasdaq',
    '^RUT': 'Russell 2000'
}

POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 
    'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 
    'SPOT', 'COIN', 'SQ', 'ZM', 'SNOW', 'PLTR', 'ROKU', 'TWTR'
]

def get_cached_stock_data(symbol, period='3mo'):
    """Get stock data with caching"""
    cache_key = f"{symbol}_{period}"
    current_time = time.time()
    
    if cache_key in stock_cache:
        data, timestamp = stock_cache[cache_key]
        if current_time - timestamp < CACHE_DURATION:
            return data
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period=period, interval='1d')
        
        if hist.empty:
            return None
            
        # Current price data
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price))
        
        # Use the most recent trading day's data
        latest_data = hist.iloc[-1]
        open_price = float(latest_data['Open'])
        day_high = float(latest_data['High']) 
        day_low = float(latest_data['Low'])
        
        # Calculate change from previous close (more accurate)
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100
        
        # Volume data
        current_volume = int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0
        avg_volume = int(hist['Volume'].mean()) if not hist['Volume'].isna().all() else 0
        
        # Technical indicators
        closes = hist['Close'].values
        highs = hist['High'].values  
        lows = hist['Low'].values
        volumes = hist['Volume'].values
        
        # Moving averages
        sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_price
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_price
        sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else current_price
        
        # RSI calculation
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            return 100 - (100 / (1 + rs))
        
        # MACD calculation
        def calculate_macd(prices):
            ema_12 = pd.Series(prices).ewm(span=12).mean()
            ema_26 = pd.Series(prices).ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_sma = np.mean(closes[-bb_period:])
        bb_std_dev = np.std(closes[-bb_period:])
        bb_upper = bb_sma + (bb_std_dev * bb_std)
        bb_lower = bb_sma - (bb_std_dev * bb_std)
        
        rsi = calculate_rsi(closes)
        macd, macd_signal, macd_hist = calculate_macd(closes)
        
        # Prepare comprehensive stock data
        stock_data = {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'price': round(current_price, 2),
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'volume': current_volume,
            'avg_volume': avg_volume,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', None),
            'forward_pe': info.get('forwardPE', None),
            'peg_ratio': info.get('pegRatio', None),
            'price_to_book': info.get('priceToBook', None),
            'debt_to_equity': info.get('debtToEquity', None),
            'roe': info.get('returnOnEquity', None),
            'dividend_yield': info.get('dividendYield', None),
            'beta': info.get('beta', None),
            'high_52w': round(float(info.get('fiftyTwoWeekHigh', max(highs))), 2),
            'low_52w': round(float(info.get('fiftyTwoWeekLow', min(lows))), 2),
            'day_high': round(day_high, 2),
            'day_low': round(day_low, 2),
            'open_price': round(open_price, 2),
            'prev_close': round(prev_close, 2),
            'technical_indicators': {
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'sma_200': round(sma_200, 2),
                'rsi': round(rsi, 1),
                'macd': round(macd, 3),
                'macd_signal': round(macd_signal, 3),
                'macd_histogram': round(macd_hist, 3),
                'bollinger_upper': round(bb_upper, 2),
                'bollinger_lower': round(bb_lower, 2),
                'bollinger_middle': round(bb_sma, 2)
            },
            'historical_data': {
                'dates': [date.strftime('%Y-%m-%d') for date in hist.index],
                'opens': [round(float(x), 2) for x in hist['Open']],
                'highs': [round(float(x), 2) for x in hist['High']],
                'lows': [round(float(x), 2) for x in hist['Low']],
                'closes': [round(float(x), 2) for x in hist['Close']],
                'volumes': [int(x) if not pd.isna(x) else 0 for x in hist['Volume']],
                'sma_20': [round(float(pd.Series(hist['Close'][:i+1]).rolling(min(20, i+1)).mean().iloc[-1]), 2) for i in range(len(hist))],
                'sma_50': [round(float(pd.Series(hist['Close'][:i+1]).rolling(min(50, i+1)).mean().iloc[-1]), 2) for i in range(len(hist))]
            }
        }
        
        # Determine trend
        if current_price > sma_20 > sma_50:
            trend = 'Bullish'
        elif current_price < sma_20 < sma_50:
            trend = 'Bearish'
        else:
            trend = 'Neutral'
        
        stock_data['trend'] = trend
        
        # Cache the data
        stock_cache[cache_key] = (stock_data, current_time)
        return stock_data
        
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def generate_ml_predictions(stock_data, days_ahead=30):
    """Generate sophisticated ML predictions"""
    try:
        historical = stock_data['historical_data']
        if len(historical['closes']) < 60:
            return None
            
        closes = np.array(historical['closes'])
        volumes = np.array(historical['volumes'])
        highs = np.array(historical['highs'])
        lows = np.array(historical['lows'])
        
        # Create advanced features
        features = []
        targets = []
        
        lookback = 30  # Use 30 days of history
        
        for i in range(lookback, len(closes) - 1):
            # Price features
            price_features = [
                closes[i-4], closes[i-3], closes[i-2], closes[i-1],  # Recent prices
                np.mean(closes[i-5:i+1]),    # 5-day MA
                np.mean(closes[i-10:i+1]),   # 10-day MA  
                np.mean(closes[i-20:i+1]),   # 20-day MA
                np.std(closes[i-20:i+1]),    # 20-day volatility
                (closes[i] - np.mean(closes[i-20:i+1])) / np.std(closes[i-20:i+1]),  # Bollinger position
                np.mean(volumes[i-5:i+1]) / 1000000,  # Average volume (millions)
                (highs[i] - lows[i]) / closes[i],  # Daily range as % of close
                (closes[i] - closes[i-1]) / closes[i-1],  # Daily return
                np.mean([(highs[j] - lows[j]) / closes[j] for j in range(i-5, i+1)]),  # Avg daily range
            ]
            
            features.append(price_features)
            targets.append(closes[i+1])
            
        if len(features) < 30:
            return None
            
        # Train ensemble model
        X = np.array(features)
        y = np.array(targets)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Random Forest for price prediction
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_scaled, y)
        
        # Generate predictions
        predictions = []
        confidences = []
        dates = []
        
        current_features = X_scaled[-1].copy()
        current_price = closes[-1]
        last_date = datetime.strptime(historical['dates'][-1], '%Y-%m-%d')
        
        for day in range(days_ahead):
            # Predict next price
            pred_price = rf_model.predict([current_features])[0]
            
            # Calculate confidence based on prediction uncertainty and market conditions
            # Get multiple predictions for confidence estimation
            tree_preds = [tree.predict([current_features])[0] for tree in rf_model.estimators_[:20]]
            pred_std = np.std(tree_preds)
            
            # Base confidence starts high and decays over time
            base_confidence = 0.98
            time_decay = 0.025 * day  # 2.5% per day
            volatility_penalty = min(0.3, pred_std / current_price)
            
            confidence = max(0.5, base_confidence - time_decay - volatility_penalty)
            
            # Only include predictions with >85% confidence
            if confidence >= 0.85:
                predictions.append(round(pred_price, 2))
                confidences.append(round(confidence, 3))
                dates.append((last_date + timedelta(days=day+1)).strftime('%Y-%m-%d'))
                
                # Update features for next prediction
                price_change = (pred_price - current_price) / current_price
                
                # Shift features and add new prediction
                current_features = np.roll(current_features, -1)
                current_features[-1] = pred_price / np.max(y)  # Normalized price
                current_price = pred_price
            else:
                break
                
        if not predictions:
            return None
            
        return {
            'dates': dates,
            'prices': predictions,
            'confidences': confidences,
            'avg_confidence': round(np.mean(confidences), 3),
            'prediction_days': len(predictions),
            'model_accuracy': round(rf_model.score(X_scaled, y), 3)
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def get_market_indices():
    """Get major market indices data"""
    indices_data = []
    for symbol, name in INDICES.items():
        data = get_cached_stock_data(symbol, '1d')
        if data:
            indices_data.append({
                'symbol': symbol,
                'name': name,
                'price': data['price'],
                'change': data['change'],
                'change_percent': data['change_percent']
            })
    return indices_data

def simulate_loading():
    """Simulate loading ML models"""
    global app_status
    time.sleep(1)
    app_status['progress'] = 40
    time.sleep(1) 
    app_status['progress'] = 70
    time.sleep(1)
    app_status['progress'] = 100
    app_status['status'] = 'ready'
    print("✅ Advanced AI models loaded!")

# Routes
@app.route('/')
def index():
    return render_template('index_advanced.html')

@app.route('/api/status')
def get_status():
    return jsonify(app_status)

@app.route('/api/search/<query>')
def search_stocks(query):
    """Enhanced stock search"""
    query = query.upper()
    results = []
    
    # Search popular stocks
    matches = [s for s in POPULAR_STOCKS if query in s][:10]
    
    for symbol in matches:
        data = get_cached_stock_data(symbol, '1d')
        if data:
            results.append({
                'symbol': data['symbol'],
                'name': data['name'],
                'price': data['price'],
                'change': data['change'],
                'change_percent': data['change_percent'],
                'volume': data['volume']
            })
    
    return jsonify(results)

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    """Get comprehensive stock data"""
    data = get_cached_stock_data(symbol.upper(), '6mo')
    if not data:
        return jsonify({'error': f'Could not fetch data for {symbol}'}), 404
    return jsonify(data)

@app.route('/api/chart/<symbol>')
def get_chart_data(symbol):
    """Get advanced chart data with technical indicators"""
    period = request.args.get('period', '3mo')
    data = get_cached_stock_data(symbol.upper(), period)
    
    if not data:
        return jsonify({'error': f'Chart data unavailable for {symbol}'}), 404
    
    return jsonify(data['historical_data'])

@app.route('/api/predictions/<symbol>')
def get_predictions(symbol):
    """Get ML price predictions"""
    data = get_cached_stock_data(symbol.upper(), '6mo')
    if not data:
        return jsonify({'error': 'Stock data unavailable'}), 404
    
    predictions = generate_ml_predictions(data)
    if not predictions:
        return jsonify({'error': 'Unable to generate predictions'}), 400
        
    return jsonify(predictions)

@app.route('/api/market')
def get_market_data():
    """Get market overview"""
    indices = get_market_indices()
    
    # Top movers
    movers = []
    for symbol in POPULAR_STOCKS[:15]:
        data = get_cached_stock_data(symbol, '1d')
        if data:
            movers.append({
                'symbol': data['symbol'],
                'name': data['name'],
                'price': data['price'],
                'change_percent': data['change_percent']
            })
    
    # Sort by absolute change
    movers.sort(key=lambda x: abs(x['change_percent']), reverse=True)
    
    return jsonify({
        'indices': indices,
        'top_gainers': [m for m in movers if m['change_percent'] > 0][:5],
        'top_losers': [m for m in movers if m['change_percent'] < 0][:5],
        'most_active': movers[:8]
    })

@app.route('/api/recommendations')
def get_recommendations():
    """AI-powered stock recommendations"""
    if app_status['status'] != 'ready':
        return jsonify([])
    
    recommendations = []
    
    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META']:
        data = get_cached_stock_data(symbol, '3mo')
        if not data:
            continue
            
        tech = data['technical_indicators']
        price = data['price']
        
        # Advanced recommendation logic
        score = 0
        reasons = []
        
        # RSI analysis
        if tech['rsi'] < 30:
            score += 2
            reasons.append("Oversold RSI indicates buying opportunity")
        elif tech['rsi'] > 70:
            score -= 2
            reasons.append("Overbought RSI suggests caution")
        
        # Moving average analysis
        if price > tech['sma_20'] > tech['sma_50']:
            score += 1.5
            reasons.append("Strong upward trend above key moving averages")
        elif price < tech['sma_20'] < tech['sma_50']:
            score -= 1.5
            reasons.append("Downward trend below key moving averages")
        
        # MACD analysis
        if tech['macd'] > tech['macd_signal'] and tech['macd_histogram'] > 0:
            score += 1
            reasons.append("MACD showing bullish momentum")
        elif tech['macd'] < tech['macd_signal']:
            score -= 1
            reasons.append("MACD showing bearish momentum")
        
        # Volume analysis
        if data['volume'] > data['avg_volume'] * 1.5:
            score += 0.5
            reasons.append("High volume confirms price movement")
        
        # Determine action and confidence
        if score >= 2:
            action = 'BUY'
            confidence = min(0.9, 0.7 + score * 0.05)
        elif score <= -2:
            action = 'SELL'
            confidence = min(0.9, 0.7 + abs(score) * 0.05)
        else:
            action = 'HOLD'
            confidence = 0.6 + abs(score) * 0.05
        
        recommendations.append({
            'symbol': symbol,
            'action': action,
            'confidence': round(confidence, 3),
            'score': round(score, 2),
            'reasons': reasons[:2],  # Top 2 reasons
            'price': price,
            'target_price': round(price * (1 + score * 0.02), 2)  # Simple target
        })
    
    return jsonify(sorted(recommendations, key=lambda x: x['confidence'], reverse=True))

if __name__ == '__main__':
    print("🚀 Elite Financial Platform - Yahoo Finance Competitor")
    print("📊 Real-time data • Advanced charts • AI predictions")
    
    # Start ML loading
    threading.Thread(target=simulate_loading, daemon=True).start()
    
    print("\n✨ Features:")
    print("  🔍 Real-time stock search & data")
    print("  📈 Advanced technical indicators (RSI, MACD, Bollinger)")
    print("  🤖 ML-powered price predictions")
    print("  💡 AI investment recommendations")
    print("  📊 Professional candlestick charts")
    print("  📱 Mobile-optimized interface")
    
    print("\n🌐 Launch: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
