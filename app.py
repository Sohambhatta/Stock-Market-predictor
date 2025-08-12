"""
Stock Market Analysis Web Application
Modern Flask backend with advanced financial analysis
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from datetime import datetime, timedelta
import yfinance as yf

# Import our modules
from advanced_financial_analysis import AdvancedFinancialAnalysis
from live_news_collector import LiveNewsCollector
from stock_recommender import StockRecommendationEngine

app = Flask(__name__)
CORS(app)

# Initialize analysis engines
financial_analyzer = AdvancedFinancialAnalysis()
news_collector = LiveNewsCollector()
recommender = StockRecommendationEngine()

# Popular stocks for search suggestions
POPULAR_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation',
    'META': 'Meta Platforms Inc.',
    'NFLX': 'Netflix Inc.',
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ Trust',
    'BRK.B': 'Berkshire Hathaway Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart Inc.',
    'PG': 'Procter & Gamble Co.',
    'UNH': 'UnitedHealth Group Inc.',
    'DIS': 'The Walt Disney Company',
    'HD': 'The Home Depot Inc.',
    'MA': 'Mastercard Incorporated'
}

@app.route('/')
def index():
    """Serve the main web application"""
    return render_template('index.html')

@app.route('/api/search/<query>')
def search_stocks(query):
    """Search for stocks by symbol or company name"""
    try:
        query = query.upper().strip()
        results = []
        
        # Search in popular stocks
        for symbol, name in POPULAR_STOCKS.items():
            if query in symbol or query.lower() in name.lower():
                results.append({
                    'symbol': symbol,
                    'name': name,
                    'type': 'stock'
                })
        
        # If no results in popular stocks, try yfinance
        if not results and len(query) >= 2:
            try:
                ticker = yf.Ticker(query)
                info = ticker.info
                if info.get('symbol'):
                    results.append({
                        'symbol': query,
                        'name': info.get('longName', query),
                        'type': 'stock'
                    })
            except:
                pass
        
        return jsonify({'results': results[:10]})  # Limit to 10 results
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/<symbol>')
def analyze_stock(symbol):
    """Get comprehensive stock analysis"""
    try:
        symbol = symbol.upper().strip()
        
        # Get comprehensive analysis
        analysis = financial_analyzer.get_comprehensive_data(symbol)
        
        if 'error' in analysis:
            return jsonify(analysis), 400
        
        # Get news analysis
        try:
            news_analysis = news_collector.get_sentiment_for_stock(symbol)
            analysis['news_sentiment'] = news_analysis
        except:
            analysis['news_sentiment'] = {'error': 'News analysis unavailable'}
        
        # Create charts data
        charts_data = create_charts_data(analysis['historical_data'], symbol)
        analysis['charts'] = charts_data
        
        # Remove large historical data from response (already in charts)
        analysis.pop('historical_data', None)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations')
def get_recommendations():
    """Get top stock recommendations"""
    try:
        # Get recommendations for popular stocks
        recommendations = []
        
        for symbol in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']:
            try:
                analysis = financial_analyzer.get_comprehensive_data(symbol, period="6mo")
                if 'error' not in analysis:
                    rec = analysis['recommendation']
                    rec['symbol'] = symbol
                    rec['company_name'] = analysis['company_name']
                    rec['current_price'] = round(analysis['current_price'], 2)
                    rec['sector'] = analysis['sector']
                    recommendations.append(rec)
            except:
                continue
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({'recommendations': recommendations[:10]})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-overview')
def market_overview():
    """Get market overview with major indices"""
    try:
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000',
            '^VIX': 'VIX (Fear Index)'
        }
        
        market_data = []
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = current - previous
                    change_pct = (change / previous) * 100 if previous != 0 else 0
                    
                    market_data.append({
                        'symbol': symbol,
                        'name': name,
                        'price': round(current, 2),
                        'change': round(change, 2),
                        'change_percent': round(change_pct, 2)
                    })
            except:
                continue
        
        return jsonify({'market_data': market_data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trending')
def get_trending():
    """Get trending stocks based on news mentions"""
    try:
        # Get trending symbols from news
        trending_symbols = ['AAPL', 'TSLA', 'NVDA', 'META', 'GOOGL']  # Simplified for demo
        trending_data = []
        
        for symbol in trending_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[0] if len(hist) > 1 else current
                    change_pct = ((current - previous) / previous) * 100 if previous != 0 else 0
                    
                    trending_data.append({
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'price': round(current, 2),
                        'change_percent': round(change_pct, 2),
                        'volume': info.get('volume', 0)
                    })
            except:
                continue
        
        # Sort by absolute change percentage
        trending_data.sort(key=lambda x: abs(x['change_percent']), reverse=True)
        
        return jsonify({'trending': trending_data[:8]})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_charts_data(hist_data, symbol):
    """Create Plotly charts data for the stock"""
    try:
        # Candlestick chart
        candlestick = go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name=f'{symbol} Price'
        )
        
        candlestick_fig = go.Figure(data=[candlestick])
        candlestick_fig.update_layout(
            title=f'{symbol} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            height=400
        )
        
        # Volume chart
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            name='Volume',
            marker_color='rgba(158, 71, 251, 0.6)'
        ))
        
        volume_fig.update_layout(
            title=f'{symbol} Volume',
            xaxis_title='Date',
            yaxis_title='Volume',
            template='plotly_dark',
            height=200
        )
        
        # Technical indicators chart
        tech_fig = go.Figure()
        
        # Price line
        tech_fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            name='Close Price',
            line=dict(color='white', width=2)
        ))
        
        # Moving averages
        if 'SMA_20' in hist_data.columns:
            tech_fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))
        
        if 'SMA_50' in hist_data.columns:
            tech_fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['SMA_50'],
                name='SMA 50',
                line=dict(color='blue', width=1)
            ))
        
        # Bollinger Bands
        if all(col in hist_data.columns for col in ['BB_High', 'BB_Low']):
            tech_fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['BB_High'],
                name='BB Upper',
                line=dict(color='rgba(255,255,255,0.3)', width=1),
                fill=None
            ))
            
            tech_fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['BB_Low'],
                name='BB Lower',
                line=dict(color='rgba(255,255,255,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(255,255,255,0.1)'
            ))
        
        tech_fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            height=400
        )
        
        # RSI chart
        rsi_fig = go.Figure()
        if 'RSI' in hist_data.columns:
            rsi_fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['RSI'],
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            
            # RSI levels
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            rsi_fig.add_hline(y=50, line_dash="dot", line_color="gray")
        
        rsi_fig.update_layout(
            title='RSI (Relative Strength Index)',
            xaxis_title='Date',
            yaxis_title='RSI',
            template='plotly_dark',
            height=200,
            yaxis=dict(range=[0, 100])
        )
        
        return {
            'candlestick': json.dumps(candlestick_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'volume': json.dumps(volume_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'technical': json.dumps(tech_fig, cls=plotly.utils.PlotlyJSONEncoder),
            'rsi': json.dumps(rsi_fig, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
