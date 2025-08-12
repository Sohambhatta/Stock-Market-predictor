import yfinance as yf
import requests
from datetime import datetime
import time

def get_yahoo_data(symbol):
    """Get data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1d")
        
        return {
            'source': 'Yahoo Finance (yfinance)',
            'price': info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 0),
            'previous_close': info.get('previousClose', 0),
            'open': info.get('open', hist['Open'].iloc[-1] if not hist.empty else 0),
            'day_high': info.get('dayHigh', hist['High'].iloc[-1] if not hist.empty else 0),
            'day_low': info.get('dayLow', hist['Low'].iloc[-1] if not hist.empty else 0),
            'volume': info.get('volume', hist['Volume'].iloc[-1] if not hist.empty else 0),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {'error': str(e)}

def get_alternative_yahoo_data(symbol):
    """Get data using direct Yahoo Finance API"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        if 'chart' in data and data['chart']['result']:
            result = data['chart']['result'][0]
            meta = result['meta']
            
            return {
                'source': 'Yahoo Finance (Direct API)',
                'price': meta.get('regularMarketPrice', 0),
                'previous_close': meta.get('previousClose', 0),
                'open': meta.get('regularMarketOpen', 0),
                'day_high': meta.get('regularMarketDayHigh', 0),
                'day_low': meta.get('regularMarketDayLow', 0),
                'volume': meta.get('regularMarketVolume', 0),
                'market_cap': 'N/A (not in this API)',
                'pe_ratio': 'N/A (not in this API)',
                'fifty_two_week_high': meta.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': meta.get('fiftyTwoWeekLow', 0),
                'last_updated': datetime.fromtimestamp(meta.get('regularMarketTime', time.time())).strftime('%Y-%m-%d %H:%M:%S')
            }
    except Exception as e:
        return {'error': str(e)}

def compare_data_sources(symbol):
    """Compare data from different sources"""
    print(f"🔍 Comparing stock data for {symbol.upper()}")
    print("=" * 80)
    
    # Get data from different sources
    yahoo_yfinance = get_yahoo_data(symbol)
    yahoo_direct = get_alternative_yahoo_data(symbol)
    
    sources = [yahoo_yfinance, yahoo_direct]
    
    # Print comparison
    for i, source_data in enumerate(sources):
        if 'error' in source_data:
            print(f"\n❌ {source_data.get('source', f'Source {i+1}')} - Error: {source_data['error']}")
            continue
            
        print(f"\n📊 {source_data['source']}:")
        print(f"   Current Price: ${source_data['price']:.2f}")
        print(f"   Previous Close: ${source_data['previous_close']:.2f}")
        print(f"   Open: ${source_data['open']:.2f}")
        print(f"   Day High: ${source_data['day_high']:.2f}")
        print(f"   Day Low: ${source_data['day_low']:.2f}")
        print(f"   Volume: {source_data['volume']:,}")
        print(f"   Market Cap: {source_data['market_cap']}")
        print(f"   P/E Ratio: {source_data['pe_ratio']}")
        print(f"   52W High: ${source_data['fifty_two_week_high']:.2f}")
        print(f"   52W Low: ${source_data['fifty_two_week_low']:.2f}")
        print(f"   Last Updated: {source_data['last_updated']}")
    
    # Calculate differences
    if len(sources) >= 2 and 'error' not in sources[0] and 'error' not in sources[1]:
        price_diff = abs(sources[0]['price'] - sources[1]['price'])
        price_diff_percent = (price_diff / sources[0]['price']) * 100 if sources[0]['price'] > 0 else 0
        
        print(f"\n📈 Price Comparison:")
        print(f"   Difference: ${price_diff:.2f} ({price_diff_percent:.3f}%)")
        
        if price_diff_percent > 0.1:
            print(f"   ⚠️  WARNING: Price difference >0.1% - sources may be out of sync")
        else:
            print(f"   ✅ Price difference is minimal - data sources are consistent")

if __name__ == "__main__":
    # Test with popular stocks
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    for stock in test_stocks:
        compare_data_sources(stock)
        print("\n" + "="*80 + "\n")
        time.sleep(1)  # Be nice to APIs
    
    print("🎯 Data Source Analysis Complete!")
    print("\n💡 Key Points:")
    print("   • Yahoo Finance data can have slight delays (15-20 minutes)")
    print("   • Different APIs may show different timestamps")
    print("   • After-hours trading affects current vs previous close")
    print("   • Market cap calculations can vary between sources")
    print("   • Volume numbers may differ due to reporting timing")
