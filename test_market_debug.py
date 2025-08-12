import yfinance as yf
import pandas as pd
from datetime import datetime

def test_market_data():
    """Test market data to see what's going wrong"""
    print("🔍 Testing Market Data Calculation...")
    
    indices_symbols = ['^GSPC', '^DJI', '^IXIC']
    indices_names = ['S&P 500', 'Dow Jones', 'Nasdaq']
    
    for symbol, name in zip(indices_symbols, indices_names):
        print(f"\n📊 Testing {name} ({symbol}):")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='5d', interval='1d')
            
            print(f"Info keys: {list(info.keys())[:10]}...")
            print(f"History shape: {hist.shape}")
            print(f"History columns: {hist.columns.tolist()}")
            print(f"Last 2 closes: {hist['Close'].tail(2).tolist()}")
            
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                print(f"Current price: ${current_price}")
                
                # Try different ways to get previous close
                prev_close_info = info.get('previousClose')
                prev_close_hist = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                
                print(f"Previous close (info): {prev_close_info}")
                print(f"Previous close (hist): {prev_close_hist}")
                
                # Use the better option
                prev_close = prev_close_info if prev_close_info else prev_close_hist
                print(f"Using prev_close: ${prev_close}")
                
                if prev_close and prev_close > 0:
                    change = current_price - prev_close
                    change_percent = (change / prev_close) * 100
                    
                    print(f"Change: ${change:.2f}")
                    print(f"Change %: {change_percent:.2f}%")
                    print(f"✅ SUCCESS!")
                else:
                    print("❌ No valid previous close found")
            else:
                print("❌ No historical data")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_market_data()
