import yfinance as yf
import pandas as pd
import numpy as np

print("Testing real stock data fetch...")
try:
    # Test Apple stock
    aapl = yf.Ticker("AAPL")
    info = aapl.info
    hist = aapl.history(period="5d")
    
    print(f"✅ Successfully fetched AAPL data!")
    print(f"Current Price: ${info.get('currentPrice', 'N/A')}")
    print(f"Company: {info.get('longName', 'Apple Inc.')}")
    print(f"Last 5 days of data: {len(hist)} rows")
    print("📊 Real data integration working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Run: pip install yfinance pandas numpy scikit-learn flask")
