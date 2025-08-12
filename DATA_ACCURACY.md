# 📊 **Data Accuracy & Source Information**

## **Why Stock Prices May Differ Between Platforms**

### **Our Data Source: Yahoo Finance**
- **Provider**: Yahoo Finance via yfinance Python library
- **Update Frequency**: Real-time during market hours
- **Delay**: Typically 0-20 minutes depending on exchange
- **Coverage**: NYSE, NASDAQ, and major global exchanges

### **Common Differences You Might See:**

#### **1. 🕐 Timing Differences**
- **Google Finance**: May update every 15-20 minutes
- **Yahoo Finance**: Updates vary by exchange (0-20 minutes)
- **Bloomberg**: Often real-time but expensive data
- **Robinhood**: Real-time but different calculation methods

#### **2. 💰 Price Calculation Methods**
- **Current Price**: Last traded price vs bid/ask midpoint
- **Previous Close**: Different exchanges report different closing times
- **After-Hours**: Some show after-hours, others don't
- **Volume**: Can include or exclude after-hours volume

#### **3. 📈 Fundamental Data Variations**
- **P/E Ratio**: Trailing 12 months vs forward P/E vs adjusted P/E
- **Market Cap**: Shares outstanding calculations differ
- **52-Week High/Low**: Different start dates or adjustment methods
- **Volume**: Regular hours only vs including pre/post market

#### **4. 🌍 Exchange Differences**
- **Currency Conversion**: Real-time vs end-of-day rates
- **International Stocks**: ADRs vs local exchange prices
- **Dividend Adjustments**: Ex-dividend date handling varies

### **Our Data Quality Standards:**

✅ **What We Guarantee:**
- Data directly from Yahoo Finance (same as Yahoo's website)
- Real-time updates during market hours
- Accurate historical data for charts
- Professional-grade technical indicators

⚠️ **Potential Discrepancies:**
- 15-20 minute delay during heavy trading
- After-hours prices may not match Google Finance
- P/E ratios calculated using trailing 12-month earnings
- Volume includes regular trading hours only

### **How to Verify Data Accuracy:**

1. **Cross-Reference**: Compare with Yahoo Finance website directly
2. **Check Timestamps**: Look for "Last Updated" time in our app
3. **Market Hours**: Prices freeze after 4 PM EST until next trading day
4. **Volume Spikes**: High volume days may have slight reporting delays

### **Why We Chose Yahoo Finance:**

- **Free & Reliable**: No API limits or costs
- **Comprehensive**: Includes technical indicators, fundamentals
- **Global Coverage**: International stocks and indices
- **Historical Data**: Years of accurate historical prices
- **Industry Standard**: Used by many financial applications

---

**💡 Pro Tip**: If you see different prices on Google Finance, it's usually due to timing differences or after-hours trading. Our data matches Yahoo Finance exactly because we use their official API.

**🔄 Data Updates**: Market data refreshes automatically every 30 seconds during trading hours.
