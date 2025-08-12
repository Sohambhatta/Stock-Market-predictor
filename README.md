# 📈 AI-Powered Stock Sentiment Analyzer & Recommendation System

A comprehensive BERT-based system that analyzes live financial news to provide intelligent stock recommendations.

## 🚀 What This Does

This system:
1. **📰 Collects live financial news** from multiple sources
2. **🧠 Analyzes sentiment** using a fine-tuned BERT model  
3. **🎯 Extracts stock symbols** mentioned in the news
4. **💡 Generates buy/sell/hold recommendations** with confidence scores
5. **⚖️ Assesses risk levels** for each recommendation
6. **📊 Provides detailed analysis reports**

## 🏗️ Project Structure

```
├── main.py                      # Main interface - choose training or live analysis
├── live_analyzer.py             # Complete live analysis pipeline
├── live_news_collector.py       # Collects live financial news
├── stock_recommender.py         # Generates stock recommendations
├── data_utils.py               # Data preprocessing utilities
├── model_trainer.py            # BERT model training
├── visualization.py            # Charts and reports
├── setup.py                    # Automated setup script
├── config_template.py          # Configuration template
├── requirements.txt            # Package dependencies
├── README.md                   # This file
└── data/                       # Data directory (auto-created)
    ├── finance_train.csv       # Training data (auto-downloaded)
    └── finance_test.csv        # Test data (auto-downloaded)
```

## ⚡ Quick Start

### Option 1: Live Stock Analysis (Recommended)
```bash
# 1. Setup
python setup.py

# 2. Run live analysis
python main.py
# Choose option 2 for live analysis
```

### Option 2: Train Your Own Model First
```bash
# 1. Setup
python setup.py

# 2. Run training + live analysis  
python main.py
# Choose option 3 for both training and live analysis
```

## 🔧 Setup for Better Results

### Get Free API Keys (Optional but Recommended)

1. **NewsAPI** (for live news): https://newsapi.org/
   - Free: 1000 requests/month
   
2. **Alpha Vantage** (for stock prices): https://www.alphavantage.co/
   - Free: 500 requests/day

3. **Create config file:**
   ```bash
   # Copy the template
   cp config_template.py config.py
   
   # Edit config.py and add your API keys
   ```

## 📊 Sample Output

```
================================================================================
STOCK RECOMMENDATIONS REPORT
Generated: 2025-01-15 14:30:25
================================================================================

1. AAPL - BUY
   Confidence: 85.2%
   Sentiment Score: +0.67
   Current Price: $185.50
   Target Price: $197.25 (+6.3%)
   Risk Level: LOW
   Reasoning: Very positive news sentiment. High news volume (12 articles). High confidence recommendation. Low risk level.

2. TSLA - SELL  
   Confidence: 72.1%
   Sentiment Score: -0.54
   Current Price: $225.80
   Target Price: $213.65 (-5.4%)
   Risk Level: MEDIUM
   Reasoning: Negative news sentiment. Moderate news volume (8 articles). Moderate confidence recommendation. Medium risk level.

3. GOOGL - BUY
   Confidence: 68.3%
   Sentiment Score: +0.45
   Current Price: $142.30
   Target Price: $148.70 (+4.5%)
   Risk Level: LOW
   Reasoning: Positive news sentiment. Moderate news volume (6 articles). Moderate confidence recommendation. Low risk level.
```

## 🎯 Features

### Live News Analysis
- **Multi-source collection**: Financial news from major outlets
- **Smart stock extraction**: Automatically identifies mentioned stocks
- **Real-time processing**: Analyzes news as it's published

### Advanced Sentiment Analysis  
- **BERT-based model**: State-of-the-art NLP for financial text
- **Fine-tuned on financial data**: Specialized for market language
- **Confidence scoring**: Know how reliable each prediction is

### Intelligent Recommendations
- **Buy/Sell/Hold actions**: Clear trading recommendations  
- **Risk assessment**: LOW/MEDIUM/HIGH risk levels
- **Target prices**: Estimated price movements
- **Detailed reasoning**: Understand why each recommendation was made

### Risk Management
- **Confidence thresholds**: Filter out low-confidence predictions
- **Volume requirements**: Ensure sufficient news coverage
- **Volatility assessment**: Account for stock price stability
- **Diversification**: Recommendations across multiple stocks

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: BERT-base-uncased (110M parameters)
- **Fine-tuning**: Specialized on financial sentiment data
- **Classes**: Negative (0), Neutral (1), Positive (2)
- **Performance**: ~85% accuracy on financial text

### Data Processing
- **Tokenization**: BERT WordPiece with special tokens
- **Sequence Length**: 128 tokens (optimal for news articles)  
- **Preprocessing**: Automatic cleaning and normalization
- **Batch Processing**: Efficient GPU utilization

### Recommendation Algorithm
1. **Sentiment Aggregation**: Combine sentiment from multiple articles
2. **Confidence Calculation**: Based on volume and consistency
3. **Risk Assessment**: Multiple risk factors considered
4. **Action Determination**: Threshold-based buy/sell/hold decisions
5. **Target Price Estimation**: Sentiment-based price movement prediction

## 📈 Use Cases

### For Individual Investors
- **Daily market screening**: Find stocks with positive/negative news momentum
- **Risk assessment**: Understand the risk level of potential investments
- **Sentiment tracking**: Monitor how news affects specific stocks

### For Traders
- **Short-term opportunities**: Identify stocks with strong sentiment shifts
- **Risk management**: Avoid high-risk recommendations during volatile periods
- **Market timing**: Use sentiment as a complementary indicator

### For Researchers
- **Market analysis**: Study correlation between news sentiment and price movements
- **Model development**: Experiment with different sentiment analysis approaches
- **Backtesting**: Historical analysis of sentiment-based strategies

## ⚠️ Important Disclaimers

🚨 **This is for educational purposes only**
- Not financial advice
- Always do your own research  
- Consider multiple factors beyond sentiment
- Past performance doesn't guarantee future results
- Markets are inherently unpredictable

🔍 **Limitations**
- Sentiment analysis isn't perfect
- News can be misleading or manipulated
- Market movements depend on many factors
- Model predictions have uncertainty
- API rate limits may restrict data collection

## 🔮 Future Enhancements

- **Real-time streaming**: Live news feed processing
- **Portfolio optimization**: Multi-stock portfolio recommendations  
- **Technical analysis**: Combine with price pattern analysis
- **Options strategies**: Recommendations for options trading
- **Backtesting framework**: Historical performance analysis
- **Mobile app**: Real-time notifications and alerts
- **Social media integration**: Twitter/Reddit sentiment analysis

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional news sources
- Better stock symbol extraction
- Enhanced risk assessment
- UI/UX improvements
- Performance optimizations

## 📄 License

This project is for educational use. Please ensure compliance with API terms of service and financial regulations in your jurisdiction.

---

**Happy Trading! 📈** (Remember: Always invest responsibly!)
