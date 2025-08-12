@echo off
title Elite Finance - Yahoo Finance Competitor
color 0A

echo.
echo ================================================
echo    ELITE FINANCE PLATFORM
echo    Professional Stock Analysis ^& Trading Platform  
echo    Real-time data • AI predictions • Technical analysis
echo ================================================
echo.

echo Installing required packages...
pip install yfinance pandas numpy scikit-learn flask plotly

echo.
echo ✨ Features:
echo   📊 Real Yahoo Finance data
echo   🕯️  Professional candlestick charts  
echo   📈 Technical indicators (RSI, MACD, Bollinger Bands)
echo   🤖 Machine learning price predictions
echo   💡 AI-powered investment recommendations
echo   📱 Mobile-responsive design
echo   ⚡ Live market data feeds

echo.
echo Starting Elite Finance Platform...
echo 🌐 Open your browser to: http://localhost:5000
echo.

cd /d "%~dp0"
python app_yahoo_competitor.py

echo.
echo Platform stopped. Press any key to exit.
pause
