@echo off
title Elite Finance Platform
color 0B

echo.
echo ================================================
echo               ELITE FINANCE
echo     Real Yahoo Finance Data Integration
echo ================================================
echo.
echo 📊 Data Source: Yahoo Finance (yfinance library)
echo 🕐 Update Frequency: Real-time during market hours
echo 📈 Market Hours: 9:30 AM - 4:00 PM EST (Mon-Fri)
echo.
echo 💡 Why prices might differ from other sources:
echo    • Different data providers (Yahoo vs Google vs Bloomberg)
echo    • Update timing (real-time vs 15-minute delays)
echo    • After-hours vs regular trading hours
echo    • Different calculation methods for ratios
echo.

echo Installing packages...
pip install yfinance pandas numpy scikit-learn flask plotly requests >nul 2>&1

echo.
echo Starting Elite Finance Platform...
echo.
echo 🌐 Open: http://localhost:5000
echo 📱 Mobile optimized and desktop ready
echo.

python app_yahoo_competitor.py

echo.
pause
