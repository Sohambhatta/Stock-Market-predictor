@echo off
title Elite Stock Analysis - Real-Time Data
color 0A

echo.
echo ===============================================
echo    ELITE STOCK ANALYSIS PLATFORM
echo    Real Yahoo Finance Data ^& AI Predictions
echo ===============================================
echo.
echo Installing required packages...
pip install yfinance pandas numpy scikit-learn flask

echo.
echo Starting the application with REAL stock data...
echo.

cd /d "%~dp0"
python app_real_data.py

echo.
echo Application stopped.
pause
