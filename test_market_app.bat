@echo off
echo ========================================
echo    TESTING ADAPTIVE PREDICTIONS APP
echo ========================================
echo.
echo 🔧 Starting Flask app with market overview...
echo 🌐 URL: http://localhost:5000
echo 📊 Check browser console for market debug logs
echo.
echo Instructions:
echo 1. App will start and load market data automatically
echo 2. Open browser dev tools (F12) to see console logs
echo 3. Look for market overview section above search box
echo 4. Check console for any JavaScript errors
echo.
cd /d "c:\Users\soham\Coding Projects\Stock-Market-predictor"
python app_adaptive_predictions.py
pause
