"""
Stock Market Analysis Web App Launcher
Quick launcher for the Flask web application
"""

import sys
import os
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'pandas', 'yfinance', 'plotly', 
        'scikit-learn', 'numpy', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def main():
    print("🚀 Stock Market Analysis Web Application Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the project root directory")
        input("Press Enter to exit...")
        return
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("✅ Packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages automatically")
            print("Please run: pip install -r requirements.txt")
            input("Press Enter to continue anyway...")
    else:
        print("✅ All dependencies satisfied!")
    
    # Start the Flask app
    print("\n🌐 Starting web application...")
    print("📍 URL: http://localhost:5000")
    print("🔄 Loading... (this may take a moment)")
    
    try:
        # Give user a moment to see the message
        time.sleep(2)
        
        # Open browser
        webbrowser.open('http://localhost:5000')
        
        # Start Flask app
        os.system('python app.py')
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        input("Press Enter to exit...")

if __name__ == '__main__':
    main()
