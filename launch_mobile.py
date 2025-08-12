"""
Fast Mobile-Optimized Stock Analysis Launcher
"""

import webbrowser
import time
import sys
import subprocess

def main():
    print("🚀 Starting Mobile-Optimized Stock Analysis App")
    print("📱 Optimized for fast loading and mobile devices")
    print("=" * 50)
    
    try:
        # Start the optimized app
        print("Starting server on port 5001...")
        print("🌐 URL: http://localhost:5001")
        print("📱 Mobile-friendly interface loading...")
        
        # Give it a moment then open browser
        time.sleep(1)
        webbrowser.open('http://localhost:5001')
        
        # Run the optimized Flask app
        import subprocess
        import os
        os.chdir(r"c:\Users\soham\Coding Projects\Stock-Market-predictor")
        subprocess.run([sys.executable, "app_optimized.py"])
        
    except KeyboardInterrupt:
        print("\n⏹️ App stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTrying to run directly...")
        try:
            exec(open('app_optimized.py').read())
        except Exception as e2:
            print(f"❌ Direct run failed: {e2}")
            input("Press Enter to exit...")

if __name__ == '__main__':
    main()
