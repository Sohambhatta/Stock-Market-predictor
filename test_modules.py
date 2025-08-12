"""
Quick test to verify all modules are working
"""

try:
    print("Testing imports...")
    
    # Test basic imports
    from advanced_financial_analysis import AdvancedFinancialAnalysis
    from live_news_collector import LiveNewsCollector
    from stock_recommender import StockRecommendationEngine
    
    print("✓ All imports successful!")
    
    # Test module initialization
    print("\nTesting module initialization...")
    
    financial_analyzer = AdvancedFinancialAnalysis()
    print("✓ AdvancedFinancialAnalysis initialized")
    
    news_collector = LiveNewsCollector()
    print("✓ LiveNewsCollector initialized")
    
    recommender = StockRecommendationEngine()
    print("✓ StockRecommendationEngine initialized")
    
    print("\n✓ All modules working correctly!")
    print("\nYou can now run the Flask app with: python app.py")
    print("Or use the batch file: start_app.bat")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing packages")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("There may be an issue with the modules")
    
input("\nPress Enter to continue...")
