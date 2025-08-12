import requests
import json

def test_market_api():
    """Test the market API endpoint directly"""
    print("🌐 Testing Market API Endpoint...")
    
    try:
        # Test the API endpoint
        response = requests.get('http://localhost:5000/api/market', timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response Success!")
            print(f"Indices count: {len(data.get('indices', []))}")
            
            if data.get('indices'):
                print("\n📊 Indices Data:")
                for index in data['indices'][:2]:  # Show first 2
                    print(f"  {index['name']}: ${index['price']} ({index['change']:+.2f}, {index['change_percent']:+.2f}%)")
            else:
                print("❌ No indices data in response")
                
            print(f"\nFull response keys: {list(data.keys())}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error - Is the Flask app running?")
        print("   Start the app with: start_adaptive.bat")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_market_api()
