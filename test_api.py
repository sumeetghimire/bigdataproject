#!/usr/bin/env python3
"""
Test script to check if Flask API endpoints are working
"""

import requests
import json

def test_api_endpoint(url, endpoint_name):
    """Test a single API endpoint"""
    try:
        print(f"\n🔍 Testing {endpoint_name}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'error' in data:
                print(f"❌ {endpoint_name}: {data['error']}")
                return False
            else:
                print(f"✅ {endpoint_name}: OK ({len(data)} items)" if isinstance(data, list) else f"✅ {endpoint_name}: OK")
                return True
        else:
            print(f"❌ {endpoint_name}: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {endpoint_name}: Connection error - {str(e)}")
        return False
    except Exception as e:
        print(f"❌ {endpoint_name}: Error - {str(e)}")
        return False

def main():
    """Test all API endpoints"""
    base_url = "http://localhost:5000"
    
    endpoints = [
        ("/api/data/summary", "Data Summary"),
        ("/api/data/endangerment-distribution", "Endangerment Distribution"),
        ("/api/data/feature-importance", "Feature Importance"),
        ("/api/data/model-performance", "Model Performance"),
        ("/api/data/speaker-vs-endangerment", "Speaker vs Endangerment"),
        ("/api/data/geographic-distribution", "Geographic Distribution"),
        ("/api/data/family-distribution", "Family Distribution"),
        ("/api/data/transmission-distribution", "Transmission Distribution"),
        ("/api/data/languages", "Languages Data")
    ]
    
    print("🚀 Testing Flask API Endpoints")
    print("=" * 50)
    
    # Test if server is running
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✅ Flask server is running (HTTP {response.status_code})")
    except:
        print("❌ Flask server is not running!")
        print("Please start the server with: python app.py")
        return
    
    # Test each endpoint
    results = []
    for endpoint, name in endpoints:
        url = base_url + endpoint
        success = test_api_endpoint(url, name)
        results.append((name, success))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} endpoints working")
    
    if passed == total:
        print("🎉 All API endpoints are working correctly!")
    else:
        print("⚠️  Some endpoints have issues. Check the Flask server logs for details.")

if __name__ == "__main__":
    main()

