#!/usr/bin/env python3
"""
Test script for the Language Extinction Risk Prediction Web Application
"""

import sys
import os
from pathlib import Path
import pandas as pd

def test_data_file():
    """Test if the data file exists and is readable"""
    print("🔍 Testing data file...")
    
    data_path = Path('data/enhanced_sample_data.csv')
    if not data_path.exists():
        print("❌ Data file not found!")
        return False
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data file loaded successfully: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ Error reading data file: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'flask',
        'pandas',
        'numpy',
        'yaml',
        'json'
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_flask_app():
    """Test if the Flask app can be created"""
    print("🔍 Testing Flask app creation...")
    
    try:
        from demo_app import app, load_demo_data
        print("✅ Flask app imported successfully")
        
        # Test data loading
        load_demo_data()
        print("✅ Demo data loaded successfully")
        
        # Test app creation
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Flask app responds to requests")
                return True
            else:
                print(f"❌ Flask app returned status code: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing Flask app: {e}")
        return False

def test_api_endpoints():
    """Test if API endpoints work"""
    print("🔍 Testing API endpoints...")
    
    try:
        from demo_app import app, load_demo_data
        load_demo_data()
        
        with app.test_client() as client:
            endpoints = [
                '/api/data/summary',
                '/api/data/endangerment-distribution',
                '/api/data/feature-importance',
                '/api/data/model-performance',
                '/api/data/family-distribution',
                '/api/data/transmission-distribution'
            ]
            
            for endpoint in endpoints:
                response = client.get(endpoint)
                if response.status_code == 200:
                    print(f"✅ {endpoint}")
                else:
                    print(f"❌ {endpoint}: Status {response.status_code}")
                    return False
            
            return True
            
    except Exception as e:
        print(f"❌ Error testing API endpoints: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Language Extinction Risk Prediction - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Data File", test_data_file),
        ("Imports", test_imports),
        ("Flask App", test_flask_app),
        ("API Endpoints", test_api_endpoints)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} Test")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✅ {test_name} test passed!")
        else:
            print(f"❌ {test_name} test failed!")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        print("\n🚀 To start the application, run:")
        print("   python run_demo.py")
        print("   or")
        print("   python demo_app.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the application.")
        sys.exit(1)

if __name__ == '__main__':
    main()
