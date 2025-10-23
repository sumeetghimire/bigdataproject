#!/usr/bin/env python3
"""
Startup script for the Language Extinction Risk Prediction Demo
"""

import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['flask', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_file():
    """Check if the sample data file exists"""
    data_file = Path('data/enhanced_sample_data.csv')
    if not data_file.exists():
        print("❌ Sample data file not found!")
        print(f"   Expected: {data_file.absolute()}")
        print("\n📁 Make sure the data file exists in the data/ directory")
        return False
    
    print(f"✅ Data file found: {data_file}")
    return True

def main():
    """Main startup function"""
    print("🌍 Language Extinction Risk Prediction Dashboard")
    print("=" * 50)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    # Check data file
    print("📊 Checking data file...")
    if not check_data_file():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("\n🚀 Starting Flask application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the demo app
    try:
        from demo_app import app, load_demo_data
        load_demo_data()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
