#!/usr/bin/env python3
"""
Run the Language Extinction Dashboard on port 5001
"""

from demo_app import app, load_demo_data

if __name__ == '__main__':
    try:
        load_demo_data()
        print("✅ Demo data loaded successfully!")
        print("🚀 Starting Flask application on port 5001...")
        print("📱 Open your browser and go to: http://localhost:5001")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("💡 Try running: python3 run_app_port_5001.py")
