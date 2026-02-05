#!/bin/bash

# 6G Channel ML Platform Startup Script

echo "🚀 Starting 6G Channel ML Platform..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

# Create necessary directories
mkdir -p data models

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting web server..."
echo "   Open your browser at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
python app.py

