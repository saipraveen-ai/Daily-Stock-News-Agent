#!/bin/bash

# Daily Stock News Agent - Setup & Run Script
# This script sets up the environment and runs the interactive agent

echo "🚀 Daily Stock News Agent - Setup & Run"
echo "========================================"

# Get the script directory and ensure we're in the right place
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv .venv
    echo "✅ Virtual environment created!"
fi

# Set environment variables to use virtual environment
export VIRTUAL_ENV="$SCRIPT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.9/site-packages"

echo "🔧 Using virtual environment at: $VIRTUAL_ENV"

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "import yt_dlp, whisper, yaml, markdown" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    python3 -m pip install -r requirements.txt
    echo "✅ Dependencies installed!"
else
    echo "✅ All dependencies are available!"
fi

# Check configuration
if [ ! -f "config.yaml" ]; then
    echo "⚠️  config.yaml not found. Please ensure configuration is set up."
fi

# Verify we have the interactive agent script
if [ ! -f "interactive_agent.py" ]; then
    echo "❌ interactive_agent.py not found in $(pwd)"
    echo "📁 Available Python files:"
    ls -la *.py
    exit 1
fi

# Run the interactive agent
echo ""
echo "🎯 Starting Interactive Daily Stock News Agent..."
echo "================================================"
python3 interactive_agent.py
