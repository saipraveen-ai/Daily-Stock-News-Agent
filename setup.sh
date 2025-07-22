#!/bin/bash
"""
Daily Stock News Agent - Quick Setup Script

This script automatically installs all required dependencies including system packages.
"""

set -e  # Exit on any error

echo "🚀 Daily Stock News Agent - Quick Setup"
echo "======================================"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is designed for macOS. For other systems, please install dependencies manually."
    echo "   Required system packages: ffmpeg"
    echo "   Then run: pip install -r requirements.txt"
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew found"
fi

# Install ffmpeg (required by Whisper)
echo "📦 Installing ffmpeg (required for Whisper audio processing)..."
if ! command -v ffmpeg &> /dev/null; then
    echo "   Installing ffmpeg via Homebrew..."
    brew install ffmpeg
else
    echo "   ✅ ffmpeg already installed"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
if [[ "$PYTHON_VERSION" < "3.8" ]]; then
    echo "❌ Python 3.8+ required. Current version: $PYTHON_VERSION"
    exit 1
else
    echo "✅ Python version: $PYTHON_VERSION"
fi

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/videos
mkdir -p data/transcripts  
mkdir -p data/analysis
mkdir -p data/reports
mkdir -p data/pipeline_states
mkdir -p logs
echo "   ✅ Directories created"

# Verify Whisper installation
echo "🎤 Verifying Whisper installation..."
if python -c "import whisper; model = whisper.load_model('base'); print('✅ Whisper working correctly')" 2>/dev/null; then
    echo "   ✅ Whisper installation verified"
else
    echo "   ❌ Whisper installation failed. This might be due to NumPy compatibility."
    echo "   Trying to fix NumPy compatibility..."
    pip install "numpy<2.0"
    
    if python -c "import whisper; model = whisper.load_model('base'); print('✅ Whisper working after NumPy fix')" 2>/dev/null; then
        echo "   ✅ Whisper fixed and working"
    else
        echo "   ❌ Whisper still not working. Manual troubleshooting required."
        echo "   Please check the SETUP_GUIDE.md for detailed instructions."
    fi
fi

# Test ffmpeg integration
echo "🔧 Testing ffmpeg integration..."
if ffmpeg -version &> /dev/null; then
    echo "   ✅ ffmpeg working correctly"
else
    echo "   ❌ ffmpeg not working. Please check installation."
fi

echo ""
echo "🎉 Setup completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the virtual environment: source .venv/bin/activate"
echo "   2. Run the modular interactive agent: python modular_interactive_agent.py"
echo "   3. Or run the original agent: python interactive_agent.py"
echo ""
echo "📚 For detailed configuration, see:"
echo "   - SETUP_GUIDE.md for comprehensive setup instructions"
echo "   - README.md for usage examples"
echo ""
echo "🆘 If you encounter issues:"
echo "   1. Check that all dependencies installed correctly"
echo "   2. Ensure ffmpeg is in your PATH: which ffmpeg"
echo "   3. Test Whisper: python -c 'import whisper; print(\"OK\")'"
echo ""
