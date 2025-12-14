#!/bin/bash

echo "🚀 Setting up Enhanced Vehicle Recognition Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Found Python version: $PYTHON_VERSION"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi

print_status "Creating virtual environment..."
python3 -m venv enhanced_venv

print_status "Activating virtual environment..."
source enhanced_venv/bin/activate

print_status "Upgrading pip..."
pip install --upgrade pip

print_status "Installing enhanced dependencies..."
pip install -r enhanced_requirements.txt

# Check if YOLO model exists, if not download it
if [ ! -f "yolo11n.pt" ]; then
    print_status "Downloading YOLO v11 nano model..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # This will download the model
print('✅ YOLO v11 model downloaded successfully')
"
fi

# Create output directories
print_status "Creating output directories..."
mkdir -p outputs
mkdir -p exports
mkdir -p models

print_success "Enhanced setup complete!"
echo ""
echo "🎯 To run the enhanced project:"
echo "   1. Activate the virtual environment: source enhanced_venv/bin/activate"
echo "   2. Run the enhanced system: python enhanced_test.py"
echo ""
echo "✨ New Features Available:"
echo "   🔧 Automatic camera calibration using vehicle dimensions"
echo "   📊 Enhanced CSV export with detailed analytics" 
echo "   🎯 Kalman filter velocity extraction"
echo "   📈 Real-time performance monitoring"
echo "   🚗 Improved vehicle tracking and speed display"
echo ""
echo "⌨️  Enhanced Controls:"
echo "   'q' - Quit application"
echo "   's' - Save current analytics"
echo "   'c' - Force calibration reset"
echo ""
echo "To deactivate the virtual environment later: deactivate"
