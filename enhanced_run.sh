#!/bin/bash

echo "🚀 Starting Enhanced Vehicle Recognition System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if enhanced virtual environment exists
if [ ! -d "enhanced_venv" ]; then
    print_error "Enhanced virtual environment not found. Please run enhanced_setup.sh first."
    echo "Run: ./enhanced_setup.sh"
    exit 1
fi

# Activate virtual environment
print_status "Activating enhanced virtual environment..."
source enhanced_venv/bin/activate

# Check if enhanced dependencies are installed
print_status "Checking dependencies..."
if ! python -c "import ultralytics, cv2, cvzone, pandas, numpy" 2>/dev/null; then
    print_error "Enhanced dependencies not found. Please run enhanced_setup.sh first."
    echo "Run: ./enhanced_setup.sh"
    exit 1
fi

# Check if enhanced modules exist
if [ ! -f "enhanced_test.py" ] || [ ! -f "sort_enhanced.py" ] || [ ! -f "calibration.py" ] || [ ! -f "data_export.py" ]; then
    print_error "Enhanced modules not found. Please ensure all enhanced files are present:"
    echo "   - enhanced_test.py"
    echo "   - sort_enhanced.py" 
    echo "   - calibration.py"
    echo "   - data_export.py"
    exit 1
fi

print_success "All dependencies and modules found!"

# Display system info
print_status "System Information:"
echo "   🐍 Python: $(python --version)"
echo "   🧠 Available RAM: $(free -h | awk '/^Mem:/ { print $7 }')" 2>/dev/null || echo "   🧠 RAM info not available"
echo "   💾 Disk space: $(df -h . | awk 'NR==2 { print $4 }')" 2>/dev/null || echo "   💾 Disk info not available"

# Run the enhanced application
print_status "Launching Enhanced Vehicle Recognition System..."
echo ""
print_success "🎯 Enhanced features active:"
echo "   ✅ Auto-calibration using vehicle dimensions"
echo "   ✅ Kalman velocity extraction"
echo "   ✅ Enhanced CSV export"
echo "   ✅ Real-time analytics"
echo ""

python enhanced_test.py

# Deactivate virtual environment
deactivate
print_status "Enhanced application finished."
