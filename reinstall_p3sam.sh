#!/bin/bash

# Reinstall P3-SAM package (useful after making changes)

set -e  # Exit on error

echo "=========================================="
echo "P3-SAM Package Reinstallation"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -f "P3-SAM/setup.py" ]; then
    echo "Error: Please run this script from the Hunyuan3D-Part root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Navigate to P3-SAM directory
cd P3-SAM

echo "Uninstalling existing P3-SAM package..."
pip uninstall -y p3sam 2>/dev/null || echo "  (package not currently installed)"

echo ""
echo "Installing P3-SAM package in editable mode..."
echo ""

# Install in editable mode
pip install -e .

echo ""
echo "=========================================="
echo "Reinstallation Complete!"
echo "=========================================="
echo ""
echo "Testing installation..."
python -c "from p3sam.demo import AutoMask; print('✓ Import successful!')" || echo "✗ Import failed!"
echo ""

