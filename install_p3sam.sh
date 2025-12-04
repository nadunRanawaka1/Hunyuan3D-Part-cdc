#!/bin/bash

# Install P3-SAM package for use from any directory

set -e  # Exit on error

echo "=========================================="
echo "P3-SAM Package Installation"
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

echo "Installing P3-SAM package in editable mode..."
echo ""

# Install in editable mode
pip install -e .

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "You can now use P3-SAM from any directory:"
echo ""
echo "1. Command line:"
echo "   p3sam-auto-mask --mesh_path mesh.glb --output_path results/"
echo ""
echo "2. Python import:"
echo "   from demo.auto_mask import AutoMask"
echo ""
echo "3. Run example:"
echo "   cd .."
echo "   python example_use_p3sam.py --mesh_path P3-SAM/demo/assets/1.glb --output_path test_output/"
echo ""
echo "See INSTALL.md for more details."
echo ""

