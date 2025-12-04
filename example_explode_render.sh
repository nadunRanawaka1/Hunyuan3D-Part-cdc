#!/bin/bash
# Example script showing how to use the exploded mesh rendering pipeline

echo "=========================================="
echo "Exploded Mesh Rendering Example"
echo "=========================================="
echo ""

# Check if results directory exists
if [ ! -d "P3-SAM/demo/results" ]; then
    echo "No results directory found. Please run auto_mask.py first!"
    echo ""
    echo "Example:"
    echo "  python P3-SAM/demo/auto_mask.py --mesh_path P3-SAM/demo/assets/1.glb --output_path P3-SAM/demo/results/example1"
    exit 1
fi

# Find a results directory
RESULTS_DIR=$(find P3-SAM/demo/results -maxdepth 1 -type d | grep -v "^P3-SAM/demo/results$" | head -n 1)

if [ -z "$RESULTS_DIR" ]; then
    echo "No mesh results found in P3-SAM/demo/results/"
    echo ""
    echo "Please run auto_mask.py first:"
    echo "  python P3-SAM/demo/auto_mask.py --mesh_path your_mesh.glb --output_path P3-SAM/demo/results/mesh1"
    exit 1
fi

echo "Found results directory: $RESULTS_DIR"
echo ""

# Example 1: Basic rendering with default settings
echo "Example 1: Basic rendering (default settings)"
echo "----------------------------------------------"
python explode_automask_results.py --results_dir "$RESULTS_DIR" \
    --output_dir "${RESULTS_DIR}/exploded_basic"
echo ""

# Example 2: Larger explosion with more views
echo "Example 2: Larger explosion with 8 camera angles"
echo "--------------------------------------------------"
python explode_automask_results.py --results_dir "$RESULTS_DIR" \
    --explosion_scale 0.7 \
    --angles "0,30 45,30 90,30 135,30 180,30 225,30 270,30 315,30" \
    --output_dir "${RESULTS_DIR}/exploded_large"
echo ""

# Example 3: High resolution for publication
echo "Example 3: High resolution rendering"
echo "-------------------------------------"
python explode_automask_results.py --results_dir "$RESULTS_DIR" \
    --explosion_scale 0.5 \
    --resolution 2560 1440 \
    --output_dir "${RESULTS_DIR}/exploded_hires"
echo ""

echo "=========================================="
echo "Examples complete!"
echo "=========================================="
echo ""
echo "Check the output directories:"
echo "  - ${RESULTS_DIR}/exploded_basic/"
echo "  - ${RESULTS_DIR}/exploded_large/"
echo "  - ${RESULTS_DIR}/exploded_hires/"
echo ""
echo "Open the PNG files to view the rendered images,"
echo "or open exploded_mesh.glb in a 3D viewer!"

