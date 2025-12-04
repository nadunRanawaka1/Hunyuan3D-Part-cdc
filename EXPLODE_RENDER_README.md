# Exploded Mesh Rendering Scripts

This directory contains scripts to create exploded views of partitioned meshes from auto_mask.py and render them with numerical labels from multiple angles.

## Overview

These scripts take the output from `auto_mask.py` (or any partitioned mesh), separate it into individual parts, create an exploded view using `explode_mesh()` from XPart, and render labeled views from multiple camera angles.

## Scripts

### 1. `explode_automask_results.py` (Recommended)

**Quick and easy script** that automatically finds mesh and face_ids from auto_mask.py output directory.

#### Usage:

```bash
# Basic usage - point to auto_mask.py results directory
python explode_automask_results.py --results_dir results/1

# Larger explosion effect
python explode_automask_results.py --results_dir results/1 --explosion_scale 0.8

# High resolution renders
python explode_automask_results.py --results_dir results/1 --resolution 3840 2160

# More camera angles (8 views around the object)
python explode_automask_results.py --results_dir results/1 --angles "0,30 45,30 90,30 135,30 180,30 225,30 270,30 315,30"

# Custom output directory
python explode_automask_results.py --results_dir results/1 --output_dir my_custom_renders
```

#### Output:

- `exploded_renders/` directory containing:
  - `render_view_00_az0_el30.png` - Rendered view from azimuth 0°, elevation 30°
  - `render_view_01_az90_el30.png` - Rendered view from azimuth 90°, elevation 30°
  - ... (one image per camera angle)
  - `exploded_mesh.glb` - The exploded mesh file (can be opened in 3D viewers)
  - `summary.txt` - Information about the rendering

### 2. `explode_and_render.py` (Advanced)

**Full-featured script** with more control over inputs and outputs.

#### Usage:

```bash
# Specify mesh and face_ids explicitly
python explode_and_render.py \
    --mesh_path results/1/auto_mask_mesh_final.glb \
    --face_ids results/1/final_face_ids.npy \
    --output_dir custom_output

# With custom parameters
python explode_and_render.py \
    --mesh_path path/to/mesh.glb \
    --face_ids path/to/face_ids.npy \
    --explosion_scale 0.6 \
    --resolution 2560 1440 \
    --angles "0,30 90,30 180,30 270,30"
```

## Parameters

### Common Parameters:

- `--explosion_scale`: Controls how far apart the parts are pushed (default: 0.4)
  - Smaller values (0.2-0.3): Subtle separation
  - Medium values (0.4-0.6): Clear separation
  - Larger values (0.7-1.0): Dramatic explosion effect

- `--resolution WIDTH HEIGHT`: Image resolution in pixels (default: 1920 1080)
  - HD: 1920 1080
  - 4K: 3840 2160
  - Custom: any resolution you need

- `--angles`: Camera positions as "azimuth,elevation" pairs separated by spaces
  - Azimuth: horizontal rotation (0-360°)
  - Elevation: vertical angle (-90 to 90°)
  - Default: "0,30 90,30 180,30 270,30 45,60 225,-30" (6 views)
  - Example for 8 views: "0,30 45,30 90,30 135,30 180,30 225,30 270,30 315,30"

## Installation Requirements

The scripts require the following Python packages:

```bash
pip install trimesh numpy matplotlib pillow pyrender
```

Optional for better rendering:
```bash
pip install pyglet
```

## How It Works

1. **Load Mesh**: Loads the partitioned mesh and face_ids from auto_mask.py output
2. **Separate Parts**: Extracts each part as a separate mesh based on face_ids
3. **Create Scene**: Assembles parts into a trimesh.Scene with unique colors
4. **Explode**: Uses `explode_mesh()` to push parts away from the center
5. **Render**: Renders from multiple camera angles
6. **Label**: Adds numerical labels at each part's center position
7. **Save**: Exports images and the exploded mesh file

## Examples

### Example 1: Quick Visualization

```bash
# Run auto_mask.py first
python P3-SAM/demo/auto_mask.py --mesh_path data/chair.glb --output_path results/chair

# Then explode and render
python explode_automask_results.py --results_dir results/chair
```

### Example 2: Publication-Quality Renders

```bash
# High resolution with more views
python explode_automask_results.py --results_dir results/chair \
    --explosion_scale 0.5 \
    --resolution 3840 2160 \
    --angles "0,30 30,30 60,30 90,30 120,30 150,30 180,30 210,30 240,30 270,30 300,30 330,30" \
    --output_dir chair_publication_renders
```

### Example 3: Different Explosion Scales

```bash
# Generate multiple explosion scales for comparison
for scale in 0.2 0.4 0.6 0.8; do
    python explode_automask_results.py --results_dir results/chair \
        --explosion_scale $scale \
        --output_dir renders/chair_scale_$scale
done
```

## Viewing the Results

### View Images:
Simply open the PNG files in `exploded_renders/` directory with any image viewer.

### View 3D Exploded Mesh:
The `exploded_mesh.glb` can be opened in:
- Blender
- MeshLab
- Online viewers like: https://gltf-viewer.donmccurdy.com/
- Any GLB/GLTF viewer

## Troubleshooting

### "No face_ids.npy found"
The script will try to extract part information from mesh face colors. If this fails, save face_ids when running auto_mask.py:
```python
np.save('results/my_mesh/face_ids.npy', face_ids)
```

### "Could not render with pyrender"
The script will fall back to matplotlib-based rendering. For better quality, install pyrender:
```bash
pip install pyrender pyglet
```

### Labels not showing or mispositioned
This can happen if the camera projection calculation doesn't match the renderer. Try different camera angles or adjust the label positioning logic in `add_part_labels()`.

### Out of memory errors
Reduce resolution:
```bash
python explode_automask_results.py results/mesh --resolution 1280 720
```

## Advanced: Customizing the Rendering

You can modify `explode_and_render.py` to customize:

1. **Colors**: Edit the `create_scene_from_parts()` function to change the colormap
2. **Label Style**: Edit the `add_part_labels()` function to change font, colors, or layout
3. **Lighting**: If using pyrender, add custom lights in `render_scene_with_labels()`
4. **Camera Paths**: Generate custom camera angles programmatically

Example custom angles (circular orbit):
```python
import numpy as np
angles = [(az, 30) for az in np.linspace(0, 360, 36, endpoint=False)]
```

## Integration with Auto-Mask Pipeline

For a complete workflow:

```bash
#!/bin/bash
# Complete pipeline script

MESH="data/my_model.glb"
OUTPUT="results/my_model"

# 1. Run segmentation
python P3-SAM/demo/auto_mask.py \
    --mesh_path $MESH \
    --output_path $OUTPUT \
    --point_num 100000 \
    --prompt_num 400

# 2. Create exploded views
python explode_automask_results.py --results_dir $OUTPUT \
    --explosion_scale 0.5 \
    --resolution 1920 1080

echo "Complete! Check $OUTPUT/exploded_renders/"
```

## Citation

If you use these scripts in your research, please cite the original Hunyuan3D-Part paper:

```bibtex
@article{hunyuan3d-part,
  title={Hunyuan3D-Part: 3D Part Segmentation and Decomposition},
  author={...},
  journal={...},
  year={2024}
}
```

## License

These scripts are part of the Hunyuan3D-Part project and follow the same license terms.

