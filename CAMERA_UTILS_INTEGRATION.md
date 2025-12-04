# Camera Utils Integration - Complete! ✅

## Summary

Successfully integrated the professional camera sampling functions from `XPart/partgen/utils/camera_utils.py` into the exploded mesh rendering system.

## What Changed

### 1. **New Camera Sampling Modes** 🎥

Now supports **8 different camera sampling modes**:

| Mode | Views | Description |
|------|-------|-------------|
| `angles` | Custom | Manual azimuth/elevation (backward compatible) |
| `uniform` | Adjustable | Random uniform sampling on sphere |
| `standard` | Adjustable | Ring pattern with top/bottom |
| `icosahedron` | 12 | Even distribution (icosahedron vertices) |
| `dodecahedron` | 20 | Dense even distribution |
| `octohedron` | 6 | Axis-aligned orthogonal views |
| `cube` | 8 | Corner views (isometric-style) |
| `tetrahedron` | 4 | Minimal quick preview |

### 2. **Camera Utils Functions Used**

- ✅ `view_matrix()` - Creates camera-to-world transformation matrices
- ✅ `sample_view_matrices()` - Uniform random sampling
- ✅ `sample_view_matrices_polyhedra()` - Polyhedra-based sampling

### 3. **Updated Files**

#### `explode_and_render.py`
- Added `torch` import
- Added `camera_utils` imports
- New parameters: `camera_mode`, `num_views`
- Replaced manual camera matrix generation with `view_matrix()`
- Support for all 8 camera modes
- Automatic calculation of azimuth/elevation for fallback renderers

#### `explode_automask_results.py`
- New command-line arguments: `--camera_mode`, `--num_views`
- Passes parameters through to render function

#### Documentation
- **`CAMERA_MODES_GUIDE.md`** - Complete guide to all camera modes
- **`LATEST_FIXES.md`** - Fixed camera positioning and labeling issues

## Usage Examples

### Basic (Backward Compatible)
```bash
# Old way still works!
python explode_automask_results.py --results_dir results/chair
```

### Uniform Random Sampling (12 views)
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode uniform \
    --renderer pyrender
```

### Icosahedron (12 evenly distributed views)
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode icosahedron \
    --renderer pyrender
```

### Standard Ring (adjustable)
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode standard \
    --num_views 16 \
    --renderer pyrender
```

### Dense Coverage (20 views)
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode dodecahedron \
    --renderer pyrender
```

## Technical Details

### Camera Coordinate System
- **Up vector**: (0, 0, 1) - Z-axis up
- **Lookat**: Scene centroid (auto-calculated)
- **Distance**: `max(scene_extents) * 2.5` (auto-calculated)
- **Convention**: Camera-to-world matrices (right, up, -forward)

### Matrix Format
- 4x4 homogeneous transformation matrices
- Generated as PyTorch tensors, converted to numpy for rendering
- Compatible with all renderers (PyRender, Blender, Matplotlib)

### Fallback Handling
- Matplotlib renderer needs azimuth/elevation angles
- Automatically calculated from camera matrix when using non-angles modes
- Smooth fallback if PyRender/Blender fail

## Benefits

✅ **Professional camera sampling** - Uses the same camera_utils as XPart  
✅ **Mathematically optimal distributions** - Polyhedra vertices provide even coverage  
✅ **Flexible** - 8 different modes for different use cases  
✅ **Backward compatible** - Old angle specification still works  
✅ **Automatic** - Distance and lookat calculated from scene bounds  
✅ **Consistent** - Same camera system across all renderers  

## Output File Naming

**Manual angles mode:**
- `render_az0_el30.png`
- `render_az90_el45.png`

**Other modes:**
- `render_view00.png`
- `render_view01.png`
- `render_view02.png`

## Command-Line Arguments

### New Arguments
```
--camera_mode {angles,uniform,standard,icosahedron,dodecahedron,octohedron,cube,tetrahedron}
    Camera sampling mode (default: angles)

--num_views INT
    Number of views for uniform/standard modes
    (default: 12 for uniform, 8 for standard)
```

### Existing Arguments (Still Work)
```
--angles "az,el az,el ..."
    Manual camera angles (used with --camera_mode angles)

--renderer {auto,pyrender,blender,matplotlib}
    Rendering backend

--explosion_scale FLOAT
    Explosion scale factor

--resolution WIDTH HEIGHT
    Image resolution
```

## Testing

To test the new camera modes:

```bash
# Test each mode
for mode in tetrahedron octohedron cube icosahedron dodecahedron uniform standard; do
    echo "Testing $mode mode..."
    python explode_automask_results.py \
        --results_dir results/test_mesh \
        --camera_mode $mode \
        --output_dir results/test_mesh/cam_$mode \
        --renderer pyrender
done
```

## Troubleshooting

### "Import torch could not be resolved"
```bash
pip install torch
```

### "Import camera_utils could not be resolved"
Make sure you're running from the project root. The script adds the path automatically.

### Rendered objects not visible?
This was fixed! Now includes:
- Gray background for PyRender
- Better lighting (3 lights from different angles)
- Black edges for Matplotlib
- Debug output showing part colors

## Next Steps

You can now generate professional multi-view renders with:
- ✅ Proper camera matrices from camera_utils
- ✅ Visible meshes with good lighting
- ✅ Numerical labels on each part
- ✅ Multiple camera sampling strategies

Enjoy! 🎉

---

## See Also

- `CAMERA_MODES_GUIDE.md` - Detailed guide for each camera mode
- `LATEST_FIXES.md` - Camera positioning and labeling fixes
- `QUICK_START_EXPLODE.md` - Quick start guide
- `EXPLODE_RENDER_README.md` - Full documentation

