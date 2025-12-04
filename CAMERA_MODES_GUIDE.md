# Camera Modes Guide

The exploded mesh renderer now uses the professional camera sampling functions from `XPart/partgen/utils/camera_utils.py`!

## Available Camera Modes

### 1. `angles` (Default - Manual Specification)
Specify exact azimuth and elevation angles for each view.

**Usage:**
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode angles \
    --angles "0,30 90,30 180,30 270,30"
```

**Best for**: Custom views, specific angles, presentations

---

### 2. `uniform` - Random Uniform Sampling
Samples camera views uniformly on a sphere (random but well-distributed).

**Usage:**
```bash
# 12 random views (default)
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode uniform

# 24 random views
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode uniform \
    --num_views 24
```

**Best for**: Comprehensive coverage, data augmentation, training data

---

### 3. `standard` - Ring Pattern with Top/Bottom
Creates views in a ring pattern at specific elevation with top and bottom views.

**Usage:**
```bash
# 8 views in ring + 2 top/bottom (default)
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode standard

# 16 views in ring
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode standard \
    --num_views 16
```

**Best for**: Clean systematic views, turntables, product photography style

---

### 4. `icosahedron` - 12 Evenly Distributed Views
Places cameras at the vertices of an icosahedron (12 views).

**Usage:**
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode icosahedron
```

**Best for**: Even coverage with moderate number of views

---

### 5. `dodecahedron` - 20 Evenly Distributed Views
Places cameras at the vertices of a dodecahedron (20 views).

**Usage:**
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode dodecahedron
```

**Best for**: Dense even coverage, thorough documentation

---

### 6. `octohedron` - 6 Axis-Aligned Views
Places cameras along the 6 primary axes (front, back, left, right, top, bottom).

**Usage:**
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode octohedron
```

**Best for**: Orthogonal views, technical drawings, blueprints

---

### 7. `cube` - 8 Corner Views
Places cameras at the 8 corners of a cube.

**Usage:**
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode cube
```

**Best for**: Isometric-style views, 3D visualization

---

### 8. `tetrahedron` - 4 Minimal Views
Places cameras at the 4 vertices of a tetrahedron.

**Usage:**
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode tetrahedron
```

**Best for**: Quick preview with minimal views

---

## Comparison Table

| Mode | # Views | Distribution | Best Use Case |
|------|---------|--------------|---------------|
| **angles** | Custom | Manual | Presentations, specific shots |
| **uniform** | 12 (adjustable) | Random uniform | Training data, comprehensive |
| **standard** | 8 (adjustable) | Ring + poles | Turntables, product shots |
| **icosahedron** | 12 | Even polyhedra | Balanced coverage |
| **dodecahedron** | 20 | Even polyhedra | Dense coverage |
| **octohedron** | 6 | Axis-aligned | Technical drawings |
| **cube** | 8 | Corner views | Isometric views |
| **tetrahedron** | 4 | Minimal | Quick preview |

---

## Examples

### Example 1: Quick Preview (4 views)
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode tetrahedron \
    --renderer pyrender
```

### Example 2: Standard Turntable (8 views)
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode standard \
    --renderer pyrender
```

### Example 3: Comprehensive Coverage (24 random views)
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode uniform \
    --num_views 24 \
    --renderer pyrender
```

### Example 4: Dense Polyhedra Views (20 views)
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode dodecahedron \
    --renderer pyrender
```

### Example 5: Custom Manual Angles
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode angles \
    --angles "0,0 45,30 90,45 180,30 270,30" \
    --renderer pyrender
```

### Example 6: High-Resolution Icosahedron Views
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode icosahedron \
    --resolution 3840 2160 \
    --renderer pyrender
```

---

## Output File Naming

**For manual angles mode:**
- `render_az0_el30.png` (azimuth 0°, elevation 30°)
- `render_az90_el45.png` (azimuth 90°, elevation 45°)

**For other modes:**
- `render_view00.png`
- `render_view01.png`
- `render_view02.png`
- etc.

---

## Technical Details

### Camera Utils Integration

The camera positions are generated using functions from `XPart/partgen/utils/camera_utils.py`:

- `view_matrix()` - Converts camera position + lookat into 4x4 transformation matrix
- `sample_view_matrices()` - Random uniform sampling on sphere
- `sample_view_matrices_polyhedra()` - Samples at polyhedra vertices

### Coordinate System

- **Lookat**: Scene centroid (automatically calculated)
- **Up vector**: (0, 0, 1) - Z-axis up
- **Distance**: Automatically calculated as `max(scene_extents) * 2.5`
- **Convention**: Camera-to-world matrices (right, up, -forward)

---

## Tips & Tricks

### 1. Choosing the Right Mode

**For presentations/papers:**
- Use `angles` mode for carefully composed shots
- Use `standard` for clean turntable sequences

**For ML/training:**
- Use `uniform` with high `num_views` (20-50)
- Provides random but well-distributed coverage

**For documentation:**
- Use `dodecahedron` (20 views) for thorough coverage
- Use `octohedron` (6 views) for axis-aligned technical views

### 2. Combining with Other Options

```bash
# High-quality dense coverage
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode dodecahedron \
    --renderer pyrender \
    --explosion_scale 0.6 \
    --resolution 2560 1440
```

### 3. Batch Processing Different Modes

```bash
# Try multiple modes
for mode in tetrahedron octohedron icosahedron; do
    python explode_automask_results.py \
        --results_dir results/chair \
        --camera_mode $mode \
        --output_dir results/chair/renders_$mode
done
```

### 4. Creating Animations

```bash
# Uniform sampling with many views for smooth animation
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode uniform \
    --num_views 72 \
    --renderer pyrender

# Then create video with ffmpeg
cd results/chair/exploded_renders
ffmpeg -framerate 24 -pattern_type glob -i 'render_view*.png' -c:v libx264 animation.mp4
```

---

## Migration from Old Angles System

**Old way:**
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --angles "0,30 90,30 180,30 270,30"
```

**New way (backward compatible):**
```bash
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode angles \
    --angles "0,30 90,30 180,30 270,30"
```

**Or use new modes:**
```bash
# Similar coverage with standard mode
python explode_automask_results.py \
    --results_dir results/chair \
    --camera_mode standard \
    --num_views 4
```

---

## Summary

✅ **8 different camera sampling modes**
✅ **Professional camera_utils integration**
✅ **Automatic distance calculation**
✅ **Backward compatible with manual angles**
✅ **Flexible number of views**
✅ **Works with all renderers (pyrender/blender/matplotlib)**

Choose the mode that best fits your needs and enjoy professional-quality multi-view renders! 🎥

