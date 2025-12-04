# Quick Start: Exploded Mesh Rendering

## TL;DR - Get Started in 30 Seconds

```bash
# 1. Run segmentation (if you haven't already)
python P3-SAM/demo/auto_mask.py --mesh_path your_mesh.glb --output_path results/my_mesh

# 2. Create exploded views
python explode_automask_results.py --results_dir results/my_mesh

# 3. View results
# Open: results/my_mesh/exploded_renders/render_view_*.png
```

Done! 🎉

## ⚠️ Got Rendering Errors?

If you see `requires 'pip install "pyglet<2"'` or similar:

```bash
# Use PyRender offscreen mode (NO pyglet needed!)
pip install pyrender
python explode_automask_results.py --results_dir results/my_mesh --renderer pyrender

# OR use Blender if installed
python explode_automask_results.py --results_dir results/my_mesh --renderer blender

# OR use matplotlib (lower quality but always works)
python explode_automask_results.py --results_dir results/my_mesh --renderer matplotlib
```

See `RENDERING_BACKENDS.md` for detailed info.

## What This Does

Takes a 3D mesh segmented by auto_mask.py and creates:
- **Exploded view** - Parts separated for easy visualization
- **Multiple angles** - 6 different camera perspectives by default
- **Labeled parts** - Each part numbered for identification
- **3D file** - `exploded_mesh.glb` you can open in Blender/MeshLab

## Common Use Cases

### 1. Quick Visualization (Default)
```bash
python explode_automask_results.py --results_dir results/chair
```
**Output**: 6 views at 1920x1080, explosion scale 0.4

### 2. Dramatic Explosion Effect
```bash
python explode_automask_results.py --results_dir results/chair --explosion_scale 0.8
```
**Use when**: Parts are tightly packed, need more separation

### 3. Subtle Separation
```bash
python explode_automask_results.py --results_dir results/chair --explosion_scale 0.2
```
**Use when**: Just want to see part boundaries clearly

### 4. 360° Turntable (8 views)
```bash
python explode_automask_results.py --results_dir results/chair \
    --angles "0,30 45,30 90,30 135,30 180,30 225,30 270,30 315,30"
```
**Use when**: Need to see object from all sides

### 5. High-Resolution Publication Images
```bash
python explode_automask_results.py --results_dir results/chair \
    --resolution 3840 2160 \
    --explosion_scale 0.5
```
**Use when**: Need 4K images for papers/presentations

### 6. Top and Bottom Views
```bash
python explode_automask_results.py --results_dir results/chair \
    --angles "0,80 90,80 180,80 270,80 0,-80 180,-80"
```
**Use when**: Important details on top/bottom of object

## Parameter Quick Reference

| Parameter | Default | What it does | Typical range |
|-----------|---------|--------------|---------------|
| `--explosion_scale` | 0.4 | How far parts move | 0.2 (subtle) - 1.0 (dramatic) |
| `--resolution` | 1920 1080 | Image size | 1280x720 (fast) - 3840x2160 (4K) |
| `--angles` | 6 views | Camera positions | See examples below |
| `--renderer` | auto | Rendering backend | auto, pyrender, blender, matplotlib |
| `--output_dir` | auto | Where to save | any directory path |

## Camera Angle Examples

Format: `"azimuth,elevation"` (azimuth=horizontal rotation, elevation=up/down)

**4 Cardinal Directions** (front, right, back, left):
```
"0,30 90,30 180,30 270,30"
```

**8 All-Around Views**:
```
"0,30 45,30 90,30 135,30 180,30 225,30 270,30 315,30"
```

**Top-Down Emphasis**:
```
"0,60 90,60 180,60 270,60 0,80 0,30"
```

**Dynamic Mix** (high, medium, low):
```
"0,60 45,45 90,30 135,15 180,30 225,45 270,60 315,45"
```

## File Organization After Running

```
results/my_mesh/
├── auto_mask_mesh_final.glb          # Original segmented mesh
├── final_face_ids.npy                # Part labels
└── exploded_renders/                 # NEW: Output directory
    ├── render_view_00_az0_el30.png   # View 1
    ├── render_view_01_az90_el30.png  # View 2
    ├── ...                           # More views
    ├── exploded_mesh.glb             # 3D exploded mesh
    └── summary.txt                   # Rendering info
```

## Troubleshooting One-Liners

**Can't find mesh files?**
```bash
# List what's in your results directory
ls -la results/my_mesh/
```

**Need to see available results directories?**
```bash
# Find all auto_mask results
find P3-SAM/demo/results -name "*.glb"
```

**Want to batch process multiple meshes?**
```bash
# Process all results at once
for dir in results/*/; do
    python explode_automask_results.py --results_dir "$dir"
done
```

**Check if rendering worked?**
```bash
# Count output images
ls results/my_mesh/exploded_renders/*.png | wc -l
```

## Tips & Tricks

### Tip 1: Different Scales for Different Objects
- **Mechanical parts** (gears, engines): 0.6-0.8 (larger explosion)
- **Furniture** (chairs, tables): 0.4-0.5 (medium)
- **Small objects** (toys, tools): 0.3-0.4 (subtle)

### Tip 2: Creating Animations
Generate many views and combine into video:
```bash
# Generate 72 frames (5° increments)
python explode_automask_results.py --results_dir results/chair \
    --angles "$(seq 0 5 355 | awk '{printf "%d,30 ", $1}')"

# Then use ffmpeg to create video
cd results/chair/exploded_renders
ffmpeg -framerate 24 -pattern_type glob -i 'render_view_*.png' -c:v libx264 animation.mp4
```

### Tip 3: Comparing Explosion Scales
```bash
# Generate multiple scales at once
for scale in 0.3 0.5 0.7; do
    python explode_automask_results.py --results_dir results/chair \
        --explosion_scale $scale \
        --output_dir results/chair/exploded_$scale
done
```

## Need More Control?

Use the advanced script directly:
```bash
python explode_and_render.py \
    --mesh_path path/to/specific/mesh.glb \
    --face_ids path/to/specific/face_ids.npy \
    --explosion_scale 0.5 \
    --resolution 2560 1440 \
    --angles "custom angles here" \
    --output_dir custom/output/path
```

## Getting Help

**See all options:**
```bash
python explode_automask_results.py --help
```

**Run example script:**
```bash
./example_explode_render.sh
```

**Read full documentation:**
```bash
cat EXPLODE_RENDER_README.md        # Full documentation
cat RENDERING_BACKENDS.md           # Rendering backend guide (pyrender/blender/matplotlib)
```

---

**Questions?** 
- Check `EXPLODE_RENDER_README.md` for detailed documentation
- Check `RENDERING_BACKENDS.md` for rendering issues and backend options

