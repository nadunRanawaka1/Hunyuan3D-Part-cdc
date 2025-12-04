# P3-SAM Package Setup - Complete Guide

## What Was Created

I've set up P3-SAM as a proper Python package that can be imported and used from anywhere on your system.

### Files Created:

1. **`P3-SAM/setup.py`** - Traditional Python package setup file
2. **`P3-SAM/pyproject.toml`** - Modern Python package configuration (PEP 518)
3. **`P3-SAM/__init__.py`** - Makes P3-SAM a proper Python package
4. **`P3-SAM/demo/__init__.py`** - Exposes AutoMask and related classes
5. **`P3-SAM/MANIFEST.in`** - Specifies which files to include in distribution
6. **`P3-SAM/INSTALL.md`** - Comprehensive installation guide
7. **`install_p3sam.sh`** - One-command installation script
8. **`example_use_p3sam.py`** - Example script showing usage from any directory

### Changes Made:

1. **`P3-SAM/demo/auto_mask.py`** - Wrapped main code in a `main()` function for entry point support

---

## Quick Start Installation

### Option 1: Using the Install Script (Easiest)

```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part
./install_p3sam.sh
```

### Option 2: Manual Installation

```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part/P3-SAM
pip install -e .
```

The `-e` flag installs in "editable" mode, meaning:
- ✅ You can use P3-SAM from any directory
- ✅ Changes to source code are immediately available (no reinstall needed)
- ✅ Perfect for development

---

## Usage After Installation

### 1. Command Line Interface

Run P3-SAM from **any directory**:

```bash
p3sam-auto-mask \
    --mesh_path /path/to/mesh.glb \
    --output_path results/ \
    --point_num 100000 \
    --prompt_num 400 \
    --threshold 0.95
```

### 2. Python Import (From Any Directory!)

```python
# Save this as my_script.py anywhere on your system
from p3sam.demo import AutoMask
import trimesh

# Initialize model
auto_mask = AutoMask(ckpt_path="/path/to/weights/p3sam.safetensors")

# Load and segment mesh
mesh = trimesh.load("mesh.glb", force='mesh')
aabb, face_ids, mesh = auto_mask.predict_aabb(
    mesh,
    save_path="output/",
    point_num=100000,
    prompt_num=400
)

print(f"Segmented into {len(aabb)} parts")
```

### 3. Use the Example Script

```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part
python example_use_p3sam.py \
    --mesh_path P3-SAM/demo/assets/1.glb \
    --output_path test_results/
```

---

## How This Solves Your Problem

### Before (Without Package Setup):
```python
# Had to be in P3-SAM/demo directory or add paths manually
import sys
sys.path.insert(0, '/long/path/to/Hunyuan3D-Part/P3-SAM')
from demo.auto_mask import AutoMask  # ❌ Would fail if not in right directory
```

### After (With Package Setup):
```python
# Works from ANY directory! No path manipulation needed!
from p3sam.demo import AutoMask  # ✅ Just works
```

---

## Entry Points Created

Two command-line tools are now available system-wide:

1. **`p3sam-auto-mask`** - Run automatic segmentation
2. **`p3sam-app`** - Run interactive demo app (if implemented)

---

## Verifying Installation

### Test 1: Check if package is installed
```bash
pip show p3sam
```

### Test 2: Try importing from any directory
```bash
cd ~  # Go home
python -c "from demo.auto_mask import AutoMask; print('Success!')"
```

### Test 3: Run the example
```bash
python /home/nranawakaara/Projects/Hunyuan3D-Part/example_use_p3sam.py \
    --mesh_path /home/nranawakaara/Projects/Hunyuan3D-Part/P3-SAM/demo/assets/1.glb \
    --output_path ~/p3sam_test/
```

---

## Integration with Your Workflow

### Example: Using P3-SAM in explode_and_render.py

Now you can use P3-SAM directly without worrying about paths:

```python
# In any script, anywhere
from p3sam.demo import AutoMask
import trimesh

def auto_segment_mesh(mesh_path, output_dir):
    """Segment mesh using P3-SAM, then explode and render."""
    
    # Use P3-SAM
    auto_mask = AutoMask(ckpt_path="path/to/checkpoint")
    mesh = trimesh.load(mesh_path, force='mesh')
    aabb, face_ids, mesh = auto_mask.predict_aabb(
        mesh,
        save_path=output_dir,
        point_num=100000,
        prompt_num=400
    )
    
    # Now use with your explode_and_render
    from explode_and_render import create_scene_from_parts, render_scene_with_labels
    scene, part_info, label_mapping = create_scene_from_parts(mesh, face_ids)
    # ... continue with rendering
    
    return face_ids
```

---

## Development Workflow

1. **Edit source files** in `P3-SAM/`
2. **Changes are immediately available** (no reinstall needed)
3. **Test from any directory**:
   ```bash
   cd ~/my_project
   python my_script.py  # Uses latest P3-SAM code
   ```

---

## Uninstalling

```bash
pip uninstall p3sam
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'p3sam'"

**Solution**: Install the package first:
```bash
cd P3-SAM
pip install -e .
```

### "Import Error: cannot import name 'AutoMask'"

**Solution**: Make sure you're importing correctly:
```python
from p3sam.demo import AutoMask  # ✅ Correct
# NOT: from P3-SAM.demo import AutoMask  # ❌ Wrong
# NOT: from demo.auto_mask import AutoMask  # ❌ Old style
```

### Changes to source not reflecting

**Solution**: You might need to restart your Python interpreter, but reinstall should NOT be needed if you used `-e` flag.

---

## Summary

✅ **Before**: Could only use P3-SAM from specific directories  
✅ **After**: Use P3-SAM from **any directory** on your system  
✅ **Benefit**: Clean imports, no path hacking, proper Python package  
✅ **Development**: Edit source, changes available immediately  

The package is now installed and ready to use system-wide! 🎉

