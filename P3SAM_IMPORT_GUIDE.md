# P3-SAM Import Guide

## Clean and Simple Import Structure

After installation, use this **single, clean import pattern**:

```python
from p3sam.demo import AutoMask
```

That's it! No confusion, no multiple ways to import.

---

## Installation

```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part
./reinstall_p3sam.sh
```

Or manually:

```bash
cd P3-SAM
pip install -e .
```

---

## Usage Examples

### Basic Usage

```python
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
    prompt_num=400,
    threshold=0.95,
    post_process=True
)

print(f"Segmented into {len(aabb)} parts")
```

### Advanced Usage

```python
from p3sam.demo import AutoMask, P3SAM, mesh_sam
from p3sam.model import build_P3SAM, load_state_dict
import trimesh
import numpy as np

# Use the lower-level mesh_sam function
model = P3SAM()
model.load_state_dict(ckpt_path="/path/to/weights")
model.eval()
model.cuda()

mesh = trimesh.load("mesh.glb", force='mesh')

aabb, face_ids, mesh = mesh_sam(
    [model, model],
    mesh,
    save_path="output/",
    point_num=100000,
    prompt_num=400,
    threshold=0.95,
    post_process=True,
    show_info=True
)
```

---

## Complete Import Reference

### From `p3sam.demo`:

```python
from p3sam.demo import AutoMask     # Main class for easy usage
from p3sam.demo import P3SAM        # Model class
from p3sam.demo import mesh_sam     # Core segmentation function
```

### From `p3sam.model`:

```python
from p3sam.model import build_P3SAM      # Model builder
from p3sam.model import load_state_dict  # Checkpoint loader
```

---

## Testing Your Installation

Run the test script from any directory:

```bash
python /home/nranawakaara/Projects/Hunyuan3D-Part/test_p3sam_install.py
```

Or quick test:

```bash
python -c "from p3sam.demo import AutoMask; print('✓ P3-SAM installed correctly!')"
```

---

## Package Structure

```
p3sam/
├── __init__.py           # Package root
├── model.py              # Model building functions
├── demo/
│   ├── __init__.py       # Exports: AutoMask, P3SAM, mesh_sam
│   ├── auto_mask.py      # Main segmentation implementation
│   ├── app.py            # Interactive demo app
│   └── ...
└── utils/
    └── chamfer3D/        # Utility functions
```

---

## Why This Structure?

✅ **Single import pattern**: No confusion about which way to import  
✅ **Clean namespace**: `p3sam.demo` clearly indicates demo/main usage  
✅ **Explicit**: You know exactly where things come from  
✅ **Pythonic**: Follows standard Python package conventions  

---

## Common Patterns

### Pattern 1: Quick Segmentation

```python
from p3sam.demo import AutoMask
import trimesh

auto_mask = AutoMask(ckpt_path="weights/p3sam.safetensors")
mesh = trimesh.load("input.glb", force='mesh')
aabb, face_ids, mesh = auto_mask.predict_aabb(mesh, save_path="results/")
```

### Pattern 2: Batch Processing

```python
from p3sam.demo import AutoMask
import trimesh
from pathlib import Path

auto_mask = AutoMask(ckpt_path="weights/p3sam.safetensors")

for mesh_file in Path("inputs").glob("*.glb"):
    print(f"Processing {mesh_file.name}...")
    mesh = trimesh.load(mesh_file, force='mesh')
    output_dir = f"results/{mesh_file.stem}"
    aabb, face_ids, _ = auto_mask.predict_aabb(
        mesh,
        save_path=output_dir,
        show_info=False
    )
    print(f"  → Found {len(aabb)} parts")
```

### Pattern 3: Integration with Other Tools

```python
from p3sam.demo import AutoMask
import trimesh
import numpy as np

def segment_and_analyze(mesh_path):
    """Segment mesh and return part statistics."""
    auto_mask = AutoMask(ckpt_path="weights/p3sam.safetensors")
    mesh = trimesh.load(mesh_path, force='mesh')
    
    aabb, face_ids, mesh = auto_mask.predict_aabb(
        mesh,
        save_path="temp/",
        show_info=False
    )
    
    # Analyze parts
    unique_parts = np.unique(face_ids[face_ids >= 0])
    stats = {}
    for part_id in unique_parts:
        part_faces = np.sum(face_ids == part_id)
        stats[int(part_id)] = {
            'face_count': int(part_faces),
            'percentage': float(part_faces / len(face_ids) * 100)
        }
    
    return stats

# Use it
stats = segment_and_analyze("model.glb")
print(f"Found {len(stats)} parts")
for part_id, info in stats.items():
    print(f"  Part {part_id}: {info['face_count']} faces ({info['percentage']:.1f}%)")
```

---

## Troubleshooting

### Import Error: "No module named 'p3sam'"

**Solution**: Install the package first:
```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part
./reinstall_p3sam.sh
```

### Import Error: "No module named 'p3sam.demo'"

**Solution**: Make sure you installed the latest version:
```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part
./reinstall_p3sam.sh
```

### Still having issues?

Run the test script to diagnose:
```bash
python /home/nranawakaara/Projects/Hunyuan3D-Part/test_p3sam_install.py
```

---

## Summary

**The ONE way to import P3-SAM:**

```python
from p3sam.demo import AutoMask
```

That's all you need to remember! 🎉

