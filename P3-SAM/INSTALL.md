# P3-SAM Installation Guide

## Quick Install

### Option 1: Development Install (Recommended for local usage)

Install in editable mode so you can use the package from anywhere while still being able to edit the source:

```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part/P3-SAM
pip install -e .
```

### Option 2: Regular Install

```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part/P3-SAM
pip install .
```

### Option 3: Install with Optional Dependencies

```bash
# Install with demo dependencies (viser, gradio)
pip install -e ".[demo]"

# Install with development dependencies
pip install -e ".[dev]"

# Install everything
pip install -e ".[demo,dev]"
```

## Usage After Installation

### 1. Command Line Usage

After installation, you can use P3-SAM from anywhere:

```bash
# Run auto-mask from any directory
p3sam-auto-mask \
    --ckpt_path /path/to/weights/p3sam.safetensors \
    --mesh_path /path/to/mesh.glb \
    --output_path /path/to/output \
    --point_num 100000 \
    --prompt_num 400
```

### 2. Python API Usage

```python
# Import from anywhere
from demo.auto_mask import AutoMask
import trimesh

# Load model
auto_mask = AutoMask(ckpt_path="/path/to/weights/p3sam.safetensors")

# Load mesh
mesh = trimesh.load("mesh.glb", force='mesh')

# Run segmentation
aabb, face_ids, mesh = auto_mask.predict_aabb(
    mesh,
    save_path="output/",
    point_num=100000,
    prompt_num=400,
    threshold=0.95,
    post_process=True,
    save_mid_res=True,
    show_info=True,
)

print(f"Found {len(aabb)} parts")
```

### 3. Programmatic Usage Example

```python
#!/usr/bin/env python3
"""Example script using installed P3-SAM package."""

import sys
import trimesh
import numpy as np
from pathlib import Path
from demo.auto_mask import AutoMask

def segment_mesh(mesh_path, output_dir, checkpoint_path):
    """Segment a mesh into parts."""
    
    # Initialize AutoMask
    auto_mask = AutoMask(
        ckpt_path=checkpoint_path,
        point_num=100000,
        prompt_num=400,
        threshold=0.95,
        post_process=True
    )
    
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # Run segmentation
    aabb, face_ids, mesh = auto_mask.predict_aabb(
        mesh,
        save_path=output_dir,
        save_mid_res=True,
        show_info=True,
    )
    
    # Get part information
    unique_parts = np.unique(face_ids[face_ids >= 0])
    print(f"\nSegmentation complete!")
    print(f"  Found {len(unique_parts)} parts")
    print(f"  Part IDs: {unique_parts.tolist()}")
    print(f"  Results saved to: {output_dir}")
    
    return aabb, face_ids, mesh

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python example.py <mesh_path> <output_dir> [checkpoint_path]")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    output_dir = sys.argv[2]
    checkpoint_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    aabb, face_ids, mesh = segment_mesh(mesh_path, output_dir, checkpoint_path)
```

## Verifying Installation

```python
# Test import
python -c "from demo.auto_mask import AutoMask; print('P3-SAM installed successfully!')"
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in a directory outside of the P3-SAM source:

```bash
cd ~  # Go to home directory
python -c "from demo.auto_mask import AutoMask"
```

### CUDA Errors

Make sure PyTorch is installed with CUDA support:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Sonata Dependencies

If you get errors about Sonata, make sure you've installed it:

```bash
# Install Sonata from Facebook Research
pip install git+https://github.com/facebookresearch/sonata.git
```

### XPart Dependencies

The package requires XPart modules. Make sure the path in `model.py` is correct:

```python
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'XPart/partgen'))
```

## Uninstalling

```bash
pip uninstall p3sam
```

## Development Workflow

For active development:

1. Install in editable mode: `pip install -e .`
2. Make changes to source files
3. Changes are immediately available (no reinstall needed)
4. Run tests: `pytest tests/` (if tests exist)
5. Format code: `black .` (if black is installed)

## Environment Variables

You can set environment variables for default paths:

```bash
export P3SAM_CHECKPOINT=/path/to/weights/p3sam.safetensors
export P3SAM_OUTPUT_DIR=/path/to/default/output
```

