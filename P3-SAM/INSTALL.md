# P3-SAM Installation Guide

## Quick Install

### Option 1: Development Install (Recommended for local usage)

Install all dependencies
```bash
mamba create -n 3D_part python=3.10
mamba activate 3D_part
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install spconv-cu126
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
pip install huggingface_hub timm omegaconf
pip install viser fpsample trimesh numba gradio
cd utils/chamfer3D
python setup.py install
```

Install in editable mode so you can use the package from anywhere while still being able to edit the source:

```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part/P3-SAM
pip install -e .
```

Then, for XPART

```bash
cd /home/nranawakaara/Projects/Hunyuan3D-Part/XPart
pip install -r requirements.txt
pip install scikit-image diffusers
pip install lightning
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
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


