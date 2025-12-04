#!/usr/bin/env python3
"""
Quick script to explode and render auto_mask.py results.
This script automatically finds the mesh and face_ids from auto_mask.py output directory.

Usage:
    python explode_automask_results.py results/my_mesh
    python explode_automask_results.py results/my_mesh --explosion_scale 0.6
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Import the main rendering script
from explode_and_render import load_partitioned_mesh, create_scene_from_parts, render_scene_with_labels

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'XPart', 'partgen'))
from utils.mesh_utils import explode_mesh


def find_automask_outputs(results_dir):
    """
    Find the mesh and face_ids files from auto_mask.py output directory.
    
    Args:
        results_dir: Path to results directory (e.g., 'results/1')
    
    Returns:
        mesh_path: Path to the mesh file
        face_ids_path: Path to the face_ids numpy file
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Look for mesh files (prefer final versions)
    mesh_candidates = [
        'auto_mask_mesh_final.glb',
        'auto_mask_mesh_final_post.glb',
        'auto_mask_mesh_filtered_2.glb',
        'auto_mask_mesh.glb',
    ]
    
    mesh_path = None
    for candidate in mesh_candidates:
        path = results_dir / candidate
        if path.exists():
            mesh_path = path
            break
    
    # If no specific file found, search for any .glb file
    if mesh_path is None:
        glb_files = list(results_dir.glob('*.glb'))
        if glb_files:
            # Prefer files with "final" or "mesh" in the name
            for glb in glb_files:
                if 'final' in glb.name.lower() or 'mesh' in glb.name.lower():
                    mesh_path = glb
                    break
            if mesh_path is None:
                mesh_path = glb_files[0]
    
    if mesh_path is None:
        raise FileNotFoundError(f"No mesh file found in {results_dir}")
    
    # Look for face_ids
    face_ids_candidates = [
        'final_face_ids.npy',
        'face_ids.npy',
        mesh_path.stem + '_face_ids.npy',
    ]
    
    face_ids_path = None
    for candidate in face_ids_candidates:
        path = results_dir / candidate
        if path.exists():
            face_ids_path = path
            break
    
    # Search for any .npy file with 'face' in the name
    if face_ids_path is None:
        npy_files = list(results_dir.glob('*face*.npy'))
        if npy_files:
            face_ids_path = npy_files[0]
    
    if face_ids_path is None:
        # Try to extract face_ids from the mesh itself if it has per-face colors
        print(f"Warning: No face_ids.npy found. Will try to extract from mesh colors.")
        face_ids_path = None
    
    print(f"Found mesh: {mesh_path}")
    if face_ids_path:
        print(f"Found face_ids: {face_ids_path}")
    
    return str(mesh_path), str(face_ids_path) if face_ids_path else None


def extract_face_ids_from_mesh(mesh):
    """
    Extract face_ids from mesh face colors (as a fallback).
    
    Args:
        mesh: trimesh.Trimesh with face colors
    
    Returns:
        face_ids: numpy array of face labels
    """
    import numpy as np
    
    if not hasattr(mesh.visual, 'face_colors'):
        raise ValueError("Mesh has no face colors and no face_ids file found")
    
    print("Extracting face_ids from mesh face colors...")
    face_colors = mesh.visual.face_colors
    
    # Group faces by color
    unique_colors = {}
    face_ids = np.zeros(len(face_colors), dtype=np.int32)
    
    for i, color in enumerate(face_colors):
        color_key = tuple(color[:3])  # Ignore alpha
        if color_key not in unique_colors:
            unique_colors[color_key] = len(unique_colors)
        face_ids[i] = unique_colors[color_key]
    
    print(f"Extracted {len(unique_colors)} unique parts from mesh colors")
    return face_ids


def main():
    parser = argparse.ArgumentParser(
        description="Explode and render auto_mask.py results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with auto_mask.py output directory
  python explode_automask_results.py results/1
  
  # Larger explosion
  python explode_automask_results.py results/1 --explosion_scale 0.8
  
  # Custom output directory and resolution
  python explode_automask_results.py results/1 --output_dir my_renders --resolution 3840 2160
  
  # More camera angles
  python explode_automask_results.py results/1 --angles "0,30 45,30 90,30 135,30 180,30 225,30 270,30 315,30"
        """
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        help='Path to auto_mask.py results directory (e.g., results/1)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for renders (default: results_dir/exploded_renders)'
    )
    parser.add_argument(
        '--explosion_scale',
        type=float,
        default=0.4,
        help='Explosion scale factor (default: 0.4)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        nargs=2,
        default=[1920, 1080],
        help='Image resolution (default: 1920 1080)'
    )
    parser.add_argument(
        '--angles',
        type=str,
        default='0,30 90,30 180,30 270,30 45,60 225,-30',
        help='Camera angles as "azimuth,elevation" pairs (used with --camera_mode angles)'
    )
    parser.add_argument(
        '--camera_mode',
        type=str,
        choices=['angles', 'uniform', 'standard', 'icosahedron', 'dodecahedron', 'octohedron', 'cube', 'tetrahedron'],
        default='icosahedron',
        help='Camera sampling mode (default: angles - manual specification)'
    )
    parser.add_argument(
        '--num_views',
        type=int,
        default=None,
        help='Number of views for uniform/standard modes (default: 12 for uniform, 8 for standard)'
    )
    parser.add_argument(
        '--renderer',
        type=str,
        choices=['auto', 'pyrender', 'blender', 'matplotlib'],
        default='auto',
        help='Rendering backend: auto (detect best), pyrender (offscreen, no pyglet), blender, or matplotlib (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'exploded_renders')
    
    print("=" * 70)
    print("Auto-Mask Results Exploder and Renderer")
    print("=" * 70)
    print()
    
    # Find mesh and face_ids
    try:
        mesh_path, face_ids_path = find_automask_outputs(args.results_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Load mesh and face_ids
    try:
        if face_ids_path:
            mesh, face_ids = load_partitioned_mesh(mesh_path, face_ids_path)
        else:
            import trimesh
            mesh = trimesh.load(mesh_path, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            face_ids = extract_face_ids_from_mesh(mesh)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        sys.exit(1)
    
    # Create scene with separate parts
    print()
    scene, part_info, label_mapping = create_scene_from_parts(mesh, face_ids)
    
    # Save label mapping to JSON
    import json
    mapping_path = os.path.join(args.output_dir, 'label_mapping.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"\nLabel mapping saved to: {mapping_path}")
    
    # Explode the scene
    print()
    print(f"Exploding mesh with scale={args.explosion_scale}...")
    exploded_scene = explode_mesh(scene, explosion_scale=args.explosion_scale)
    
    # Parse camera angles (only used if camera_mode='angles')
    camera_angles = []
    if args.camera_mode == 'angles':
        for angle_str in args.angles.split():
            azimuth, elevation = map(float, angle_str.split(','))
            camera_angles.append((azimuth, elevation))
    
    # Render from multiple angles
    print()
    print(f"Rendering views...")
    render_scene_with_labels(
        exploded_scene,
        part_info,
        camera_angles,
        args.output_dir,
        resolution=tuple(args.resolution),
        explosion_scale=args.explosion_scale,
        renderer=args.renderer,
        camera_mode=args.camera_mode,
        num_views=args.num_views
    )
    
    # Save the exploded mesh
    exploded_mesh_path = os.path.join(args.output_dir, 'exploded_mesh.glb')
    exploded_scene.export(exploded_mesh_path)
    print(f"\nExploded mesh saved to: {exploded_mesh_path}")
    
    # Create a summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Exploded Mesh Rendering Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Source mesh: {mesh_path}\n")
        f.write(f"Face IDs: {face_ids_path or 'Extracted from mesh colors'}\n")
        f.write(f"Number of parts: {len(part_info)}\n")
        f.write(f"Explosion scale: {args.explosion_scale}\n")
        f.write(f"Resolution: {args.resolution[0]}x{args.resolution[1]}\n")
        f.write(f"Number of views: {len(camera_angles)}\n\n")
        f.write(f"Part Details:\n")
        f.write(f"-" * 50 + "\n")
        for part_id, info in sorted(part_info.items()):
            f.write(f"Part {part_id}: {info['vertex_count']} vertices, "
                   f"center={info['center']}\n")
    
    print(f"Summary saved to: {summary_path}")
    print()
    print("=" * 70)
    print("Complete! Check the output directory for renders.")
    print("=" * 70)


if __name__ == '__main__':
    main()

