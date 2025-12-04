#!/usr/bin/env python3
"""
Script to load a partitioned mesh from auto_mask.py, explode it, 
render from multiple angles, and label each part.

Usage:
    python explode_and_render.py --mesh_path results/mesh.glb --face_ids results/face_ids.npy --output_dir exploded_renders
"""

import os
import sys
import numpy as np
import trimesh
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
import torch
import shutil
from tqdm import tqdm
import json
import colorsys


# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'XPart', 'partgen'))
from utils.mesh_utils import explode_mesh
from utils.camera_utils import sample_view_matrices, sample_view_matrices_polyhedra, view_matrix


def load_partitioned_mesh(mesh_path, face_ids_path=None, verbose=False):
    """
    Load a mesh and its face_ids (part labels).
    
    Args:
        mesh_path: Path to the mesh file (.glb, .ply, etc.)
        face_ids_path: Path to face_ids numpy array. If None, try to infer from mesh_path
    
    Returns:
        mesh: trimesh.Trimesh object
        face_ids: numpy array of face labels
    """
    print(f"Explode and render: Loading mesh from: {mesh_path} for rendering")
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # If mesh is a Scene, extract the main geometry
    if isinstance(mesh, trimesh.Scene):
        if verbose:
            print("Loaded a Scene, extracting main geometry...")
        mesh = mesh.dump(concatenate=True)
    
    if face_ids_path is None:
        # Try to infer face_ids path
        base_path = Path(mesh_path).parent
        face_ids_candidates = [
            base_path / 'final_face_ids.npy',
            Path(mesh_path).with_suffix('.npy'),
            Path(mesh_path.replace('.glb', '_face_ids.npy')),
            base_path / 'face_ids.npy',
        
        ]
        
        for candidate in face_ids_candidates:
            if candidate.exists():
                face_ids_path = str(candidate)
                break
        
        if face_ids_path is None:
            raise FileNotFoundError(
                f"Could not find face_ids file. Please specify with --face_ids. "
                f"Searched: {[str(c) for c in face_ids_candidates]}"
            )
    
    if verbose:
        print(f"Loading face_ids from: {face_ids_path}")
    face_ids = np.load(face_ids_path)
    
    if verbose:
        print(f"Mesh info: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Face IDs: shape={face_ids.shape}, unique parts={len(np.unique(face_ids[face_ids >= 0]))}")
    
    return mesh, face_ids


def create_scene_from_parts(mesh, face_ids, verbose=False):
    """
    Create a trimesh.Scene with each part as a separate geometry.
    
    Args:
        mesh: trimesh.Trimesh object
        face_ids: numpy array of face labels
    
    Returns:
        scene: trimesh.Scene with separate geometries for each part
        part_info: dict mapping part_id -> (center, color, vertex_count)
        label_mapping: dict mapping old_label -> new_label (0, 1, 2, ...)
    """
    if verbose:
        print("Creating scene from parts...")
    scene = trimesh.Scene()
    part_info = {}
    
    unique_ids = np.unique(face_ids)
    valid_ids = unique_ids[unique_ids >= 0]  # Filter out -1, -2, etc.
    
    # Create label mapping: old_id -> new_id (0, 1, 2, ...)
    label_mapping = {int(old_id): new_id for new_id, old_id in enumerate(valid_ids)}
    if verbose:
        print(f"Created label mapping: {len(label_mapping)} parts (0 to {len(label_mapping)-1})")
    
    # Generate high-contrast distinct colors for each part
    # Using HSV colorspace with varying hue for maximum distinction
    colors = []
    n_parts = len(valid_ids)
    for i in range(n_parts):
        # Use golden ratio for hue distribution (maximally distinct colors)
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio
        saturation = 0.9
        value = 0.85
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    
    for idx, part_id in enumerate(valid_ids):
        # Extract faces for this part
        part_mask = face_ids == part_id
        part_faces_indices = np.where(part_mask)[0]
        
        if len(part_faces_indices) == 0:
            continue
        
        # Get the faces and create submesh
        part_faces = mesh.faces[part_faces_indices]
        
        # Find all vertices used by these faces
        vertices_used = np.unique(part_faces.flatten())
        
        # Create a mapping from old vertex indices to new ones
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertices_used)}
        
        # Extract vertices and remap faces
        part_vertices = mesh.vertices[vertices_used]
        part_faces_remapped = np.array([
            [vertex_map[v] for v in face] 
            for face in part_faces
        ])
        
        # Create submesh for this part
        part_mesh = trimesh.Trimesh(
            vertices=part_vertices,
            faces=part_faces_remapped,
            process=False
        )
        
        # Set color
        color_rgb = np.array(colors[idx % len(colors)]) * 255
        part_mesh.visual.face_colors = np.tile(
            np.append(color_rgb, 255), 
            (len(part_mesh.faces), 1)
        ).astype(np.uint8)
        
        # Calculate center and average normal direction
        center = part_vertices.mean(axis=0)
        
        # Calculate average normal (for visibility check)
        face_normals = np.cross(
            part_vertices[part_faces_remapped[:, 1]] - part_vertices[part_faces_remapped[:, 0]],
            part_vertices[part_faces_remapped[:, 2]] - part_vertices[part_faces_remapped[:, 0]]
        )
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10)
        avg_normal = face_normals.mean(axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
        
        # Get new label ID (0, 1, 2, ...)
        new_label = label_mapping[int(part_id)]
        
        # Add to scene with new label name
        geom_name = f"part_{new_label}"
        scene.add_geometry(part_mesh, geom_name=geom_name)
        
        # Store info with new label
        part_info[new_label] = {
            'center': center,
            'color': color_rgb,
            'vertex_count': len(part_vertices),
            'geom_name': geom_name,
            'normal': avg_normal,
            'old_label': int(part_id)
        }
        
        if verbose:
            print(f"  Part {new_label} (was {part_id}): {len(part_vertices)} vertices, center={center}")
    
    if verbose:
        print(f"Created scene with {len(scene.geometry)} parts")
    return scene, part_info, label_mapping


def render_with_pyrender_offscreen(scene, camera_transform, resolution, verbose=False):
    """
    Render using pyrender in offscreen mode (no pyglet needed).
    
    Args:
        scene: trimesh.Scene
        camera_transform: 4x4 camera matrix
        resolution: (width, height) tuple
    
    Returns:
        tuple: (PIL Image, depth_buffer) or (None, None) if failed
    """
    try:
        import pyrender
        from io import BytesIO
        
        # Create pyrender scene with gray background
        pr_scene = pyrender.Scene(
            ambient_light=[0.4, 0.4, 0.4],
            bg_color=[0.95, 0.95, 0.95, 1.0]  # Light gray background
        )
        
        if verbose:
            print(f"  PyRender: Rendering {len(scene.geometry)} parts...")
        
        # Add all geometries from trimesh scene
        for geom_name, geometry in scene.geometry.items():
            if hasattr(geometry, 'vertices'):
                # Convert face colors to vertex colors for pyrender compatibility
                mesh_to_render = geometry.copy()
                
                if verbose:
                    print(f"    Part {geom_name}: {len(geometry.vertices)} vertices, {len(geometry.faces)} faces")
                
                if hasattr(geometry.visual, 'face_colors'):
                    # Convert face colors to vertex colors
                    face_colors = geometry.visual.face_colors
                    vertex_colors = np.zeros((len(geometry.vertices), 4), dtype=np.uint8)
                    
                    # Average colors from all faces that use each vertex
                    vertex_count = np.zeros(len(geometry.vertices), dtype=np.int32)
                    vertex_color_sum = np.zeros((len(geometry.vertices), 4), dtype=np.float32)
                    
                    for face_idx, face in enumerate(geometry.faces):
                        for vertex_idx in face:
                            vertex_color_sum[vertex_idx] += face_colors[face_idx].astype(np.float32)
                            vertex_count[vertex_idx] += 1
                    
                    # Average the colors
                    for i in range(len(vertex_colors)):
                        if vertex_count[i] > 0:
                            vertex_colors[i] = (vertex_color_sum[i] / vertex_count[i]).astype(np.uint8)
                        else:
                            vertex_colors[i] = [200, 200, 200, 255]
                    
                    avg_color = np.mean(vertex_colors[:, :3], axis=0)

                    if verbose:
                        print(f"      Color: RGB({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})")
                    
                    # Create new mesh with vertex colors
                    mesh_to_render = trimesh.Trimesh(
                        vertices=geometry.vertices,
                        faces=geometry.faces,
                        vertex_colors=vertex_colors,
                        process=False
                    )
                
                # Convert to pyrender mesh
                try:
                    pr_mesh = pyrender.Mesh.from_trimesh(mesh_to_render, smooth=False)
                    pr_scene.add(pr_mesh)
                except Exception as e:
                    # Fallback: create mesh without colors
                    print(f"  Warning: Could not add colors for {geom_name}, using default material")
                    plain_mesh = trimesh.Trimesh(
                        vertices=geometry.vertices,
                        faces=geometry.faces,
                        process=False
                    )
                    pr_mesh = pyrender.Mesh.from_trimesh(plain_mesh, smooth=False)
                    pr_scene.add(pr_mesh)
        
        # Check if any meshes were added
        if len(pr_scene.meshes) == 0:
            if verbose:
                print("  Warning: No meshes added to pyrender scene!")
            return None, None
        
        if verbose:
            print(f"  Added {len(pr_scene.meshes)} meshes to scene")
        
        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        pr_scene.add(camera, pose=camera_transform)
        
        # Add multiple lights for better illumination from different angles
        # Main light from camera
        light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        pr_scene.add(light1, pose=camera_transform)
        
        # Fill light from opposite side
        light_transform2 = camera_transform.copy()
        light_transform2[:3, 3] = -camera_transform[:3, 3] * 0.5
        light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
        pr_scene.add(light2, pose=light_transform2)
        
        # Top light
        light_transform3 = np.eye(4)
        light_transform3[:3, 3] = [0, 0, 10]
        light3 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        pr_scene.add(light3, pose=light_transform3)
        
        # Render offscreen
        if verbose:
            print(f"  Rendering at {resolution[0]}x{resolution[1]}...")
        renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
        color, depth = renderer.render(pr_scene)
        renderer.delete()
        
        if verbose:
            print(f"  Render complete")
        
        # Convert to PIL Image
        img = Image.fromarray(color)
        return img, depth
        
    except Exception as e:
        print(f"  Pyrender offscreen failed: {e}")
        return None, None


def render_with_blender(scene, camera_pos, camera_target, resolution, output_path):
    """
    Render using Blender (subprocess call).
    
    Args:
        scene: trimesh.Scene
        camera_pos: camera position [x, y, z]
        camera_target: camera target [x, y, z]
        resolution: (width, height) tuple
        output_path: where to save the image
    
    Returns:
        tuple: (PIL Image, None) or (None, None) if failed (no depth buffer support)
    """
    try:
        import subprocess
        import tempfile
        
        # Save scene to temporary file
        temp_dir = tempfile.mkdtemp()
        scene_path = os.path.join(temp_dir, 'scene.glb')
        scene.export(scene_path)
        
        # Create Blender Python script
        blender_script = f"""
import bpy
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import the scene
bpy.ops.import_scene.gltf(filepath='{scene_path}')

# Set up camera
cam_data = bpy.data.cameras.new('Camera')
cam_obj = bpy.data.objects.new('Camera', cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Set camera position and target
cam_obj.location = {tuple(camera_pos)}
target_location = {tuple(camera_target)}

# Point camera at target
direction = tuple(target_location[i] - cam_obj.location[i] for i in range(3))
rot_quat = direction_to_rotation(direction)
cam_obj.rotation_euler = rot_quat.to_euler()

def direction_to_rotation(direction):
    import mathutils
    direction = mathutils.Vector(direction).normalized()
    up = mathutils.Vector((0, 0, 1))
    
    # Handle edge case where direction is parallel to up
    if abs(direction.dot(up)) > 0.999:
        up = mathutils.Vector((0, 1, 0))
    
    right = direction.cross(up).normalized()
    up = right.cross(direction).normalized()
    
    mat = mathutils.Matrix((right, up, -direction)).transposed().to_4x4()
    return mat.to_quaternion()

cam_obj.rotation_euler = direction_to_rotation(direction).to_euler()

# Set up lighting
light_data = bpy.data.lights.new('Light', 'SUN')
light_data.energy = 2.0
light_obj = bpy.data.objects.new('Light', light_data)
bpy.context.scene.collection.objects.link(light_obj)
light_obj.location = cam_obj.location

# Set render settings
bpy.context.scene.render.resolution_x = {resolution[0]}
bpy.context.scene.render.resolution_y = {resolution[1]}
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = '{output_path}'

# Render
bpy.ops.render.render(write_still=True)
"""
        
        script_path = os.path.join(temp_dir, 'render_script.py')
        with open(script_path, 'w') as f:
            f.write(blender_script)
        
        # Run Blender
        print("  Running Blender...")
        result = subprocess.run(
            ['blender', '--background', '--python', script_path],
            capture_output=True,
            timeout=60
        )
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return Image.open(output_path), None
        else:
            print(f"  Blender failed: {result.stderr.decode()[:200]}")
            return None, None
            
    except Exception as e:
        print(f"  Blender rendering failed: {e}")
        return None, None


def render_with_matplotlib(scene, azimuth, elevation, resolution, verbose=False):
    """
    Render using matplotlib (fallback, lower quality).
    
    Args:
        scene: trimesh.Scene
        azimuth: horizontal angle in degrees
        elevation: vertical angle in degrees
        resolution: (width, height) tuple
    
    Returns:
        tuple: (PIL Image, None) - no depth buffer support
    """
    from io import BytesIO
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    if verbose:
        print(f"  Matplotlib: Rendering {len(scene.geometry)} parts...")
    
    for geom_name, geometry in scene.geometry.items():
        if hasattr(geometry, 'vertices'):
            vertices = geometry.vertices
            faces = geometry.faces
            
            if verbose:
                print(f"    Part {geom_name}: {len(vertices)} vertices, {len(faces)} faces")
            
            # Get color - convert to tuple for matplotlib
            if hasattr(geometry.visual, 'face_colors'):
                color_array = geometry.visual.face_colors[0][:3] / 255.0
                color = tuple(float(c) for c in color_array)
            else:
                color = (0.7, 0.7, 0.7)
            
            # Create mesh collection for better rendering
            mesh_faces = vertices[faces]
            
            # Create 3D polygon collection
            poly_collection = Poly3DCollection(
                mesh_faces,
                facecolors=color,
                edgecolors='black',
                linewidths=0.1,
                alpha=0.9
            )
            ax.add_collection3d(poly_collection)
    
    # Set axis limits based on scene bounds
    bounds = scene.bounds
    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    ax.set_zlim(bounds[0, 2], bounds[1, 2])
    
    # Set view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Set equal aspect ratio
    try:
        ax.set_box_aspect([
            bounds[1, 0] - bounds[0, 0],
            bounds[1, 1] - bounds[0, 1],
            bounds[1, 2] - bounds[0, 2]
        ])
    except:
        ax.set_box_aspect([1, 1, 1])
    
    # Turn off axis
    ax.set_axis_off()
    
    # Save to buffer with white background
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                pad_inches=0.1, facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    
    return img, None


def render_scene_with_labels(scene, part_info, camera_angles, output_dir, 
                              resolution=(1920, 1080), explosion_scale=0.4,
                              renderer='auto', camera_mode='angles', num_views=None, verbose=False):
    """
    Render the scene from multiple angles with part labels.
    
    Args:
        scene: trimesh.Scene to render
        part_info: dict with part information
        camera_angles: list of (azimuth, elevation) tuples in degrees (used if camera_mode='angles')
        output_dir: directory to save renders
        resolution: (width, height) tuple
        explosion_scale: scale factor for explosion
        renderer: 'auto', 'pyrender', 'blender', or 'matplotlib'
        camera_mode: 'angles' (manual azimuth/elevation), 'uniform' (random uniform), 
                     'standard', 'icosahedron', 'dodecahedron', etc.
        num_views: number of views (used for uniform/polyhedra modes)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect available renderers
    available_renderers = []
    
    if renderer == 'auto' or renderer == 'pyrender':
        try:
            import pyrender
            available_renderers.append('pyrender')
        except:
            pass
    
    if renderer == 'auto' or renderer == 'blender':
        try:
            import subprocess
            result = subprocess.run(['blender', '--version'], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                available_renderers.append('blender')
        except:
            pass
    
    if renderer == 'auto' or renderer == 'matplotlib':
        available_renderers.append('matplotlib')
    
    # Choose renderer
    if renderer == 'auto':
        if 'pyrender' in available_renderers:
            chosen_renderer = 'pyrender'
        elif 'blender' in available_renderers:
            chosen_renderer = 'blender'
        else:
            chosen_renderer = 'matplotlib'
    else:
        if renderer in available_renderers:
            chosen_renderer = renderer
        else:
            print(f"Warning: Renderer '{renderer}' not available. Falling back to matplotlib.")
            chosen_renderer = 'matplotlib'
    
    if verbose:
        print(f"Using renderer: {chosen_renderer}")
        print()
    
    # Calculate appropriate camera distance based on scene bounds
    scene_extents = scene.extents
    scene_scale = np.max(scene_extents)
    # Move closer: 2.0 -> 1.7 (approximately 0.3-0.5m closer scaled to object size)
    camera_distance = scene_scale * 1.25  # Distance as multiple of scene size
    
    if verbose: 
        print(f"Scene bounds: {scene_extents}, using camera distance: {camera_distance:.2f}")
    
    # Generate camera matrices using camera_utils
    lookat_position = torch.from_numpy(scene.centroid).float()
    
    if camera_mode == 'angles':
        # Manual azimuth/elevation specification (backward compatibility)
        if verbose:
            print(f"Camera mode: manual angles ({len(camera_angles)} views)")
        camera_transforms = []
        for azimuth, elevation in camera_angles:
            azimuth_rad = np.radians(azimuth)
            elevation_rad = np.radians(elevation)
            
            # Calculate camera position
            camera_pos = np.array([
                camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad),
                camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
                camera_distance * np.sin(elevation_rad)
            ])
            camera_pos_torch = torch.from_numpy(camera_pos + scene.centroid).float()
            
            # Use camera_utils.view_matrix
            cam_matrix = view_matrix(
                camera_pos_torch,
                lookat_position,
                up=torch.tensor([0.0, 0.0, 1.0])
            )
            camera_transforms.append((cam_matrix[0].numpy(), azimuth, elevation))
        
    elif camera_mode == 'uniform':
        # Uniform random sampling
        n = num_views if num_views is not None else 12
        if verbose:
            print(f"Camera mode: uniform random ({n} views)")
        cam_matrices = sample_view_matrices(n, camera_distance, lookat_position)
        camera_transforms = [(cam_matrices[i].numpy(), i, None) for i in range(n)]
        
    elif camera_mode in ['standard', 'icosahedron', 'dodecahedron', 'octohedron', 'cube', 'tetrahedron']:
        # Polyhedra-based sampling
        kwargs = {}
        if camera_mode == 'standard':
            kwargs['n'] = num_views if num_views is not None else 8
            kwargs['elevation'] = 15
            if verbose:
                print(f"Camera mode: standard ({kwargs['n']} views, elevation={kwargs['elevation']}°)")
        else:
            if verbose:
                print(f"Camera mode: {camera_mode}")
        
        cam_matrices = sample_view_matrices_polyhedra(
            camera_mode,
            camera_distance,
            lookat_position,
            **kwargs
        )
        camera_transforms = [(cam_matrices[i].numpy(), i, None) for i in range(len(cam_matrices))]
    
    else:
        raise ValueError(f"Unknown camera_mode: {camera_mode}")
    
    # Render from all camera positions
    
    for view_idx, (camera_transform, az_or_idx, el) in enumerate(tqdm(camera_transforms, desc="Rendering views")):
        if el is not None:
            # Manual angles mode
            if verbose:
                print(f"Rendering view {view_idx + 1}/{len(camera_transforms)}: "
                  f"azimuth={az_or_idx}°, elevation={el}°")
            view_name = f"az{az_or_idx}_el{el}"
        else:
            # Other modes
            if verbose:
                print(f"Rendering view {view_idx + 1}/{len(camera_transforms)}")
            view_name = f"view{view_idx:02d}"
        
        # Render using chosen backend
        output_path = os.path.join(output_dir, f"render_{view_name}.png")
        img = None
        
        # Extract camera position and target for renderers that need it
        camera_eye = camera_transform[:3, 3]
        camera_target = scene.centroid
        
        # Calculate azimuth and elevation for matplotlib fallback
        cam_to_target = camera_target - camera_eye
        distance = np.linalg.norm(cam_to_target)
        if distance > 1e-6:
            cam_to_target_norm = cam_to_target / distance
            # Elevation (angle from XY plane)
            elevation_calc = np.degrees(np.arcsin(np.clip(cam_to_target_norm[2], -1, 1)))
            # Azimuth (angle in XY plane)
            azimuth_calc = np.degrees(np.arctan2(cam_to_target_norm[1], cam_to_target_norm[0]))
        else:
            elevation_calc = 0
            azimuth_calc = 0
        
        depth_buffer = None
        
        if chosen_renderer == 'pyrender':
            img, depth_buffer = render_with_pyrender_offscreen(scene, camera_transform, resolution)
            if img is None:
                print("  Pyrender failed, falling back to matplotlib...")
                img, depth_buffer = render_with_matplotlib(scene, azimuth_calc, elevation_calc, resolution)
        
        elif chosen_renderer == 'blender':
            img, depth_buffer = render_with_blender(scene, camera_eye, camera_target, resolution, output_path)
            if img is None:
                print("  Blender failed, falling back to matplotlib...")
                img, depth_buffer = render_with_matplotlib(scene, azimuth_calc, elevation_calc, resolution)
        
        else:  # matplotlib
            img, depth_buffer = render_with_matplotlib(scene, azimuth_calc, elevation_calc, resolution)
        
        # Add labels to the image
        img = add_part_labels(img, scene, part_info, camera_transform, resolution, depth_buffer)
        
        # Save image
        img.save(output_path)
        if verbose:
            print(f"Saved render to: {output_path}")
    
    print(f"\nAll renders saved to: {output_dir}")


def add_part_labels(img, scene, part_info, camera_transform, resolution, depth_buffer=None, verbose=False):
    """
    Add numerical labels to each part in the rendered image.
    Only shows labels for parts that are visible from the camera view.
    
    Args:
        img: PIL Image
        scene: trimesh.Scene
        part_info: dict with part information
        camera_transform: 4x4 camera transformation matrix (camera to world)
        resolution: (width, height) tuple
        depth_buffer: numpy array of depth values (optional, for occlusion testing)
    
    Returns:
        img: PIL Image with labels added
    """
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Try to load a larger, bolder font (increased by 8pts: 48 -> 56)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except Exception as e:
        print(f"Warning: Could not load DejaVuSans-Bold font: {e}")
        print("Falling back to Arial...")
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception as e:
            print(f"Warning: Could not load Arial font: {e}")
            print("Falling back to default font...")
            font = ImageFont.load_default()
    
    # Get camera position and view direction
    camera_pos = camera_transform[:3, 3]
    # Camera forward direction (looks down -Z in camera space, which is +Z column in transform)
    camera_forward = camera_transform[:3, 2]
    
    # Invert camera transform to get view matrix (world to camera space)
    try:
        view_matrix = np.linalg.inv(camera_transform)
    except:
        print("Warning: Could not invert camera transform for labels")
        return img
    
    # Create projection matrix
    fov = np.pi / 3.0  # 60 degrees
    aspect = resolution[0] / resolution[1]
    near = 0.01
    far = 1000.0
    
    f = 1.0 / np.tan(fov / 2.0)
    projection_matrix = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])
    
    # Project part centers to screen coordinates and check visibility
    labels_to_draw = []
    
    for part_id, info in part_info.items():
        center_3d = info['center']
        part_normal = info['normal']
        
        # Calculate view direction from part center to camera
        view_direction = camera_pos - center_3d
        view_direction = view_direction / (np.linalg.norm(view_direction) + 1e-10)
        
        # Check if part is facing the camera (visibility check)
        # Dot product > 0 means part is facing camera
        dot_product = np.dot(part_normal, view_direction)
        if dot_product < 0.1:  # Threshold to avoid grazing angles
            continue  # Part is not visible from this view
        
        # Transform to camera space
        center_homogeneous = np.append(center_3d, 1.0)
        center_camera = view_matrix @ center_homogeneous
        
        # Check if behind camera
        if center_camera[2] >= 0:  # In front of camera (OpenGL convention: camera looks down -Z)
            continue
        
        # Apply projection
        center_clip = projection_matrix @ center_camera
        
        # Perspective divide
        if abs(center_clip[3]) < 1e-6:
            continue
            
        center_ndc = center_clip[:3] / center_clip[3]
        
        # Check if within view frustum
        if abs(center_ndc[0]) > 1.5 or abs(center_ndc[1]) > 1.5:
            continue
        
        # Convert to screen coordinates
        x = int((center_ndc[0] + 1) * resolution[0] / 2)
        y = int((1 - center_ndc[1]) * resolution[1] / 2)
        
        # Check if within image bounds
        if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
            depth = -center_camera[2]  # Store depth for sorting
            
            # Occlusion test using depth buffer if available
            if depth_buffer is not None:
                # Check a small region around the projected point to be more robust
                # (since the part center might not perfectly align with rendered pixels)
                radius = 20  # Check 20-pixel radius
                x_min = max(0, x - radius)
                x_max = min(resolution[0] - 1, x + radius)
                y_min = max(0, y - radius)
                y_max = min(resolution[1] - 1, y + radius)
                
                # Get minimum depth in the region (closest surface to camera)
                region_depth = depth_buffer[y_min:y_max+1, x_min:x_max+1]
                min_rendered_depth = np.min(region_depth)
                
                # If part center is significantly behind the rendered surface, it's occluded
                # Allow small tolerance for numerical precision and explosion displacement
                depth_tolerance = 0.1  # 10% tolerance
                if depth > min_rendered_depth * (1 + depth_tolerance):
                    continue  # Part is occluded, skip labeling
            
            color = info['color']  # Get segment color
            labels_to_draw.append((part_id, x, y, depth, color))
    
    # Sort labels by depth (draw far ones first)
    labels_to_draw.sort(key=lambda l: l[3], reverse=True)
    
    occlusion_status = "with depth-based occlusion testing" if depth_buffer is not None else "without occlusion testing (depth buffer unavailable)"
    if verbose:
        print(f"  Drawing {len(labels_to_draw)} visible labels ({occlusion_status})")
    
    # Draw labels with segment colors
    for part_id, x, y, depth, color in labels_to_draw:
        label_text = str(part_id)
        
        # Convert color to integers
        label_color = tuple(int(c) for c in color[:3])
        
        # Calculate contrasting outline color (black or white)
        brightness = (label_color[0] * 0.299 + label_color[1] * 0.587 + label_color[2] * 0.114)
        outline_color = (0, 0, 0, 255) if brightness > 128 else (255, 255, 255, 255)
        text_color = (0, 0, 0, 255) if brightness > 128 else (255, 255, 255, 255)
        
        # Get text bounding box
        bbox = draw.textbbox((x, y), label_text, font=font, anchor="mm")
        
        # Draw background with segment color (semi-transparent)
        padding = 10
        bg_bbox = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding
        ]
        bg_color = label_color + (200,)  # Add alpha channel
        draw.rectangle(bg_bbox, fill=bg_color, outline=outline_color, width=3)
        
        # Draw label text centered with contrasting color
        draw.text((x, y), label_text, fill=text_color, font=font, anchor="mm")
    
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Explode and render a partitioned mesh with labels"
    )
    parser.add_argument(
        '--mesh_path', 
        type=str, 
        required=True,
        help='Path to the partitioned mesh file (.glb, .ply, etc.)'
    )
    parser.add_argument(
        '--face_ids', 
        type=str, 
        default=None,
        help='Path to face_ids numpy array (optional, will try to auto-detect)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Output directory for rendered images'
    )
    parser.add_argument(
        '--explosion_scale', 
        type=float, 
        default=0.4,
        help='Scale factor for explosion (default: 0.4)'
    )
    parser.add_argument(
        '--resolution', 
        type=int, 
        nargs=2, 
        default=[1920, 1080],
        help='Image resolution as width height (default: 1920 1080)'
    )
    parser.add_argument(
        '--angles', 
        type=str, 
        default='0,30 90,30 180,30 270,30 0,60 0,-30',
        help='Camera angles as "azimuth,elevation" pairs separated by spaces (used with --camera_mode angles)'
    )
    parser.add_argument(
        '--camera_mode',
        type=str,
        choices=['angles', 'uniform', 'standard', 'icosahedron', 'dodecahedron', 'octohedron', 'cube', 'tetrahedron'],
        default='icosahedron',
        help='Camera sampling mode (default: icosahedron)'
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

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output and debugging'
    )
    
    args = parser.parse_args()
    
    # Parse camera angles (only used if camera_mode='angles')
    camera_angles = []
    if args.camera_mode == 'angles':
        for angle_str in args.angles.split():
            azimuth, elevation = map(float, angle_str.split(','))
            camera_angles.append((azimuth, elevation))
    
    print("=" * 70)
    print("Exploded Mesh Renderer with Labels")
    print("=" * 70)
    
    # Load mesh and face IDs
    mesh, face_ids = load_partitioned_mesh(args.mesh_path, args.face_ids, args.verbose)
    
    # Create scene with separate parts
    scene, part_info, label_mapping = create_scene_from_parts(mesh, face_ids, args.verbose)
    

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.mesh_path), 'exploded_renders')
    os.makedirs(args.output_dir, exist_ok=True)
    # Save label mapping to JSON
    mapping_path = os.path.join(args.output_dir, 'label_mapping.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"\nLabel mapping saved to: {mapping_path}")
    if args.verbose:
        print(f"Label mapping: {label_mapping}")

    # Create and save new face_ids array, as well as mask of labels to faces
    label_to_faces = np.zeros((len(label_mapping), len(face_ids)), dtype=np.int32)
    for old_id, new_id in label_mapping.items():
        label_to_faces[new_id, face_ids == old_id] = 1
        face_ids[face_ids == old_id] = new_id
    
    # Save label to faces mapping as compressed numpy archive (efficient for large arrays)
    label_to_faces_path = os.path.join(args.output_dir, 'label_to_faces_mask.npz')
    np.savez_compressed(label_to_faces_path, label_to_faces)
    print(f"Label to faces mapping saved to: {label_to_faces_path}")
    
    np.save(os.path.join(args.output_dir, 'face_ids.npy'), face_ids)
    print(f"New face_ids saved to: {os.path.join(args.output_dir, 'face_ids.npy')}")

    # Copy original mesh
    original_mesh_path = os.path.join(Path(args.mesh_path).parent, 'original_mesh.glb')
    shutil.copy(original_mesh_path, os.path.join(args.output_dir, 'original_mesh.glb'))

    # Copy segmented mesh
    segmented_original_mesh_path = os.path.join(args.output_dir, 'original_segmented_mesh.glb')
    mesh.export(segmented_original_mesh_path)
    print(f"Original mesh saved to: {segmented_original_mesh_path}")
    
    # Explode the scene
    print(f"\nExploding mesh with scale={args.explosion_scale}...")
    exploded_scene = explode_mesh(scene, explosion_scale=args.explosion_scale, verbose=args.verbose)
    
    # Render from multiple angles
    print(f"\nRendering views...")
    render_scene_with_labels(
        exploded_scene, 
        part_info, 
        camera_angles,
        args.output_dir,
        resolution=tuple(args.resolution),
        explosion_scale=args.explosion_scale,
        renderer=args.renderer,
        camera_mode=args.camera_mode,
        num_views=args.num_views,
        verbose=args.verbose
    )
    
    # Also save the exploded mesh
    exploded_mesh_path = os.path.join(args.output_dir, 'exploded_mesh.glb')
    exploded_scene.export(exploded_mesh_path)
    print(f"\nExploded mesh saved to: {exploded_mesh_path}")
    
    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

