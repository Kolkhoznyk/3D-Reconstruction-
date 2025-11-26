
from pathlib import Path

import open3d as o3d
import numpy as np  

def load_camera_poses() -> list[tuple[np.ndarray, np.ndarray]]:
    current_path = Path(__file__).parent
    cameras = []
    with open(str(current_path / "params" / "frames.txt"), "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            elems = line.split()
            _ = int(elems[0])
            _ = int(elems[1])
            qvec = np.array([float(elems[2]), float(elems[3]),
                            float(elems[4]), float(elems[5])])
            tvec = np.array([float(elems[6]), float(elems[7]), float(elems[8])])
            
            R = o3d.geometry.get_rotation_matrix_from_quaternion(qvec)
            C = -R.T @ tvec                  # camera center in world coords
            forward = R.T @ np.array([0,0,1])  # optical axis direction

            cameras.append((C, forward))
    return cameras

def build_camera_geometry(cameras: list[tuple[np.ndarray, np.ndarray]], radius: float =0.05,
                          axis_length: float = 1.0) -> tuple[list[o3d.geometry.TriangleMesh], 
                                                                    list[o3d.geometry.LineSet]]:
    camera_spheres = []
    camera_axes = []

    for C, forward in cameras:
        # Camera center marker (small sphere)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color([1, 0, 0])   # red
        sphere.translate(C)
        camera_spheres.append(sphere)
        # Optical axis (line)
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([C, C + forward * axis_length])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # green
        camera_axes.append(line)
    return camera_spheres, camera_axes

