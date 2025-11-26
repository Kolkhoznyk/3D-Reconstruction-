from pathlib import Path
import open3d as o3d
import numpy as np
import random
from functions import load_camera_poses, build_camera_geometry

current_path = Path(__file__).parent
pcd = o3d.io.read_point_cloud(str(current_path /"preprocess"/"clean_fused.ply"),    #try original "/fused.ply" to see difference
                              remove_nan_points=True, remove_infinite_points=True)
if not pcd.has_points():
    raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))


o3d.visualization.draw_geometries([pcd])

cameras = load_camera_poses()
camera_spheres, camera_axes = build_camera_geometry(cameras, radius=0.05, axis_length=1.0)

o3d.visualization.draw_geometries(
    [pcd] + camera_spheres + camera_axes
)

camera = random.choice(cameras) #choose a rancom vector to rotate around
rotation_axis = camera[1]
rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

pcd.translate(rotation_axis * 5.0)
rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * np.pi /3) # Rotate by pi/3 radians (60 degrees)
camera_sphere, camera_axis = build_camera_geometry([camera], radius=0.5, axis_length=20.0)
pcd.rotate(rotation_matrix, center=pcd.get_center())
o3d.visualization.draw_geometries(
    [pcd] + camera_sphere + camera_axis
)