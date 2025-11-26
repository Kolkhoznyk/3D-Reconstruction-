from pathlib import Path
import open3d as o3d
import numpy as np
import random
from functions import load_camera_poses, build_camera_geometry
print(o3d.__version__)

current_path = Path(__file__).parent
pcd = o3d.io.read_point_cloud(str(current_path /"preprocess"/"clean_fused.ply"),  #try original "/fused.ply" to see difference
                              remove_nan_points=True, remove_infinite_points=True)
if not pcd.has_points():
    raise FileNotFoundError(f"Couldn't load point cloud in {current_path}")


o3d.visualization.draw_geometries([pcd])

cameras = load_camera_poses()
camera_spheres, camera_axes = build_camera_geometry(
    cameras, radius=0.05, axis_length=1.0
)

o3d.visualization.draw_geometries(
    [pcd] + camera_spheres + camera_axes
)

camera = random.choice(cameras)  # Choose a random camera to rotate around
rotation_axis = camera[1]
rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

pcd.translate(rotation_axis * 5.0)
rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
    rotation_axis * (np.pi / 3.0)
)  # Rotate by 60 degrees
camera_sphere, camera_axis = build_camera_geometry(
    [camera], radius=0.5, axis_length=20.0
)
pcd.rotate(rotation_matrix, center=pcd.get_center())
o3d.visualization.draw_geometries(
    [pcd] + camera_sphere + camera_axis
)

#As it is written in the assignment, i assume that the tasks are sequential, so I scale after rotation and translation, 
# without scaling the cameras and vectors.

scale_factor = 5
pcd.scale(scale_factor, center=pcd.get_center())
o3d.visualization.draw_geometries(
    [pcd] + camera_sphere + camera_axis
)

#Pick two points to validate distance. After validation close the window to end the program.

validation_pcd = o3d.geometry.PointCloud()
#make a copy of pcd manually. interesting that o3d does not support a direct copy method
validation_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
if pcd.has_colors():
    validation_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
if pcd.has_normals():
    validation_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))

#Shift + Left Click to pick points!!

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="Pick exactly 2 points")
vis.add_geometry(validation_pcd)
render_opt = vis.get_render_option()
render_opt.point_size = 3.0  # make points easier to click
vis.run()
picked = vis.get_picked_points()
vis.destroy_window()

if not picked or len(picked) != 2:
    count = 0 if not picked else len(picked)
    print(f"You picked {count} points. Please pick exactly 2 points!")
else:
    pts = np.asarray(validation_pcd.points)
    p1, p2 = pts[picked[0]], pts[picked[1]]
    dist = np.linalg.norm(p1 - p2)
    print("\n--- Validation Result ---")
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Distance between picked points: {dist:.3f} meters")
    print("--------------------------------------------")