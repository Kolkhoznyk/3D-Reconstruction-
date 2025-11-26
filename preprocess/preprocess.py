from pathlib import Path
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

#This is my fight with noise and outliers. Due to not optimal lighting conditions during capture (see original capture in this folder),
# defocus sometimes and not top quality, the point cloud is somewhat noisy. This script attempts to clean the point cloud
# by removing statistical outliers, fitting and removing planes (like ground and walls) and clustering

current_path = Path(__file__).parent
pcd = o3d.io.read_point_cloud(str(current_path / "fused.ply"),
                              remove_nan_points=True, remove_infinite_points=True)
if not pcd.has_points():
    raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))


# Clean the point cloud using statistical outlier removal
pcd, ind = pcd.remove_statistical_outlier(
    nb_neighbors=50,
    std_ratio=1.0
)
o3d.visualization.draw_geometries([pcd])

#Fit plane to the point cloud and remove the plane points like ground or walls
# Hyperparameter's values increased
plane_model, inlier_indices = pcd.segment_plane(distance_threshold=0.2,
                                                        ransac_n=4,
                                                        num_iterations=10000)

best_inliers = np.full(shape=len(pcd.points, ), fill_value=False, dtype=bool)
best_inliers[inlier_indices] = True

scene_pcd = pcd.select_by_index(inlier_indices, invert=True)

scene_pcd, ind = scene_pcd.remove_statistical_outlier(
    nb_neighbors=50,
    std_ratio=1.0
)

points = np.asarray(scene_pcd.points, dtype=np.float32)
colors = np.asarray(scene_pcd.colors, dtype=np.float32)

# Cluster with DBSCAN and filter out small clusters
#This did not work well for my data, but might be useful for others
labels = DBSCAN(eps=0.1,
                        min_samples= 15).fit_predict(points)


unique_labels, counts = np.unique(labels, return_counts=True)
min_cluster_size = 200
valid_clusters = unique_labels[counts >= min_cluster_size]

mask = np.isin(labels, valid_clusters)

filtered_points = points[mask]
filtered_colors = colors[mask]

# Create a new point cloud
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors) 

# Save the cleaned point cloud
output_dir = current_path
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "clean_fused.ply"
success = o3d.io.write_point_cloud(str(output_file), scene_pcd)
if not success:
    raise IOError(f"Failed to write point cloud to {output_file}")
else:
    print(f"Saved cleaned point cloud to {output_file} ({len(filtered_pcd.points)} points)")

