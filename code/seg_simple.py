import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Load the point cloud
DATANAME = "../SLAM_output/apartment.ply"
pcd = o3d.io.read_point_cloud(DATANAME)

# Data preprocessing
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

# Statistical outlier removal
nn = 16
std_multiplier = 10
filtered_pcd = pcd.remove_statistical_outlier(nn, std_multiplier)
outliers = pcd.select_by_index(filtered_pcd[1], invert=True)
filtered_pcd = filtered_pcd[0]
# outliers.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw_geometries([filtered_pcd, outliers])

# Voxel downsampling
voxel_size = 0.01
pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size)
# o3d.visualization.draw_geometries([pcd_downsampled])

# Estimate normals
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())
radius_normals = nn_distance * 4
pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), 
                                 fast_normal_computation=True)
# o3d.visualization.draw_geometries([pcd_downsampled, outliers])

# Setting parameters
front = [0.52108284957207607, 0.058076234139224157, -0.85152792961244161]
lookat = [0.68879259921693059, -0.16751796913884687, -0.66025054788714577]
up = [-0.024151069752379704, -0.99627951317011965, -0.082727610066561055]
zoom = 0.69999999999999996
pcd = pcd_downsampled
# o3d.visualization.draw_geometries([pcd],
#     zoom=zoom,
#     front=front,
#     lookat=lookat,
#     up=up)

# 

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=nn_distance*2, min_points=3, print_progress=True))

max_label = labels.max()
print(f"Found {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])