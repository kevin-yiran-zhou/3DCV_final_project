import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Load the point cloud
DATANAME = "../SLAM_output/apartment.ply"
pcd = o3d.io.read_point_cloud(DATANAME)

# Data preprocessing
pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])