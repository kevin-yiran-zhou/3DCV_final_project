import open3d as o3d
import numpy as np

# Load the point cloud
pcd = o3d.io.read_point_cloud("SLAM_output/apartment.ply")

# Example: Rotate 180 degrees around X-axis to flip upside down
R = pcd.get_rotation_matrix_from_xyz((-np.pi * 0.48, 0, 0))
# R = pcd.get_rotation_matrix_from_xyz((-np.pi * 0.58, 0, 0))
pcd.rotate(R, center=(0, 0, 0))

# Save the rotated point cloud to a new file
o3d.io.write_point_cloud("SLAM_output/apartment_rotated.ply", pcd)

# Visualize
o3d.visualization.draw_geometries([pcd])



# import open3d as o3d

# if __name__ == "__main__":
#     # Load the PLY point cloud file
#     pcd = o3d.io.read_point_cloud("SLAM_output/conference_room_rotated.ply")
    
#     # Print basic information about the point cloud
#     print(pcd)
    
#     # Visualize the point cloud
#     o3d.visualization.draw_geometries([pcd])

