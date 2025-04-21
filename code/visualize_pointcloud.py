import open3d as o3d

DATANAME = "../SLAM_output/example.ply"
pcd = o3d.io.read_point_cloud(DATANAME)
o3d.visualization.draw_geometries([pcd])