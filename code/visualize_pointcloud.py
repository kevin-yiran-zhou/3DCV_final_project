import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# === Load and downsample ===
name = "conference_room"
pcd = o3d.io.read_point_cloud(f"../SLAM_output/{name}.ply")
pcd = pcd.voxel_down_sample(voxel_size=0.05)

# Convert to numpy
pts = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
print(f"Number of points: {pts.shape[0]}")

# === Function to set equal aspect ratio ===
def set_axes_equal(ax):
    limits = np.array([
        [np.min(pts[:, 0]), np.max(pts[:, 0])],
        [np.min(pts[:, 1]), np.max(pts[:, 1])],
        [np.min(pts[:, 2]), np.max(pts[:, 2])]
    ])
    centers = np.mean(limits, axis=1)
    spans = np.ptp(limits, axis=1)
    max_span = max(spans)

    for center, axis in zip(centers, [ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        axis(center - max_span / 2, center + max_span / 2)

# === Plot in 3D ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Interactive 3D View of Point Cloud")
set_axes_equal(ax)
plt.tight_layout()
plt.show()