import open3d as o3d
import open3d.ml as ml3d
from open3d.ml.torch.models import RandLANet, PointTransformer, KPFCNN
from open3d.ml.torch.pipelines import SemanticSegmentation
import numpy as np
import matplotlib.pyplot as plt
import torch


# Load and convert point cloud
pcd = o3d.io.read_point_cloud("../SLAM_output/apartment.ply")
points = np.asarray(pcd.points).astype(np.float32)
feats = np.asarray(pcd.colors).astype(np.float32) if pcd.has_colors() else np.ones((points.shape[0], 1), dtype=np.float32)

# Dummy label to satisfy run_inference()
labels = np.zeros((points.shape[0],), dtype=np.int32)
data = {
    "point": points,
    "feat": feats,
    "label": labels
}

# Load config
ymls = [
    "kpconv_s3dis.yml",
    "pointtransformer_s3dis.yml",
    "randlanet_s3dis.yml"
] 
pths = [
    "kpconv_s3dis_202010091238.pth",
    "pointtransformer_s3dis_2021092`41350utc.pth",
    "randlanet_s3dis_202201071330utc.pth"
]
ModelClasses = [KPFCNN, PointTransformer, RandLANet]
model_path = "/home/kevin-zhou/Desktop/UMich/Winter2025/3DCV_final_project/model/"
which = 2
cfg = ml3d.utils.Config.load_from_file(model_path + ymls[which])
cfg.model['in_channels'] = 6
cfg.model['ckpt_path'] = model_path + pths[which]

# Load model
# model = RandLANet(**cfg.model)
# model = PointTransformer(**cfg.model)
model = ModelClasses[which](**cfg.model)
device = 'cuda' if o3d.core.cuda.is_available() else 'cpu'
pipeline = SemanticSegmentation(model, dataset=None, device=device)
print("Model weight mean:", next(model.parameters()).mean().item())

# Run inference
print("Running segmentation...")
data['point'] = torch.from_numpy(data['point'])
data['feat'] = torch.from_numpy(data['feat'])
data['label'] = torch.from_numpy(data['label'])
result = pipeline.run_inference(data)

# Output
pred_labels = result['predict_labels']
unique, counts = np.unique(pred_labels, return_counts=True)
print("Predicted class distribution:")
s3dis_classes = {
    0: "ceiling", 1: "floor", 2: "wall", 3: "beam", 4: "column",
    5: "window", 6: "door", 7: "table", 8: "chair", 9: "sofa",
    10: "bookcase", 11: "board", 12: "clutter"
}
for cls, count in zip(unique, counts):
    print(f"Class {cls} ({s3dis_classes.get(cls, 'unknown')}): {count} points")

num_classes = cfg.model['num_classes']
cmap = plt.get_cmap("tab20", num_classes)  # or use "gist_ncar", "viridis", etc.
colors = cmap(np.arange(num_classes))[:, :3]  # only RGB, discard alpha
pcd.colors = o3d.utility.Vector3dVector(colors[pred_labels])

# Visualize
o3d.visualization.draw_geometries([pcd])