import tkinter as tk
from tkinter import filedialog, messagebox
import open3d as o3d
import numpy as np
import torch
from open3d.ml.torch.models import RandLANet, PointTransformer, KPFCNN
from open3d.ml.torch.pipelines import SemanticSegmentation
import open3d.ml as ml3d
import matplotlib.pyplot as plt
import os
import threading

ModelClasses = [KPFCNN, PointTransformer, RandLANet]
model_keywords = ["kpconv", "pointtransformer", "randlanet"]

LARGE_FONT = ("Arial", 26)
TITLE_FONT = ("Arial", 26, "bold")
MONO_FONT = ("Courier", 18)

class SegmentApp:
    def __init__(self, master):
        self.master = master
        master.title("3D Semantic Segmentation Tool")

        self.pcd_path = tk.StringVar()
        self.model_index = tk.IntVar()

        # === Step 1 ===
        tk.Label(master, text="Step 1: Load Point Cloud", font=TITLE_FONT).pack(pady=(30, 10))
        tk.Button(master, text="Select Point Cloud (.ply)", command=self.load_pcd, font=LARGE_FONT, height=2, width=30).pack(pady=10)
        self.pcd_label = tk.Label(master, text="No point cloud selected", font=LARGE_FONT, fg="gray")
        self.pcd_label.pack(pady=10)
        tk.Button(master, text="Visualize Original Point Cloud", command=self.visualize_pcd,
                  font=LARGE_FONT, height=1, bg="#2196F3", fg="white", width=30).pack(pady=10)

        # === Step 2 ===
        tk.Label(master, text="Step 2: Choose Model", font=TITLE_FONT).pack(pady=(100, 50))
        for i, name in enumerate(["KPFCNN", "PointTransformer (not available now)", "RandLANet"]):
            tk.Radiobutton(master, text=name, variable=self.model_index, value=i,
                           font=LARGE_FONT, anchor="center").pack(anchor="center", pady=5)

        # === Step 3 ===
        tk.Label(master, text="Step 3: Run Segmentation", font=TITLE_FONT).pack(pady=(100, 50))
        tk.Button(master, text="Run Segmentation", command=self.run_segmentation,
                  font=LARGE_FONT, bg="#4CAF50", fg="white", height=2, width=30).pack(pady=10)

        self.status_label = tk.Label(master, text="", font=LARGE_FONT, fg="green")
        self.status_label.pack(pady=10)

        self.result_label = tk.Label(master, text="", font=LARGE_FONT, fg="green")
        self.result_label.pack(pady=10, padx=20)

    def load_pcd(self):
        default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../SLAM_output"))
        path = filedialog.askopenfilename(
            initialdir=default_dir,
            filetypes=[("Point Cloud", "*.ply")],
            title="Select a Point Cloud File"
        )
        if path:
            self.pcd_path.set(path)
            filename = os.path.basename(path)
            self.pcd_label.config(text=f"Selected point cloud: {filename}", fg="black")

    def preprocess_point_cloud(self, pcd):
        # === Downsample ===
        voxel_size = 0.02
        pcd = pcd.voxel_down_sample(voxel_size)

        # === Optional: smooth to reduce jagged edges ===
        pcd = self.smooth_point_cloud(pcd, radius=0.05)

        # === Normals ===
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)
        pcd.normalize_normals()

        # === Remove outliers ===
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # === Normalize to origin ===
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        points -= centroid
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd

    def smooth_point_cloud(self, pcd, radius=0.05):
        """
        简单的点云平滑操作：对每个点的邻域点取平均位置
        """
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        smoothed_points = []

        for i in range(len(points)):
            _, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
            if len(idx) > 3:
                neighbor_pts = points[idx]
                smoothed_points.append(np.mean(neighbor_pts, axis=0))
            else:
                smoothed_points.append(points[i])  # 太孤立不动它

        pcd.points = o3d.utility.Vector3dVector(np.array(smoothed_points))
        return pcd

    def visualize_pcd(self):
        try:
            path = self.pcd_path.get()
            if not path or not os.path.isfile(path):
                raise FileNotFoundError("No valid point cloud file selected.")
            pcd = o3d.io.read_point_cloud(path)
            o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def find_model_files(self, model_keyword):
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))
        yml_file = None
        pth_file = None
        for f in os.listdir(model_dir):
            if model_keyword in f:
                if f.endswith(".yml"):
                    yml_file = os.path.join(model_dir, f)
                elif f.endswith(".pth"):
                    pth_file = os.path.join(model_dir, f)
        if not yml_file or not pth_file:
            raise FileNotFoundError(f"Model files for '{model_keyword}' not found in {model_dir}")
        return yml_file, pth_file

    def run_segmentation(self):
        def task():
            try:
                self.status_label.config(text="Running...", fg="orange")
                for widget in self.result_label.winfo_children():
                    widget.destroy()
                self.master.update_idletasks()

                model_keyword = model_keywords[self.model_index.get()]
                yml_path, pth_path = self.find_model_files(model_keyword)

                pcd = o3d.io.read_point_cloud(self.pcd_path.get())
                pcd = self.preprocess_point_cloud(pcd)

                points = np.asarray(pcd.points).astype(np.float32)
                feats = np.asarray(pcd.colors).astype(np.float32) if pcd.has_colors() else np.ones((points.shape[0], 1), dtype=np.float32)

                data = {
                    "point": torch.from_numpy(points),
                    "feat": torch.from_numpy(feats),
                    "label": torch.zeros((points.shape[0],), dtype=torch.int32)
                }

                cfg = ml3d.utils.Config.load_from_file(yml_path)
                cfg.model['in_channels'] = 6
                cfg.model['ckpt_path'] = pth_path
                model = ModelClasses[self.model_index.get()](**cfg.model)

                device = 'cuda' if o3d.core.cuda.is_available() else 'cpu'
                pipeline = SemanticSegmentation(model, dataset=None, device=device)
                result = pipeline.run_inference(data)
                pred_labels = result['predict_labels']

                num_classes = cfg.model['num_classes']
                cmap = plt.get_cmap("tab20", num_classes)
                colors = cmap(np.arange(num_classes))[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(colors[pred_labels])

                # Update UI before showing visualization
                self.status_label.config(text="Done!", fg="green")

                unique, counts = np.unique(pred_labels, return_counts=True)
                s3dis_classes = {
                    0: "ceiling", 1: "floor", 2: "wall", 3: "beam", 4: "column",
                    5: "window", 6: "door", 7: "table", 8: "chair", 9: "sofa",
                    10: "bookcase", 11: "board", 12: "clutter"
                }
                # Clear previous result labels (if any)
                for widget in self.result_label.winfo_children():
                    widget.destroy()

                tk.Label(self.result_label, text="Predicted class distribution:", font=MONO_FONT).pack(pady=(0, 10))

                for cls, count in zip(unique, counts):
                    rgb = cmap(cls)[:3]  # Ignore alpha
                    rgb = tuple(int(255 * c) for c in rgb)
                    hex_color = '#%02x%02x%02x' % rgb
                    text = f"Class {cls:2d} ({s3dis_classes[cls]}): {count:5d} points"
                    tk.Label(self.result_label, text=text, font=MONO_FONT, fg=hex_color).pack()

                # Show the Open3D window in a separate thread
                threading.Thread(target=lambda: o3d.visualization.draw_geometries([pcd])).start()

            except Exception as e:
                self.status_label.config(text="Error occurred", fg="red")
                self.result_label.config(text="")
                messagebox.showerror("Error", str(e))

        threading.Thread(target=task).start()



if __name__ == "__main__":
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.state("normal")

    app = SegmentApp(root)
    root.mainloop()