import torch
import numpy as np
import open3d as o3d
import argparse
import os
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def visualize_segmented_scene(coord_file, group_file, scene_name):
    data = torch.load(coord_file)
    coords = torch.tensor(data["coord"]).numpy()

    if isinstance(group_file, str):
        group = torch.load(group_file)
    else:
        group = data["group"].numpy()

    max_group = group.max()
    cmap = get_cmap('tab20', max_group + 1)  
    vis_colors = cmap(group)[:, :3]  

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(vis_colors.astype(np.float32))

    save_path = f"{scene_name}_segmented.ply"
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved point cloud to: {save_path}")

    label_path = f"{scene_name}_labels.npy"
    np.save(label_path, group)
    print(f"Saved labels to: {label_path}")

    try:
        o3d.visualization.draw_geometries([pcd])
    except:
        print("Unable to open OpenGL window (likely headless server).")

    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], c=vis_colors, s=0.5)
    plt.axis('equal')
    plt.title("2D projection colored by label")
    plt.savefig(f"{scene_name}_projection.png", dpi=300)
    print(f"Saved 2D projection to: {scene_name}_projection.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', type=str, required=True, help='Scene name without extension')
    parser.add_argument('--save_path', type=str, required=True, help='Path where final labels are saved')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .pth file with coord/color')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train')
    args = parser.parse_args()

    group_file = join(args.save_path, args.scene_name + '.pth')
    coord_file = join(args.data_path, args.split, args.scene_name + '.pth')

    visualize_segmented_scene(coord_file, group_file, args.scene_name)