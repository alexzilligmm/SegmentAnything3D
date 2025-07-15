import argparse
import glob
import os
import numpy as np
import torch

semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
semantic_label_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
    'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'
]

def main():
    parser = argparse.ArgumentParser(description="Generate .txt instance groundtruth from .pth files")
    parser.add_argument("--scans_dir", required=True, help="Directory with scene folders (e.g. scans/sceneXXXX_XX/)")
    parser.add_argument("--output_dir", required=True, help="Directory to store the .txt output files")
    args = parser.parse_args()

    scene_paths = sorted(glob.glob(os.path.join(args.scans_dir, "scene*/scene*_inst_nostuff.pth")))
    if not scene_paths:
        print(f"No .pth files found in `{args.scans_dir}`.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for i, pth_file in enumerate(scene_paths):
        data = torch.load(pth_file)
        xyz, rgb, superpoint, label, instance_label = data

        scene_name = os.path.basename(pth_file)[:12]
        print(f"[{i+1}/{len(scene_paths)}] Processing {scene_name}")

        instance_label_new = np.zeros(instance_label.shape, dtype=np.int32)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            if instance_mask.size == 0:
                continue
            sem_id = int(label[instance_mask[0]])
            if sem_id == -100:
                sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

        output_path = os.path.join(args.output_dir, f"{scene_name}.txt")
        np.savetxt(output_path, instance_label_new, fmt="%d")

    print(f"\nDone. {len(scene_paths)} instance maps saved to `{args.output_dir}`.")

if __name__ == "__main__":
    main()