import glob
import torch
import numpy as np
import open3d as o3d
import argparse
import os
from plyfile import PlyData
import tqdm


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def get_labels(labels_path, points):
    plydata = PlyData.read(open(labels_path, "rb"))
    label_field = None
    for field in ['label', 'scalar_Label']:
        if field in plydata['vertex'].data.dtype.names:
            label_field = field
            break
    if label_field is None:
        raise KeyError("No valid field into .labels.ply")
    
    labels = np.array(plydata['vertex'][label_field])
    assert len(labels) == points, f"{len(labels)} labels vs {points} points"
    return labels
    
def oracle_classes(labels, results):
    ids = np.unique(results)
    oracle_labels = {}
    for id in ids:
        index = np.where(results == id)[0]
        unique, counts = np.unique(labels[index], return_counts=True)
        oracle_label = unique[np.argmax(counts)]
        oracle_labels[id] = oracle_label
    return oracle_labels


def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--results_path', type=str, required=True, help='Path to the results directory')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the ground truth data')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save output files')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    results_root = args.results_path
    gt_root = args.gt_path
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)
    mask_dir = os.path.join(output_folder, "predicted_masks")
    os.makedirs(mask_dir, exist_ok=True)

    all_results = sorted(glob.glob(os.path.join(results_root, "scene*.pth")))
    
    for res_path in tqdm.tqdm(all_results, desc="Processing results"):
        results = torch.load(res_path)
        scene_name = os.path.basename(res_path).split('.')[0]
        scene_id = scene_name[:12]
        gt_file = os.path.join(gt_root, scene_name, f"{scene_name}_vh_clean_2.labels.ply")
        predictions_txt_path = os.path.join(output_folder, f"{scene_name}.txt")

        if not os.path.exists(gt_file):
            print(f"[!] Skipping {scene_id}, GT not found.")
            print(f"Expected GT file: {gt_file}")
            continue
        with open(predictions_txt_path, 'w') as pred_file:
            print(f"Processing scene: {scene_id}")
            labels = get_labels(gt_file, len(results))
            oracle_labels = oracle_classes(labels, results)

            for idx, (id, label) in enumerate(oracle_labels.items()):
                mask_rel_path = f"predicted_masks/{scene_id}_{str(idx).zfill(3)}.txt"
                mask_abs_path = os.path.join(output_folder, mask_rel_path)

                with open(mask_abs_path, 'w') as mask_f:
                    mask_f.writelines(f"{int(b)}\n" for b in results == id)

                pred_file.write(f"{mask_rel_path} {label} 1.0\n")

    print(f"\nDone. Output written to `{output_folder}`.")


if __name__ == "__main__":
    main()