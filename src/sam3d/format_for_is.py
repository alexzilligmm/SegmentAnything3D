import torch
import numpy as np
import open3d as o3d
import argparse
import os
from plyfile import PlyData

# takes a ply file with N points and ids and produces a folder that respects the ScanNet format
# classes are oracled for now...
# TODO: make this to hanlde a whole folder of results and not only one scene, for now we assume a single results .ply file is given

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
    results_path = args.results_path
    gt_path = args.gt_path
    output_folder = args.output_folder
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the .ply file
    print("Loading the results from:", results_path)
    results = torch.load(results_path)

    print("obtaining labels from ground truth data:", gt_path)
    labels = get_labels(gt_path, len(results))
    
    print("Obtaining oracle labels for the results...")
    oracle_labels = oracle_classes(labels, results)
    
    print(oracle_labels)
    
    scene_name = os.path.basename(os.path.dirname(gt_path))
    file_name = os.path.basename(gt_path)  

    print(f"Processing scene: {scene_name}, file: {file_name}")



if __name__ == "__main__":
    main()