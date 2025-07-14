#!/usr/bin/env python
"""
Modified from Relation3D / SparseConvNet ScanNet data prep.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import open3d as o3d
import plyfile
import torch

import sam3d.scannet_util as scannet_util
from segmentator import segment_mesh

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150, dtype=np.int32) * -100
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


def process_scene(fn: str):
    """Process a single scene ply and related files, save .pth"""
    scene_dir = os.path.dirname(fn)
    scene_id = os.path.basename(scene_dir)
    print(f"[{mp.current_process().name}] {scene_id}")

    # Paths
    fn_ply        = fn
    fn_labels     = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.labels.ply")
    fn_segs       = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.0.010000.segs.json")
    fn_agg        = os.path.join(scene_dir, f"{scene_id}.aggregation.json")
    out_pth       = os.path.join(scene_dir, f"{scene_id}_inst_nostuff.pth")

    plydata = plyfile.PlyData.read(open(fn_ply, "rb"))
    pts    = np.vstack([plydata['vertex'][k] for k in ('x','y','z','red','green','blue')]).T
    coords = pts[:, :3] - pts[:, :3].mean(0)
    colors = pts[:, 3:6] / 127.5 - 1

    ply_labels = plyfile.PlyData.read(open(fn_labels, "rb"))
    raw_labels = np.array(ply_labels['vertex']['label'])
    sem_labels = remapper[raw_labels]

    with open(fn_segs) as f:
        segdata = json.load(f)['segIndices']
    segid2pts = {}
    for idx, segid in enumerate(segdata):
        segid2pts.setdefault(segid, []).append(idx)

    inst_segids = []
    with open(fn_agg) as f:
        for grp in json.load(f)['segGroups']:
            label_raw = grp['label']
            cat_name  = scannet_util.g_raw2scannetv2[label_raw]
            if cat_name not in ('wall','floor'):
                inst_segids.append(grp['segments'])

    instance_labels = np.full_like(sem_labels, fill_value=-100, dtype=np.int32)
    for inst_id, segids in enumerate(inst_segids):
        pts_idx = []
        for sid in segids:
            pts_idx.extend(segid2pts[sid])
        instance_labels[pts_idx] = inst_id
        assert np.unique(sem_labels[pts_idx]).size == 1

    mesh      = o3d.io.read_triangle_mesh(fn_ply)
    verts     = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    faces     = torch.from_numpy(np.asarray(mesh.triangles, dtype=np.int64))
    superpt   = segment_mesh(verts, faces).numpy()

    torch.save((coords, colors, superpt, sem_labels, instance_labels), out_pth)
    print(f" → Saved {out_pth}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ScanNet 3D semantic+instance .pth for all scenes"
    )
    parser.add_argument(
        "--data_folder",
        required=True,
        help="Root folder (e.g. scans/) with subfolders sceneXXXX_XX/",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.data_folder, "scene*/scene*_vh_clean_2.ply")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No scenes found in `{args.data_folder}` with pattern `{pattern}`")
        return

    # Multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_scene, files)


if __name__ == "__main__":
    main()