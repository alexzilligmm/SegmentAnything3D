# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
from copy import deepcopy
from typing import Any, ItemsView, List, Tuple

import numpy as np
import torch


# Very lightly adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py

logger = logging.getLogger(__name__)


class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, torch.Tensor)
            ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray, torch.Tensor)
        ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    def keys(self):
        return self._stats.keys()

    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.float().detach().cpu().numpy()

    def merge(self, mapping: torch.Tensor) -> None:
        """

        Args:
            mapping (torch.Tensor): A tensor of shape n masks where each element is the corresponding group id.
        """
        device = mapping.device
        cluster_ids = torch.unique(mapping).tolist()

        boxes = self._stats["boxes"]
        masks = self._stats["masks"]
        ious = self._stats["iou_preds"]
        points = self._stats["points"]
        stability_scores = self._stats["stability_score"]

        assert (
            masks.dtype == torch.bool
        ), f"Expected masks dtype=torch.bool, got {masks.dtype}"

        merged_masks = []
        merged_boxes = []
        merged_ious = []
        merged_points = []
        merged_stability_scores = []

        for cid in cluster_ids:
            idxs = (mapping == cid).nonzero(as_tuple=True)[0]
            merged_masks.append(torch.any(masks[idxs], dim=0))
            merged_boxes.append(merge_boxes(boxes[idxs]))
            merged_ious.append(torch.mean(ious[idxs]))
            merged_points.append(torch.mean(points[idxs], dim=0))
            merged_stability_scores.append(torch.mean(stability_scores[idxs]))
        if not (len(merged_masks) == 0):
            self._stats["masks"] = torch.stack(merged_masks, dim=0)
            self._stats["boxes"] = torch.stack(merged_boxes, dim=0)
            self._stats["iou_preds"] = torch.stack(merged_ious, dim=0)
            self._stats["points"] = torch.stack(merged_points, dim=0)
            self._stats["stability_score"] = torch.stack(merged_stability_scores, dim=0)

def merge_crop_boxes(boxes, mapping):
    cluster_ids = torch.unique(mapping).tolist()
    merged_boxes = []
    
    for cid in cluster_ids:
        idxs = (mapping == cid).nonzero(as_tuple=True)[0]
        merged_boxes.append(merge_boxes(boxes[idxs]))
            
    return torch.stack(merged_boxes, dim=0) 


def merge_boxes(boxes: torch.Tensor) -> torch.Tensor:
    assert (
        boxes.ndim == 2 and boxes.size(1) == 4
    ), f"Expected boxes of shape [N,4], got {boxes.shape}"
    x1 = boxes[:, 0].min()
    y1 = boxes[:, 1].min()
    x2 = boxes[:, 2].max()
    y2 = boxes[:, 3].max()
    return torch.tensor([x1, y1, x2, y2], device=boxes.device, dtype=boxes.dtype)

def sample_point_cloud(n_points: int) -> np.ndarray:
    """Samples a 2D point cloud with random points in [0,1]x[0,1]."""
    points = np.random.rand(n_points, 2)
    return points


def build_all_layer_random_cloud(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Sample point clouds for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(sample_point_cloud(n_points))
    return points_by_layer


def generate_crop_efficient(
    im_size: Tuple[int, int], boxes: List[List[int]], overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Expands each box by a fraction (overlap_ratio) of its own size, not the image size.
    """
    im_h, im_w = im_size

    crop_boxes = []
    layer_idxs = []

    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    layer = 0
    for box in boxes:
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        
        if box_w * box_h >= 0.33 * im_w * im_h:
            continue
        
        layer += 1
        
        expand_w = int(math.ceil(overlap_ratio * box_w) / 2)
        expand_h = int(math.ceil(overlap_ratio * box_h) / 2)

        x0e = max(0, x1 - expand_w)
        y0e = max(0, y1 - expand_h)
        x2e = min(im_w, x2 + expand_w)
        y2e = min(im_h, y2 + expand_h)

        crop_boxes.append([x0e, y0e, x2e, y2e])
        layer_idxs.append(layer)
    return crop_boxes, layer_idxs


def get_indices_by_layer(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> List[List[int]]:
    """
    Returns a list of index‐lists, one per layer, into the crop_boxes / prompted_points arrays.

    For example, with n_layers=1 you get something like:
      [[0],       # layer 0 → the full image
       [1,2,3,4]] # layer 1 → the four sub‐crops
    """
    _, layer_idxs = generate_crop_boxes(im_size, n_layers, overlap_ratio)
    if not layer_idxs:
        return []
    max_layer = max(layer_idxs)
    # collect indices for each layer in [0..max_layer]
    return [
        [i for i, l in enumerate(layer_idxs) if l == layer]
        for layer in range(max_layer + 1)
    ]

