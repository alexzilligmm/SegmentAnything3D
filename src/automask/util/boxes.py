import torch
from torchvision.ops.boxes import distance_box_iou
import torchvision

from automask.util import compute_pairwise_intersection

def bm(boxes: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    N = boxes.size(0)
    if N == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    iou = distance_box_iou(boxes, boxes)

    mapping = union_find_clusters(iou, iou_threshold)

    return mapping


def mm(masks: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    N = masks.size(0)
    if N == 0:
        return torch.empty((0,), dtype=torch.long, device=masks.device)

    # sort masks by scores
    _, idxs = scores.sort(descending=True)
    masks_sorted = masks[idxs]

    iou = compute_pairwise_intersection(masks_sorted, device=masks.device)

    mapping_sorted = union_find_clusters(iou, iou_threshold)

    # {mask -> cluster num}
    inv_idxs = torch.empty_like(idxs)
    inv_idxs[idxs] = torch.arange(N, device=masks.device)
    mapping = mapping_sorted[inv_idxs]

    return mapping


def union_find_clusters(distance_matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Generic union-find clustering based on a distance matrix and threshold.
    Returns a mapping from each item to its cluster index.
    """
    N = distance_matrix.size(0)
    if N == 0:
        return torch.empty((0,), dtype=torch.long, device=distance_matrix.device)

    iu0, iu1 = torch.triu_indices(N, N, offset=1, device=distance_matrix.device)
    mask = distance_matrix[iu0, iu1] > threshold
    edges = list(zip(iu0[mask].tolist(), iu1[mask].tolist()))

    parent = list(range(N))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(i, j)

    mapping = torch.tensor([find(i) for i in range(N)], dtype=torch.long, device=distance_matrix.device)
    _, inverse = torch.unique(mapping, return_inverse=True)
    return inverse


def batched_mm(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Performs boxes merging in a batched fashion.

    Each index value correspond to a category, and BM
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where BM will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been processed by BM, sorted
        in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    if (
        boxes.numel() > (4000 if boxes.device.type == "cpu" else 20000)
        and not torchvision._is_tracing()
    ):
        return _batched_mm_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_mm_coordinate_trick(boxes, scores, idxs, iou_threshold)


@torch.jit._script_if_tracing
def _batched_mm_coordinate_trick(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    # strategy: in order to perform BM independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_mm = boxes + offsets[:, None]
    keep = mm(boxes_for_mm, scores, iou_threshold)
    return keep


@torch.jit._script_if_tracing
def _batched_mm_vanilla(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    # Based on Detectron2 implementation, just manually call BM() on each class independently
    mapping = torch.empty(boxes.shape[0], dtype=torch.long, device=boxes.device)
    offset = 0
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_merge = mm(boxes[curr_indices], scores[curr_indices], iou_threshold)
        mapping[curr_indices] = curr_merge + offset
        offset += int(curr_merge.max().item()) + 1
    return mapping


def batched_bm( boxes: torch.Tensor, idxs: torch.Tensor, iou_threshold: float,):
    """
    Performs boxes merging in a batched fashion.

    Each index value correspond to a category, and BM
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where BM will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been processed by BM, sorted
        in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    if (
        boxes.numel() > (4000 if boxes.device.type == "cpu" else 20000)
        and not torchvision._is_tracing()
    ):
        return _batched_bm_vanilla(boxes, idxs, iou_threshold)
    else:
        return _batched_bm_coordinate_trick(boxes, idxs, iou_threshold)
    
    
@torch.jit._script_if_tracing
def _batched_bm_coordinate_trick(
    boxes: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    # strategy: in order to perform BM independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_mm = boxes + offsets[:, None]
    keep = bm(boxes_for_mm, iou_threshold)
    return keep


@torch.jit._script_if_tracing
def _batched_bm_vanilla(
    boxes: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    # Based on Detectron2 implementation, just manually call BM() on each class independently
    mapping = torch.empty(boxes.shape[0], dtype=torch.long, device=boxes.device)
    offset = 0
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_merge = bm(boxes[curr_indices], torch.ones_like(curr_indices, dtype=torch.float32), iou_threshold)
        mapping[curr_indices] = curr_merge + offset
        offset += int(curr_merge.max().item()) + 1
    return mapping