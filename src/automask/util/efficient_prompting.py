import copy
import math
import numpy as np
import torch
from automask.util import (
    compute_pairwise_p,
    index_to_coords,
    compute_pairwise_cosine,
    compute_pairwise_iou,
)


def fps_from_median(coords: np.ndarray, k: int) -> list:
    """
    Farthest Point Sampling starting from the most central point (median).
    coords: np.ndarray of shape (N, 2)
    k: number of points to select
    Returns: list of indices in coords
    """
    # Step 1: Find the median point (minimum total distance to others)
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    median_idx = np.argmin(dist_matrix.sum(axis=1))

    selected = [median_idx]
    distances = np.full(len(coords), np.inf)

    for _ in range(1, k):
        last_selected = coords[selected[-1]]
        dist_to_last = np.linalg.norm(coords - last_selected, axis=1)
        distances = np.minimum(distances, dist_to_last)
        next_idx = np.argmax(distances)
        selected.append(next_idx)

    return selected


def get_centres(
    cluster_labels, num_tokens, attn_scores, method="mean", points_per_cluster=1
):
    """
    Compute cluster centers either by:
    - "mean": geometric center of the cluster in (x, y) space
    - "feature-space-argmax": most attended token per cluster based on attention scores
    """
    unique_labels = set(cluster_labels)
    unique_labels.discard(
        -1
    )  # TODO: check if we want the noise to be masked, maybe not, it can be useful for stuff class?

    num_patches = int(math.sqrt(num_tokens))
    cluster_centers = []

    for cluster_id in unique_labels:
        indices = np.where(cluster_labels == cluster_id)[0]

        if len(indices) == 0:
            cluster_centers.append(np.array([0, 0]))
            print(
                "[WARNING] Somehow a cluster has zero points... something has gone wrong."
            )
            continue

        if method == "mean":
            coords = np.array([index_to_coords(i, num_patches) for i in indices])
            cluster_centers.append(coords.mean(axis=0))
        elif method == "median":
            coords = np.array([index_to_coords(i, num_patches) for i in indices])
            dists = np.sum(
                np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2), axis=1
            )
            central_idx = np.argmin(dists)
            cluster_centers.append(coords[central_idx])
        elif method == "fps":
            coords = np.array([index_to_coords(i, num_patches) for i in indices])
            selected_indices = fps_from_median(coords, points_per_cluster)
            for idx in selected_indices:
                coord = coords[idx]
                cluster_centers.append(coord)
        elif method == "fmax":
            cluster_attn = attn_scores[:, indices].sum(axis=0)
            max_index = indices[torch.argmax(cluster_attn)]
            coord = index_to_coords(max_index.item(), num_patches)
            cluster_centers.append(coord)
        elif method == "fkmax":
            cluster_attn = attn_scores[:, indices].sum(axis=0)
            _, topk_pos = torch.topk(cluster_attn, k=points_per_cluster, dim=0)
            topk_indices = indices[topk_pos.cpu().numpy()]
            for idx in topk_indices:
                coord = index_to_coords(idx.item(), num_patches)
                cluster_centers.append(coord)

        else:
            raise ValueError(f"Unknown method '{method}' for get_centres")

    return np.array(cluster_centers)


def compute_distance_matrix(attn_scores, metric, device=None):
    """
    Compute distance matrix based on the specified metric.
    """
    if metric == "iou":
        binary_attn = (attn_scores > 0.0001).to(torch.uint8)
        return 1 - compute_pairwise_iou(binary_attn, device=device)
    elif metric == "cosine":
        return 1 - compute_pairwise_cosine(attn_scores, device=device)
    elif metric == "euclidean":
        return compute_pairwise_p(attn_scores, p=2, device=device)
    else:
        raise ValueError(f"Unknown metric '{metric}' for compute_distance_matrix")


def build_layer_efficient_prompt(
    cluster_alg,
    attn_scores,
    method="mean",
    metric="euclidean",
    points_per_cluster=1,
    device=None,
) -> np.ndarray:
    assert method in ["mean", "fmax", "fkmax", "median", "fps"]
    assert metric in ["iou", "cosine", "euclidean"]

    _, num_tokens = attn_scores.shape

    distance_matrix = compute_distance_matrix(attn_scores, metric, device=device)

    cluster_labels = cluster_alg.fit_predict(
        distance_matrix.cpu().numpy().astype(np.float64)
    )

    centres = get_centres(
        cluster_labels,
        num_tokens,
        attn_scores,
        method=method,
        points_per_cluster=points_per_cluster,
    )

    return centres, cluster_labels  # [K, 2]


def build_clusters(
    cluster_alg, attn_scores, metric="euclidean", device=None
) -> np.ndarray:
    assert metric in ["iou", "cosine", "euclidean"]

    distance_matrix = compute_distance_matrix(attn_scores, metric, device=device)

    cluster_labels = cluster_alg.fit_predict(
        distance_matrix.cpu().numpy().astype(np.float64)
    )

    return cluster_labels  # [K, 2]


import torch.nn.functional as F


def _cluster_labels_to_low_res_masks(
    cluster_labels, num_patches, output_hw, softmax=True
):
    unique = np.unique(cluster_labels)

    cluster_labels = cluster_labels.reshape(num_patches, num_patches)

    out_scale, out_bias = 20.0, -10.0

    masks = []
    for i in unique:
        if i == -1:
            continue
        mask = (cluster_labels == i).astype(np.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        if softmax:
            logit_mask = mask_tensor * out_scale + out_bias
        else:
            logit_mask = mask_tensor

        interp_mask = F.interpolate(
            logit_mask[None, None, ...],
            size=output_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        masks.append(interp_mask)

    return torch.stack(masks)


def patches_labels_to_masks(
    cluster_labels, img_size, num_patches, device="cuda", softmax=True
):
    low_res_masks = _cluster_labels_to_low_res_masks(
        cluster_labels, num_patches, img_size, softmax=softmax
    )
    return low_res_masks.squeeze(1).to(device)


def get_cluster_alg(base_cluster_alg, crop_size, original_size):
    ''' Computes the clustering algorithm to use based on the current crop size wrt to the original image. '''
    cluster_alg = copy.deepcopy(base_cluster_alg)
    print(
        f"[INFO] Base clustering algorithm parameters: min_samples={cluster_alg.min_samples}, min_cluster_size={cluster_alg.min_cluster_size}"
    )
    crop_h, crop_w = crop_size
    im_h, im_w = original_size
    crop_area = crop_h * crop_w
    image_area = im_h * im_w
    # TODO: tune these parameters maybe introduce some alpha and betas?
    if hasattr(cluster_alg, "min_samples") and image_area > crop_area:
        cluster_alg.min_samples = int(cluster_alg.min_samples * image_area / (2 * crop_area))
    if hasattr(cluster_alg, "min_cluster_size") and image_area > crop_area:
        cluster_alg.min_cluster_size = int(
            cluster_alg.min_cluster_size * image_area / (4 * crop_area)
        )
    print(
        f"[INFO] Updated clustering algorithm parameters: min_samples={cluster_alg.min_samples}, min_cluster_size={cluster_alg.min_cluster_size}"
    )
    return cluster_alg
