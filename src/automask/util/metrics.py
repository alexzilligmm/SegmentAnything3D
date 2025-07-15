
import torch

import torch.nn.functional as F



def load_cfg(name, overrides=[]):
    from hydra.compose import compose

    return compose(config_name=name, overrides=overrides)


def index_to_coords(index, grid_size):
    row, col = divmod(index, grid_size)
    return col / (grid_size - 1), row / (grid_size - 1)


def compute_pairwise_p(x, y=None, p=2, device="cuda"):
    """
    Compute pairwise p between x's rows and y's rows.
    """
    x = x.to(device).float()
    if y is None:
        y = x
    else:
        y = y.to(device).float()

    return torch.cdist(x, y, p=p)


def compute_pairwise_cosine(
    x: torch.Tensor, y: torch.Tensor = None, device: str = "cuda", eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute pairwise cosine distance between rows of x and rows of y:
        dist(i,j) = 1 - cosine_similarity(x_i, y_j)
    """
    x = x.to(device).float()
    if y is None:
        y = x
    else:
        y = y.to(device).float()

    x_norm = F.normalize(x, p=2, dim=1, eps=eps)  # [N, D]
    y_norm = F.normalize(y, p=2, dim=1, eps=eps)  # [M, D]

    return torch.mm(x_norm, y_norm.t())


def compute_pairwise_intersection(
    x: torch.Tensor, y=None, device="cuda", eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute pairwise intersection between x's rows and y's rows,
    normalized by the min of the two areas of each pair.
    """
    if x.dtype not in (torch.bool, torch.uint8):
        raise TypeError("x must be a boolean tensor")

    if y is None:
        y = x

    P = x.shape[0]
    G = y.shape[0]
    if P == 0 or G == 0:
        return torch.zeros((P, G), device=device, dtype=torch.float32)

    x = x.to(device).float().view(P, -1).contiguous()  # [P, H*W]
    y = y.to(device).float().view(G, -1).contiguous()  # [G, H*W]

    intersection = torch.mm(x, y.t())  # [P, G]
    x_area = x.sum(dim=1, keepdim=True)  # [P, 1]
    y_area = y.sum(dim=1, keepdim=True).t()  # [1, G]

    # mean_area = (x_area + y_area) / 2.0
    # mean_area = mean_area.clamp(min=eps)

    return intersection / torch.min(x_area, y_area).clamp(min=eps)  # [P, G]


def compute_pairwise_iou(
    x: torch.Tensor, y=None, device="cuda", eps: float = 1e-7
) -> torch.Tensor:
    """Compute pairwise IoU between x's rows and y's rows."""
    if x.dtype not in (torch.bool, torch.uint8):
        raise TypeError("x must be a boolean tensor")

    if y is None:
        y = x

    P = x.shape[0]
    G = y.shape[0]
    if P == 0 or G == 0:
        return torch.zeros((P, G), device=device, dtype=torch.float32)

    x = x.to(device).float().view(P, -1).contiguous()  # [P, H*W]
    y = y.to(device).float().view(G, -1).contiguous()  # [G, H*W]

    intersection = torch.mm(x, y.t())  # [P, G]
    area_x = x.sum(dim=1, keepdim=True)  # [P, 1]
    area_y = y.sum(dim=1, keepdim=True).t()  # [1, G]

    union = area_x + area_y - intersection
    union = union.clamp(min=eps)
    iou = intersection / union  # [P, G]

    return iou

