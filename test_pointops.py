import torch
import pointops

a = torch.rand((1, 1024, 3), device='cuda')
b = torch.rand((1, 1024, 3), device='cuda')
offset = torch.tensor([0, 1024], device='cuda', dtype=torch.int32)

idx, dist2 = pointops.knn_query(1, a, offset, b, offset)
print("KNN query output:", idx.shape, dist2.shape)