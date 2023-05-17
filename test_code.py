import torch
a = torch.randn((90000, 512), dtype=torch.half, device='cuda')
import torch.nn.functional as F

values, indices  = torch.topk(a, k=10, dim=1, largest=False, sorted=True)




print (indices)