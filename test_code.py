import torch
a = torch.randn((90000, 512), dtype=torch.half, device='cuda')
import torch.nn.functional as F

values, indices  = torch.topk(a, k=10, dim=1, largest=False, sorted=True)




a = torch.tensor([0.9606, 0.0394], device='cuda:0')

b = torch.tensor([1., 9.], device='cuda:0')




c = F.cross_entropy(b.unsqueeze(0), a.unsqueeze(0), reduction='mean')

a = list([1,2,3,4]) + list([3,4,5,6])

print (a)