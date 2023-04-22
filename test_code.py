from collections import Counter


a  = [2,2,3,4,23,1234,1232,3,4]


x = Counter(a)
#print (x)
top_1 = x.most_common(2)


import torch



A = torch.tensor([[1.0, 2, 3, 4], [2,3,4,-5]])

B = torch.tensor([[2,3,22,-5]])

C = [A, B]

print (torch.concatenate(C))


indices = torch.tensor([[1], [0]])#.repeat(1, 4)
ss = torch.zeros(A.size())
ss = torch.gather(A, 0, indices)



