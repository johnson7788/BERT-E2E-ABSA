import torch
x = torch.ones(5, 3)
print(x)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
index = torch.tensor([0, 4, 2])
res = x.index_add_(0, index, t)
print(t)
print(res)
print(res.shape)