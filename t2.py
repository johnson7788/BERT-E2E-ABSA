import torch
# x = torch.ones(5, 3)
# print(x)
t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
# index = torch.tensor([0, 4, 2])
# res = x.index_add_(0, index, t)
# print(t)
# print(res)
# print(res.shape)
# print(t.mean(dim=1))

a = torch.tensor([[[0, 1, 2]], [[5, 7, 0]]])
mask = torch.tensor([[[0, 1, 1]], [[1, 1, 0]]])
print(a.shape)
print(a.sum(0).shape)
print(mask.sum(0).shape)
new = a.sum(0)
ma = mask.sum(0)
# print(torch.div(new, ma))
print(ma)
# mask_mean = a.sum(-1) / mask.sum(-1)
# print(mask_mean.shape)
# print(mask_mean)