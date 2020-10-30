import torch

# a = torch.tensor([[[0, 1, 2]], [[5, 7, 0]]])
# mask = torch.tensor([[[0, 1, 1]], [[1, 1, 0]]])

# print(mask[:,0])
batch_res = []
for idx,i in enumerate(a):
    # print(mask[idx][0])
    active = mask[idx][0] ==1
    # print(active)
    print(i[0][active])
print(a.shape)
# print(a.sum(0).shape)
# print(mask.sum(0).shape)
# new = a.sum(0)
# ma = mask.sum(0)
# print(torch.div(new, ma))
# print(ma)
# mask_mean = a.sum(-1) / mask.sum(-1)
# print(mask_mean.shape)
# print(mask_mean)