import torch
import numpy as np
# inputs代表的意思是[batch_size, sequence_length, word_dimesion]， 那么下面（2，8，3）的意思就是有2句话，每句话有8个单词，每个单词的向量维度是3
inputs = torch.arange(1,49).view(2,8,3)
# locations代表的是[batch_size, 1, words_location]  1，代表的是有一个词语， words_location代表的是词语在句子中sequence_length的起始和结束位置
locations = np.array([[[0,2]],[[2,6]]])
# 我们要做的是把，每个句子中的这个词语的向量取出来,相加在一起，代表这个词语的向量，形状是[batch_size, 1, word_dimesion]
# print(locations.shape)
# print(input)
# print(locations)
result = []
for input,location in zip(inputs,locations):
    new = input[location[0][0]:location[0][1],:]
    final = torch.sum(input=new, dim=0)
    # print(final.shape)
    result.append(final.numpy())
res = torch.tensor(result)
print(res.shape)
output = res.view(2,1,3)
print(output.shape)