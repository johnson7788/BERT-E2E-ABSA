import torch
import numpy as np
# inputs代表的意思是[batch_size, sequence_length, word_dimesion]， 那么下面（3，8，2）的意思就是有3句话，每句话有8个单词，每个单词的向量维度是2
inputs = torch.arange(1,49).view(3,8,2)
# locations代表的是[batch_size, 1, words_location]  1，代表的是有一个词语， words_location代表的是词语在句子中sequence_length的起始和结束位置
locations = torch.randint(0,2,(3,8))
# 我们要做的是把，每个句子中的这个词语的向量取出来,相加在一起，代表这个词语的向量，形状是[batch_size, 1, word_dimesion]
print(locations.shape)
print(inputs)
print(locations)
active_loss = locations.view(-1) == 1
flatten_inputs = inputs.view(-1, 2)
print(active_loss)
print(flatten_inputs)
final = flatten_inputs[active_loss]

# flatten_locations = locations.reshape(-1,2)

# for input,location in zip(inputs,locations):
#     new = input[location[0][0]:location[0][1],:]
#     final = torch.sum(input=new, dim=0)
    # print(final.shape)
    # result.append(final.numpy())
# res = torch.tensor(result)
# print(res.shape)
# output = res.view(2,1,3)
# print(output)

"""
tensor([[[  5,   7,   9]],

        [[142, 146, 150]]])
"""