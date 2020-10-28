import torch
#一共有多少GPU
print(torch.cuda.device_count())
#当前GPU是哪个
print(torch.cuda.current_device())
#当前GPU的名字
print(torch.cuda.get_device_name(torch.cuda.current_device()))
# torch是否正在使用GPU
print(torch.cuda.is_available())