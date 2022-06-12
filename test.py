import torch
from torch import rand

x = rand((3, 10, 25)) # 3 hàng xóm, 10 điểm dữ liệu, 25 phần tử
print(x[0])
print(x[0].size())