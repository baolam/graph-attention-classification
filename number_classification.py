from GraphAttention import AttentionGraph
from torch import nn

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    
    # Ảnh đầu vào có 3 kênh (R, G, B) --> Đưa ra 16 đầu ra
    self._first_conv2d = nn.Conv2d(3, 16, 3)
    self._second_conv2d = nn.Conv2d(16, 16, 3)