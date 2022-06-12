from torch import nn
from torch import Tensor

class Input(nn.Module):
  def __init__(self, _dx : int, _da : int):
    super(Input, self).__init__()
    
    # Đầu vào chuẩn của không gian dữ liệu thực
    assert isinstance(_dx, int)
    
    # Đầu ra được chuyển vị
    assert isinstance(_da, int)
    
    self._convert = nn.Linear(_dx, _da, bias=True)
  
  def forward(self, x : Tensor) -> Tensor:
    return self._convert(x)