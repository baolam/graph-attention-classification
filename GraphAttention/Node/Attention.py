from torch import nn
from torch import rand
from torch import Tensor
from torch import zeros
from torch import sigmoid
from torch import softmax
from typing import Dict
from typing import Tuple
from abc import ABC
from ..utils import b_search
from ..utils import sort_pointer

class FakeAttention(ABC):
  def __init__(self):
    self.effect_result = None
    self.transform_result = None
    self.inp_neighbors : List[int] = []
    self.out_neighbors : List[int] = []
    self._id = None
     
  def _add_in_neighbor(self):
    raise NotImplementedError()
  
  def forward(self):
    raise NotImplementedError()
  
  def reset(self):
    raise NotImplementedError()

class Transform(nn.Module):
  def __init__(self, _da : int):
    super(Transform, self).__init__()
    
    assert isinstance(_da, int)
    self.w = rand((_da), requires_grad=True)
    self.b = rand((_da), requires_grad=True)
  
  def forward(self, x : Tensor) -> Tensor:
    return self.w * x + self.b
    
class Attention(nn.Module):
  def __init__(self, _id : int, _da : int, _get_infor : any):
    super(Attention, self).__init__()
    
    assert isinstance(_id, int)
    assert callable(_get_infor)
    
    self._id = _id
    
    self._is_inp = True # Là đỉnh đầu vào
    self._is_out = True # Là đỉnh đầu ra
    
    # Số chiều của không gian attention
    # Số chiều này mang chiều không đổi trong suốt quá trình
    # lan truyền
    self._da = _da
    
    # Xem đây là tính chất của attention này
    p = rand((_da), requires_grad=False)
    self.n = nn.Parameter(p, requires_grad=True)
    
    # Tập các đỉnh đầu vào (id của đỉnh)
    self.inp_neighbors : List[int] = []
    self.imporant_inp_neighbor = Tensor()
      
    # Tập các đỉnh đầu ra (id của đỉnh)
    self.out_neighbors : List[int] = []
    
    # Hàm này được dùng để lấy thông tin của đỉnh
    self._get_infor = _get_infor
    
    # Có tính chất hay không ?
    self._effect = nn.Linear(_da, 1, bias=True)
    self._transform = Transform(_da)
    
    # Biến này dùng để lưu trữ kết quả tính toán
    # Dùng biến này có mục đích xem đỉnh về lại một điểm đầu vào
    # Hai biến này còn có chức năng lưu trữ loại quá trình gradient
    self.effect_result = None
    self.transform_result = None
    
  def forward(self, x : Tensor):
    # Số chiều của neighbors
    # (số hàng xóm, đầu vào x, số chiều da)
    
    assert x.size()[1] == self._da

    if len(self.inp_neighbors) != 0:
      # Tiến hành truy lấy hệ số đánh giá 
      # các đỉnh hàng xóm
      x = self._build_important_neighbor(x)
      x_ = self.__calc_sum(x)
      
      # Thực hiện nhúng tính chất vào x_
      x_ = self.n * x_
    else:
      x_ = self.n * x
    
    effect = sigmoid(self._effect(x_))
    transform = self._transform(x_)
    
    # Thể hiện ảnh hưởng của tính chất lên đầu vào (bị biến đổi)
    self.effect_result = effect
    self.transform_result = effect * transform
    
  def __calc_sum(self, x_ : Tensor) -> Tensor:
    """Tính tổng các neighbor

    Args:
      x_ (Tensor): _description_

    Returns:
      Tensor: _description_
    """
    # Đầu ra có số chiều là (đầu vào x, số chiều da)
    output_size = (x_.size()[1], x_.size()[2])
    y = zeros(output_size)
    for i in range(x_.size()[0]):
      t = x_[i].reshape((self._da), -1) * self.imporant_inp_neighbor[i]
      t = t.reshape((-1, self._da))
      y = y + t
    return y
    
  def _build_important_neighbor(self, x : Tensor):
    """Xây dựng đầu vào khối đánh giá tính chất
    """
    t = zeros((len(self.inp_neighbors), x.size()[0]))
    n = zeros((len(self.inp_neighbors), x.size()[0], x.size()[1]))
    
    for j in range(len(self.inp_neighbors)):
      temp : FakeAttention = self._get_infor(self.inp_neighbors[j])
      t[j] = temp.effect_result.reshape(x.size()[0])
      n[j] = temp.transform_result
    
    self.imporant_inp_neighbor = softmax(t, dim=0) # Cột
    return n
    
  def _add_in_neighbor(self, _id : int):
    """Thêm một đỉnh đầu vào mới

    Args:
      _id (int): _description_

    Returns:
      _type_: _description_
    """
    # Kiểm tra xem đỉnh đã tồn tại chưa
    if b_search(self.inp_neighbors, _id) == -1:
      self._is_inp = True
      self._is_out = False
      self.inp_neighbors = sort_pointer(self.inp_neighbors, _id)
      return True
    
    # Kết quả trả về là ko thể thêm vào
    return False
  
  def _add_out_neighbor(self, _id : int):
    """Thêm một đỉnh đầu vào mới

    Args:
      _id (int): _description_

    Returns:
      _type_: _description_
    """
    # Kiểm tra xem đỉnh đã tồn tại chưa
    if b_search(self.out_neighbors, _id) == -1:
      self._is_inp = False
      self._is_out = True
      self.out_neighbors = sort_pointer(self.out_neighbors, _id)
      return True
    
    # Kết quả trả về là ko thể thêm vào
    return False
  
  def reset(self):
    self.effect_result = None
    self.transform_result = None