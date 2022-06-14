from torch import nn
from torch import Tensor
from typing import List
from ..Node import Attention
from ..Node import FakeAttention
from ..Node import Input

class AttentionGraph(nn.Module):
  def __init__(self, _dx, _da):
    super(AttentionGraph, self).__init__()
    
    # Tập các đỉnh đồ thị
    self._nodes : List[Attention] = nn.ModuleList()
    
    # Tập các đỉnh không có đầu vào, chỉ có đầu ra 
    # Xem đây tập đầu vào
    self._inps : List[int] = []
    
    # Tất cả các đỉnh đều là đầu ra
    # Nên không có tập đầu ra
    self._input = Input(_dx, _da)
    self._da = _da
      
  def _build_topo(self):
    visited = [False] * len(self._nodes)
    topo = []
    for i in range(len(self._inps)):
      if not visited[self._inps[i]]:
        visited[self._inps[i]] = True
        self.__dfs(topo, self._nodes[self._inps[i]], visited)
    
    # Thứ tự topo cho từng 
    # đỉnh đầu vào
    topo.reverse()
    self._topo = topo
  
  def forward(self, x : Tensor):
    x = self._input(x)
    for j in self._topo:
      self._nodes[j].forward(x)

  def reset(self):
    """Gán lại giá trị
    """
    for j in self._nodes:
      j.reset()
      
  def __dfs(self, trace, node : FakeAttention, visited):
    for j in node.out_neighbors:
      if not visited[j]:
        visited[j] = True
        self.__dfs(trace, self._nodes[j], visited)
    trace.append(node._id)
        
  def _update_inps_outs(self):
    """Phương thức được dùng để cập nhật lại kết quả đầu ra
    """
    self._inps = []
    self._outs = []
    for i in range(len(self._nodes)):
      temp_node = self._nodes[i]
      if temp_node._is_out:
        self._inps.append(i)
  
  def _add_node(self):
    n = Attention(len(self._nodes), self._da, self._get_infor)
    self._nodes.append(n)
    
  def _get_infor(self, _id : int) -> Attention:
    return self._nodes[_id]