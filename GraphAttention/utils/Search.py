import math
from typing import List

def b_search(X : List[int], vl : int):
  """Giải thuật tìm kiếm nhị phân

  Args:
    X (List[int]): _description_
    vl (int): _description_

  Returns:
    _type_: _description_
  """
  # assert isinstance(X, list)
  assert isinstance(vl, int)
  
  l = 0
  r = len(X) - 1
  m = -1
  
  while l <= r:
    m = math.floor((l + r) / 2)
    if vl < X[m]:
      l = m + 1
    elif vl == X[m]:
      return m
    else:
      r = m - 1
  
  return -1