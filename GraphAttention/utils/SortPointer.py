from typing import List

def sort_pointer(inital_state : List[int], new_vl : int) -> List[int]:
  """Sắp xếp theo kỹ thuật 2 con trỏ

  Args:
    inital_state (List[int]): _description_
    new_vl (int): _description_

  Returns:
    List[int]: _description_
  """
  # assert isinstance(inital_state, list)
  assert isinstance(new_vl, int)
  
  x = []
  stop = False
  l = 0
  if len(inital_state) == 0:
    x.append(new_vl)
    return x
    
  while l != len(inital_state) and not stop:
    if inital_state[l] > new_vl:
      x.append(new_vl)
      stop = True
    else:
      x.append(inital_state[l])
      l += 1
  
  while l != len(inital_state):
    x.append(inital_state[l])
    l += 1
  
  if not stop:
    x.append(new_vl)
    
  return x