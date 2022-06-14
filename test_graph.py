from GraphAttention import AttentionGraph
from torch import tensor
from torch import optim
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import time

graph = AttentionGraph(1, 16)
x = tensor([[0.0], [1.0], [2], [3.0], [4], [5], [6], [8], [13]])
y = tensor([[0], [1], [0.0], [1.0], [0.0], [1], [0], [0], [1]])

graph._add_node() # 0
graph._add_node() # 1
graph._add_node() # 2
graph._add_node() # 3
graph._add_node() # 4

# print(graph._nodes)

graph._nodes[2]._add_in_neighbor(0)
graph._nodes[2]._add_in_neighbor(1)
graph._nodes[2]._add_in_neighbor(3)

graph._nodes[0]._add_out_neighbor(2)
graph._nodes[1]._add_out_neighbor(2)
graph._nodes[3]._add_out_neighbor(2)

graph._nodes[4]._add_in_neighbor(2)
graph._nodes[2]._add_out_neighbor(4)

graph._update_inps_outs()
# print(graph._inps)
graph._build_topo()
print(graph._topo)

bce = nn.BCELoss()
x_ = tensor([[5.0], [6], [10]])
y_ = tensor([[1.0], [0], [0]])
optimizer = optim.Adam(graph.parameters(), lr=0.002) 

def run(x, y):
  epochs = 500
  e = 1
  losses = []
  losses_val = []
  while e <= epochs:
    n = time.time()
    graph(x)
    y_hat = graph._nodes[4].effect_result
    optimizer.zero_grad()
    loss = bce(y_hat, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    graph(x_)
    loss_val = bce(graph._nodes[4].effect_result, y_)
    losses_val.append(loss_val.item())
    print ('Epoch = {}. Loss = {}. Loss_val = {}. Predict = {}. Time = {}s'.format(e, loss.item(), loss_val.item(), None, time.time() - n))
    e += 1
  return losses, losses_val

losses, losses_val = run(x, y)
times = np.arange(0, len(losses))
graph(tensor([[9.0], [20]]))
print (graph._nodes[4].effect_result)
plt.plot(times, losses)
plt.plot(times, losses_val)
plt.show()