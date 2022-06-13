from GraphAttention import AttentionGraph
from torch import tensor
from torch import optim
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import time

graph = AttentionGraph(3, 16)
x = tensor([[0, 0.0, 0], [1, 0, 0], [2, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])
y = tensor([[1], [0], [1.0], [0.0], [1.0]])

graph._add_node() # 0
graph._add_node() # 1
graph._add_node() # 2

graph._nodes[2]._add_in_neighbor(0)
graph._nodes[2]._add_in_neighbor(1)
graph._nodes[0]._add_out_neighbor(2)
graph._nodes[1]._add_out_neighbor(2)

graph._update_inps_outs()
graph._build_topo()

bce = nn.BCELoss()
optimizer = optim.Adam(graph.parameters(), lr = 0.01) 

def run(x, y):
  epochs = 150
  e = 1
  losses = []
  while e <= epochs:
    n = time.time()
    graph(x)
    y_hat = graph._nodes[2].effect_result
    optimizer.zero_grad()
    loss = bce(y_hat, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print ('Epoch = {}. Loss = {}. Predict = {}. Time = {}s'.format(e, loss.item(), None, time.time() - n))
    e += 1
  return losses

losses = run(x, y)
print(graph._nodes[2].imporant_inp_neighbor)
times = np.arange(0, len(losses))
plt.plot(times, losses)
plt.show()