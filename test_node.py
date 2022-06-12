from GraphAttention import Model
from matplotlib import pyplot as plt
from torch import rand
from torch import nn
from torch import tensor
from torch import float64
from torch import optim
import numpy as np

def p():
  """Hàm phụ
  """
  pass

k = rand((1))
x = tensor([[0, 0.0, 0], [0, 1, 0], [0, 0, 2], [3.0, 0, 0], [0, 4.0, 0]])
y = tensor([[1], [0], [1.0], [0.0], [1.0]])

bce = nn.BCELoss()
model = Model(3, 16)
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def run(x, y):
  epochs = 200
  e = 1
  losses = []
  while e <= epochs:
    y_hat, _ = model(x)
    optimizer.zero_grad()
    loss = bce(y_hat, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print ('Epoch = {}. Loss = {}. Predict = {}'.format(e, loss.item(), y_hat.reshape(y_hat.size()[0])))
    e += 1
  return losses
