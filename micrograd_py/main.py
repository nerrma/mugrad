#!/usr/bin/env python3

# import autograd as ag
#
# a = ag.Value(1.0, "a")
# b = ag.Value(2.0, "b")
#
# print(a + b)
# print(a * b)
#
# m = ag.Value(3.0, "m")
# d = a * m + b
# d.backward()
#
# print(d)
# print(a)
# print(b)

import autograd as ag
from nn import MLP
import numpy as np

# lin-reg
xs = [[1], [2], [3], [4]]
# y = [1, 2, 3, 4]
y = [1, 4, 9, 16]

# xor
# xs = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
# y = [0, 0, 1, 1]
clf = MLP(dims=[(1, 4), (4, 4), (4, 4), (4, 1)])

nit = 900
eta = 0.001
y_hat = None
for i in range(nit):
    y_hat = [clf.forward(x) for x in xs]
    loss = sum((y_h - y_t) ** 2 for y_h, y_t in zip(y_hat, y))
    clf.zero_grad()
    loss.backward()

    for p in clf.params():
        p.data -= eta * p.grad


for y in y_hat:
    print(y.data)
