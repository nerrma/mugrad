#!/usr/bin/env python3

from nn import MLP
import random
import numpy as np

random.seed(18)
np.random.seed(18)

x = np.array(np.arange(0, 6))
y = [e**2 for e in x]
xs = x.reshape(-1, 1)

clf = MLP([(1, 16), (16, 32), (32, 16), (16, 1)], activation="tanh")

nit = 400
eta = 0.01
y_hat = None
for i in range(nit):
    y_hat = [clf(x) for x in xs]
    loss = sum((y_h - y_t) ** 2 for y_h, y_t in zip(y_hat, y)) / len(x)
    clf.zero_grad()
    loss.backward()

    if i % 10 == 0:
        print(f"iter {i}: loss {loss.data}")

    for p in clf.parameters():
        p.data -= eta * p.grad


for y in y_hat:
    print(y.data)
