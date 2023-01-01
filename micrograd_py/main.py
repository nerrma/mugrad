#!/usr/bin/env python3

import autograd as ag

a = ag.Value(1.0, "a")
b = ag.Value(2.0, "b")

print(a + b)
print(a * b)

m = ag.Value(3.0, "m")
d = a * m + b
d.backward()

print(d)
print(a)
print(b)

d.draw_graph()
