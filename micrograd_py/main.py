#!/usr/bin/env python3

import autograd as ag

a = ag.Value(1.0)
b = ag.Value(2.0)

print("add", a + b)
print("mul", a * b)
