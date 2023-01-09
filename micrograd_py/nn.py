#!/usr/bin/env python3

import random
import autodiff as ad


class Neuron:
    def __init__(self, n_in, w=None, b=None, active=True, activation=None):
        if w is None:
            self.w = [ad.Value(random.uniform(-1.0, 1.0)) for i in range(n_in)]
        else:
            self.w = w

        if b is None:
            self.b = ad.Value(0.0, label=f"b")
        else:
            self.b = b

        self.activation = activation
        self.active = active
        assert len(self.w) == n_in

    def __call__(self, x):
        assert len(x) == len(self.w)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if not self.active:
            return act
        elif self.activation == "relu":
            return act.relu()
        elif self.activation == "tanh":
            return act.tanh()
        else:
            return act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, dims, weights=None, biases=None, **kwargs):
        if weights and biases:
            self.neurons = [
                Neuron(dims[0], w=ad.Value(w), b=ad.Value(b), **kwargs)
                for w, b in zip(weights, biases)
            ]
        else:
            self.neurons = [Neuron(dims[0], **kwargs) for i in range(dims[1])]

    def __call__(self, x):
        out = [N(x) for N in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, dims, **kwargs):
        self.layers = [
            Layer(d, active=(i != len(dims) - 1), **kwargs) for i, d in enumerate(dims)
        ]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def set_params(self, p_list):
        for p, d in zip(self.parameters(), p_list):
            p = d

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
