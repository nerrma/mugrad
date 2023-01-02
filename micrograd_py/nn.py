#!/usr/bin/env python3

import random
import autograd as ag


class Neuron:
    def __init__(self, n_in, w=None, b=None, activation=None, layer=-1, neuron=-1):
        if w is None:
            self.w = [
                ag.Value(random.uniform(0, 1.0), label=f"{layer}_{neuron}_w{i}")
                for i in range(n_in)
            ]

        if b is None:
            self.b = ag.Value(0.0, label=f"b")

        self.activation = activation
        assert len(self.w) == n_in

    def __call__(self, x):
        assert len(x) == len(self.w)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # return act
        return act.relu() if self.activation else act

    def params(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, dims, weights=None, biases=None, **kwargs):
        if weights and biases:
            self.neurons = [
                Neuron(dims[0], w=ag.Value(w), b=ag.Value(b), **kwargs)
                for w, b in zip(weights, biases)
            ]
        else:
            self.neurons = [Neuron(dims[0], neuron=i, **kwargs) for i in range(dims[1])]
        self.dims = dims

    def __call__(self, x):
        out = [N(x) for N in self.neurons]
        return out[0] if len(out) == 1 else out

    def params(self):
        return [p for n in self.neurons for p in n.params()]


class MLP:
    def __init__(self, dims=[(2, 1), (1, 1)], **kwargs):
        self.layers = []
        for i, d in enumerate(dims):
            self.layers.append(Layer(d, activation=True, layer=i, **kwargs))

        self.layers[-1].activation = False

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def zero_grad(self):
        for p in self.params():
            p.grad = 0
