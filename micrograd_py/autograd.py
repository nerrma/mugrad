#!/usr/bin/env python3


class Value:
    def __init__(self, val, children=()):
        self.data = val
        self.children = children
        self.grad = 0.0
        self.label = ""
        self._backward = lambda: None

    def __repr__(self):
        return f"value {self.label}: {self.data}, grad {self.grad}"

    def __add__(self, other):
        result = Value(self.data + other.data, children=(self, other))

        return result

    def __mul__(self, other):
        result = Value(self.data * other.data, children=(self, other))

        return result
