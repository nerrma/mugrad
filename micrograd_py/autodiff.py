#!/usr/bin/env python3

from graphviz import Digraph
import math


class Value:
    def __init__(self, val, label="", op="", children=()):
        self.data = val
        self.children = children
        self.grad = 0.0
        self._backward = lambda: None
        self._op = op
        self.label = label
        self.topo = []

    def __repr__(self):
        return f"Value(label={self.label}, data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        result = Value(
            self.data + other.data,
            children=(self, other),
            op="+",
        )

        def __backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = __backward
        return result

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=str(other))
        result = Value(
            self.data * other.data,
            children=(self, other),
            op="*",
        )

        def __backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = __backward

        return result

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        result = Value(self.data**other, children=(self,))

        def __backward():
            self.grad += other * (self.data ** (other - 1)) * result.grad

        result._backward = __backward
        return result

    def exp(self):
        result = Value(
            math.exp(self.data),
            children=(self.data,),
            op="exp",
        )

        def __backward():
            self.grad += result.data * result.grad

        result._backward = __backward
        return result

    def gen_topo(self):
        # construct topo sorted graph
        # as the graph is fully connected we do not need to call dfs for each vertex
        seen = set()
        topo = []

        def dfs(v, p):
            for u in v.children:
                if u not in seen:
                    seen.add(u)
                    dfs(u, v)
            topo.append(v)

        dfs(self, Value(0.0, label="blank"))
        return topo[::-1]

    def relu(self):
        result = Value(
            0.0 if self.data < 0.0 else self.data, children=(self,), label="ReLU"
        )

        def __backward():
            self.grad += (result.data > 0.0) * result.grad

        result._backward = __backward
        return result

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, children=(self,), label="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # call backward on all children
        self.grad = 1.0
        for v in self.gen_topo():
            v._backward()

    def zero_grad(self):
        for v in self.gen_topo():
            v.grad = 0

    def draw_graph(self):
        nodes, edges = set(), set()
        seen = set()

        def dfs(v):
            nodes.add(v)
            for u in v.children:
                if u not in nodes:
                    nodes.add(u)
                    edges.add((u, v))
                    dfs(u)

        dfs(self)
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right
        for n in nodes:
            uid = str(id(n))
            # for any value in the graph, create a rectangular ('record') node for it
            dot.node(
                name=uid,
                label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                shape="record",
            )
            if n._op:
                # if this value is a result of some operation, create an op node for it
                dot.node(name=uid + n._op, label=n._op)
                # and connect this node to it
                dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            # connect n1 to the op node of n2
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot
