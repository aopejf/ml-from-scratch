"""
Project 2: a tiny autograd engine.

In project 0 the "which way is downhill" formula (prediction minus truth)
was worked out by hand. That only works for one specific model. Change the
model and you would have to redo the calculus.

This file removes that step for good. Every number is wrapped in a Value.
A Value remembers:

    data      the number itself
    parents   the Values it was made from
    local     for each parent: "if that parent goes up by 1, how much do I go up?"
    grad      filled in later: "if I go up by 1, how much does the final loss go up?"

Do any maths with Values and they quietly build a record of every step.
Calling .backward() on the final loss then walks that record in reverse
and fills in .grad for every number involved. That walk is backpropagation.

This is the same job PyTorch's loss.backward() does.
"""

import math


class Value:
    __slots__ = ("data", "grad", "parents", "local")

    def __init__(self, data, parents=(), local=()):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.local = local

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # -- The basic operations. Each one returns a new Value that remembers
    # -- its parents and the local slope towards each of them.

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # a + b: nudging either one up by 1 moves the sum up by 1.
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # a * b: nudging a up by 1 moves the product up by b, and the reverse.
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, power):
        # a ** n, where n is a plain number. Slope is n * a^(n-1).
        return Value(self.data**power, (self,), (power * self.data ** (power - 1),))

    def tanh(self):
        # Squashes any number into -1..1. Used inside the network's neurons.
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

    def sigmoid(self):
        # Squashes any number into 0..1. The same squash as project 0.
        s = 1.0 / (1.0 + math.exp(-self.data))
        return Value(s, (self,), (s * (1 - s),))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    # -- Everything else is built from the operations above.

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    # -- Backpropagation.

    def backward(self):
        """
        Fill in .grad for every Value that went into making this one.

        Step 1: list every Value in the order it was computed.
        Step 2: go through that list backwards. Each Value takes the grad
                that has arrived from above, multiplies it by its local
                slope towards each parent, and passes it down to that parent.
                (That multiplication is the chain rule.)
        """
        order = []
        seen = set()
        stack = [(self, False)]
        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                for parent in node.parents:
                    if parent not in seen:
                        stack.append((parent, False))

        self.grad = 1.0  # the loss goes up by 1 when the loss goes up by 1
        for node in reversed(order):
            for parent, slope in zip(node.parents, node.local):
                parent.grad += slope * node.grad
