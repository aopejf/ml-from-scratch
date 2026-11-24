"""
Reusable optimizers.

optimizers.py showed the three "walk downhill" rules on a toy valley.
This file packages the same three rules so any model can use them.

Every optimizer works the same way from the outside:

    optimizer = Adam(learning_rate=0.05)
    optimizer.step(params, grads)

`params` is a list of the model's adjustable numbers (NumPy arrays).
`grads` is a matching list saying which way is uphill for each one.
`step` changes the params in place, one nudge downhill.

This is the same shape PyTorch uses (optimizer.step()), so the later
projects can swap these out for the real thing without changing how
the training loop reads.
"""

import numpy as np


class GradientDescent:
    """Take a small step straight downhill."""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.learning_rate * g


class Momentum:
    """Remember where we were heading, like a rolling ball."""

    def __init__(self, learning_rate, memory=0.9):
        self.learning_rate = learning_rate
        self.memory = memory
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocity):
            v *= self.memory
            v -= self.learning_rate * g
            p += v


class Adam:
    """Give every adjustable number its own step size."""

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.avg_grad = None
        self.avg_sq_grad = None
        self.t = 0

    def step(self, params, grads):
        if self.avg_grad is None:
            self.avg_grad = [np.zeros_like(p) for p in params]
            self.avg_sq_grad = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.avg_grad, self.avg_sq_grad):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g**2
            corrected_m = m / (1 - self.beta1**self.t)
            corrected_v = v / (1 - self.beta2**self.t)
            p -= self.learning_rate * corrected_m / (np.sqrt(corrected_v) + self.eps)
