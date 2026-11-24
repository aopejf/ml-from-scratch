"""
Neurons, layers and networks, built from Values.

A neuron is the model from project 0: a weighted sum of its inputs, then a
squash. A layer is several neurons looking at the same inputs. A network is
layers in a row, each one fed by the one before.

No gradient formula appears anywhere in this file. The engine works it out.
"""

import random
import sys
from pathlib import Path

import numpy as np

from engine import Value

# The optimizers built in project 1.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_optimizers"))
from optim import Adam, GradientDescent, Momentum  # noqa: E402,F401


class Neuron:
    def __init__(self, n_inputs, squash, rng):
        scale = 1.0 / n_inputs**0.5
        self.weights = [Value(rng.uniform(-1, 1) * scale) for _ in range(n_inputs)]
        self.bias = Value(0.0)
        self.squash = squash  # "tanh" or "sigmoid"

    def __call__(self, inputs):
        total = self.bias
        for w, x in zip(self.weights, inputs):
            total = total + w * x
        return total.tanh() if self.squash == "tanh" else total.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs, n_neurons, squash, rng):
        self.neurons = [Neuron(n_inputs, squash, rng) for _ in range(n_neurons)]

    def __call__(self, inputs):
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class Network:
    """
    Network(2, [8, 8, 1]) means: 2 inputs, then a layer of 8 neurons,
    another layer of 8, and a final single neuron that outputs a
    probability between 0 and 1.

    Network(2, [1]) is one neuron on its own: exactly logistic regression.
    """

    def __init__(self, n_inputs, layer_sizes, seed=0):
        rng = random.Random(seed)
        sizes = [n_inputs] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            is_last = i == len(layer_sizes) - 1
            self.layers.append(Layer(sizes[i], sizes[i + 1], "sigmoid" if is_last else "tanh", rng))

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs[0]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def cross_entropy(predictions, truths):
    """The same loss as project 0, written with Values."""
    total = Value(0.0)
    for p, y in zip(predictions, truths):
        total = total - (y * (p + 1e-12).log() + (1 - y) * (1 - p + 1e-12).log())
    return total / len(truths)


def zero_grads(params):
    """Wipe the grads from the previous step before computing new ones."""
    for p in params:
        p.grad = 0.0


def optimizer_step(optimizer, params):
    """Hand the engine's grads to a project 1 optimizer and store the nudged numbers."""
    numbers = np.array([p.data for p in params])
    grads = np.array([p.grad for p in params])
    optimizer.step([numbers], [grads])
    for p, new in zip(params, numbers):
        p.data = float(new)
