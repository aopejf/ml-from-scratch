"""
Four checks that the engine's gradients are right.

    python checks.py
"""

import time

import numpy as np
from sklearn.datasets import load_breast_cancer

from engine import Value
from nn import Adam, Network, cross_entropy, optimizer_step, zero_grads

# ---------------------------------------------------------------------------
# Check 1. A sum you can follow by hand.
#   loss = (a * b + c) squared, with a = 2, b = -3, c = 10
# ---------------------------------------------------------------------------
print("Check 1: a tiny example, traced")
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
product = a * b
total = product + c
loss = total**2
loss.backward()

print(f"  forward:   a*b = {product.data:g}   then + c = {total.data:g}   then squared = {loss.data:g}")
print("  backward:  how much the loss moves if each number goes up by 1")
print(f"    total    {total.grad:6g}   (slope of squaring at 4 is 2*4)")
print(f"    c        {c.grad:6g}   (adding passes the 8 straight through)")
print(f"    product  {product.grad:6g}")
print(f"    a        {a.grad:6g}   (8 times b, which is -3)")
print(f"    b        {b.grad:6g}   (8 times a, which is 2)")


def measured(f, tiny=1e-6):
    return (f(tiny) - f(-tiny)) / (2 * tiny)


by_nudging = measured(lambda t: ((2.0 + t) * -3.0 + 10.0) ** 2)
print(f"  nudging a by hand and measuring gives {by_nudging:.4f}. Engine said {a.grad:g}.")


# ---------------------------------------------------------------------------
# Check 2. The famous shortcut from project 0, rediscovered.
#   One neuron, one student. Nobody told the engine "prediction minus truth".
# ---------------------------------------------------------------------------
print("\nCheck 2: does the engine rediscover 'prediction minus truth'?")
neuron = Network(2, [1], seed=1)
student, truth = [1.3, -0.4], 1.0
prediction = neuron(student)
cross_entropy([prediction], [truth]).backward()
weights = neuron.parameters()[:2]
print(f"  prediction {prediction.data:.4f}, truth {truth:g}")
for i, w in enumerate(weights):
    by_formula = (prediction.data - truth) * student[i]
    print(f"  weight {i}:  engine {w.grad:+.6f}   (prediction - truth) x input = {by_formula:+.6f}")


# ---------------------------------------------------------------------------
# Check 3. A whole network, every weight, against brute-force measurement.
# ---------------------------------------------------------------------------
print("\nCheck 3: every weight in a 2 -> 8 -> 8 -> 1 network")
rng = np.random.default_rng(0)
points = rng.normal(size=(5, 2)).tolist()
labels = [1.0, 0.0, 1.0, 1.0, 0.0]
net = Network(2, [8, 8, 1], seed=2)
params = net.parameters()


def network_loss():
    return cross_entropy([net(p) for p in points], labels)


zero_grads(params)
network_loss().backward()

worst = 0.0
for p in params:
    original = p.data

    def loss_with_nudge(t, p=p, original=original):
        p.data = original + t
        return network_loss().data

    worst = max(worst, abs(p.grad - measured(loss_with_nudge)))
    p.data = original
print(f"  {len(params)} weights checked. Biggest difference from measurement: {worst:.2e}")


# ---------------------------------------------------------------------------
# Check 4. Real data. Train the project 0 tumour classifier again, but with
#   the engine working out the gradients. Project 1 trained the same model
#   with a hand-written formula and Adam for 150 steps and ended at loss 0.0624.
# ---------------------------------------------------------------------------
print("\nCheck 4: the tumour classifier, with no hand-written gradient formula")
rng = np.random.default_rng(0)
data = load_breast_cancer()
inputs, is_cancer = data.data, (data.target == 0).astype(float)
order = rng.permutation(len(inputs))
inputs, is_cancer = inputs[order], is_cancer[order]
n_train = int(0.8 * len(inputs))
train_x, test_x = inputs[:n_train], inputs[n_train:]
train_y, test_y = is_cancer[:n_train], is_cancer[n_train:]
mean, spread = train_x.mean(axis=0), train_x.std(axis=0)
train_x, test_x = (train_x - mean) / spread, (test_x - mean) / spread

model = Network(30, [1])
params = model.parameters()
for p in params:
    p.data = 0.0  # start from zero, like projects 0 and 1
optimizer = Adam(learning_rate=0.05)
rows, truths = train_x.tolist(), train_y.tolist()
penalty = 1.0 / n_train

start = time.time()
for step in range(151):
    loss = cross_entropy([model(row) for row in rows], truths)
    for w in params[:-1]:
        loss = loss + 0.5 * penalty * w**2  # the same penalty on large weights
    if step in (0, 50, 100, 150):
        print(f"  step {step:3d}   loss {loss.data:.4f}")
    if step == 150:
        break
    zero_grads(params)
    loss.backward()
    optimizer_step(optimizer, params)

said_cancer = np.array([model(row).data for row in test_x.tolist()]) > 0.5
print(f"  accuracy on {len(test_y)} unseen patients: {np.mean(said_cancer == (test_y == 1)):.1%}")
print(f"  took {time.time() - start:.0f} seconds")
