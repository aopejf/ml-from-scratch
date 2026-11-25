"""
Train the loan default network on the hand-built engine and save its weights.

    python train.py

Takes a few minutes: the engine handles one number at a time, in plain Python, so that
every step can be read. Two models are trained the same way:
    one neuron on its own    (this is logistic regression, the banking standard)
    a network of 8 + 1       (can learn that two warning signs together are worse than their sum)
The one that does better on the "choose settings" customers is kept. The final test customers are not touched here.
"""

import time

import loan

STEPS = 4000

loans = loan.load()
table = loan.build_features(loans)
learn, choose, _ = loan.split(loans)
s = loan.scaler(table.iloc[learn])
x, y = loan.scale(table, s), loans.defaulted.to_numpy()

results = {}
for name, layers in (("one neuron", [1]), ("network", [8, 1])):
    print(f"Training: {name}")
    began = time.time()
    model, losses = loan.train(x[learn], y[learn], layers, STEPS, report=500)
    weights = loan.weights_of(model)
    score = loan.auc(y[choose], loan.predict(weights, x[choose]))
    results[name] = dict(weights=weights, losses=losses, auc=score, seconds=time.time() - began)
    print(f"  {len(model.parameters())} adjustable numbers, {results[name]['seconds']:.0f} seconds, "
          f"ranking score on the choose-settings customers: {score:.4f}")

best = max(results, key=lambda n: results[n]["auc"])
print(f"Keeping: {best}")
loan.save_weights(results[best]["weights"], s, dict(
    chosen=best, steps=STEPS,
    candidates={n: dict(auc_choose=r["auc"], seconds=r["seconds"], layers=r["weights"],
                        loss_curve=[float(sum(r["losses"][i:i + 100]) / 100) for i in range(0, STEPS, 100)]) for n, r in results.items()}))
print("Saved weights.json")
