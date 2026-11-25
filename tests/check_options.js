/*
Checks the options page's sums (JavaScript) against the Python ones.

    node tests/check_options.js

The page's Greeks are textbook formulas. Python's come from the hand-built autograd engine with
no Greek formula at all. 13_options_pricer/results.json holds the engine's answers for 42 options,
and a tree price for each. This reruns all of them in pricer.js and fails if any differ.
*/

const path = require("path");
const folder = path.join(__dirname, "..", "13_options_pricer");
const { option, tree, impliedVolatility } = require(path.join(folder, "pricer.js"));
const R = require(path.join(folder, "results.json"));

let failed = false, checked = 0;
function same(label, js, python, tolerance) {
  const ok = Math.abs(js - python) <= tolerance;
  failed ||= !ok; checked++;
  if (!ok || process.argv.includes("--all")) console.log(`${ok ? "ok  " : "FAIL"} ${label}: JavaScript ${js}, Python ${python}`);
}
const greeks = ["delta", "gamma", "vega", "theta", "rho"];

for (const kind of ["call", "put"]) {
  const js = option({ ...R.greeks.example, kind });
  same(`${kind} value`, js.value, R.greeks.table[kind].value, 1e-9);
  greeks.forEach(g => same(`${kind} ${g}`, js[g], R.greeks.table[kind][g].engine, 1e-10));
}

R.spot_checks.forEach((c, i) => {
  const js = option(c.case), scale = Math.max(1, c.case.price);
  same(`option ${i + 1} value`, js.value, c.engine.value, 1e-10 * scale);
  greeks.forEach(g => same(`option ${i + 1} ${g}`, js[g], c.engine[g], 1e-10 * scale));
  same(`option ${i + 1} tree`, tree(c.case, 200, false), c.tree, 1e-9 * scale);
  same(`option ${i + 1} tree, usable early`, tree(c.case, 200, true), c.american, 1e-9 * scale);
  same(`option ${i + 1} volatility found from its price`, impliedVolatility(js.value, c.case), c.case.volatility, 1e-6);
});

const put = R.methods.example;
same("the example put", option(put).value, R.methods.formula, 1e-9);
R.methods.tree_curve.forEach(row => same(`tree with ${row.steps} steps`, tree(put, row.steps, false), row.value, 1e-8));
same("the example put, usable early", tree(put, 1000, true), R.methods.american_put, 1e-8);

console.log(failed ? "Some checks failed." : `ok   all ${checked} option figures match Python`);
process.exit(failed ? 1 : 0);
