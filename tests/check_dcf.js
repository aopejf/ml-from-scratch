/*
Checks the valuation web page's sums (JavaScript) against the Python ones.

    node tests/check_dcf.js

11_company_valuation/results.json holds what valuation.py worked out. This reruns the base
case, every cell of the sensitivity table, the three stories and the reverse valuation in
dcf.js, and fails if any differ.
*/

const path = require("path");
const folder = path.join(__dirname, "..", "11_company_valuation");
const { dcf, solve } = require(path.join(folder, "dcf.js"));
const R = require(path.join(folder, "results.json"));

let failed = false, checked = 0;
function same(label, js, python, tolerance) {
  const ok = Math.abs(js - python) <= tolerance;
  failed ||= !ok; checked++;
  if (!ok || process.argv.includes("--all")) console.log(`${ok ? "ok  " : "FAIL"} ${label}: JavaScript ${js}, Python ${python}`);
}

const base = dcf(R.start, R.assumptions);
same("value per share", base.valuePerShare, R.base.value_per_share, 1e-6);
same("enterprise value", base.enterpriseValue, R.base.enterprise_value, 1);
same("share after year ten", base.terminalShare, R.base.terminal_share, 1e-9);
same("year-ten multiple", base.terminalMultiple, R.base.terminal_multiple, 1e-9);
base.rows.forEach((row, i) => same(`free cash flow ${row.year}`, row.freeCashFlow, R.base.rows[i].free_cash_flow, 1));

R.sensitivity.waccs.forEach((wacc, i) => R.sensitivity.terminal_growths.forEach((terminal_growth, j) =>
  same(`per share at WACC ${wacc}, growth ${terminal_growth}`, dcf(R.start, { ...R.assumptions, wacc, terminal_growth }).valuePerShare, R.sensitivity.value_per_share[i][j], 1e-6)));

for (const [name, c] of Object.entries(R.scenarios))
  same(`${name} story`, dcf(R.start, { ...R.assumptions, growth_start: c.growth_start, margin: c.margin }).valuePerShare, c.value_per_share, 1e-6);

same("growth the price needs", solve(R.start, R.assumptions, "growth_start", -0.05, 0.60), R.implied.growth_start, 1e-9);
same("margin the price needs", solve(R.start, R.assumptions, "margin", 0.001, 0.30), R.implied.margin, 1e-9);
same("return the price implies", solve(R.start, R.assumptions, "wacc", R.assumptions.terminal_growth + 0.002, 0.25), R.implied.wacc, 1e-9);

console.log(failed ? "Some checks failed." : `ok   all ${checked} valuation figures match Python`);
process.exit(failed ? 1 : 0);
