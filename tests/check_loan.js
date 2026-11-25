/*
Checks the loan page's sums (JavaScript) against the Python ones.

    node tests/check_loan.js

12_loan_default/data.js holds the test customers and what loan.py worked out. This reruns the
lending rule, the stress test and the fallback cutoff in policy.js, and fails if any differ.
The page stores each risk to six decimal places, so money figures are allowed a small rounding gap.
*/

const fs = require("fs");
const path = require("path");
const folder = path.join(__dirname, "..", "12_loan_default");
const { stress, lend, breakEven, fallbackCutoff } = require(path.join(folder, "policy.js"));
const window = {};
eval(fs.readFileSync(path.join(folder, "data.js"), "utf8"));
const D = window.DATA, C = D.customers, P = D.policy, S = D.stress;

let failed = false, checked = 0;
function same(label, js, python, tolerance) {
  const ok = Math.abs(js - python) <= tolerance;
  failed ||= !ok; checked++;
  if (!ok || process.argv.includes("--all")) console.log(`${ok ? "ok  " : "FAIL"} ${label}: JavaScript ${js}, Python ${python}`);
}
const money = 2000;  // NT$, on a book of NT$309m

const line = breakEven(P.margin, P.loss_if_default);
same("break-even risk", line, P.break_even_risk, 1e-12);

const everyone = lend(C, C.defaulted, 2, P.margin, P.loss_if_default);
same("lend to everyone", everyone.profit, P.lend_to_all, 1e-3);
same("balances", everyone.lent, P.balance, 1e-3);

const model = lend(C, C.defaulted, line, P.margin, P.loss_if_default);
same("profit with the model", model.profit, P.with_model, 1e-3);
same("share approved", model.approved, P.approved, 1e-12);
same("default rate of those approved", model.defaultRate, P.default_rate_approved, 1e-12);

D.normal_curve.forEach(row => same(`profit at a cutoff of ${row.cutoff}`, lend(C, C.defaulted, row.cutoff, P.margin, P.loss_if_default).profit, row.profit, 1e-3));

const hard = stress(C.risk, S.multiplier);
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
same("average risk, normal", mean(C.risk), S.average_risk_normal, 1e-6);
same("average risk, recession", mean(hard), S.average_risk_recession, 1e-6);
same("expected profit, normal", lend(C, C.risk, line, P.margin, P.loss_if_default).profit, S.expected_normal, money);
same("recession, normal cutoff kept", lend(C, hard, line, P.margin, P.loss_if_default).profit, S.recession_without_fallback, money);
const fallback = fallbackCutoff(C.risk, hard, line);
same("fallback cutoff", fallback, S.fallback_cutoff, 1e-5);
const careful = lend(C, hard, fallback, P.margin, P.loss_if_default);
same("recession, fallback cutoff", careful.profit, S.recession_with_fallback, money);
same("share approved on the fallback", careful.approved, S.approved_fallback, 1e-12);
D.recession_curve.forEach(row => same(`recession profit at a cutoff of ${row.cutoff}`, lend(C, hard, row.cutoff, P.margin, P.loss_if_default).profit, row.profit, money));

console.log(failed ? "Some checks failed." : `ok   all ${checked} lending figures match Python`);
process.exit(failed ? 1 : 0);
