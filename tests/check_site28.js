/*
Checks the dashboard's investment sums (JavaScript) against the Python ones.

    node tests/check_site28.js

10_car_wash_finance/results.json holds what finance.py worked out for three locations
in three volume cases. This reruns all nine in site28.js and fails if any differ.
*/

const path = require("path");
const folder = path.join(__dirname, "..", "10_car_wash_finance");
const { site28 } = require(path.join(folder, "site28.js"));
const results = require(path.join(folder, "results.json")).site28;

let failed = false;
for (const place of results.candidates) {
  for (const [name, python] of Object.entries(place.cases)) {
    const js = site28(python.cars, place.rent, place.build_cost, results.economics, results.assumptions);
    const samePayback = python.payback_years === null ? js.paybackYears === null : Math.abs(js.paybackYears - python.payback_years) < 1e-6;
    const ok = Math.abs(js.npv - python.npv) < 0.01 && Math.abs(js.irr - python.irr) < 1e-9 && samePayback;
    failed ||= !ok;
    console.log(`${ok ? "ok  " : "FAIL"} ${place.candidate}, ${name}: NPV ${Math.round(js.npv).toLocaleString()} (Python ${Math.round(python.npv).toLocaleString()}), IRR ${(js.irr * 100).toFixed(2)}%`);
  }
}
process.exit(failed ? 1 : 0);
