/*
The site-24 investment sums, in JavaScript, so the dashboard sliders can redo them instantly.
The same sums as site24_cash_flows, npv, irr and payback_years in finance.py.
tests/check_site24.js confirms the two give the same answers.
*/

function npv(rate, flows) {
  return flows.reduce((total, cash, year) => total + cash / Math.pow(1 + rate, year), 0);
}

function irr(flows) {
  let low = -0.99, high = 1.0;
  if (npv(low, flows) < 0 || npv(high, flows) > 0) return NaN;
  for (let i = 0; i < 100; i++) {
    const mid = (low + high) / 2;
    if (npv(mid, flows) > 0) low = mid; else high = mid;
  }
  return (low + high) / 2;
}

function paybackYears(flows) {
  let total = flows[0];
  for (let year = 1; year < flows.length; year++) {
    if (total + flows[year] >= 0) return year - 1 + (-total / flows[year]);
    total += flows[year];
  }
  return null;
}

// cars: washes a year once full. econ and a: the "economics" and "assumptions" blocks in results.json.
function site24(cars, rentPerMonth, buildCost, econ, a) {
  const flows = [-buildCost], profits = [];
  for (let year = 1; year <= a.years; year++) {
    const fill = a.ramp[Math.min(year, a.ramp.length) - 1];
    const growth = Math.pow(1 + a.growth, year - 1);
    const revenue = cars * fill * econ.revenue_per_car * growth;
    const costs = (cars * fill * econ.cost_per_car + econ.fixed_cost_per_year + rentPerMonth * 12) * growth + revenue * econ.card_fee_share;
    const profit = revenue - costs - (year === 1 ? a.opening_cost : 0);
    profits.push(profit);
    flows.push(profit - a.upkeep_share * revenue);
  }
  const withoutSale = flows.slice();
  flows[flows.length - 1] += a.exit_multiple * profits[profits.length - 1];
  return {
    flows, profits,
    npv: npv(a.discount_rate, flows),
    irr: irr(flows),
    paybackYears: paybackYears(withoutSale),
    matureProfit: profits[2],
  };
}

if (typeof module !== "undefined") module.exports = { npv, irr, paybackYears, site24 };
