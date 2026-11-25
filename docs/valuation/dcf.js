/*
The valuation sums, in JavaScript, so the web page's sliders can redo them instantly.
The same sums as growth_path, dcf and solve in valuation.py.
tests/check_dcf.js confirms the two give the same answers.
*/

function growthPath(a) {
  const path = [];
  for (let i = 0; i < a.years; i++) path.push(a.growth_start + (a.terminal_growth - a.growth_start) * i / (a.years - 1));
  return path;
}

// start: the facts (last year's sales, cash, debt, share count, price). a: the assumptions.
function dcf(start, a) {
  const rows = [];
  let revenue = start.revenue;
  growthPath(a).forEach((growth, i) => {
    const year = i + 1, previous = revenue;
    revenue = revenue * (1 + growth);
    const operatingProfit = revenue * a.margin;
    const afterTax = operatingProfit * (1 - a.tax_rate);
    const depreciation = revenue * a.depreciation_share;
    const capex = revenue * a.capex_share;
    const workingCapitalChange = a.working_capital_share * (revenue - previous);
    const freeCashFlow = afterTax + depreciation - capex - workingCapitalChange;
    const discount = 1 / Math.pow(1 + a.wacc, year);
    rows.push({ year: start.fiscal_year + year, growth, revenue, operatingProfit, afterTax, depreciation, capex, workingCapitalChange, freeCashFlow, discount, presentValue: freeCashFlow * discount });
  });
  const last = rows[rows.length - 1];
  const terminalValue = last.freeCashFlow * (1 + a.terminal_growth) / (a.wacc - a.terminal_growth);
  const terminalPresent = terminalValue * last.discount;
  const forecastPresent = rows.reduce((total, r) => total + r.presentValue, 0);
  const enterpriseValue = forecastPresent + terminalPresent;
  const equityValue = enterpriseValue - start.debt + start.cash;
  const valuePerShare = equityValue / start.shares;
  return {
    rows, forecastPresent, terminalValue, terminalPresent, enterpriseValue, equityValue, valuePerShare,
    gap: valuePerShare / start.price - 1,
    terminalShare: terminalPresent / enterpriseValue,
    terminalMultiple: terminalValue / (last.operatingProfit + last.depreciation),
  };
}

// The value of one assumption that makes the model's answer equal the share price, or null if none in range does.
function solve(start, a, name, low, high) {
  const value = x => dcf(start, Object.assign({}, a, { [name]: x })).valuePerShare - start.price;
  if (value(low) * value(high) > 0) return null;
  const rising = value(high) > value(low);
  for (let i = 0; i < 80; i++) {
    const mid = (low + high) / 2;
    if ((value(mid) < 0) === rising) low = mid; else high = mid;
  }
  return (low + high) / 2;
}

if (typeof module !== "undefined") module.exports = { growthPath, dcf, solve };
