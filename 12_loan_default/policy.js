/*
The lending sums for the web page. The same sums as loan.py, so the page's sliders
give the same answers as the Python analysis. tests/check_loan.js confirms they match.
*/

// Make every customer riskier so that the average rises by the multiplier, keeping their order.
function stress(risk, multiplier) {
  const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
  const target = Math.min(mean(risk) * multiplier, 0.99);
  const logOdds = risk.map(p => Math.log(p / (1 - p)));
  let low = 0, high = 10;
  for (let i = 0; i < 60; i++) {
    const middle = (low + high) / 2;
    if (mean(logOdds.map(z => 1 / (1 + Math.exp(-(z + middle))))) < target) low = middle; else high = middle;
  }
  const shift = (low + high) / 2;
  return logOdds.map(z => 1 / (1 + Math.exp(-(z + shift))));
}

// Lend to everyone whose score is below the cutoff. "outcome" is what really happened (0 or 1), or a risk for what to expect.
function lend(customers, outcome, cutoff, margin, lossIfDefault) {
  let approved = 0, profit = 0, lent = 0, defaults = 0;
  for (let i = 0; i < customers.risk.length; i++) {
    if (customers.risk[i] < cutoff) {
      approved++;
      lent += customers.balance[i];
      defaults += outcome[i];
      profit += customers.balance[i] * ((1 - outcome[i]) * margin - outcome[i] * lossIfDefault);
    }
  }
  return { approved: approved / customers.risk.length, profit, lent, defaultRate: approved ? defaults / approved : 0 };
}

const breakEven = (margin, lossIfDefault) => margin / (margin + lossIfDefault);

// In a recession the break-even rule is applied to the stressed risks. This returns the matching cutoff on today's score.
function fallbackCutoff(risk, stressed, line) {
  let cutoff = 1;  // the safest customer who no longer clears the line
  for (let i = 0; i < risk.length; i++) if (stressed[i] >= line && risk[i] < cutoff) cutoff = risk[i];
  return cutoff;
}

if (typeof module !== "undefined") module.exports = { stress, lend, breakEven, fallbackCutoff };
