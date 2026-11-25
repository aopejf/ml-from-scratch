/*
The option sums for the web page: the Black-Scholes price, the five Greeks and the tree.
A browser has no autograd engine, so the Greeks here are the textbook formulas.
tests/check_options.js confirms they match what the Python engine works out.
*/

// The share of a bell curve lying below x (Graeme West's double-precision routine).
function bellArea(x) {
  const a = Math.abs(x);
  let c = 0;
  if (a <= 37) {
    const e = Math.exp(-a * a / 2);
    if (a < 7.07106781186547) {
      let top = 3.52624965998911e-2 * a + 0.700383064443688;
      top = top * a + 6.37396220353165; top = top * a + 33.912866078383; top = top * a + 112.079291497871;
      top = top * a + 221.213596169931; top = top * a + 220.206867912376;
      let bottom = 8.83883476483184e-2 * a + 1.75566716318264;
      bottom = bottom * a + 16.064177579207; bottom = bottom * a + 86.7807322029461; bottom = bottom * a + 296.564248779674;
      bottom = bottom * a + 637.333633378831; bottom = bottom * a + 793.826512519948; bottom = bottom * a + 440.413735824752;
      c = e * top / bottom;
    } else {
      let b = a + 0.65; b = a + 4 / b; b = a + 3 / b; b = a + 2 / b; b = a + 1 / b;
      c = e / b / 2.506628274631;
    }
  }
  return x > 0 ? 1 - c : c;
}
const bellHeight = z => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

// o = { price, strike, years, rate, volatility, dividend, kind: "call" | "put" }
function option(o) {
  const swing = o.volatility * Math.sqrt(o.years);
  const d1 = (Math.log(o.price / o.strike) + (o.rate - o.dividend + 0.5 * o.volatility ** 2) * o.years) / swing;
  const d2 = d1 - swing;
  const carry = Math.exp(-o.dividend * o.years), discount = Math.exp(-o.rate * o.years);
  const sign = o.kind === "call" ? 1 : -1;
  const value = sign * (o.price * carry * bellArea(sign * d1) - o.strike * discount * bellArea(sign * d2));
  const theta = -o.price * carry * bellHeight(d1) * o.volatility / (2 * Math.sqrt(o.years))
    - sign * o.rate * o.strike * discount * bellArea(sign * d2) + sign * o.dividend * o.price * carry * bellArea(sign * d1);
  return {
    value,
    delta: sign * carry * bellArea(sign * d1),                        // per $1 on the share
    gamma: carry * bellHeight(d1) / (o.price * swing),               // change in delta per $1
    vega: o.price * carry * bellHeight(d1) * Math.sqrt(o.years) / 100, // per 1 point of volatility
    theta: theta / 365,                                                // per day
    rho: sign * o.strike * o.years * discount * bellArea(sign * d2) / 100, // per 1 point of interest
  };
}

// A tree of up and down moves. american: true lets the holder use the option early.
function tree(o, steps, american) {
  const dt = o.years / steps, up = Math.exp(o.volatility * Math.sqrt(dt)), down = 1 / up;
  const chanceUp = (Math.exp((o.rate - o.dividend) * dt) - down) / (up - down), discount = Math.exp(-o.rate * dt);
  const sign = o.kind === "call" ? 1 : -1;
  const worth = new Float64Array(steps + 1);
  for (let i = 0; i <= steps; i++) worth[i] = Math.max(sign * (o.price * up ** i * down ** (steps - i) - o.strike), 0);
  for (let step = steps - 1; step >= 0; step--)
    for (let i = 0; i <= step; i++) {
      worth[i] = discount * (chanceUp * worth[i + 1] + (1 - chanceUp) * worth[i]);
      if (american) worth[i] = Math.max(worth[i], sign * (o.price * up ** i * down ** (step - i) - o.strike));
    }
  return worth[0];
}

// The volatility that makes the formula match a price, by halving the range.
function impliedVolatility(target, o) {
  let low = 0.001, high = 5;
  if (!(option({ ...o, volatility: low }).value < target && target <= option({ ...o, volatility: high }).value)) return NaN;
  for (let i = 0; i < 100; i++) {
    const middle = (low + high) / 2;
    if (option({ ...o, volatility: middle }).value < target) low = middle; else high = middle;
  }
  return (low + high) / 2;
}

if (typeof module !== "undefined") module.exports = { bellArea, option, tree, impliedVolatility };
