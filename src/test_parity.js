// Checks that docs/engine.js reproduces the Python models' predictions.
// Run: python src/decision_model.py && node src/test_parity.js
const fs = require('fs');
const E = require('../docs/engine.js');
const M = JSON.parse(fs.readFileSync(__dirname + '/../outputs/calculator_models.json'));
const P = JSON.parse(fs.readFileSync(__dirname + '/../outputs/parity_cases.json'));
let worst = 0;
for (const [name, key, rows] of [['coach', 'coach', P.inputs], ['conv', 'conv', P.inputs],
                                 ['passrate', 'passrate', P.inputs], ['wp', 'wp', P.wp_inputs]]) {
  let m = 0;
  rows.forEach((r, i) => { m = Math.max(m, Math.abs(E.predict(M.models[key], r) - P[name][i])); });
  console.log(`${name.padEnd(9)} max abs diff vs Python: ${m.toExponential(2)}`);
  worst = Math.max(worst, m);
}
if (worst > 1e-4) { console.error('PARITY FAILED'); process.exit(1); }
console.log('parity OK');
