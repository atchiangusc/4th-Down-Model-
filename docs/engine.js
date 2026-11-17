// 4th-down decision engine: runs the exported LightGBM models in the browser.
// Mirrors src/win_prob.py (decision_values) and src/decision_model.py exactly;
// src/test_parity.js checks this file against the Python predictions.
(function (root) {
  'use strict';

  // Sum raw log-odds over trees, then apply the logistic function.
  function predict(model, row) {
    const x = model.features.map(f => row[f]);
    let z = 0;
    for (const t of model.trees) {
      let node = t.r;
      while (node >= 0) {
        const n = t.n[node];
        node = x[n[0]] <= n[1] ? n[2] : n[3];
      }
      z += t.l[~node];
    }
    return 1 / (1 + Math.exp(-z));
  }

  function wpRow(sd, gsr, hsr, yl, down, ytg, toPos, toDef, spread) {
    gsr = Math.max(gsr, 0); hsr = Math.max(hsr, 0);
    yl = Math.min(Math.max(yl, 1), 99);
    const elapsed = (3600 - Math.min(gsr, 3600)) / 3600;
    return {
      score_differential: sd, diff_time_ratio: sd * Math.exp(4 * elapsed),
      game_seconds_remaining: gsr, half_seconds_remaining: hsr, yardline_100: yl,
      down: down, ydstogo: ytg, posteam_timeouts_remaining: toPos,
      defteam_timeouts_remaining: toDef, posteam_spread: spread,
      spread_time: spread * Math.exp(-4 * elapsed)
    };
  }

  // s: {ydstogo, yardline_100, game_seconds_remaining, half_seconds_remaining,
  //     score_differential, posteam_timeouts_remaining, defteam_timeouts_remaining,
  //     off_rating, def_rating, qb_rating, wind, outdoors, temp, spread}
  function evaluate(M, s) {
    const T = M.tables, wp = M.models.wp;
    const yd = s.ydstogo, yl = s.yardline_100, sd = s.score_differential;
    const g = s.game_seconds_remaining, h = s.half_seconds_remaining;
    const to = s.posteam_timeouts_remaining, tdf = s.defteam_timeouts_remaining;
    const sp = s.spread || 0;
    const g2 = g - T.play_seconds, h2 = h - T.play_seconds;
    const W = (sd_, gg, hh, yy, ytg, a, b, spr) => predict(wp, wpRow(sd_, gg, hh, yy, 1, ytg, a, b, spr));

    // Conversion odds by play type, and how often coaches pass here
    const base = Object.assign({}, s, { yardline_capped: Math.min(yl, 40) });
    const pRun = predict(M.models.conv, Object.assign({}, base, { is_pass: 0 }));
    const pPass = predict(M.models.conv, Object.assign({}, base, { is_pass: 1 }));
    const passRate = predict(M.models.passrate, base);
    const pConv = passRate * pPass + (1 - passRate) * pRun;
    const coachGo = predict(M.models.coach, s);

    // Go for it: success -> 1st down (or TD), failure -> opponent's ball at the spot
    const gain = T.success_gain[Math.min(Math.max(yd, 1), 15) - 1];
    const newYl = yd >= yl ? 0 : Math.round(yl - gain);
    let wpSucc;
    if (newYl <= 0) wpSucc = 1 - W(-(sd + 7), g2, h2, T.kickoff_yl, 10, tdf, to, -sp);
    else { const y = Math.min(Math.max(newYl, 1), 99); wpSucc = W(sd, g2, h2, y, Math.min(10, y), to, tdf, sp); }
    const wpFail = 1 - W(-sd, g2, h2, 100 - yl, 10, tdf, to, -sp);

    // Field goal
    const dist = yl + 18;
    const z = T.fg.intercept + T.fg.dist * dist + T.fg.wind * s.wind * s.outdoors;
    const pFg = 1 / (1 + Math.exp(-z));
    const wpMake = 1 - W(-(sd + 3), g2, h2, T.kickoff_yl, 10, tdf, to, -sp);
    const wpMiss = 1 - W(-sd, g2, h2, Math.min(80, 93 - yl), 10, tdf, to, -sp);
    const wpFg = pFg * wpMake + (1 - pFg) * wpMiss;

    // Punt
    const opp = Math.round(T.punt_opp_yl[Math.min(Math.max(yl, 1), 99) - 1]);
    const wpPunt = 1 - W(-sd, g - 8, h - 8, opp, 10, tdf, to, -sp);

    const wpGo = pConv * wpSucc + (1 - pConv) * wpFail;
    const options = [{ k: 'go', wp: wpGo, ok: true }, { k: 'fg', wp: wpFg, ok: dist <= 66 },
                     { k: 'punt', wp: wpPunt, ok: yl >= 30 }];
    const avail = options.filter(o => o.ok);
    avail.sort((a, b) => b.wp - a.wp);
    const bestAlt = Math.max(...avail.filter(o => o.k !== 'go').map(o => o.wp), -1);
    const breakeven = bestAlt < 0 ? 0 : Math.min(Math.max((bestAlt - wpFail) / (wpSucc - wpFail), 0), 1);
    return { pRun, pPass, passRate, pConv, coachGo, wpGo, wpFg, wpPunt, pFg, dist, wpSucc, wpFail,
             options, best: avail[0].k, margin: avail.length > 1 ? avail[0].wp - avail[1].wp : 1, breakeven };
  }

  const api = { predict, evaluate, wpRow };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FourthDown = api;
})(this);
