"""Win probability model + the pieces needed to value each 4th-down choice.

- WP model: P(team with the ball wins | score, clock, field position, down,
  distance, timeouts). LightGBM, trained on every regulation snap 2016-2024,
  tested on 2025.
- Field goal model: P(make | kick distance, wind), logistic regression, 2020-2025.
- Punt table: average field position the opponent gets after a punt, by spot.
- Success-gain table: average yards gained on a successful conversion, by distance.

Run directly to train and save everything to outputs/."""
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from features import load_plays

warnings.filterwarnings("ignore")
WP_FEATS = ["score_differential", "diff_time_ratio", "game_seconds_remaining",
            "half_seconds_remaining", "yardline_100", "down", "ydstogo",
            "posteam_timeouts_remaining", "defteam_timeouts_remaining", "posteam_spread", "spread_time"]
WP_MONO = {"posteam_spread": 1, "spread_time": 1, "score_differential": 1, "diff_time_ratio": 1, "yardline_100": -1,
           "posteam_timeouts_remaining": 1, "defteam_timeouts_remaining": -1}
WP_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                 subsample=0.7, subsample_freq=1, colsample_bytree=0.9, reg_lambda=5, verbose=-1,
                 random_state=7)
KICKOFF_YL = 70          # opponent's average start after a kickoff (own 30)
PLAY_SECONDS = 6         # clock used by a 4th-down play


def add_dtr(df):
    """Score lead matters more as time runs out (same idea as nflfastR's model)."""
    elapsed = (3600 - df.game_seconds_remaining.clip(0, 3600)) / 3600
    df["diff_time_ratio"] = df.score_differential * np.exp(4 * elapsed)
    # Vegas spread from the offense's view (+ = favored); its weight fades as the game goes on.
    # Including it stops the model from crediting "having the ball" for being the better team.
    if "posteam_spread" not in df:
        df["posteam_spread"] = np.where(df.posteam == df.home_team, df.spread_line, -df.spread_line)
    df["spread_time"] = df.posteam_spread * np.exp(-4 * elapsed)
    return df


def train_wp(plays):
    d = plays[plays.down.notna() & plays.posteam.notna() & (plays.qtr <= 4) & plays.pos_win.notna()
              & plays.play_type.isin(["run", "pass", "punt", "field_goal", "no_play"])].copy()
    d = add_dtr(d[d.spread_line.notna()].copy())
    # Note: we tried training on drive starts only (to strip "offense is rolling"
    # information from mid-drive snaps), but late-game drive starts are too rare
    # (~30 per game state) and that model was badly overconfident. All snaps it is.
    d["w"] = 1.0
    tr, te = d[d.season <= 2024], d[d.season == 2025]
    cons = [WP_MONO.get(f, 0) for f in WP_FEATS]
    m = lgb.LGBMClassifier(n_estimators=3000, monotone_constraints=cons, **WP_PARAMS)
    va = tr[tr.season == 2024]
    t0 = tr[tr.season <= 2023]
    m.fit(t0[WP_FEATS], t0.pos_win, sample_weight=t0.w, eval_set=[(va[WP_FEATS], va.pos_win)],
          eval_sample_weight=[va.w], callbacks=[lgb.early_stopping(100, verbose=False)])
    n = m.best_iteration_
    m = lgb.LGBMClassifier(n_estimators=n, monotone_constraints=cons, **WP_PARAMS).fit(
        tr[WP_FEATS], tr.pos_win, sample_weight=tr.w)
    p = m.predict_proba(te[WP_FEATS])[:, 1]
    print(f"WP model: {len(tr):,} train snaps, {len(te):,} test snaps (2025), trees={n}")
    print(f"  2025 AUC={roc_auc_score(te.pos_win, p):.3f}  Brier={brier_score_loss(te.pos_win, p):.4f}  "
          f"logloss={log_loss(te.pos_win, p):.4f}")
    late = te.game_seconds_remaining < 300
    print(f"  last 5 min of 2025 games: Brier={brier_score_loss(te.pos_win[late], p[late.values]):.4f}")
    cal = pd.DataFrame({"p": p, "y": te.pos_win.values})
    cal["b"] = pd.cut(cal.p, np.linspace(0, 1, 11), labels=False, include_lowest=True)
    print(cal.groupby("b").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(3).to_string())
    # Final model on all seasons
    m = lgb.LGBMClassifier(n_estimators=n, monotone_constraints=cons, **WP_PARAMS).fit(
        d[WP_FEATS], d.pos_win, sample_weight=d.w)
    return m


def train_fg(plays):
    fg = plays[(plays.play_type == "field_goal") & (plays.season >= 2020) & plays.kick_distance.notna()].copy()
    fg["made"] = (fg.field_goal_result == "made").astype(int)
    X = np.column_stack([fg.kick_distance, fg.wind * fg.outdoors])
    lr = LogisticRegression(C=10, max_iter=1000).fit(X, fg.made)
    coef = {"intercept": float(lr.intercept_[0]), "dist": float(lr.coef_[0][0]),
            "wind": float(lr.coef_[0][1])}
    for dist in (30, 40, 50, 55, 60):
        z = coef["intercept"] + coef["dist"] * dist
        print(f"  FG make prob {dist} yds (calm): {1 / (1 + np.exp(-z)):.3f}")
    print(f"  FG model on {len(fg):,} kicks; wind coef per mph {coef['wind']:.4f}")
    return coef


def punt_table(plays):
    """Opponent's yardline_100 on their next snap after a punt from each spot."""
    p = plays[plays.posteam.notna() & plays.down.notna()].copy()
    p["next_pos"] = p.groupby("game_id").posteam.shift(-1)
    p["next_yl"] = p.groupby("game_id").yardline_100.shift(-1)
    p["next_type"] = p.groupby("game_id").play_type.shift(-1)
    pu = p[(p.play_type == "punt") & (p.next_pos == p.defteam) & (p.next_type != "extra_point")
           & p.next_yl.notna()]
    t = pu.groupby("yardline_100").next_yl.agg(["mean", "size"])
    t = t.reindex(range(1, 100))
    sm = t["mean"].rolling(7, center=True, min_periods=1).mean()
    sm = sm.interpolate().bfill().ffill()
    print(f"  punts used: {len(pu):,}; from own 30 opp starts at {sm[70]:.1f} to go, from opp 40: {sm[40]:.1f}")
    return [round(float(v), 2) for v in sm.values]


def success_gain(plays):
    s = plays[(plays.down == 4) & plays.play_type.isin(["run", "pass"]) & (plays.converted == 1)
              & (plays.touchdown != 1)]
    g = s.groupby(s.ydstogo.clip(upper=15)).yards_gained.mean().reindex(range(1, 16)).interpolate()
    print("  avg gain on successful conversions:", g.round(1).to_dict())
    return [round(float(v), 2) for v in g.values]


def wp_frame(wp_model, sd, gsr, hsr, yl, down, ytg, to_pos, to_def):
    df = pd.DataFrame({"score_differential": sd, "game_seconds_remaining": np.clip(gsr, 0, None),
                       "half_seconds_remaining": np.clip(hsr, 0, None), "yardline_100": np.clip(yl, 1, 99),
                       "down": down, "ydstogo": ytg, "posteam_timeouts_remaining": to_pos,
                       "defteam_timeouts_remaining": to_def, "posteam_spread": 0.0})  # even matchup
    df = add_dtr(df)
    return wp_model.predict_proba(df[WP_FEATS])[:, 1]


def decision_values(wp_model, fg_coef, punt_yl, gain, s):
    """Vectorized: WP of each choice for states s (DataFrame with ydstogo, yardline_100,
    game_seconds_remaining, half_seconds_remaining, score_differential, timeouts, wind, outdoors).
    Mirrors the JavaScript in docs/index.html exactly."""
    yd, yl = s.ydstogo.values.astype(int), s.yardline_100.values.astype(int)
    sd, g, h = s.score_differential.values, s.game_seconds_remaining.values, s.half_seconds_remaining.values
    to, tdf = s.posteam_timeouts_remaining.values, s.defteam_timeouts_remaining.values
    g2, h2 = g - PLAY_SECONDS, h - PLAY_SECONDS
    ones, tens = np.ones(len(s)), np.full(len(s), 10)
    gain_v = np.array(gain)[np.clip(yd, 1, 15) - 1]
    new_yl = np.where(yd >= yl, 0, np.round(yl - gain_v))
    td = new_yl <= 0
    wp_td = 1 - wp_frame(wp_model, -(sd + 7), g2, h2, KICKOFF_YL * ones, ones, tens, tdf, to)
    wp_first = wp_frame(wp_model, sd, g2, h2, np.clip(new_yl, 1, 99), ones, np.minimum(10, np.clip(new_yl, 1, 99)), to, tdf)
    wp_succ = np.where(td, wp_td, wp_first)
    wp_fail = 1 - wp_frame(wp_model, -sd, g2, h2, 100 - yl, ones, tens, tdf, to)
    dist = yl + 18
    z = fg_coef["intercept"] + fg_coef["dist"] * dist + fg_coef["wind"] * s.wind.values * s.outdoors.values
    p_fg = 1 / (1 + np.exp(-z))
    wp_make = 1 - wp_frame(wp_model, -(sd + 3), g2, h2, KICKOFF_YL * ones, ones, tens, tdf, to)
    wp_miss = 1 - wp_frame(wp_model, -sd, g2, h2, np.minimum(80, 93 - yl), ones, tens, tdf, to)
    wp_fg = p_fg * wp_make + (1 - p_fg) * wp_miss
    opp = np.array(punt_yl)[np.clip(yl, 1, 99) - 1]
    wp_punt = 1 - wp_frame(wp_model, -sd, g - 8, h - 8, np.round(opp), ones, tens, tdf, to)
    return pd.DataFrame({"wp_succ": wp_succ, "wp_fail": wp_fail, "p_fg": p_fg, "wp_fg": wp_fg,
                         "wp_punt": wp_punt, "fg_ok": dist <= 66, "punt_ok": yl >= 30})


def main():
    plays = load_plays()
    wp = train_wp(plays)
    wp.booster_.save_model("outputs/wp_model.txt")
    fg = train_fg(plays)
    pt = punt_table(plays)
    gn = success_gain(plays)
    json.dump({"fg": fg, "punt_opp_yl": pt, "success_gain": gn, "kickoff_yl": KICKOFF_YL,
               "play_seconds": PLAY_SECONDS}, open("outputs/decision_tables.json", "w"), indent=1)


if __name__ == "__main__":
    main()
