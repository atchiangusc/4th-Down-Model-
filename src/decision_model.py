"""4th-down decision models that power the calculator (docs/index.html).

Model A (coach):       P(NFL team goes for it | situation, timeouts), 2021-2025
Model B (conversion):  P(convert | went for it, situation, run or pass, offense
                       strength, defense strength, QB rating, weather), 2016-2025
Model C (play call):   P(pass | went for it, situation), 2021-2025
Plus the win probability model and tables from win_prob.py, which turn the
conversion odds into a recommendation: go, punt, or kick.

All four LightGBM models are exported as compact JSON trees so the page can run
them in the browser for any exact input (no lookup grid, no backend).

Also back-tests the recommendation against what coaches actually did."""
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from features import load_plays
from win_prob import decision_values, WP_FEATS

warnings.filterwarnings("ignore")
SIT = ["ydstogo", "yardline_100", "game_seconds_remaining", "half_seconds_remaining", "score_differential"]
TO = ["posteam_timeouts_remaining", "defteam_timeouts_remaining"]
COACH_FEATS = SIT + TO
CONV_FEATS = ["ydstogo", "yardline_capped", "game_seconds_remaining", "half_seconds_remaining",
              "score_differential", "is_pass", "off_rating", "def_rating", "qb_rating", "wind", "outdoors", "temp"]
PASS_FEATS = ["ydstogo", "yardline_capped", "game_seconds_remaining", "half_seconds_remaining",
              "score_differential"] + TO
MONO = {"ydstogo": -1, "off_rating": 1, "def_rating": 1, "qb_rating": 1}
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=16, min_child_samples=40,
              subsample=0.8, subsample_freq=1, reg_lambda=3.0, verbose=-1, random_state=7)


def fit(df, target, feats, mono=MONO):
    X, y = df[feats], df[target].values
    oof, iters = np.zeros(len(df)), []
    cons = [mono.get(f, 0) for f in feats]
    for tr, va in GroupKFold(5).split(X, y, df.game_id):
        m = lgb.LGBMClassifier(n_estimators=3000, monotone_constraints=cons, **PARAMS)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        iters.append(m.best_iteration_)
    n = int(np.mean(iters))
    print(f"  {target:<9} n={len(df):>6,}  CV AUC={roc_auc_score(y, oof):.3f}  "
          f"Brier={brier_score_loss(y, oof):.4f}  trees={n}")
    return lgb.LGBMClassifier(n_estimators=n, monotone_constraints=cons, **PARAMS).fit(X, y)


def export_trees(booster, feats):
    """Flatten LightGBM trees: each node is [feature, threshold, left, right];
    a child index < 0 points at leaf ~index. Leaf values are raw log-odds."""
    dump = booster.dump_model()
    assert dump["feature_names"] == feats
    trees = []
    for t in dump["tree_info"]:
        nodes, leaves = [], []

        def walk(n):
            if "leaf_value" in n:
                leaves.append(round(n["leaf_value"], 6))
                return ~(len(leaves) - 1)
            assert n["decision_type"] == "<="
            i = len(nodes)
            nodes.append(None)
            left, right = walk(n["left_child"]), walk(n["right_child"])
            nodes[i] = [n["split_feature"], float(f"{n['threshold']:.10g}"), left, right]  # keep tiny thresholds like 1e-35
            return i

        root = walk(t["tree_structure"])
        trees.append({"n": nodes, "l": leaves, "r": root})
    return {"features": feats, "trees": trees}


def main():
    p = load_plays()
    fd = p[(p.down == 4) & p.play_type.isin(["run", "pass", "punt", "field_goal"])
           & (p.qb_kneel != 1) & (p.qb_spike != 1) & (p.qtr <= 4)].copy()
    fd["go"] = fd.play_type.isin(["run", "pass"]).astype(int)
    print(f"All 4th downs 2016-2025 (Q1-Q4): {len(fd):,}")
    print("  go-for-it rate by season:", fd.groupby("season").go.mean().round(3).to_dict())

    recent = fd[fd.season >= 2021].reset_index(drop=True)
    # Fake punts / fake FGs are surprise plays, not normal attempts: leave them out
    fakes = fd.is_fake == 1
    print(f"  fake punts/FGs removed from conversion data: {fakes.sum()}")
    att = fd[(fd.go == 1) & ~fakes].reset_index(drop=True)
    coach = fit(recent, "go", COACH_FEATS)
    conv = fit(att, "converted", CONV_FEATS)
    passm = fit(att[att.season >= 2021].reset_index(drop=True), "is_pass", PASS_FEATS, mono={})
    wp = lgb.Booster(model_file="outputs/wp_model.txt")
    for name, m in [("coach_go", coach), ("conversion", conv), ("pass_rate", passm)]:
        m.booster_.save_model(f"outputs/{name}_model.txt")

    # How much do the new features move the odds? 4th & 2 at the opp 40, Q3, tied, pass
    ex = pd.DataFrame([dict(ydstogo=2, yardline_100=40, yardline_capped=40, game_seconds_remaining=1350,
                            half_seconds_remaining=1350, score_differential=0, is_pass=1,
                            off_rating=0.0, def_rating=0.0, qb_rating=0.0, wind=0, outdoors=0, temp=70)])
    teams = fd.drop_duplicates(["game_id", "posteam"])
    pct = {c: teams[c].quantile([0.1, 0.5, 0.9]).values for c in ["off_rating", "def_rating", "qb_rating"]}
    print("  Effect check, 4th & 2 pass at opp 40 (10th / 50th / 90th percentile):")
    for c in pct:
        vals = []
        for v in pct[c]:
            e = ex.copy(); e[c] = v
            vals.append(conv.predict_proba(e[CONV_FEATS])[0, 1])
        print(f"    {c:<11} " + " / ".join(f"{v:.3f}" for v in vals))
    for w in (0, 10, 20):
        e = ex.copy(); e["outdoors"] = 1; e["wind"] = w; e["temp"] = 55
        print(f"    wind {w:>2} mph   {conv.predict_proba(e[CONV_FEATS])[0, 1]:.3f}")

    # ---- Back-test: recommendation vs. what coaches did (2021-2025) ----
    tables = json.load(open("outputs/decision_tables.json"))
    bt = recent[recent.is_fake == 0].copy()
    bt["passrate"] = passm.predict_proba(bt[PASS_FEATS])[:, 1]
    r_, p_ = bt.copy(), bt.copy()
    r_["is_pass"], p_["is_pass"] = 0, 1
    bt["p_conv"] = (bt.passrate * conv.predict_proba(p_[CONV_FEATS])[:, 1]
                    + (1 - bt.passrate) * conv.predict_proba(r_[CONV_FEATS])[:, 1])

    class W:  # tiny wrapper so decision_values can call predict_proba on the WP booster
        def predict_proba(self, X):
            q = wp.predict(X[WP_FEATS]); return np.column_stack([1 - q, q])
    dv = decision_values(W(), tables["fg"], tables["punt_opp_yl"], tables["success_gain"], bt)
    bt = pd.concat([bt.reset_index(drop=True), dv], axis=1)
    bt["wp_go"] = bt.p_conv * bt.wp_succ + (1 - bt.p_conv) * bt.wp_fail
    alt = np.where(bt.fg_ok & bt.punt_ok, np.maximum(bt.wp_fg, bt.wp_punt),
                   np.where(bt.fg_ok, bt.wp_fg, np.where(bt.punt_ok, bt.wp_punt, -1)))
    alt = np.where(alt < 0, np.maximum(bt.wp_fg, bt.wp_punt), alt)
    bt["go_gain"] = bt.wp_go - alt
    # Skip blowouts where nothing matters
    live = bt[(bt.wp_go > 0.05) & (bt.wp_go < 0.95)]
    strong = live[live.go_gain >= 0.02]
    by = strong.groupby("season").go.agg(["mean", "size"])
    print("  When the model says GO by 2+ win-prob points, coaches went for it:")
    print(by.round(3).to_string())
    kick_call = live[live.go_gain <= -0.02]
    print(f"  When the model says KICK/PUNT by 2+ points, coaches went for it {kick_call.go.mean():.1%} "
          f"of {len(kick_call):,} times")
    missed = strong[strong.go == 0]
    print(f"  WP left on the table 2021-25 by not going in clear spots: "
          f"{missed.go_gain.sum():.1f} wins total ({missed.go_gain.sum() / 32 / 5:.2f} per team-season)")
    by.to_csv("outputs/backtest_by_season.csv")
    tm = strong[strong.season == 2025].groupby("posteam").agg(clear_go_spots=("go", "size"), went=("go", "mean"))
    tm.sort_values("went", ascending=False).to_csv("outputs/backtest_2025_by_team.csv")
    print("  2025 most / least aggressive in clear GO spots (min 8):")
    t8 = tm[tm.clear_go_spots >= 8].sort_values("went")
    print(pd.concat([t8.tail(3), t8.head(3)]).round(2).to_string())
    bt[["season", "week", "game_id", "play_id", "posteam", "ydstogo", "yardline_100", "score_differential",
        "game_seconds_remaining", "play_type", "go", "p_conv", "wp_go", "wp_fg", "wp_punt", "go_gain"]].to_csv(
        "outputs/backtest_plays.csv", index=False)

    # ---- Export for the browser ----
    fd["yd_b"] = fd.ydstogo.clip(upper=15)
    fd["zone"] = ((fd.yardline_100 - 1) // 10).astype(int)
    hist = fd[fd.season >= 2021].groupby(["yd_b", "zone"]).agg(n=("go", "size"), go=("go", "mean")).reset_index()
    hc = fd[fd.go == 1].groupby(["yd_b", "zone"]).agg(att=("converted", "size"), conv=("converted", "mean")).reset_index()
    hist = hist.merge(hc, on=["yd_b", "zone"], how="left").fillna(0)
    q = np.linspace(0, 1, 101)
    out = {
        "models": {"coach": export_trees(coach.booster_, COACH_FEATS),
                   "conv": export_trees(conv.booster_, CONV_FEATS),
                   "passrate": export_trees(passm.booster_, PASS_FEATS),
                   "wp": export_trees(wp, WP_FEATS)},
        "tables": tables,
        # Rating value at each percentile 0..100, so the page can use sliders
        "pct": {c: [round(float(v), 4) for v in teams[c].quantile(q).values] for c in pct},
        "n_run": {int(k): int(v) for k, v in att[att.play_type == "run"].ydstogo.clip(upper=15).value_counts().items()},
        "n_pass": {int(k): int(v) for k, v in att[att.play_type == "pass"].ydstogo.clip(upper=15).value_counts().items()},
        "hist": {f"{int(r.yd_b)}_{int(r.zone)}": [int(r.n), round(r.go, 3), int(r.att), round(r.conv, 3)]
                 for r in hist.itertuples()},
    }
    json.dump(out, open("outputs/calculator_models.json", "w"), separators=(",", ":"))

    # Parity cases: the page's JS must reproduce these predictions
    rng = np.random.default_rng(1)
    cases = recent.sample(300, random_state=1).reset_index(drop=True)
    cases["is_pass"] = rng.integers(0, 2, len(cases))
    from win_prob import add_dtr
    wpx = add_dtr(cases.assign(down=4).copy())
    par = {"inputs": cases[sorted(set(COACH_FEATS + CONV_FEATS + PASS_FEATS))].to_dict("records"),
           "wp_inputs": wpx[WP_FEATS].to_dict("records"),
           "coach": coach.predict_proba(cases[COACH_FEATS])[:, 1].tolist(),
           "conv": conv.predict_proba(cases[CONV_FEATS])[:, 1].tolist(),
           "passrate": passm.predict_proba(cases[PASS_FEATS])[:, 1].tolist(),
           "wp": wp.predict(wpx[WP_FEATS]).tolist()}
    json.dump(par, open("outputs/parity_cases.json", "w"))


if __name__ == "__main__":
    main()
