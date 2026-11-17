"""Situation-only 4th-down models for the decision calculator.

Model A (coach model):    P(NFL team goes for it | situation), trained on every
                          4th down from 2021-2025 so it reflects today's coaches.
Model B (conversion):     P(convert | went for it, situation, run or pass),
                          trained on all 4th-down attempts 2016-2025.
Model C (play call):      P(pass | went for it, situation), 2021-2025, i.e. how
                          often today's coaches throw when they go for it.

Inputs are only what you'd know before the play: yards to go, yard line,
quarter + clock, score differential. (The personnel model in train.py needs
the alignment at the snap, which a coach doesn't have when deciding.)"""
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import nflreadpy as nfl
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
FEATS = ["ydstogo", "yardline_100", "game_seconds_remaining", "half_seconds_remaining",
         "score_differential"]
CONV_FEATS = FEATS + ["is_pass"]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=16, min_child_samples=40,
              subsample=0.8, subsample_freq=1, reg_lambda=3.0, verbose=-1, random_state=7)

def load():
    cols = ["game_id", "play_id", "season", "season_type", "down", "ydstogo", "yardline_100",
            "play_type", "qb_kneel", "qb_spike", "fourth_down_converted",
            "game_seconds_remaining", "half_seconds_remaining", "score_differential", "qtr"]
    pbp = nfl.load_pbp(list(range(2016, 2026))).select(cols).to_pandas()
    fd = pbp[(pbp.down == 4) & pbp.play_type.isin(["run", "pass", "punt", "field_goal"])
             & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1) & (pbp.qtr <= 4)].copy()
    fd["go"] = fd.play_type.isin(["run", "pass"]).astype(int)
    fd["converted"] = (fd.fourth_down_converted == 1).astype(int)
    fd["is_pass"] = (fd.play_type == "pass").astype(int)
    return fd

def fit(df, target, mono, feats=FEATS):
    """Pick tree count by game-grouped CV, report CV AUC/Brier, refit on all."""
    X, y = df[feats], df[target].values
    oof, iters = np.zeros(len(df)), []
    cons = [mono.get(f, 0) for f in feats]
    for tr, va in GroupKFold(5).split(X, y, df.game_id):
        m = lgb.LGBMClassifier(n_estimators=3000, monotone_constraints=cons, **PARAMS)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]; iters.append(m.best_iteration_)
    print(f"  {target}: n={len(df)}  CV AUC={roc_auc_score(y, oof):.3f}  "
          f"Brier={brier_score_loss(y, oof):.4f}  trees={int(np.mean(iters))}")
    m = lgb.LGBMClassifier(n_estimators=int(np.mean(iters)), monotone_constraints=cons, **PARAMS)
    return m.fit(X, y)

def main():
    fd = load()
    print(f"All 4th downs 2016-2025 (reg+post, Q1-Q4): {len(fd)}")
    print(fd.groupby("season").go.mean().round(3).to_string())

    coach = fit(fd[fd.season >= 2021].reset_index(drop=True), "go", {"ydstogo": -1})
    att = fd[fd.go == 1].reset_index(drop=True)
    print("  (conversion without run/pass, for comparison)")
    fit(att, "converted", {"ydstogo": -1})
    conv = fit(att, "converted", {"ydstogo": -1}, CONV_FEATS)
    passm = fit(att[att.season >= 2021].reset_index(drop=True), "is_pass", {}, FEATS)
    coach.booster_.save_model("outputs/coach_go_model.txt")
    conv.booster_.save_model("outputs/situation_conversion_model.txt")
    passm.booster_.save_model("outputs/coach_pass_rate_model.txt")

    # Prediction grid for the web calculator (precomputed so the page has no backend)
    YDS = list(range(1, 16))                       # 15 = "15+"
    YL = list(range(1, 100))
    # (label, quarter, clock seconds left in quarter) representative points
    TIME = [("Q1", 1, 450), ("Q2", 2, 450), ("Q2 <2:00", 2, 60), ("Q3", 3, 450),
            ("Q4 10:00+", 4, 750), ("Q4 5-10:00", 4, 450), ("Q4 2-5:00", 4, 210), ("Q4 <2:00", 4, 60)]
    SCORE = [("Down 9+", -12), ("Down 4-8", -6), ("Down 1-3", -2), ("Tied", 0),
             ("Up 1-3", 2), ("Up 4-8", 6), ("Up 9+", 12)]
    rows = []
    for t_i, (_, q, clk) in enumerate(TIME):
        g = (4 - q) * 900 + clk
        h = ((2 - q) * 900 + clk) if q <= 2 else ((4 - q) * 900 + clk)
        for s_i, (_, sd) in enumerate(SCORE):
            for yd in YDS:
                for yl in YL:
                    if yd > yl:            # can't need more yards than the end zone
                        continue
                    rows.append((t_i, s_i, yd, yl, g, h, sd))
    grid = pd.DataFrame(rows, columns=["t", "s", "ydstogo", "yardline_100",
                                       "game_seconds_remaining", "half_seconds_remaining", "score_differential"])
    grid["go"] = coach.predict_proba(grid[FEATS])[:, 1]
    grid["is_pass"] = 0
    grid["run"] = conv.predict_proba(grid[CONV_FEATS])[:, 1]
    grid["is_pass"] = 1
    grid["pass"] = conv.predict_proba(grid[CONV_FEATS])[:, 1]
    grid["passrate"] = passm.predict_proba(grid[FEATS])[:, 1]
    # Blended conversion = what actually happens given today's run/pass mix
    grid["conv"] = grid.passrate * grid["pass"] + (1 - grid.passrate) * grid.run

    # Historical counts for similar spots (distance bucket x 10-yard zone), all game states
    fd["yd_b"] = fd.ydstogo.clip(upper=15)
    fd["zone"] = ((fd.yardline_100 - 1) // 10).astype(int)
    hist = fd[fd.season >= 2021].groupby(["yd_b", "zone"]).agg(n=("go", "size"), go=("go", "mean")).reset_index()
    hc = fd[fd.go == 1].groupby(["yd_b", "zone"]).agg(att=("converted", "size"), conv=("converted", "mean")).reset_index()
    hist = hist.merge(hc, on=["yd_b", "zone"], how="left").fillna(0)

    # Sample sizes by distance and play type, so the page can flag thin estimates
    runs_by_yd = att[att.play_type == "run"].ydstogo.clip(upper=15).value_counts().to_dict()
    pass_by_yd = att[att.play_type == "pass"].ydstogo.clip(upper=15).value_counts().to_dict()
    out = {"time": [t[0] for t in TIME], "score": [s[0] for s in SCORE],
           "n_run": {int(k): int(v) for k, v in runs_by_yd.items()},
           "n_pass": {int(k): int(v) for k, v in pass_by_yd.items()}}
    # Stored as whole percents in arrays indexed [ydstogo-1][yardline-1]; -1 = impossible spot
    for k in ("go", "run", "pass", "passrate", "conv"):
        out[k] = {}
        for (t, s_), sub in grid.groupby(["t", "s"]):
            arr = np.full((len(YDS), len(YL)), -1, dtype=int)
            arr[sub.ydstogo.values - 1, sub.yardline_100.values - 1] = np.round(sub[k].values * 100).astype(int)
            out[k][f"{t}_{s_}"] = arr.tolist()
    out["hist"] = {f"{int(r.yd_b)}_{int(r.zone)}": [int(r.n), round(r.go, 3), int(r.att), round(r.conv, 3)]
                   for r in hist.itertuples()}
    json.dump(out, open("outputs/calculator_grid.json", "w"), separators=(",", ":"))
    print("grid rows:", len(grid))

if __name__ == "__main__":
    main()
