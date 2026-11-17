"""Situation-only 4th-down models for the decision calculator.

Model A (coach model):    P(NFL team goes for it | situation), trained on every
                          4th down from 2021-2025 so it reflects today's coaches.
Model B (conversion):     P(convert | went for it, situation), trained on all
                          4th-down attempts 2016-2025.

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
    return fd

def fit(df, target, mono):
    """Pick tree count by game-grouped CV, report CV AUC/Brier, refit on all."""
    X, y = df[FEATS], df[target].values
    oof, iters = np.zeros(len(df)), []
    cons = [mono.get(f, 0) for f in FEATS]
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
    conv = fit(fd[fd.go == 1].reset_index(drop=True), "converted", {"ydstogo": -1})
    coach.booster_.save_model("outputs/coach_go_model.txt")
    conv.booster_.save_model("outputs/situation_conversion_model.txt")

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
    grid["conv"] = conv.predict_proba(grid[FEATS])[:, 1]

    # Historical counts for similar spots (distance bucket x 10-yard zone), all game states
    fd["yd_b"] = fd.ydstogo.clip(upper=15)
    fd["zone"] = ((fd.yardline_100 - 1) // 10).astype(int)
    hist = fd[fd.season >= 2021].groupby(["yd_b", "zone"]).agg(n=("go", "size"), go=("go", "mean")).reset_index()
    hc = fd[fd.go == 1].groupby(["yd_b", "zone"]).agg(att=("converted", "size"), conv=("converted", "mean")).reset_index()
    hist = hist.merge(hc, on=["yd_b", "zone"], how="left").fillna(0)

    out = {"time": [t[0] for t in TIME], "score": [s[0] for s in SCORE],
           "go": {}, "conv": {}, "hist": {}}
    for (t, s), sub in grid.groupby(["t", "s"]):
        key = f"{t}_{s}"
        out["go"][key] = {f"{r.ydstogo}_{r.yardline_100}": int(round(r.go * 1000)) for r in sub.itertuples()}
        out["conv"][key] = {f"{r.ydstogo}_{r.yardline_100}": int(round(r.conv * 1000)) for r in sub.itertuples()}
    # compress: per (t,s) store arrays indexed [yd-1][yl-1], -1 where impossible
    for k in ("go", "conv"):
        for key, d in out[k].items():
            out[k][key] = [[d.get(f"{yd}_{yl}", -1) for yl in YL] for yd in YDS]
    out["hist"] = {f"{int(r.yd_b)}_{int(r.zone)}": [int(r.n), round(r.go, 3), int(r.att), round(r.conv, 3)]
                   for r in hist.itertuples()}
    json.dump(out, open("outputs/calculator_grid.json", "w"), separators=(",", ":"))
    print("grid rows:", len(grid))

if __name__ == "__main__":
    main()
