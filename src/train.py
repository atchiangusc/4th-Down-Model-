"""Train and evaluate 4th-down conversion models.
Time split: train 2016-2023, test 2024-2025 (out-of-time, like real deployment).
Ablation: does knowing the defense (box count, DL/LB/DB) add predictive value
beyond distance, field position and game context?"""
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

df = pd.read_parquet("data/fourth_downs.parquet")
df["log_ydstogo"] = np.log1p(df.ydstogo)
df["goal_to_go"] = (df.ydstogo >= df.yardline_100).astype(int)
df["offense_formation"] = df.offense_formation.fillna("UNKNOWN").astype("category")

DISTANCE = ["ydstogo", "log_ydstogo"]
CONTEXT = DISTANCE + ["yardline_100", "goal_to_go", "is_pass", "shotgun", "no_huddle",
                      "score_differential", "game_seconds_remaining", "wp",
                      "home", "outdoors", "spread_line"]
OFFENSE = ["off_OL", "off_RB", "off_TE", "off_WR", "offense_formation"]
# def_LB is omitted because DL + LB + DB = 11 (it is implied)
DEFENSE = ["defenders_in_box", "def_DL", "def_DB", "box_minus_blockers"]
DEFENSE_LOGIT = ["defenders_in_box", "def_DL", "def_DB"]  # box_minus_blockers is a linear combo

train = df[df.season <= 2023].reset_index(drop=True)
test = df[df.season >= 2024].reset_index(drop=True)
y_tr, y_te = train.converted, test.converted
base_rate = y_tr.mean()

def metrics(y, p):
    return {"auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, p),
            "brier_skill_vs_base": 1 - brier_score_loss(y, p) / brier_score_loss(y, np.full(len(y), base_rate))}

LGB_PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=8,
                  min_child_samples=60, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=7)

def fit_lgb(feats):
    # Pick n_estimators with game-grouped CV on training years only
    mono = [-1 if f in ("ydstogo", "log_ydstogo") else 0 for f in feats]
    best = []
    for tr_i, va_i in GroupKFold(5).split(train, y_tr, train.game_id):
        m = lgb.LGBMClassifier(n_estimators=2000, monotone_constraints=mono, **LGB_PARAMS)
        m.fit(train.loc[tr_i, feats], y_tr[tr_i], eval_set=[(train.loc[va_i, feats], y_tr[va_i])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        best.append(m.best_iteration_)
    n = int(np.mean(best))
    m = lgb.LGBMClassifier(n_estimators=n, monotone_constraints=mono, **LGB_PARAMS)
    m.fit(train[feats], y_tr)
    return m

def fit_logit(feats):
    X_tr = pd.get_dummies(train[feats], drop_first=True, dtype=float)
    X_te = pd.get_dummies(test[feats], drop_first=True, dtype=float).reindex(columns=X_tr.columns, fill_value=0)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1], m, X_tr.columns

results, preds = {}, {}
p, _, _ = fit_logit(DISTANCE); results["1. Logistic: distance only"] = metrics(y_te, p); preds["distance"] = p
p, _, _ = fit_logit(CONTEXT + OFFENSE); results["2. Logistic: context + offense"] = metrics(y_te, p)
p, logit_full, logit_cols = fit_logit(CONTEXT + OFFENSE + DEFENSE_LOGIT)
results["3. Logistic: + defense"] = metrics(y_te, p); preds["logit_full"] = p
m_ctx = fit_lgb(CONTEXT + OFFENSE)
p = m_ctx.predict_proba(test[CONTEXT + OFFENSE])[:, 1]
results["4. LightGBM: context + offense"] = metrics(y_te, p); preds["lgb_ctx"] = p
FULL = CONTEXT + OFFENSE + DEFENSE
m_full = fit_lgb(FULL)
p = m_full.predict_proba(test[FULL])[:, 1]
results["5. LightGBM: + defense (full)"] = metrics(y_te, p); preds["lgb_full"] = p

res = pd.DataFrame(results).T.round(4)
print(f"Train plays {len(train)}, test plays {len(test)}, test conv rate {y_te.mean():.3f}\n")
print(res.to_string())
res.to_csv("outputs/model_comparison.csv")

# Bootstrap the AUC gain from adding defense (resample test games)
rng = np.random.default_rng(0)
games = test.game_id.unique()
idx_by_game = test.groupby("game_id").indices
gains = []
for _ in range(1000):
    g = rng.choice(games, len(games), replace=True)
    ix = np.concatenate([idx_by_game[k] for k in g])
    gains.append(roc_auc_score(y_te[ix], preds["lgb_full"][ix]) - roc_auc_score(y_te[ix], preds["lgb_ctx"][ix]))
lo, hi = np.percentile(gains, [2.5, 97.5])
print(f"\nAUC gain from defensive features: {np.mean(gains):+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")

# Feature importance via SHAP values (LightGBM built-in TreeSHAP)
contrib = m_full.predict(test[FULL], pred_contrib=True)[:, :-1]
imp = pd.Series(np.abs(contrib).mean(0), index=FULL).sort_values(ascending=False)
imp.to_csv("outputs/shap_importance.csv", header=["mean_abs_shap"])
print("\nMean |SHAP| (log-odds):\n", imp.round(4).to_string())

# Logistic coefficients for defensive features (interpretable direction)
coefs = pd.Series(logit_full[-1].coef_[0], index=logit_cols)
print("\nStandardized logistic coefs (defense):\n", coefs[DEFENSE_LOGIT].round(3).to_string())

test_out = test[["season", "week", "game_id", "play_id", "posteam", "defteam", "ydstogo",
                 "yardline_100", "is_pass", "defenders_in_box", "def_DL", "converted"]].copy()
test_out["p_convert"] = preds["lgb_full"]
test_out["p_distance_only"] = preds["distance"]
test_out.to_csv("outputs/test_predictions.csv", index=False)
m_full.booster_.save_model("outputs/lgb_full_model.txt")
json.dump({"features": FULL, "gain_ci": [float(lo), float(hi)], "gain_mean": float(np.mean(gains))},
          open("outputs/run_summary.json", "w"), indent=2)
