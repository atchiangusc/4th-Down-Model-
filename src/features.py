"""Shared play-level data + engineered features (team strength, QB, weather).

Every rating uses only games played BEFORE the current one, so nothing a team
does in today's game leaks into today's features.

Ratings are decayed, shrunk averages of EPA (expected points added) per play:
    rating = (decayed sum of EPA + K * prior) / (decayed play count + K)
- Decay: each older game counts DECAY times as much as the next one (0.9 => a
  game ~7 games ago counts half as much). Carries across seasons.
- Shrinkage (K pseudo-plays at the prior) keeps small samples from producing
  extreme ratings, e.g. a backup QB after 12 dropbacks.

Run directly to build and cache data/plays.parquet."""
import os
import numpy as np
import pandas as pd
import nflreadpy as nfl

SEASONS = list(range(2016, 2026))
CACHE = "data/plays.parquet"
DECAY = 0.9
TEAM_K, TEAM_PRIOR = 250, 0.0      # league-average prior for teams
QB_K, QB_PRIOR = 150, -0.10        # unknown QBs start below average (backups)

COLS = ["game_id", "play_id", "season", "week", "season_type", "game_date",
        "home_team", "away_team", "posteam", "defteam", "posteam_type",
        "down", "ydstogo", "yardline_100", "play_type", "qb_kneel", "qb_spike",
        "qb_dropback", "qb_scramble", "pass", "rush", "epa", "yards_gained", "touchdown",
        "passer_player_id", "passer_player_name", "fourth_down_converted",
        "game_seconds_remaining", "half_seconds_remaining", "qtr",
        "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "result", "temp", "wind", "roof", "spread_line",
        "field_goal_result", "kick_distance", "return_yards", "touchback", "punt_blocked",
        "desc"]


def decayed_rating(df, key, k, prior):
    """df: one row per (key, game) in time order with columns s (EPA sum), n (plays).
    Returns the pre-game rating for each row."""
    out = np.empty(len(df))
    for _, idx in df.groupby(key, sort=False).indices.items():
        S = N = 0.0
        for i in idx:                       # idx is already in time order
            out[i] = (S + k * prior) / (N + k)
            S = DECAY * S + df.s.iat[i]
            N = DECAY * N + df.n.iat[i]
    return out


def build():
    pbp = nfl.load_pbp(SEASONS).select(COLS).to_pandas()
    pbp["game_date"] = pd.to_datetime(pbp.game_date)
    pbp = pbp.sort_values(["game_date", "game_id", "play_id"]).reset_index(drop=True)

    # --- Team offense / defense strength (EPA per run or pass play) ---
    plays = pbp[pbp.play_type.isin(["run", "pass"]) & pbp.epa.notna()]
    games = plays.groupby(["game_id", "game_date", "posteam", "defteam"]).epa.agg(s="sum", n="size").reset_index()
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    games["off_rating"] = decayed_rating(games.rename(columns={"posteam": "team"}), "team", TEAM_K, TEAM_PRIOR)
    games["def_rating"] = decayed_rating(games.rename(columns={"defteam": "team"}), "team", TEAM_K, TEAM_PRIOR)
    # def_rating = EPA per play the defense has ALLOWED: higher is a worse defense

    # --- Quarterback: EPA per dropback, for the team's primary QB in this game ---
    db = pbp[(pbp.qb_dropback == 1) & pbp.epa.notna() & pbp.passer_player_id.notna()]
    qbg = db.groupby(["game_id", "game_date", "posteam", "passer_player_id", "passer_player_name"]).epa.agg(
        s="sum", n="size").reset_index().sort_values(["game_date", "game_id"]).reset_index(drop=True)
    qbg["qb_rating"] = decayed_rating(qbg, "passer_player_id", QB_K, QB_PRIOR)
    primary = qbg.sort_values("n", ascending=False).drop_duplicates(["game_id", "posteam"])
    primary = primary[["game_id", "posteam", "passer_player_name", "qb_rating"]].rename(
        columns={"passer_player_name": "qb_name"})

    pbp = pbp.merge(games[["game_id", "posteam", "off_rating"]], on=["game_id", "posteam"], how="left")
    dmap = games[["game_id", "defteam", "def_rating"]]
    pbp = pbp.merge(dmap, on=["game_id", "defteam"], how="left")
    pbp = pbp.merge(primary, on=["game_id", "posteam"], how="left")
    pbp["qb_rating"] = pbp.qb_rating.fillna(QB_PRIOR)

    # --- Weather: domes and closed roofs have no wind and a constant 70F ---
    indoor = pbp.roof.isin(["dome", "closed"])
    pbp["outdoors"] = (~indoor).astype(int)
    pbp.loc[indoor, "wind"] = 0
    pbp.loc[indoor, "temp"] = 70
    pbp["wind"] = pbp.wind.fillna(pbp.loc[~indoor, "wind"].median())
    pbp["temp"] = pbp.temp.fillna(pbp.loc[~indoor, "temp"].median())
    pbp["wind"] = pbp.wind.clip(upper=40)   # a few readings (e.g. 71 mph) are data errors

    # Labels used by several models
    pbp["is_pass"] = (pbp.play_type == "pass").astype(int)
    pbp["converted"] = (pbp.fourth_down_converted == 1).astype(int)
    # Fake punts / field goals: runs or passes snapped from a kicking formation
    pbp["is_fake"] = (pbp.play_type.isin(["run", "pass"])
                      & pbp.desc.str.contains(r"\((?:Punt|Field Goal) formation\)", case=False, na=False)).astype(int)
    pbp = pbp.drop(columns="desc")
    # Field position only affects conversion near the goal line (compressed field);
    # beyond the opp 40 it mostly reflects which teams choose to go, so cap it.
    pbp["yardline_capped"] = pbp.yardline_100.clip(upper=40)
    home = pbp.posteam == pbp.home_team
    pbp["pos_win"] = np.where(pbp.result == 0, np.nan,
                              np.where(home, pbp.result > 0, pbp.result < 0)).astype(float)
    os.makedirs("data", exist_ok=True)
    pbp.to_parquet(CACHE, index=False)
    return pbp


def load_plays():
    return pd.read_parquet(CACHE) if os.path.exists(CACHE) else build()


if __name__ == "__main__":
    p = build()
    print(p.shape)
    g = p.drop_duplicates(["game_id", "posteam"])
    print(g[["off_rating", "def_rating", "qb_rating", "wind", "temp"]].describe().round(3))
    # Sanity: 2025 top/bottom offenses and QBs entering their latest game
    last = g[g.season == 2025].sort_values("game_date").drop_duplicates("posteam", keep="last")
    print(last.nlargest(5, "off_rating")[["posteam", "off_rating"]])
    print(last.nlargest(5, "qb_rating")[["posteam", "qb_name", "qb_rating"]])
    print(last.nsmallest(3, "def_rating")[["defteam", "def_rating"]])
