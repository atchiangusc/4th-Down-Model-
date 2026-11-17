"""Build a 4th-down 'go for it' dataset joining nflverse play-by-play with
participation (personnel, box count) data. Participation: 2016-2022 from NGS,
2023+ from FTN Data via nflverse (CC-BY-SA 4.0)."""
import re
import nflreadpy as nfl
import pandas as pd

SEASONS = list(range(2016, 2026))
PBP_COLS = ["game_id", "play_id", "season", "week", "posteam", "defteam", "down",
            "ydstogo", "yardline_100", "play_type", "qb_kneel", "qb_spike",
            "fourth_down_converted", "fourth_down_failed", "shotgun", "no_huddle",
            "score_differential", "game_seconds_remaining", "qtr", "wp",
            "posteam_timeouts_remaining", "defteam_timeouts_remaining",
            "roof", "surface", "spread_line", "posteam_type"]

# 2016-22 (NGS) strings look like "4 DL, 2 LB, 5 DB"; 2023+ (FTN) list exact
# positions like "3 CB, 2 DE, 2 DT, 1 FS, 2 ILB, 1 SS". Map both to groups.
GROUPS = {
    "DL": {"DL", "DE", "DT", "NT"},
    "LB": {"LB", "ILB", "MLB", "OLB"},
    "DB": {"DB", "CB", "FS", "SS", "S", "SAF"},
    "OL": {"OL", "C", "G", "T"},
    "RB": {"RB", "FB", "HB"},
    "TE": {"TE"},
    "WR": {"WR"},
}

def count_pos(s, group):
    if not isinstance(s, str):
        return None
    return sum(int(n) for n, pos in re.findall(r"(\d+)\s+([A-Z]+)", s)
               if pos in GROUPS[group])

def main():
    pbp = nfl.load_pbp(SEASONS).select(PBP_COLS).to_pandas()
    part = nfl.load_participation(SEASONS).select(
        ["nflverse_game_id", "play_id", "offense_formation", "offense_personnel",
         "defenders_in_box", "defense_personnel"]).to_pandas()
    part = part.rename(columns={"nflverse_game_id": "game_id"})

    fd = pbp[(pbp.down == 4) & pbp.play_type.isin(["run", "pass"])
             & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1)].copy()
    fd = fd.merge(part, on=["game_id", "play_id"], how="left")

    fd["converted"] = (fd.fourth_down_converted == 1).astype(int)
    fd["is_pass"] = (fd.play_type == "pass").astype(int)
    for pos in ["DL", "LB", "DB"]:
        fd[f"def_{pos}"] = fd.defense_personnel.map(lambda s: count_pos(s, pos))
    for pos in ["RB", "TE", "WR", "OL"]:
        fd[f"off_{pos}"] = fd.offense_personnel.map(lambda s: count_pos(s, pos))
    # Older strings omit the 5 standard linemen and only list extras ("6 OL")
    fd.loc[fd.off_OL == 0, "off_OL"] = 5
    # Box count relative to offensive blockers: OL(5) + TE + RB
    fd["box_minus_blockers"] = fd.defenders_in_box - (fd.off_OL + fd.off_TE + fd.off_RB)
    fd["home"] = (fd.posteam_type == "home").astype(int)
    fd["outdoors"] = fd.roof.isin(["outdoors", "open"]).astype(int)

    before = len(fd)
    fd = fd.dropna(subset=["defenders_in_box", "def_DL", "off_WR"])
    print(f"4th-down go plays: {before}, with personnel/box data: {len(fd)}")
    fd.to_parquet("data/fourth_downs.parquet", index=False)
    print(fd.groupby("season").agg(plays=("converted", "size"), conv_rate=("converted", "mean")))

if __name__ == "__main__":
    main()
