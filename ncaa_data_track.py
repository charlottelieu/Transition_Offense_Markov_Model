import pandas as pd
import requests
from collections import defaultdict
TARGET_TEAM  = 47
TRANSITION_WINDOW = 10   
 
#Fetch the ESPN play-by-play data for a given game ID. 
# Returns a list of play dictionaries.
 
def get_game_data(game_id: str) -> list[dict]:
    url = (
            f"https://cdn.espn.com/core/mens-college-basketball/playbyplay"
            f"?gameId={game_id}&xhr=1"
    )
    response = requests.get(url, timeout=15)
    data = response.json()
    try:
        return data["gamepackageJSON"]["plays"]
    except KeyError:
        print("Could not find plays in response")
        return []
 
 
# Clock conversion helper: converts "MM:SS" to total seconds (int).
# Returns None if format is invalid.
 
def clock_to_seconds(clock_str: str) -> int | None:
    try:
        m, s = clock_str.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return None
 
 
# Classifies each play into a standardized event type based on the text and play type. 
_SHOT_TYPE_KEYWORDS = [
    "jump shot", "layup", "lay-up", "dunk", "tip shot", "hook shot",
    "floating", "turnaround", "step back", "pull-up", "pull up",
    "three point", "three-point", "3-point", "3pt", "shot",
]
 
def classify_event(text: str, play_type: str) -> str:
    t  = text.lower()
    pt = play_type.lower()
 
    is_shot_type = any(kw in pt for kw in _SHOT_TYPE_KEYWORDS)
 
    # ESPN uses two tenses across different game feeds:
    #   past:    "Player made/missed Two-Point Jump Shot."
    #   present: "Player makes/misses 25-foot three point jumper"
    _made   = "made"   in t or "makes"  in t
    _missed = "missed" in t or "misses" in t
 
    if is_shot_type:
        if _made:
            return "MAKE"
        elif _missed:
            return "MISS"
        return "MISS"
 
    # Free throws — before generic made/missed check
    if "free throw" in t or "free throw" in pt:
        if _made:
            return "FT_MAKE"
        elif _missed:
            return "FT_MISS"
        return "FT"
 
    # Everything else is text based
    if _made:
        return "MAKE"
    elif _missed:
        return "MISS"
    elif "offensive rebound" in t:
        return "OREB"
    elif "defensive rebound" in t or "rebound" in t:
        return "DREB"
    elif "turnover" in t or "bad pass" in t or "lost ball" in t:
        return "TOV"
    elif "foul" in t:
        return "FOUL"
    elif "steal" in t:
        return "STL"
    elif "block" in t:
        return "BLK"
 
    return "OTHER"
 
 
#Extract fields of interest from the raw play data and return a cleaned DataFrame.
 
def extract_plays(plays: list[dict]) -> pd.DataFrame:
    rows = []
    for play in plays:
        clock = play.get("clock", {})
        time  = clock.get("displayValue")
        if not time:
            continue
        seconds = clock_to_seconds(time)
        if seconds is None:
            continue
 
        team_id   = int(play.get("team", {}).get("id", 0))
        text      = play.get("text", "")
        play_type = play.get("type", {}).get("text", "")
        seq       = int(play.get("sequenceNumber", 0))
        period    = play.get("period", {}).get("number", None)
 
        rows.append({
            "time":     time,
            "seconds":  seconds,
            "sequence": seq,
            "team_id":  team_id,
            "text":     text,
            "type":     play_type,
            "event":    classify_event(text, play_type),
            "period":   period,
        })
 
    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid plays found.")
        return df
 
    df = df.sort_values(
        by=["period", "seconds", "sequence"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return df
 
 
#Filter only the relevant plays
 
def filter_plays(df: pd.DataFrame) -> pd.DataFrame:
    keep = {"MAKE", "MISS", "OREB", "DREB", "TOV", "FOUL"}
    return df[df["event"].isin(keep)].reset_index(drop=True)
 
 
#Diagonistic helper

def diagnose(df_raw: pd.DataFrame) -> None:
    print("\n=== Event distribution (raw, before filter) ===")
    print(df_raw["event"].value_counts().to_string())
 
    print("\n=== Sample MAKE rows ===")
    makes = df_raw[df_raw["event"] == "MAKE"]
    print(f"  Count: {len(makes)}")
    if not makes.empty:
        print(makes[["text", "type", "event"]].head(8).to_string(index=False))
 
    print("\n=== Rows with 'made' in text that were NOT classified as MAKE ===")
    suspect = df_raw[
        df_raw["text"].str.contains("made", case=False, na=False)
        & (df_raw["event"] != "MAKE")
    ]
    if suspect.empty:
        print("  None — good!")
    else:
        print(suspect[["text", "type", "event"]].to_string(index=False))
    print()
 
 
#helpers -----------
 
 
def is_three_pointer(text: str) -> bool:
    """Detect 3-point shots from event text (handles made/makes/missed/misses formats)."""
    t = text.lower()
    return (
        "three point" in t          # "makes 25-foot three point jumper"
        or "three-point" in t       # "made Three-Point Jump Shot"
        or "3-point" in t
        or "3pt" in t
    )
 
 
def is_oob_turnover(text: str) -> bool:
    lower = text.lower()
    return "out of bounds" in lower or " oob" in lower
 
 
# ── possession builder ────────
 
def build_possessions(df: pd.DataFrame, target_team: int) -> pd.DataFrame:
    teams = df["team_id"].unique().tolist()
    if len(teams) != 2:
        raise ValueError(f"Expected exactly 2 teams; found {teams}")
    opponent = next(t for t in teams if t != target_team)
 
    records = []
 
    for period, grp in df.groupby("period"):
        grp  = grp.sort_values(
            ["seconds", "sequence"], ascending=[False, True]
        ).reset_index(drop=True)
        rows = grp.to_dict("records")
 
        poss_owner:   int | None = None
        poss_start:   int | None = None
        has_stoppage: bool       = False
 
        def commit(end_state: str, end_sec: int) -> None:
            if poss_owner != target_team:
                return
            elapsed     = poss_start - end_sec
            start_state = "T" if (elapsed <= TRANSITION_WINDOW and not has_stoppage) else "H"
            records.append(dict(
                period=period,
                start_state=start_state,
                end_state=end_state,
                start_sec=poss_start,
                end_sec=end_sec,
                elapsed=elapsed,
            ))
 
        def new_poss(owner: int, start_sec: int, dead_ball: bool = False) -> None:
            nonlocal poss_owner, poss_start, has_stoppage
            poss_owner   = owner
            poss_start   = start_sec
            has_stoppage = dead_ball
 
        for i, row in enumerate(rows):
            team  = row["team_id"]
            event = row["event"]
            sec   = row["seconds"]
            text  = row["text"]
 
            def upcoming(n: int = 5) -> list[dict]:
                return rows[i + 1 : i + 1 + n]
 
            # ── LAZY INIT ─────────────────────────────────────────────────
            # poss_owner is None at period start (and after any untracked gap).
            # Backfill ownership from the first possession-revealing event so
            # the opening possession of each period is never silently dropped.
            if poss_owner is None:
                if event in ("MAKE", "MISS", "TOV"):
                    # This team had the ball. We don't know the true start time,
                    # so we use the current clock → elapsed = 0 → classified T.
                    new_poss(team, sec, dead_ball=False)
                elif event == "DREB":
                    # Opponent missed → this team rebounds. Give opponent a
                    # synthetic possession so DREB handler can close it as E.
                    opp = opponent if team == target_team else target_team
                    new_poss(opp, sec, dead_ball=False)
                elif event in ("FOUL", "OREB"):
                    continue   # can't reconstruct cleanly; wait for clearer event
 
            if event == "MAKE":
                next_rows = upcoming(2)
                and1_foul = any(
                    r["event"] == "FOUL"
                    and r["team_id"] != team
                    and r["seconds"] == sec
                    for r in next_rows
                )
                shot_type = "S3" if is_three_pointer(text) else "S2"
                commit("F" if and1_foul else shot_type, sec)
                new_poss(
                    opponent if team == target_team else target_team,
                    sec, dead_ball=True,
                )
 
            elif event == "TOV":
                commit("TO", sec)
                new_poss(
                    opponent if team == target_team else target_team,
                    sec, dead_ball=is_oob_turnover(text),
                )
 
            elif event == "DREB":
                if poss_owner is not None and poss_owner != team:
                    commit("E", sec)
                new_poss(team, sec, dead_ball=False)
 
            elif event == "OREB":
                pass  # possession continues; start clock does NOT reset
 
            elif event == "MISS":
                pass  # outcome resolved by the following DREB/OREB
 
            elif event == "FOUL":
                if poss_owner is None:
                    continue
 
                committing_team    = team
                is_offensive_foul  = (committing_team == poss_owner)
 
                if is_offensive_foul:
                    commit("TO", sec)
                    new_poss(
                        opponent if committing_team == target_team else target_team,
                        sec, dead_ball=True,
                    )
                else:
                    ahead = upcoming(6)
                    ft_keywords = ("Free Throw", "FreeThrow", "FTA", "FTM")
                    has_ft = any(
                        r["event"] in ("FT_MAKE", "FT_MISS", "FT", "FTA", "FTM")
                        or any(kw in r.get("text", "") for kw in ft_keywords)
                        for r in ahead
                    )
                    if has_ft and poss_owner == target_team:
                        commit("F", sec)
                        new_poss(opponent, sec, dead_ball=True)
                    else:
                        has_stoppage = True  # non-shooting foul → forces H
 
    return pd.DataFrame(records)
 
 
# ── summary ────
 
REQUIRED_COLS = [
    "transition_possessions", "halfcourt_possessions",
    "transition_2pt",         "halfcourt_2pt",
    "transition_3pt",         "halfcourt_3pt",
    "transition_TOV",         "halfcourt_TOV",
    "transition_F",           "halfcourt_F",
    "transition_empty",       "halfcourt_empty",
]
 
END_STATE_MAP = {"S2": "2pt", "S3": "3pt", "TO": "TOV", "E": "empty", "F": "F"}
 
 
def summarize(poss_df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = defaultdict(int)
    for _, row in poss_df.iterrows():
        prefix = "transition" if row["start_state"] == "T" else "halfcourt"
        counts[f"{prefix}_possessions"] += 1
        suffix = END_STATE_MAP.get(row["end_state"])
        if suffix:
            counts[f"{prefix}_{suffix}"] += 1
    return pd.DataFrame([{col: counts.get(col, 0) for col in REQUIRED_COLS}])
 
 
# ── main ─────────────
def main() -> pd.DataFrame:
    game_id = "401851477"

    # 1. Fetch Plays
    plays  = get_game_data(game_id)
    df_raw = extract_plays(plays)

    #diagnose(df_raw)
 
    # 3. Filter
    df = filter_plays(df_raw)
    df.to_csv("clean_plays.csv", index=False)
    print("Saved clean_plays.csv")
 
    # 4. Enforce dtypes
    for col in ("seconds", "sequence", "team_id", "period"):
        df[col] = df[col].astype(int)
 
    # 5. Build possessions
    poss_df = build_possessions(df, TARGET_TEAM)

    print("=== Possession Detail ===")
    if not poss_df.empty:
        print(poss_df.to_string(
            index=False,
            columns=["period", "start_state", "end_state",
                     "start_sec", "end_sec", "elapsed"],
        ))
    else:
        print("No completed possessions found.")
 
    summary_df = summarize(poss_df)
    print("\n=== Summary Counts ===")
    print(summary_df.T.rename(columns={0: "count"}).to_string())
    print("\n=== Output CSV Row ===")
    print(summary_df.to_csv(index=False))
 
    return summary_df
 
 
if __name__ == "__main__":
    main()