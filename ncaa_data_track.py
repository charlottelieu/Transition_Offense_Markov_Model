# load required libraries
import pandas as pd
import requests
from collections import defaultdict

# define global constants for the markov model
TARGET_TEAM  = 47
TRANSITION_WINDOW = 10   
 
# fetch the espn play-by-play data for a given game id
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
 
 
# clock conversion helper that converts "MM:SS" to total seconds (int)
def clock_to_seconds(clock_str: str) -> int | None:
    try:
        m, s = clock_str.split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return None
 
 
# define keywords for classifying shot types
_SHOT_TYPE_KEYWORDS = [
    "jump shot", "layup", "lay-up", "dunk", "tip shot", "hook shot",
    "floating", "turnaround", "step back", "pull-up", "pull up",
    "three point", "three-point", "3-point", "3pt", "shot",
]
 
 # classify each play into a standardized event type based on the text and play type
def classify_event(text: str, play_type: str) -> str:
    t  = text.lower()
    pt = play_type.lower()
 
    # check if the play type matches any predefined shot keywords
    is_shot_type = any(kw in pt for kw in _SHOT_TYPE_KEYWORDS)

    _made   = "made"   in t or "makes"  in t
    _missed = "missed" in t or "misses" in t
 
    # return make or miss if the event is a recognized shot type
    if is_shot_type:
        if _made:
            return "MAKE"
        elif _missed:
            return "MISS"
        return "MISS"
 
    # classify free throws
    if "free throw" in t or "free throw" in pt:
        if _made:
            return "FT_MAKE"
        elif _missed:
            return "FT_MISS"
        return "FT"
 
    # classify everything else text based
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
 
 
# extract fields of interest from the raw play data and return a cleaned dataframe
def extract_plays(plays: list[dict]) -> pd.DataFrame:
    rows = []
    for play in plays:
        clock = play.get("clock", {})
        time  = clock.get("displayValue")
        if not time:
            continue

        # convert string clock to seconds
        seconds = clock_to_seconds(time)
        if seconds is None:
            continue
        
        # extract play metadata
        team_id   = int(play.get("team", {}).get("id", 0))
        text      = play.get("text", "")
        play_type = play.get("type", {}).get("text", "")
        seq       = int(play.get("sequenceNumber", 0))
        period    = play.get("period", {}).get("number", None)
 
        # append parsed play data to rows list
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
    
    #convert list of dictionaries to a pandas dataframe
    df = pd.DataFrame(rows)

    # handle edge case for empty dataframes
    if df.empty:
        print("No valid plays found.")
        return df
 
    # sort plays chronologically and reset index
    df = df.sort_values(
        by=["period", "seconds", "sequence"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return df
 
# filter only the relevant plays for the markov state space
def filter_plays(df: pd.DataFrame) -> pd.DataFrame:
    keep = {"MAKE", "MISS", "OREB", "DREB", "TOV", "FOUL"}
    return df[df["event"].isin(keep)].reset_index(drop=True)
 
 
# diagnostic helper to view raw events
def diagnose(df_raw: pd.DataFrame) -> None:
    print("\n Raw event distribution")
    print(df_raw["event"].value_counts().to_string())
 
    print("\n Sample MAKE rows")
    makes = df_raw[df_raw["event"] == "MAKE"]
    print(f"  Count: {len(makes)}")
    if not makes.empty:
        print(makes[["text", "type", "event"]].head(8).to_string(index=False))
 
    print("\n Sample rows with 'made' that weren't classified as MAKE")
    suspect = df_raw[
        df_raw["text"].str.contains("made", case=False, na=False)
        & (df_raw["event"] != "MAKE")
    ]
    if suspect.empty:
        print("none")
    else:
        print(suspect[["text", "type", "event"]].to_string(index=False))
    print()
 
 
# ── helpers ─────────────────────────────────────────────────────────────────
 
 # detect 3-point shots
def is_three_pointer(text: str) -> bool:
    t = text.lower()
    return (
        "three point" in t          # "makes 25-foot three point jumper"
        or "three-point" in t       # "made Three-Point Jump Shot"
        or "3-point" in t
        or "3pt" in t
    )
 
 # detect turnovers
def is_oob_turnover(text: str) -> bool:
    lower = text.lower()
    return "out of bounds" in lower or " oob" in lower
 
 
# ── possession builder ──────────────────────────────────────────────────────

# group raw plays into possession sequences
def build_possessions(df: pd.DataFrame, target_team: int) -> pd.DataFrame:
    teams = df["team_id"].unique().tolist()

    # validate that exactly two teams are present
    if len(teams) != 2:
        raise ValueError(f"2 teams: {teams}")
    opponent = next(t for t in teams if t != target_team)
 
    records = []
 
    # iterate through plays grouped by game period
    for period, grp in df.groupby("period"):

        # sort plays chronologically within the period
        grp  = grp.sort_values(
            ["seconds", "sequence"], ascending=[False, True]
        ).reset_index(drop=True)
        rows = grp.to_dict("records")
 
        # initialize tracking variables for current possession
        poss_owner:   int | None = None
        poss_start:   int | None = None
        has_stoppage: bool       = False
 
        # function logs completed possessions
        def commit(end_state: str, end_sec: int) -> None:

            # only record possessions for the target team
            if poss_owner != target_team:
                return
            
            # calculate possession duration in seconds
            elapsed     = poss_start - end_sec

            # classify start state as transition or half-court based on duration and stoppages
            start_state = "T" if (elapsed <= TRANSITION_WINDOW and not has_stoppage) else "H"

            # append possession data to records list
            records.append(dict(
                period=period,
                start_state=start_state,
                end_state=end_state,
                start_sec=poss_start,
                end_sec=end_sec,
                elapsed=elapsed,
            ))
 
        #function initializes a new possession
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
 
            # helper to peek at upcoming events in the sequence
            def upcoming(n: int = 5) -> list[dict]:
                return rows[i + 1 : i + 1 + n]
 
            # ── lazy initialization ───────────────────────────────────────────────
            if poss_owner is None:
                if event in ("MAKE", "MISS", "TOV"):
                   # true start time unknown so default to current clock
                    new_poss(team, sec, dead_ball=False)
                elif event == "DREB":
                    # opponent missed and this team rebounds so start a new possession and DREB handler can close it as E.
                    opp = opponent if team == target_team else target_team
                    new_poss(opp, sec, dead_ball=False)
                elif event in ("FOUL", "OREB"):
                    continue  
 
            # handle made field goals
            if event == "MAKE":
                # check immediate upcoming plays for an and-1 foul
                next_rows = upcoming(2)
                and1_foul = any(
                    r["event"] == "FOUL"
                    and r["team_id"] != team
                    and r["seconds"] == sec
                    for r in next_rows
                )
                # classify as 3-point or 2-point make
                shot_type = "S3" if is_three_pointer(text) else "S2"

                # commit possession as foul if and-1, otherwise standard make
                commit("F" if and1_foul else shot_type, sec)

                # transfer possession to opponent via dead ball
                new_poss(
                    opponent if team == target_team else target_team,
                    sec, dead_ball=True,
                )
            # handle turnovers
            elif event == "TOV":
                commit("TO", sec)
                new_poss(
                    opponent if team == target_team else target_team,
                    sec, dead_ball=is_oob_turnover(text),
                )

            # handle defensive rebounds
            elif event == "DREB":
                if poss_owner is not None and poss_owner != team:
                    commit("E", sec)
                new_poss(team, sec, dead_ball=False)

            # handle offensive rebounds (posession continues)
            elif event == "OREB":
                pass  
 
            # handle missed shots (possession continues until other team posession)
            elif event == "MISS":
                pass 
 
            # handle fouls
            elif event == "FOUL":
                if poss_owner is None:
                    continue
 
                # determine if foul was committed by the offense
                committing_team    = team
                is_offensive_foul  = (committing_team == poss_owner)

                # offensive fouls terminate possession as a turnover
                if is_offensive_foul:
                    commit("TO", sec)
                    new_poss(
                        opponent if committing_team == target_team else target_team,
                        sec, dead_ball=True,
                    )
                else:
                    # check upcoming plays for resulting free throws
                    ahead = upcoming(6)
                    ft_keywords = ("Free Throw", "FreeThrow", "FTA", "FTM")
                    has_ft = any(
                        r["event"] in ("FT_MAKE", "FT_MISS", "FT", "FTA", "FTM")
                        or any(kw in r.get("text", "") for kw in ft_keywords)
                        for r in ahead
                    )
                    # commit target team shooting fouls as absorbing state F
                    if has_ft and poss_owner == target_team:
                        commit("F", sec)
                        new_poss(opponent, sec, dead_ball=True)
                    else:
                        #non-shooting foul forces transient state H without ending possession
                        has_stoppage = True 
 
    return pd.DataFrame(records)
 
 
# ── summary ─────────────────────────────────────────────────────────────────

# define required columns for the output dataframe
REQUIRED_COLS = [
    "transition_possessions", "halfcourt_possessions",
    "transition_2pt",         "halfcourt_2pt",
    "transition_3pt",         "halfcourt_3pt",
    "transition_TOV",         "halfcourt_TOV",
    "transition_F",           "halfcourt_F",
    "transition_empty",       "halfcourt_empty",
]
 
 # map markov absorbing states to suffixes
END_STATE_MAP = {"S2": "2pt", "S3": "3pt", "TO": "TOV", "E": "empty", "F": "F"}
 
# calculate summary statistics for transition and half-court states
def summarize(poss_df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = defaultdict(int)

    # determine prefix based on transient state (T or H)
    for _, row in poss_df.iterrows():
        prefix = "transition" if row["start_state"] == "T" else "halfcourt"
        counts[f"{prefix}_possessions"] += 1

        # determine suffix based on absorbing state
        suffix = END_STATE_MAP.get(row["end_state"])
        if suffix:
            counts[f"{prefix}_{suffix}"] += 1
    return pd.DataFrame([{col: counts.get(col, 0) for col in REQUIRED_COLS}])
 
# ── main ────────────────────────────────────────────────────────────────────

# main execution function
def main() -> pd.DataFrame:
    game_id = "401851477"

    #fetch plays
    plays  = get_game_data(game_id)
    df_raw = extract_plays(plays)

    #diagnose(df_raw)
 
    #filter
    df = filter_plays(df_raw)
    df.to_csv("clean_plays.csv", index=False)
    print("Saved clean_plays.csv")
 
    #enforce dtypes
    for col in ("seconds", "sequence", "team_id", "period"):
        df[col] = df[col].astype(int)
 
    #build possessions
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