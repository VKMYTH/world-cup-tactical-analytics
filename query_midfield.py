"""
Query-layer function for the `controlling_midfield` concept.

HARDENED after validation on the 2022 final: midfield control is TEMPORAL and
GAME-STATE-SENSITIVE. A single whole-match share averages opposite phases into a
meaningless number, so this returns a per-window timeline plus a goal timeline
for context, not just one aggregate.

Middle third = 40 <= x <= 80 (symmetric, so no coordinate-frame issue).
"""
from statsbombpy import sb
from clock import add_clock, p90_factor, ordered_buckets, bucket_label, in_play
import warnings
warnings.filterwarnings("ignore")

MID_LOW, MID_HIGH = 40, 80
FINAL_THIRD_X = 80
WINDOW = 15                 # minutes per timeline bucket
CONTROL_THRESHOLD = 0.60    # share in a window to count as "controlled"


def _x(loc):
    return loc[0] if isinstance(loc, list) else None


def _mid_mask(ev):
    return (ev.x >= MID_LOW) & (ev.x <= MID_HIGH) & (
        ((ev.type == "Pass") & (ev.pass_outcome.isna())) | (ev.type == "Carry"))


def _goal_timeline(ev):
    # Exclude the penalty shootout (period 5): StatsBomb logs shootout kicks as
    # Shot/Goal events, which would otherwise appear as a burst of "goals" at ~120'.
    inplay = in_play(ev)
    goals = []
    shots = inplay[(inplay.type == "Shot") & (inplay.shot_outcome == "Goal")]
    for _, r in shots.iterrows():
        goals.append((int(r.minute), r.team))
    for _, r in inplay[inplay.type == "Own Goal Against"].iterrows():
        goals.append((int(r.minute), f"(OG) not {r.team}"))
    return sorted(goals)


def controlling_midfield(match_id, ev=None):
    if ev is None:
        ev = sb.events(match_id=match_id)
    ev = ev.copy()
    ev["x"] = ev["location"].apply(_x)
    ev["pass_end_x"] = ev["pass_end_location"].apply(_x)
    ev["carry_end_x"] = ev["carry_end_location"].apply(_x)

    teams = [t for t in ev["team"].dropna().unique()]
    p90 = p90_factor(ev)          # nominal 90/120, never the shootout clock

    completed_all = ev[(ev.type == "Pass") & (ev.pass_outcome.isna())]
    pass_counts = completed_all.team.value_counts()
    total_passes = pass_counts.sum()

    mid = ev[_mid_mask(ev)]
    mid_total = len(mid)

    # ---- per-team whole-match aggregates (context only) ----
    team_stats = {}
    for team in teams:
        opp = [t for t in teams if t != team][0]
        mid_passes = ev[(ev.x >= MID_LOW) & (ev.x <= MID_HIGH)
                        & (ev.team == team) & (ev.type == "Pass")]
        mid_completed = mid_passes[mid_passes.pass_outcome.isna()]
        prog_p = mid_completed[mid_completed.pass_end_x > FINAL_THIRD_X]
        prog_c = ev[(ev.x >= MID_LOW) & (ev.x <= MID_HIGH) & (ev.team == team)
                    & (ev.type == "Carry") & (ev.carry_end_x > FINAL_THIRD_X)]
        prog = len(prog_p) + len(prog_c)
        team_stats[team] = {
            "opponent": opp,
            "overall_possession": round(pass_counts.get(team, 0) / total_passes, 2),
            "mid_third_share_WHOLE_MATCH": round(len(mid[mid.team == team]) / mid_total, 2),
            "mid_third_completion": round(len(mid_completed) / len(mid_passes), 2) if len(mid_passes) else None,
            "progressions": prog,
            "progressions_p90": round(prog * p90, 1),
        }

    # ---- temporal timeline (the real answer for this concept) ----
    # Period-aware buckets: raw `minute` is NOT monotonic across periods (P1
    # runs to 52', P2 restarts at 45'), so naive minute-binning merges first-half
    # stoppage with the early second half. See clock.py.
    mid_c = add_clock(mid, WINDOW)
    timeline = []
    for b in ordered_buckets(mid_c):
        w = mid_c[mid_c.bucket == b]
        n = len(w)
        if n < 20:  # too few events to read control
            continue
        shares = {t: round(len(w[w.team == t]) / n, 2) for t in teams}
        top = max(shares, key=shares.get)
        dominant = top if shares[top] >= CONTROL_THRESHOLD else None
        timeline.append({"window": bucket_label(b, WINDOW), "events": n,
                         "shares": shares, "dominant": dominant})

    verdict = _verdict(timeline, team_stats, teams)
    return {"teams": team_stats, "timeline": timeline,
            "goals": _goal_timeline(ev), "verdict": verdict}


def _verdict(timeline, team_stats, teams):
    # compress consecutive windows with the same dominant team into runs
    runs = []
    for w in timeline:
        d = w["dominant"]
        if runs and runs[-1][0] == d:
            runs[-1][1].append(w["window"])
        else:
            runs.append([d, [w["window"]]])
    parts = []
    for team, wins in runs:
        span = f"{wins[0].split('-')[0]}-{wins[-1].split('-')[1]}"
        parts.append(f"{team if team else 'contested'} ({span}')")
    whole = " / ".join(f"{t} {team_stats[t]['mid_third_share_WHOLE_MATCH']}" for t in teams)
    return (f"Whole-match share ({whole}) is an AVERAGE and hides the swing. "
            f"Control by phase: {' -> '.join(parts)}. "
            f"Read against the scoreline — a leading side often cedes the middle by choice.")


if __name__ == "__main__":
    import sys
    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 3869685
    out = controlling_midfield(mid)
    print("GOALS:", out["goals"])
    print("\nWHOLE-MATCH (context):")
    for t, r in out["teams"].items():
        print(f"  {t}: mid_share={r['mid_third_share_WHOLE_MATCH']} "
              f"overall_poss={r['overall_possession']} prog/90={r['progressions_p90']}")
    print("\nTIMELINE:")
    for w in out["timeline"]:
        print(f"  {w['window']:>8}'  {w['shares']}  dominant={w['dominant']}")
    print("\nVERDICT:", out["verdict"])