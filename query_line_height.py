"""
Query-layer function for `defensive_line_height` — the first 360 concept.

WHY 360: event data only records where the BALL was touched. Defending is mostly
NOT touching the ball — it is standing in the right place, holding a line. Event
data is structurally blind to that. 360 freeze-frames give the position of all 22
players at each event, so we can read where the back line actually SAT.

COORDINATE FRAME (verified empirically, not assumed — acting team's keeper sits
at mean x=10.6, opponent's at x=111.8 on match 3869220):
  Freeze-frame locations are in the ACTING TEAM's frame; the acting team attacks
  toward x=120. So to measure MOROCCO's line we read Morocco's players out of
  SPAIN's events, where they appear as `teammate: false` and Morocco's OWN GOAL
  is at x=120.
  => Morocco's line height in Morocco's own frame = 120 - x.
  Getting this inversion backwards silently flips every answer.

FALLBACK: `defensive_action_height` (mean x of defensive actions) needs events
only, so the concept still answers on competitions without 360 (e.g. Copa America
2024). It is weaker on purpose — it says where actions HAPPENED, not where the
line SAT — and the two are reported side by side so the gap is visible.
"""
from statsbombpy import sb
from clock import in_play, p90_factor
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PITCH_LEN = 120
BACK_LINE_N = 4          # how many deepest outfielders constitute "the line"
# VISIBILITY: 360 freeze-frames only contain players in the BROADCAST SHOT. When
# the ball is far from the defending team, its back line is off-camera and the
# "deepest visible players" are actually its forwards. Measured on 3869220: at a
# 6-defender floor, Morocco's line reads 72.2 when Spain builds from the back —
# an impossible line deep in Spain's half. corr(ball x, measured height) = -0.70.
MIN_VISIBLE_DEFENDERS = 8

# Ball-position bands (in the ATTACKING team's frame) used to condition the read.
# CANONICAL band = opponent in the middle third: good visibility, and the phase
# where "how high is their line" is actually a meaningful question.
BALL_BANDS = [("opp_own_third", 0, 40), ("middle_third", 40, 80),
              ("opp_attacking_third", 80, 120)]
CANONICAL_BAND = "middle_third"
DEF_ACTION_TYPES = {"Pressure", "Interception", "Clearance", "Ball Recovery", "Duel"}


def _x(loc):
    return loc[0] if isinstance(loc, list) else None


def _load_frames(match_id):
    try:
        return sb.frames(match_id=match_id, fmt="dict")
    except Exception:
        return None


def line_height_360(match_id, ev=None, frames=None, open_play_only=True):
    """Mean height of each team's back line while the OPPONENT has the ball.

    Returns None if no 360 data exists for the match — callers must fall back.
    """
    frames = frames if frames is not None else _load_frames(match_id)
    if not frames:
        return None
    if ev is None:
        ev = sb.events(match_id=match_id)
    ev = in_play(ev)

    keep = ev
    if open_play_only:
        # set pieces distort shape completely (everyone in the box on a corner)
        keep = keep[keep.play_pattern == "Regular Play"]

    uuid2team = dict(zip(keep.id, keep.team))
    teams = [t for t in ev.team.dropna().unique()]

    ball_x = dict(zip(keep.id, keep.location.apply(_x)))

    # defending team -> band -> list of line heights
    heights = {t: {b[0]: [] for b in BALL_BANDS} for t in teams}
    skipped = {t: 0 for t in teams}

    for fr in frames:
        acting = uuid2team.get(fr["event_uuid"])
        if acting is None:
            continue
        bx = ball_x.get(fr["event_uuid"])
        if bx is None:
            continue
        defending = [t for t in teams if t != acting]
        if not defending:
            continue
        defending = defending[0]

        # opponents in this frame ARE the defending team
        opp = [p for p in fr["freeze_frame"]
               if not p["teammate"] and not p["keeper"]]
        if len(opp) < MIN_VISIBLE_DEFENDERS:
            skipped[defending] += 1
            continue

        xs = sorted((p["location"][0] for p in opp), reverse=True)
        back = xs[:BACK_LINE_N]          # deepest = highest x = nearest own goal
        h = PITCH_LEN - float(np.mean(back))   # into the defender's own frame
        for name, lo, hi in BALL_BANDS:
            if lo <= bx < hi:
                heights[defending][name].append(h)
                break

    out = {}
    for t in teams:
        bands = {}
        for name, _, _ in BALL_BANDS:
            arr = np.array(heights[t][name])
            bands[name] = ({"mean": round(float(arr.mean()), 1),
                            "sd": round(float(arr.std()), 1),
                            "frames": len(arr)} if len(arr) >= 20 else None)
        if not any(bands.values()):
            out[t] = None
            continue
        canon = bands.get(CANONICAL_BAND)
        out[t] = {
            # headline = CANONICAL BAND only, never a whole-match mean. A
            # whole-match average is not measurable without bias: loose
            # visibility floors admit off-camera artifacts, tight ones only
            # capture compressed deep phases. Condition, don't average.
            "line_height_360": canon["mean"] if canon else None,
            "line_height_sd": canon["sd"] if canon else None,
            "frames_used": canon["frames"] if canon else 0,
            "measured_when": f"opponent in possession, ball in {CANONICAL_BAND}",
            "by_ball_position": bands,
            "frames_skipped_low_visibility": skipped[t],
        }
    return out


def defensive_action_height(match_id, ev=None):
    """Events-only fallback: mean x of a team's defensive actions (own frame)."""
    if ev is None:
        ev = sb.events(match_id=match_id)
    ev = in_play(ev).copy()
    ev["x"] = ev["location"].apply(_x)
    teams = [t for t in ev.team.dropna().unique()]
    out = {}
    for t in teams:
        d = ev[(ev.team == t) & (ev.type.isin(DEF_ACTION_TYPES))].dropna(subset=["x"])
        out[t] = {
            "defensive_action_height": round(float(d.x.mean()), 1) if len(d) else None,
            "n_actions": len(d),
        }
    return out


def defensive_line_height(match_id, ev=None):
    """Full concept: 360 primary + events-only fallback, reported together."""
    if ev is None:
        ev = sb.events(match_id=match_id)
    teams = [t for t in ev.team.dropna().unique()]

    f360 = line_height_360(match_id, ev=ev)
    fallback = defensive_action_height(match_id, ev=ev)
    has_360 = f360 is not None and any(v for v in f360.values())

    results = {}
    for t in teams:
        opp = [x for x in teams if x != t][0]
        primary = (f360 or {}).get(t) or {}
        fb = fallback.get(t, {})
        entry = {
            "team": t,
            "opponent": opp,
            "method": "360" if primary else "fallback (events only)",
            "has_360": bool(primary),
            **primary,
            **fb,
        }
        entry["read"] = _verdict(entry)
        results[t] = entry

    if has_360:
        results["_method_note"] = (
            "Line height read from 360 freeze-frames, CONDITIONED on ball "
            "position and restricted to high-visibility frames (>=8 outfield "
            "defenders in shot). A whole-match average is deliberately NOT "
            "reported: 360 only captures players in the broadcast shot, so when "
            "the ball is far away the back line is off-camera and the deepest "
            "visible players are actually forwards. Unconditioned, the measure "
            "correlates -0.70 with ball position — i.e. it substantially "
            "measures where the BALL was, not where the LINE sat. See "
            "by_ball_position for the honest profile.")
    else:
        results["_method_note"] = (
            "NO 360 DATA for this match — using the events-only fallback. This "
            "reflects where defensive actions occurred, not where the line sat; "
            "a pinned-back team clears from its own box, which is the ball, not "
            "the shape. Treat as directional only.")
    return results


# Bands (starting points — tune against known matches, do not treat as gospel)
DEEP_LINE = 30      # own third => deep block
HIGH_LINE = 45      # at/beyond halfway => high line


def _verdict(e):
    lh = e.get("line_height_360")
    if lh is None and e.get("by_ball_position"):
        return ("360 present but too few high-visibility frames in the canonical "
                "band to read a line height honestly.")
    if lh is None:
        dah = e.get("defensive_action_height")
        if dah is None:
            return "no defensive actions recorded"
        return (f"FALLBACK (no 360): mean defensive-action height {dah} — where "
                f"actions happened, not where the line sat. Directional only.")
    if lh < DEEP_LINE:
        band = "deep block (line in own third)"
    elif lh < HIGH_LINE:
        band = "mid block"
    else:
        band = "high line"
    prof = e.get("by_ball_position") or {}
    trail = ""
    if prof.get("opp_attacking_third"):
        trail = (f" Drops to {prof['opp_attacking_third']['mean']} when pressed "
                 f"back into their own third.")
    return (f"With the ball in the middle third, back line sat {lh} from own goal "
            f"({e.get('line_height_sd')} sd, {e.get('frames_used')} frames) "
            f"=> {band}.{trail}")


if __name__ == "__main__":
    import sys, json
    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 3869220
    r = defensive_line_height(mid)
    note = r.pop("_method_note")
    for t, e in r.items():
        print(f"\n=== {e['team']} (vs {e['opponent']}) ===")
        for k, v in e.items():
            if k not in ("team", "opponent", "read"):
                print(f"  {k}: {v}")
        print(f"  READ: {e['read']}")
    print(f"\nNOTE: {note}")
