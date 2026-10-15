"""
Query-layer function for the `pressing_intensity` concept.

Turns the KB data_signature into real numbers for one match, with the
possession-imbalance guardrail baked in so PPDA is never reported naked.

Returns a structured dict per team that the synthesis layer can render into a
grounded sentence. Nothing here hallucinates: every field traces to an event.
"""
from statsbombpy import sb
from clock import p90_factor, in_play
import warnings
warnings.filterwarnings("ignore")

# Tunable thresholds (community-standard starting points; sanity-check & tune)
PRESS_ZONE_X = 40        # pressing team's defensive actions counted above this x
OPP_PASS_MAX_X = 72      # opponent passes counted in their own ~60% (below this x)
FINAL_THIRD_X = 80       # "high" press threshold
SKEW_POSSESSION = 0.65   # >this share => PPDA distorted, mark unreliable

# Conventional PPDA denominator: tackles + interceptions + challenges + fouls.
# Pressures are deliberately EXCLUDED — they occur in the hundreds per match and
# would collapse PPDA far below the standard scale, breaking the interp bands.
# Pressures are still captured separately as the high-press signal.
DEF_ACTIONS = {"Interception", "Foul Committed", "Dribbled Past"}


def _x(loc):
    return loc[0] if isinstance(loc, list) else None


def pressing_intensity(match_id, ev=None):
    """Compute the hardened pressing_intensity signature for both teams."""
    if ev is None:
        ev = sb.events(match_id=match_id)
    # drop the penalty shootout: its shots/pressures are not open-play football
    ev = in_play(ev).copy()
    ev["x"] = ev["location"].apply(_x)

    teams = [t for t in ev["team"].dropna().unique()]
    # possession share via completed passes
    completed = ev[(ev.type == "Pass") & (ev.pass_outcome.isna())]
    pass_counts = completed.team.value_counts()
    total_passes = pass_counts.sum()

    # Normalize per 90 using NOMINAL played time (90 or 120), never the raw
    # clock — raw minute.max() is polluted by the shootout. See clock.py.
    p90 = p90_factor(ev)

    # PPDA reliability is a property of the MATCH, not one team: if either side
    # is possession-dominant, both teams' PPDA is distorted. Flag both.
    max_share = max(pass_counts.get(t, 0) / total_passes for t in teams)
    match_ppda_reliable = max_share <= SKEW_POSSESSION

    def is_tackle_duel(row):
        return row.type == "Duel" and "Tackle" in str(row.get("duel_type", ""))

    results = {}
    for team in teams:
        opp = [t for t in teams if t != team][0]

        opp_passes = ev[(ev.team == opp) & (ev.type == "Pass")
                        & (ev.pass_outcome.isna()) & (ev.x < OPP_PASS_MAX_X)]

        zone = ev[(ev.team == team) & (ev.x > PRESS_ZONE_X)]
        def_actions = zone[zone.type.isin(DEF_ACTIONS)
                           | zone.apply(is_tackle_duel, axis=1)]
        n_def = len(def_actions)

        ppda = round(len(opp_passes) / n_def, 2) if n_def else None
        high_press = ev[(ev.team == team) & (ev.type == "Pressure")
                        & (ev.x > FINAL_THIRD_X)]
        high_in_zone = def_actions[def_actions.x > FINAL_THIRD_X]
        high_press_share = round(len(high_in_zone) / n_def, 2) if n_def else None

        poss = round(pass_counts.get(team, 0) / total_passes, 2)
        high_press_p90 = round(len(high_press) * p90, 1)

        results[team] = {
            "team": team,
            "opponent": opp,
            "possession_share": poss,
            "ppda": ppda,
            "ppda_reliable": match_ppda_reliable,
            "high_pressures": len(high_press),
            "high_pressures_p90": high_press_p90,
            "high_press_share": high_press_share,
            "opp_passes_allowed": len(opp_passes),
            "press_zone_def_actions": n_def,
            "read": _verdict(ppda, match_ppda_reliable, poss,
                             len(high_press), high_press_p90),
        }
    return results


# a team pressing high generates ~30+ final-third pressures per 90; a low block <20
HIGH_PRESS_P90 = 30


def _verdict(ppda, reliable, poss, high_pressures, high_press_p90):
    """Honest one-line interpretation with the guardrail applied."""
    if not reliable:
        verdict = "pressed high" if high_press_p90 >= HIGH_PRESS_P90 else "sat off / defended deep"
        return (f"PPDA ({ppda}) UNRELIABLE under skewed possession ({int(poss*100)}%). "
                f"Length-normalized high pressures = {high_press_p90}/90 => {verdict}.")
    if ppda is None:
        return "No defensive actions in pressing zone."
    if ppda < 10:
        band = "aggressive high press"
    elif ppda < 15:
        band = "moderate press"
    else:
        band = "passive / contain-and-block"
    return f"PPDA {ppda} at balanced possession ({int(poss*100)}%) => {band}."


if __name__ == "__main__":
    import sys, json
    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 3869220  # Morocco-Spain
    out = pressing_intensity(mid)
    for team, r in out.items():
        print(f"\n=== {r['team']} (vs {r['opponent']}) ===")
        for k, v in r.items():
            if k not in ("team", "opponent", "read"):
                print(f"  {k}: {v}")
        print(f"  READ: {r['read']}")