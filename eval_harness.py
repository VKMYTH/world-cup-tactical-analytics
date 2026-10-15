"""
Eval harness for the 2022 WC tactical pipeline.

THREE SEPARABLE GRADERS — scored independently so a failure tells you WHICH
layer broke, not just that "the answer was bad":

  1. routing      — did the router resolve the right match + right concept?
                    Includes NEGATIVE cases: refusing an ambiguous question is
                    a PASS; guessing is a FAIL.
  2. computation  — do the computed numbers match externally-verifiable truth
                    (scorelines, possession shares, goal minutes)? This is the
                    layer that catches a metric quietly lying.
  3. faithfulness — does the generated ANSWER respect the KB guardrails, or does
                    it parrot the impressive-looking raw number? Needs an API
                    key; skipped otherwise.

Usage:
    python eval_harness.py             # routing + computation (no API key needed)
    python eval_harness.py --live      # adds faithfulness grader
"""
import sys
import yaml
import importlib
from router import route
from synthesis import synthesize

GOLDEN = yaml.safe_load(open("golden_set.yaml"))["cases"]
_KB = {c["id"]: c for c in yaml.safe_load(open("concept_kb.yaml"))["concepts"]}

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def _call_query(path, match_id):
    mod, fn = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), fn)(match_id)


# ----------------------------- grader 1: routing -----------------------------

def grade_routing(case):
    exp = case["expect"]
    r = route(case["question"])
    got_id = r["match"].get("match_id")
    got_concepts = [c["id"] for c in r["concepts"]]
    top = got_concepts[0] if got_concepts else None

    fails = []
    if exp.get("match_id") is None and "match_id" in exp:
        if got_id is not None:
            fails.append(f"should have REFUSED, guessed match {got_id}")
    elif exp.get("match_id") != got_id:
        fails.append(f"match: expected {exp.get('match_id')}, got {got_id}")

    # ground truth is either `concept` (strict) or `concept_any` (a set of
    # legitimate readings — see the header of golden_set.yaml)
    if "concept_any" in exp:
        if top not in exp["concept_any"]:
            fails.append(f"concept: expected one of {exp['concept_any']}, got {top}")
    elif "concept" in exp:
        if exp["concept"] is None:
            if top is not None:
                fails.append(f"should match NO concept, got {top}")
        elif top != exp["concept"]:
            fails.append(f"concept: expected {exp['concept']}, got {top}")

    return (not fails), fails, r


# --------------------------- grader 2: computation ---------------------------

def _get(result, path, team=None):
    """Pull a value from a query result, handling both result shapes."""
    if team:
        if team in result:                      # pressing shape: {team: {...}}
            return result[team].get(path)
        if "teams" in result and team in result["teams"]:   # midfield shape
            return result["teams"][team].get(path)
        return None
    return result.get(path)


def grade_computation(case):
    exp = case["expect"]
    checks = exp.get("numeric", [])
    has_goals = "goals" in exp
    has_swing = "timeline_swing" in exp
    has_clock = "clock" in exp
    if not (checks or has_goals or has_swing or has_clock):
        return None, [], None                   # no computation assertions

    concept_id = exp.get("concept") or exp["concept_any"][0]
    kb_entry = _KB[concept_id]
    fn = {"pressing_intensity": "query_pressing.pressing_intensity",
          "controlling_midfield": "query_midfield.controlling_midfield",
          "defensive_line_height": "query_line_height.defensive_line_height",
          }.get(concept_id)
    if not fn:
        return None, [f"no query fn for {concept_id}"], None

    result = _call_query(fn, exp["match_id"])
    fails = []

    for c in checks:
        val = _get(result, c["path"], c.get("team"))
        label = f"{c.get('team','')}.{c['path']}"
        if val is None:
            fails.append(f"{label}: MISSING")
            continue
        if "equals" in c and val != c["equals"]:
            fails.append(f"{label}: expected {c['equals']}, got {val}")
        if "min" in c and val < c["min"]:
            fails.append(f"{label}: {val} < min {c['min']}")
        if "max" in c and val > c["max"]:
            fails.append(f"{label}: {val} > max {c['max']}")

    if has_goals:
        goals = result.get("goals", [])
        mins = [m for m, _ in goals]
        if len(goals) != exp["goals"]["count"]:
            fails.append(f"goals: expected {exp['goals']['count']}, got {len(goals)} -> {mins}")
        if mins != exp["goals"]["minutes"]:
            fails.append(f"goal minutes: expected {exp['goals']['minutes']}, got {mins}")

    if "clock" in exp:
        spec = exp["clock"]
        # no window may extend past nominal full/extra time
        for w in result.get("timeline", []):
            end = int(w["window"].rstrip("+").split("-")[1])
            if end > spec["max_window_end"]:
                fails.append(f"phantom window {w['window']} exceeds "
                             f"{spec['max_window_end']}' (shootout leaking into clock?)")
        # Assert on the CLOCK ITSELF, not just the timeline. (First version of
        # this check asserted only on the midfield timeline — but shootout events
        # are penalty-spot shots, never mid-third passes, so they could never
        # reach that path. The mutation passed. Test where the bug actually lives.)
        if spec.get("single_period_buckets"):
            from clock import add_clock
            from statsbombpy import sb as _sb
            _ev = add_clock(_sb.events(match_id=exp["match_id"]))
            # 1) the shootout must be excluded from the clock entirely
            leaked = sorted(set(_ev.period.unique()) - {1, 2, 3, 4})
            if leaked:
                fails.append(f"non-football periods leaked into clock: {leaked} "
                             f"(period 5 = penalty shootout)")
            # 2) no bucket may start at/after nominal ET end
            late = sorted({int(b) for b in _ev.bucket.dropna().unique()
                           if b >= spec["max_window_end"]})
            if late:
                fails.append(f"buckets at/after {spec['max_window_end']}': {late}")
            # 3) every bucket must hold exactly one period
            impure = _ev.groupby("bucket").period.nunique()
            bad = [int(b) for b, n in impure.items() if n > 1]
            if bad:
                fails.append(f"buckets mixing >1 period: {bad} "
                             f"(non-monotonic `minute` across periods)")

    if has_swing:
        doms = {w["dominant"] for w in result.get("timeline", []) if w["dominant"]}
        if len(doms) < 2:
            fails.append(f"timeline should swing (both teams dominant somewhere), got {doms}")

    return (not fails), fails, result


# -------------------------- grader 3: faithfulness ---------------------------

def grade_faithfulness(case, result):
    exp = case["expect"]
    spec = exp.get("faithfulness")
    if not spec:
        return None, [], None
    text = synthesize(case["question"], _KB[exp["concept"]], result)
    low = text.lower()
    fails = []
    any_hits = [m for m in spec.get("must_mention_any", []) if m.lower() in low]
    if spec.get("must_mention_any") and not any_hits:
        fails.append(f"mentions none of {spec['must_mention_any']}")
    for bad in spec.get("must_not_claim", []):
        if bad.lower() in low:
            fails.append(f"makes forbidden claim: '{bad}'")
    return (not fails), fails, text


# --------------------------------- runner ------------------------------------

def run(live=False):
    tallies = {"routing": [0, 0], "computation": [0, 0], "faithfulness": [0, 0]}

    for case in GOLDEN:
        print(f"\n{case['id']}")
        print(f"  Q: {case['question']}")

        ok, fails, routed = grade_routing(case)
        tallies["routing"][1] += 1
        tallies["routing"][0] += ok
        print(f"  {'PASS' if ok else 'FAIL'} routing" +
              ("" if ok else "  -> " + "; ".join(fails)))

        ok_c, fails_c, result = grade_computation(case)
        if ok_c is not None:
            tallies["computation"][1] += 1
            tallies["computation"][0] += ok_c
            print(f"  {'PASS' if ok_c else 'FAIL'} computation" +
                  ("" if ok_c else "  -> " + "; ".join(fails_c)))

        if live and result is not None:
            ok_f, fails_f, text = grade_faithfulness(case, result)
            if ok_f is not None:
                tallies["faithfulness"][1] += 1
                tallies["faithfulness"][0] += ok_f
                print(f"  {'PASS' if ok_f else 'FAIL'} faithfulness" +
                      ("" if ok_f else "  -> " + "; ".join(fails_f)))

    print("\n" + "=" * 60)
    for k, (p, n) in tallies.items():
        if n:
            print(f"  {k:<14} {p}/{n}  ({round(100*p/n)}%)")
        else:
            print(f"  {k:<14} skipped")
    print("=" * 60)


if __name__ == "__main__":
    run(live="--live" in sys.argv)