"""
Match clock helpers.

WHY THIS MODULE EXISTS — two real bugs, both found by inspecting a real match:

  BUG 1: `ev.minute.max()` includes PERIOD 5 (the penalty shootout). In the 2022
         final the shootout occupies minutes 120-125, so the raw max is 125 —
         a clock reading that corresponds to no actual football. This inflated
         the timeline range (minting a phantom "120-135" window) and corrupted
         the per-90 denominator.

  BUG 2 (worse): StatsBomb's `minute` is NOT MONOTONIC ACROSS PERIODS. Period 1
         runs to minute 52 (45 + 7 stoppage); period 2 then RESTARTS at 45. So
         binning on raw `minute` silently merges first-half stoppage time with
         the early second half. Verified on the final: the naive "45-60" window
         held 55 period-1 events and 106 period-2 events — two distinct phases
         of the match averaged into one number.
         Same collision at 90 (P2 stoppage vs ET start) and 105 (P3 vs P4).

FIX: bin by (period, minute-within-period) using football-native boundaries, and
folding each period's stoppage into that period's FINAL bucket (labelled "+").
This is both correct and how a coach reads a game: "the last 15 of the first
half" includes stoppage, and never bleeds into the second half.
"""

# nominal [start, end) of each real-football period. Period 5 = shootout: excluded.
PERIOD_BOUNDS = {1: (0, 45), 2: (45, 90), 3: (90, 105), 4: (105, 120)}
REAL_PERIODS = set(PERIOD_BOUNDS)


def in_play(ev):
    """Drop the penalty shootout (period 5) — it is not football in open play."""
    return ev[ev.period.isin(REAL_PERIODS)]


def played_minutes(ev):
    """Nominal played minutes: 120 if the match went to extra time, else 90.

    Uses NOMINAL rather than raw wall-clock so per-90 figures stay comparable
    across matches (standard football-analytics practice) and are never skewed
    by how much stoppage a referee happened to add.
    """
    periods = set(ev.period.unique()) & REAL_PERIODS
    return 120 if periods & {3, 4} else 90


def p90_factor(ev):
    """Multiplier to convert a raw count into a per-90 rate."""
    return 90 / played_minutes(ev)


def bucket_start(period, minute, window=15):
    """Period-aware bucket start. Stoppage folds into that period's last bucket.

    Returns None for the shootout so it can never enter a timeline.
    """
    if period not in PERIOD_BOUNDS:
        return None
    lo, hi = PERIOD_BOUNDS[period]
    m = max(min(minute, hi - 1), lo)          # clamp stoppage inside the period
    start = lo + ((m - lo) // window) * window
    return min(start, hi - window)


def bucket_label(start, window=15):
    """'30-45+' marks a bucket that absorbs its period's stoppage time."""
    end = start + window
    absorbs = any(end == hi for _, hi in PERIOD_BOUNDS.values())
    return f"{start}-{end}{'+' if absorbs else ''}"


def add_clock(ev, window=15):
    """Attach period-aware `bucket` + `bucket_label` columns; drops the shootout."""
    ev = in_play(ev).copy()
    ev["bucket"] = [bucket_start(p, m, window)
                    for p, m in zip(ev.period, ev.minute)]
    ev["bucket_label"] = [bucket_label(b, window) if b is not None else None
                          for b in ev["bucket"]]
    return ev


def ordered_buckets(ev, window=15):
    """All real buckets present, in true chronological order."""
    return sorted(b for b in ev["bucket"].dropna().unique())
