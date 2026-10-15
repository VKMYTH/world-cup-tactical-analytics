"""
Visualization layer — the "receipts".

Turns the SAME hardened query results into pitch maps and charts. These are not
decoration: the pressure map is visual proof that the numbers came from real
located events, which is the whole thesis of the project.

Design rule (mirrors the rest of the pipeline): viz reads ONLY from the event
data / result dicts. It never re-derives or re-interprets — no new logic here.

COORDINATE NOTE: StatsBomb normalizes every team to attack toward x=120 in its
OWN event rows. So plotting both teams' events on one pitch without flipping is
correct *per team* but must be drawn as SEPARATE panels — never overlaid, or
you'd imply they were attacking the same way. Hence side-by-side panels.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from statsbombpy import sb
from clock import bucket_start, bucket_label, in_play
import warnings
warnings.filterwarnings("ignore")

BG = "#0e1117"
LINE = "#c7d0d9"
ACCENT = "#00d4aa"
ACCENT2 = "#ff6b6b"


def _x(loc):
    return loc[0] if isinstance(loc, list) else None


def _y(loc):
    return loc[1] if isinstance(loc, list) else None


def _prep(ev):
    ev = in_play(ev).copy()      # shootout shots/pressures are not open play
    ev["x"] = ev["location"].apply(_x)
    ev["y"] = ev["location"].apply(_y)
    return ev


def pressure_map(match_id, ev=None, figsize=(13, 6)):
    """Side-by-side pressure locations per team — the deep-block vs high-press proof."""
    if ev is None:
        ev = sb.events(match_id=match_id)
    ev = _prep(ev)
    teams = [t for t in ev["team"].dropna().unique()]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(BG)
    pitch = Pitch(pitch_type="statsbomb", pitch_color=BG, line_color=LINE,
                  linewidth=1)

    for ax, team, color in zip(axes, teams, (ACCENT, ACCENT2)):
        pitch.draw(ax=ax)
        p = ev[(ev.team == team) & (ev.type == "Pressure")].dropna(subset=["x", "y"])
        # heat-ish kde + points so both density and raw events are legible
        if len(p) > 5:
            pitch.kdeplot(p.x, p.y, ax=ax, fill=True, levels=50,
                          thresh=0.05, cmap="mako", alpha=0.7, zorder=1)
        pitch.scatter(p.x, p.y, ax=ax, s=28, color=color, edgecolors="white",
                      linewidth=0.4, alpha=0.85, zorder=2)
        high = p[p.x > 80]
        ax.set_title(f"{team} — pressures: {len(p)}  (final third: {len(high)})",
                     color="white", fontsize=11, pad=8)
        # mark the final-third line: the "pressing high" boundary
        ax.axvline(80, color="white", ls="--", lw=1, alpha=0.5, zorder=3)

    fig.suptitle("Where each team pressed  →  attacking direction left-to-right",
                 color=LINE, fontsize=10, y=0.04)
    fig.tight_layout()
    return fig


def midfield_timeline(result, figsize=(11, 4.5)):
    """Control-share swing over time — the point of the temporal metric."""
    tl = result["timeline"]
    teams = list(result["teams"].keys())
    t0, t1 = teams[0], teams[1]
    labels = [w["window"] for w in tl]
    s0 = [w["shares"].get(t0, 0) for w in tl]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = range(len(labels))
    ax.axhline(0.5, color=LINE, ls="--", lw=1, alpha=0.6)
    ax.fill_between(x, s0, 0.5, where=[v >= 0.5 for v in s0],
                    color=ACCENT, alpha=0.75, interpolate=True, label=t0)
    ax.fill_between(x, s0, 0.5, where=[v < 0.5 for v in s0],
                    color=ACCENT2, alpha=0.75, interpolate=True, label=t1)
    ax.plot(x, s0, color="white", lw=1.6, zorder=3)

    # Goal markers — mapped through the PERIOD-AWARE clock, not minute/15.
    # Raw minute/15 misplaces any goal in stoppage time (a 45+2' goal would land
    # in the second-half bucket). Match the goal's label to the axis labels.
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    for minute, scorer in result.get("goals", []):
        placed = None
        for period in (1, 2, 3, 4):
            b = bucket_start(period, minute)
            if b is None:
                continue
            lab = bucket_label(b)
            if lab in label_to_idx:
                placed = label_to_idx[lab]
                break
        if placed is not None:
            ax.axvline(placed, color="#ffd166", lw=1.4, alpha=0.9, zorder=4)
            ax.text(placed, 1.02, f"⚽{minute}'", color="#ffd166",
                    fontsize=7.5, ha="center")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=LINE, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mid-third share", color=LINE, fontsize=9)
    ax.tick_params(colors=LINE)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.legend(facecolor=BG, edgecolor="#333", labelcolor=LINE, fontsize=9,
              loc="lower right")
    ax.set_title("Midfield control by phase — the whole-match average hides this swing",
                 color="white", fontsize=11, pad=10)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from query_midfield import controlling_midfield
    # 1) pressure map: Morocco 0-0 Spain — the deep block vs high press
    fig = pressure_map(3869220)
    fig.savefig("/home/claude/preview_pressure.png", dpi=110, facecolor=BG)
    print("saved preview_pressure.png")
    # 2) timeline: the final — the swing
    res = controlling_midfield(3869685)
    fig2 = midfield_timeline(res)
    fig2.savefig("/home/claude/preview_timeline.png", dpi=110, facecolor=BG)
    print("saved preview_timeline.png")