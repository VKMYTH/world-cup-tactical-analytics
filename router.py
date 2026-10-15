"""
Router: raw question -> (match_id, concept_ids) so the pipeline can run without
a human wiring it by hand.

Two jobs, deliberately kept separate because they have different trust models:

  1. resolve_match  — DETERMINISTIC. Extract team/stage entities from the text
     and look them up in the real StatsBomb matches table. The LLM is never
     allowed to emit a match_id (it would hallucinate one); IDs only ever come
     from a real lookup.

  2. select_concepts — RETRIEVAL over the concept KB. Lexical baseline here
     (token overlap vs each concept's aliases + definition). Swap in embedding
     similarity for the production version — the interface stays the same.
"""
import os
import re
import yaml
from functools import lru_cache
from statsbombpy import sb
from retriever import get_retriever
import warnings
warnings.filterwarnings("ignore")

COMP_ID, SEASON_ID = 43, 106  # 2022 Men's World Cup

# a few common aliases -> StatsBomb team names
TEAM_ALIASES = {
    "south korea": "Korea Republic",
    "korea": "Korea Republic",
    "iran": "IR Iran",
    "usa": "United States",
    "us": "United States",
}


@lru_cache(maxsize=1)
def _matches():
    return sb.matches(competition_id=COMP_ID, season_id=SEASON_ID)


@lru_cache(maxsize=1)
def _team_names():
    m = _matches()
    return sorted(set(m.home_team) | set(m.away_team))


def _find_teams(q):
    ql = q.lower()
    found = []
    for name in _team_names():
        if name.lower() in ql:
            found.append(name)
    for alias, canonical in TEAM_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", ql) and canonical not in found:
            found.append(canonical)
    return found


# checked in order — compound stages BEFORE bare "final" so that "semifinal"
# and "quarterfinal" don't collide with the "final" substring
STAGE_PRIORITY = [
    ("round of 16", "Round of 16"),
    ("last 16", "Round of 16"),
    ("quarter-final", "Quarter-finals"),
    ("quarterfinal", "Quarter-finals"),
    ("quarter", "Quarter-finals"),
    ("semi-final", "Semi-finals"),
    ("semifinal", "Semi-finals"),
    ("semi", "Semi-finals"),
    ("third place", "3rd Place Final"),
    ("group", "Group Stage"),
    ("final", "Final"),  # last: only matches a bare "final"
]


def _find_stage(q):
    ql = q.lower()
    for kw, stage in STAGE_PRIORITY:
        if kw in ql:
            return stage
    return None


def resolve_match(question):
    """Return {'match_id', 'label'} or {'error', 'candidates'} — never guesses."""
    m = _matches()
    teams = _find_teams(question)
    stage = _find_stage(question)

    cand = m
    if teams:
        for t in teams:
            cand = cand[(cand.home_team == t) | (cand.away_team == t)]
    if stage:
        cand = cand[cand.competition_stage == stage]

    if len(teams) >= 2:
        t1, t2 = teams[0], teams[1]
        cand = m[((m.home_team == t1) & (m.away_team == t2)) |
                 ((m.home_team == t2) & (m.away_team == t1))]

    if len(cand) == 1:
        r = cand.iloc[0]
        return {"match_id": int(r.match_id),
                "label": f"{r.home_team} {r.home_score}-{r.away_score} {r.away_team} ({r.competition_stage})"}

    if len(cand) == 0:
        return {"error": "no match found",
                "teams_seen": teams, "stage_seen": stage}
    return {"error": "ambiguous — add a team or stage",
            "candidates": [f"{r.home_team} v {r.away_team} ({r.competition_stage})"
                           for _, r in cand.iterrows()][:8]}


# ----------------------------- concept retrieval -----------------------------

# ----------------------------- concept retrieval -----------------------------
# Backend is pluggable (see retriever.py). Default stays LEXICAL until
# compare_retrievers.py shows embeddings actually win on the golden set —
# measured baseline: lexical is 100% on alias-worded questions but only 2/6 on
# real paraphrases. Switch with:  RETRIEVER=embedding python ...
RETRIEVER_KIND = os.environ.get("RETRIEVER", "lexical")


@lru_cache(maxsize=2)
def _retriever(kind):
    return get_retriever(kind)


def select_concepts(question, k=2):
    """Ranked [(concept_id, score)]. Scores are NOT comparable across backends:
    lexical = token counts, embedding = cosine similarity."""
    return _retriever(RETRIEVER_KIND).select(question, k=k)


# concept_id -> query function name (dispatch table)
CONCEPT_QUERY = {
    "pressing_intensity": "query_pressing.pressing_intensity",
    "controlling_midfield": "query_midfield.controlling_midfield",
    "defensive_line_height": "query_line_height.defensive_line_height",
    # progression_buildup, zone_14_threat: query fns TBD
}


def route(question):
    """Full route: question -> match + concepts + dispatch targets."""
    match = resolve_match(question)
    concepts = select_concepts(question)
    return {
        "question": question,
        "match": match,
        "concepts": [
            {"id": cid, "score": s, "query_fn": CONCEPT_QUERY.get(cid, "TBD")}
            for cid, s in concepts
        ],
    }


if __name__ == "__main__":
    tests = [
        "How aggressively did Spain press against Morocco?",
        "Who controlled the midfield in the final?",
        "Did Argentina press high in the semifinal?",
        "How did Croatia build up against Brazil?",
        "Who controlled midfield in Argentina vs Croatia?",
        "What happened in the final?",            # no concept -> should be empty
        "How did England press?",                 # team played several -> ambiguous
    ]
    import json
    for q in tests:
        print(json.dumps(route(q), indent=2, default=str))
        print("-" * 60)