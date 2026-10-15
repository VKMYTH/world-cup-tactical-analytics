"""
Concept retrieval — pluggable backends behind one interface.

WHY: the lexical baseline scores 100% on golden questions we wrote using alias
vocabulary, but only 2/6 on real paraphrases. That gap is overfitting, not
skill. Measured, not assumed — see the para_* cases in golden_set.yaml.

Diagnostic from the baseline: "Did Morocco just sit back and absorb everything?"
routes to `progression_buildup`, because the token "back" collides with the
alias "playing out from the back". Token overlap, zero semantics.

Backends:
  LexicalRetriever   — token overlap. No deps, no downloads. The baseline to beat.
  EmbeddingRetriever — dense vectors over each concept's semantic text.
                       Encodes MEANING, so "sit back and absorb" can reach
                       pressing_intensity without sharing a word with it.

Swap with:
    from retriever import get_retriever
    r = get_retriever("embedding")     # or "lexical"
    r.select("did they sit deep?", k=2)   -> [(concept_id, score), ...]

NOTE ON VALIDATION: the embedding backend could not be validated in the dev
sandbox (no network access to the model host). Run `python compare_retrievers.py`
locally to get the real verdict against the golden set BEFORE trusting it. If it
does not beat 2/6 on the paraphrase cases, it has not earned its dependency.
"""
import re
import yaml
from functools import lru_cache

KB_PATH = "concept_kb.yaml"
_WORD = re.compile(r"[a-z]+")

STOP = set(_WORD.findall(
    "the a an of in to how did do does was were they team match against play "
    "played who what when where why is are and or"))


@lru_cache(maxsize=1)
def load_concepts():
    return yaml.safe_load(open(KB_PATH))["concepts"]


def concept_text(c):
    """The semantic surface of a concept — what a retriever matches against.

    Deliberately name + aliases + definition + tactical_context. NOT the
    data_signature: that is implementation detail (column names, thresholds) and
    would only add noise to the vector.
    """
    return " ".join([
        c["name"],
        " ".join(c.get("aliases", [])),
        c.get("definition", ""),
        c.get("tactical_context", ""),
    ]).strip()


class LexicalRetriever:
    """Token-overlap baseline. Cheap, no deps, overfits to alias vocabulary."""
    name = "lexical"

    def __init__(self, min_score=2.0):
        self.min_score = min_score

    def _tokens(self, s):
        return set(_WORD.findall(s.lower())) - STOP

    def select(self, question, k=2):
        q = self._tokens(question)
        out = []
        for c in load_concepts():
            bag = self._tokens(concept_text(c))
            alias = self._tokens(" ".join(c.get("aliases", [])))
            score = len(q & bag) + 1.5 * len(q & alias)
            if score >= self.min_score:
                out.append((c["id"], round(score, 1)))
        out.sort(key=lambda x: -x[1])
        return out[:k]


# BGE models are TRAINED with an instruction prefix on the query side. Omitting
# it measurably degrades retrieval. v1 of this class omitted it — one of three
# bugs that made embeddings LOSE to lexical (14/20 vs 16/20) on the golden set.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingRetriever:
    """Dense-vector semantic retrieval — MULTI-VECTOR, max-pooled.

    Uses fastembed (ONNX, ~50MB) not sentence-transformers/torch (~2GB): a
    5-concept KB does not need a 2GB dependency, and torch strains the Streamlit
    Cloud free tier.

    THREE FIXES OVER v1 (which lost to lexical — measured, see git history):

    1. QUERY PREFIX. BGE expects an instruction prefix on queries. v1 embedded
       raw question text.

    2. MULTI-VECTOR, NOT ONE BLOB. v1 embedded name+aliases+definition+context as
       a SINGLE vector, so sharp alias phrases ("high press") were diluted by
       paragraphs of prose. Now each alias and the definition get their own
       vector and we take the MAX similarity — a question needs to match ONE
       facet of a concept well, not the average of all of them.

    3. CALIBRATED FLOOR, NOT A GUESS. v1 hardcoded 0.55 and got squeezed: real
       paraphrases fell BELOW it while "What happened in the final?" (which must
       match nothing) sat ABOVE it. Run calibrate_threshold.py to sweep it
       against the golden set instead of guessing.

    The floor exists so the retriever can still return NOTHING. Embeddings always
    have a nearest neighbour, so without it the system silently loses the ability
    to say "I don't know" — a behaviour the golden set explicitly protects.
    """
    name = "embedding"

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", min_score=0.55,
                 use_prefix=True):
        from fastembed import TextEmbedding
        import numpy as np
        self.np = np
        self.model = TextEmbedding(model_name=model_name)
        self.concepts = load_concepts()
        self.min_score = min_score
        self.use_prefix = use_prefix

        # multi-vector: one row per facet, mapped back to its concept
        texts, self.owner = [], []
        for i, c in enumerate(self.concepts):
            facets = [c["name"], c.get("definition", "")] + list(c.get("aliases", []))
            for f in facets:
                if f and f.strip():
                    texts.append(f.strip())
                    self.owner.append(i)
        self.vectors = self._normalize(np.array(list(self.model.embed(texts))))

    def _normalize(self, v):
        return v / self.np.linalg.norm(v, axis=1, keepdims=True)

    def _embed_query(self, q):
        text = (BGE_QUERY_PREFIX + q) if self.use_prefix else q
        return self._normalize(self.np.array(list(self.model.embed([text]))))[0]

    def scores(self, question):
        """Max-pooled cosine per concept. Exposed so the floor can be calibrated."""
        qv = self._embed_query(question)
        sims = self.vectors @ qv
        best = {}
        for owner, s in zip(self.owner, sims):
            cid = self.concepts[owner]["id"]
            if s > best.get(cid, -1):
                best[cid] = float(s)
        return best

    def select(self, question, k=2):
        best = self.scores(question)
        ranked = sorted(best.items(), key=lambda x: -x[1])[:k]
        return [(cid, round(s, 3)) for cid, s in ranked if s >= self.min_score]


class HybridRetriever:
    """Lexical + embedding fusion — the standard production answer.

    Rationale: the two fail differently. Lexical is precise on alias vocabulary
    and blind to paraphrase; embeddings generalize but drift onto near-neighbours.
    Fusing keeps lexical's precision where it fires and falls back to semantics
    where it does not. Scores are normalized per-backend before fusion since
    token counts and cosines are not comparable.
    """
    name = "hybrid"

    def __init__(self, alpha=0.5, **kw):
        self.lex = LexicalRetriever(min_score=0.0)
        self.emb = EmbeddingRetriever(min_score=0.0, **kw)
        self.alpha = alpha          # weight on embedding
        self.min_score = kw.get("min_score", 0.35)

    def select(self, question, k=2):
        emb = self.emb.scores(question)
        lex_raw = dict(self.lex.select(question, k=99))
        lex_max = max(lex_raw.values()) if lex_raw else 0.0
        fused = {}
        for c in load_concepts():
            cid = c["id"]
            e = emb.get(cid, 0.0)
            l = (lex_raw.get(cid, 0.0) / lex_max) if lex_max else 0.0
            fused[cid] = self.alpha * e + (1 - self.alpha) * l
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(cid, round(s, 3)) for cid, s in ranked if s >= self.min_score]


def get_retriever(kind="lexical", **kw):
    if kind == "lexical":
        return LexicalRetriever(**kw)
    if kind == "embedding":
        return EmbeddingRetriever(**kw)
    if kind == "hybrid":
        return HybridRetriever(**kw)
    raise ValueError(f"unknown retriever: {kind}")