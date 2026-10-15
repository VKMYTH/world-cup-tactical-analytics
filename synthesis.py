"""
Synthesis layer.

Takes (question, concept-from-KB, hardened query-result dict) and produces a
grounded natural-language answer.

Design principle: the query layer has ALREADY done the judgement (guardrails,
reliability flags, temporal reads, the "territorial control != superiority"
rule) and emitted it as `read`/`verdict` strings inside the result dict. The LLM
is deliberately given the NARROW job of presenting that evidence packet fluently
for the specific question asked. It does not get to re-interpret the numbers —
that is what keeps it from hallucinating the analysis back in.

Running the real synthesis needs an Anthropic API key in ANTHROPIC_API_KEY.
Use dry_run=True to assemble and inspect the exact prompt without an API call.
"""
import json

# Configurable. Use a current model string for your account.
MODEL = "claude-sonnet-5"
MAX_TOKENS = 700

SYSTEM_PROMPT = """\
You are a football tactical analyst. You answer questions about specific matches \
using ONLY the structured evidence provided to you. That evidence has already \
been computed and sanity-checked by a trusted pipeline; your job is to present \
it faithfully and readably, NOT to add analysis it doesn't support.

Rules:
1. Every quantitative claim must come directly from the COMPUTED EVIDENCE. Use \
the actual numbers. Never invent stats, events, scorelines, xG, or player \
actions that are not present in the evidence.
2. Obey the INTERPRETATION RULES and RELIABILITY GUARDRAIL exactly. If a metric \
is flagged unreliable, do NOT lead with it — use the stated fallback signal. If \
the rules say territorial control is not superiority, never imply it is.
3. The evidence contains pre-computed 'read' / 'verdict' strings. These encode \
the correct, already-hardened judgement. Weave them into a fluent answer for \
THIS question — do not contradict or re-derive them.
4. If the evidence does not contain what is needed to answer, say so plainly \
rather than guessing.
5. Be concise and concrete, like a good analyst briefing a coach. Lead with the \
direct answer, then the supporting numbers.\
"""


def _clean(obj):
    """Coerce numpy/pandas scalar types to native Python so JSON is clean."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


def assemble_evidence(concept, result):
    """Turn a KB concept + query result into a labelled evidence block."""
    result = _clean(result)
    parts = [
        f"CONCEPT: {concept['name']}",
        f"DEFINITION: {concept['definition'].strip()}",
    ]
    if concept.get("interpretation"):
        parts.append("INTERPRETATION RULES (apply exactly; do not contradict):\n"
                     + concept["interpretation"].strip())
    if concept.get("reliability_guardrail"):
        parts.append("RELIABILITY GUARDRAIL:\n"
                     + concept["reliability_guardrail"].strip())
    parts.append("COMPUTED EVIDENCE (the only facts you may use):\n"
                 + json.dumps(result, indent=2, default=str))
    return "\n\n".join(parts)


def build_user_message(question, concept, result):
    return (f"QUESTION: {question}\n\n"
            f"{assemble_evidence(concept, result)}\n\n"
            "Answer the question using only the evidence above.")


def synthesize(question, concept, result, model=MODEL, dry_run=False):
    user = build_user_message(question, concept, result)
    if dry_run:
        return {"system": SYSTEM_PROMPT, "user": user}
    import anthropic  # imported lazily so dry_run needs no SDK/key
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


if __name__ == "__main__":
    import yaml
    from query_pressing import pressing_intensity
    from query_midfield import controlling_midfield

    kb = {c["id"]: c for c in
          yaml.safe_load(open("concept_kb.yaml"))["concepts"]}

    # Example 1: pressing, Morocco vs Spain (guardrail case)
    q1 = "How aggressively did Spain press against Morocco?"
    r1 = pressing_intensity(3869220)["Spain"]
    print("=" * 70, "\nEXAMPLE 1 (pressing / Spain v Morocco)\n", "=" * 70)
    print(synthesize(q1, kb["pressing_intensity"], r1, dry_run=True)["user"])

    # Example 2: midfield, the final (temporal case)
    q2 = "Who controlled the midfield in the final?"
    r2 = controlling_midfield(3869685)
    print("\n" + "=" * 70, "\nEXAMPLE 2 (midfield / final)\n", "=" * 70)
    print(synthesize(q2, kb["controlling_midfield"], r2, dry_run=True)["user"])
