"""
End-to-end pipeline: a raw question in, a grounded answer out.

    answer("Did Argentina press high in the semifinal?")

Chains the three layers:
    router.route      -> match_id + concept_ids   (deterministic match lookup)
    <dispatch>        -> hardened query on real StatsBomb data
    synthesis         -> grounded natural-language answer

Dispatch is dynamic: each concept in the KB names its query function
("module.function"), which we import and call with the resolved match_id.
"""
import importlib
import yaml
from router import route
from synthesis import synthesize

_KB = {c["id"]: c for c in yaml.safe_load(open("concept_kb.yaml"))["concepts"]}


def _call_query(query_fn_path, match_id):
    module_name, fn_name = query_fn_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)(match_id)


def answer(question, top_k=1, dry_run=False):
    r = route(question)
    match = r["match"]

    if "match_id" not in match:                 # ambiguous / not found
        return {"question": question, "resolved": False, "reason": match}
    if not r["concepts"]:
        return {"question": question, "resolved": False,
                "reason": "no tactical concept matched (open-ended question)",
                "match": match["label"]}

    answers = []
    for c in r["concepts"][:top_k]:
        if c["query_fn"] == "TBD":               # concept exists, query fn not built yet
            answers.append({"concept": c["id"], "status": "query_fn not implemented"})
            continue
        result = _call_query(c["query_fn"], match["match_id"])
        out = synthesize(question, _KB[c["id"]], result, dry_run=dry_run)
        answers.append({"concept": c["id"], "result": result, "answer": out})

    return {"question": question, "resolved": True,
            "match": match["label"], "answers": answers}


if __name__ == "__main__":
    for q in ["Did Argentina press high in the semifinal?",
              "Who controlled the midfield in the final?"]:
        res = answer(q, dry_run=True)
        print("=" * 70)
        print("Q:", res["question"])
        print("MATCH:", res.get("match"))
        for a in res["answers"]:
            print("CONCEPT:", a["concept"])
            # dry_run => 'answer' is the assembled prompt dict; show it flowed real numbers
            user = a["answer"]["user"]
            evidence_start = user.index("COMPUTED EVIDENCE")
            print("  ...evidence handed to LLM (first 240 chars):")
            print("  " + user[evidence_start:evidence_start + 240].replace("\n", "\n  "))
