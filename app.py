"""
Streamlit UI — a thin window onto the pipeline. NO new logic lives here.

    streamlit run app.py

Deliberately scoped to three things (answer / pitch map / timeline). The
substance of this project is the concept KB and the guardrails; the UI is a
window onto them, not the product. Resist adding more until the remaining
concepts are built.
"""
import os
import streamlit as st
import yaml
from statsbombpy import sb


from router import route
from synthesis import synthesize
from viz import pressure_map, midfield_timeline
import importlib
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="WC22 Tactical Q&A", page_icon="⚽", layout="wide")

KB = {c["id"]: c for c in yaml.safe_load(open("concept_kb.yaml"))["concepts"]}
QUERY_FN = {
    "pressing_intensity": "query_pressing.pressing_intensity",
    "controlling_midfield": "query_midfield.controlling_midfield",
}


@st.cache_data(show_spinner=False)
def _events(match_id):
    return sb.events(match_id=match_id)


@st.cache_data(show_spinner=False)
def _run_query(fn_path, match_id):
    mod, fn = fn_path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), fn)(match_id)


st.title("World Cup 2022 Q&A")
st.caption("Answers grounded in real StatsBomb event data. ")

with st.sidebar:
    st.subheader("How to ask")
    st.markdown(
        "- Name **both teams** (*Spain vs Morocco*) or **team + round** "
        "(*Argentina in the semifinal*)\n"
        "- Built concepts: **pressing**, **midfield control**\n"
        "- Ambiguous questions are **refused, not guessed**"
    )
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    st.info("API key detected — full answers on." if has_key else
            "No ANTHROPIC_API_KEY — showing evidence only (no prose).")

examples = [
    "How aggressively did Spain press against Morocco?",
    "Who controlled the midfield in the final?",
    "Did Argentina press high in the semifinal?",
]
picked = st.selectbox("Try an example", [""] + examples)
question = st.text_input("Your question", value=picked or "")

if question:
    r = route(question)
    match, concepts = r["match"], r["concepts"]

    if "match_id" not in match:
        st.error(f"**Couldn't resolve the match — refusing to guess.** "
                 f"{match.get('error','')}")
        if match.get("candidates"):
            st.write("Did you mean:")
            for c in match["candidates"]:
                st.write(f"- {c}")
        st.stop()

    st.success(f"**Match:** {match['label']}")

    if not concepts:
        st.warning("No tactical concept matched — this looks open-ended. "
                   "Try asking about pressing or midfield control.")
        st.stop()

    c = concepts[0]
    cid = c["id"]
    st.caption(f"Concept: `{cid}`  ·  retrieval score {c['score']}")

    if cid not in QUERY_FN:
        st.warning(f"Concept `{cid}` is in the KB but its query function "
                   "isn't built yet.")
        st.stop()

    with st.spinner("Computing from event data…"):
        result = _run_query(QUERY_FN[cid], match["match_id"])

    # ---- the answer ----
    st.subheader("Answer")
    if has_key:
        with st.spinner("Synthesizing…"):
            st.markdown(synthesize(question, KB[cid], result))
    else:
        st.info("Set ANTHROPIC_API_KEY for the written answer. "
                "The computed evidence below is what it would be built from.")

    # ---- the receipts ----
    st.subheader("The receipts")
    ev = _events(match["match_id"])

    if cid == "pressing_intensity":
        st.pyplot(pressure_map(match["match_id"], ev=ev))
        cols = st.columns(2)
        for col, (team, r_) in zip(cols, result.items()):
            with col:
                st.metric(f"{team} — PPDA", r_["ppda"],
                          "reliable" if r_["ppda_reliable"] else "UNRELIABLE (skewed possession)",
                          delta_color="normal" if r_["ppda_reliable"] else "inverse")
                st.metric("High pressures /90", r_["high_pressures_p90"])
                st.metric("Possession", f"{int(r_['possession_share']*100)}%")
                st.caption(r_["read"])

    elif cid == "controlling_midfield":
        st.pyplot(midfield_timeline(result))
        st.info(result["verdict"])
        cols = st.columns(2)
        for col, (team, r_) in zip(cols, result["teams"].items()):
            with col:
                st.metric(f"{team} — mid-third share (whole match)",
                          r_["mid_third_share_WHOLE_MATCH"], "average — see timeline",
                          delta_color="off")
                st.metric("Possession", f"{int(r_['overall_possession']*100)}%")
                st.metric("Progressions /90", r_["progressions_p90"])

    with st.expander("Raw computed evidence (what the LLM was given)"):
        st.json(result)
