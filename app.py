"""Streamlit UI. Run with:  streamlit run app.py

Designed for explainability: next to every answer it shows the retrieved
evidence, similarity scores, which sources the answer actually cited,
and the exact prompt sent to the LLM.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from workplace_rag.config import load_config  # noqa: E402
from workplace_rag.factory import load_pipeline  # noqa: E402

EXAMPLES = [
    "How much statutory sick pay can an employee get?",
    "Can an employee ask to work from home?",
    "How much notice do I need to give before making someone redundant?",
    "What counts as a reasonable adjustment for a disabled worker?",
]

st.set_page_config(page_title="Workplace Rights Assistant", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Loading models and index…")
def get_pipeline():
    return load_pipeline()


cfg = load_config()
pipeline = get_pipeline()

with st.sidebar:
    st.header("Settings")
    pipeline.top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, cfg["retrieval"]["top_k"])
    topics = ["(all topics)"] + list(cfg["corpus"].keys())
    topic = st.selectbox("Restrict to topic", topics)
    st.divider()
    st.caption(f"LLM: **{pipeline.llm.provider}** · `{pipeline.llm.model}`")
    st.caption(f"Embeddings: `{cfg['embeddings']['model_name']}`")
    st.caption(f"Not-found threshold: {pipeline.min_score}")
    st.caption("General information from GOV.UK, not legal advice.")

st.title("⚖️ Workplace Rights Assistant")
st.write("Answers UK employee-rights questions from official GOV.UK guidance, with sources.")

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex):
        st.session_state["q"] = ex

question = st.text_input("Your question", key="q", placeholder="e.g. Is holiday pay included in a zero-hours contract?")

if question:
    with st.spinner("Retrieving guidance and drafting an answer…"):
        r = pipeline.answer(question, topic=None if topic == "(all topics)" else topic)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Answer")
        st.markdown(r.answer)
        if r.guardrail:
            st.info(f"The LLM was not called. {r.guardrail}.")
        elif r.invalid_citations:
            st.error(f"Answer cites sources that don't exist: {r.invalid_citations}")
        elif not r.cited:
            st.warning("The answer contains no citations — treat with caution.")
        else:
            st.success(f"All citations valid · uses sources {r.cited}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Retrieval", f"{r.retrieval_s:.2f}s")
        m2.metric("Generation", f"{r.generation_s:.2f}s")
        m3.metric("Tokens in / out", f"{r.prompt_tokens or 0} / {r.completion_tokens or 0}")

    with right:
        st.subheader("Evidence")
        for s in r.sources:
            used = s.rank in r.cited
            label = f"{'✅' if used else '▫️'} [{s.rank}] {s.title} › {s.heading}  —  score {s.score:.2f}"
            with st.expander(label, expanded=used):
                st.progress(max(0.0, min(1.0, s.score)))
                st.write(s.body)
                st.caption(f"Topic: {s.topic} · [Open on GOV.UK]({s.url})")

    if r.messages:
        with st.expander("🔍 Exact prompt sent to the LLM"):
            for m in r.messages:
                st.markdown(f"**{m['role']}**")
                st.code(m["content"], language=None)
