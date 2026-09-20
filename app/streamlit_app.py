"""AmbiSense - Streamlit demo UI (Phase 6).

Thin presentation layer over the same pipeline the CLI uses
(``ambisense.pipeline.analyze``): this file computes nothing about ambiguity
itself, it only lays the existing report out for a browser, the way
``cli/render.py`` lays it out for a terminal.

Run with:

    streamlit run app/streamlit_app.py

Streamlit re-executes this whole script on every user interaction, so the
expensive one-time setup (loading settings, the spaCy pipeline, and the LLM
adjudicator) is wrapped in ``st.cache_resource`` and built once per server
process rather than once per click.
"""

from __future__ import annotations

import html
import json

import streamlit as st

from ambisense.config import ConfigError, load_settings
from ambisense.llm import build_adjudicator
from ambisense.llm.health import check_server, ollama_pull_hint, ollama_serve_hint
from ambisense.pipeline import SetupError, analyze
from ambisense.preprocessing import InputValidationError, LinguisticAnalyzer, ModelLoadError
from ambisense.schemas import AdjudicationReport, RewriteStatus, SentenceVerdict

st.set_page_config(page_title="AmbiSense", page_icon="\U0001F50D", layout="wide")

#: icon, accent colour, human label - one entry per SentenceVerdict.
VERDICT_STYLE = {
    SentenceVerdict.AMBIGUOUS: ("\U0001F7E7", "#b45309", "Ambiguous"),
    SentenceVerdict.NOT_AMBIGUOUS: ("\U0001F7E9", "#15803d", "Not ambiguous"),
    SentenceVerdict.UNCERTAIN: ("⬜", "#57534e", "Uncertain"),
    SentenceVerdict.INCOMPLETE: ("\U0001F7E8", "#a16207", "Incomplete analysis"),
    SentenceVerdict.NO_CANDIDATES: ("\U0001F7E9", "#15803d", "No candidates found"),
}


@st.cache_resource
def get_settings():
    return load_settings()


@st.cache_resource
def get_analyzer(_settings) -> LinguisticAnalyzer:
    return LinguisticAnalyzer(_settings.nlp.spacy_model)


@st.cache_resource
def get_adjudicator(_settings):
    return build_adjudicator(_settings)


def load_demo_cases(settings) -> list[dict]:
    """The same file ``--live-llm-test`` uses - one set of demo sentences."""
    path = settings.project_root / "data" / "examples" / "adjudication_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def render_highlighted_text(text: str, report: AdjudicationReport) -> str:
    """Wrap genuine-verdict spans in ``<mark>``; everything else is escaped.

    Only candidates the LLM confirmed as genuine are highlighted, not every
    rule-based candidate - the detectors over-flag by design (Section 11.2 of
    the README), so highlighting their raw output would mislead a reader
    into thinking every flagged span is a confirmed ambiguity.
    """
    spans = sorted(
        (
            (item.candidate.char_start, item.candidate.char_end, item.candidate)
            for item in report.adjudications
            if item.is_genuine
        ),
        key=lambda entry: entry[0],
    )
    pieces: list[str] = []
    cursor = 0
    for start, end, candidate in spans:
        if start < cursor:
            continue  # overlapping span; skip rather than emit broken markup
        pieces.append(html.escape(text[cursor:start]))
        tooltip = html.escape(f"{candidate.type_hint.value}: {candidate.explanation}")
        pieces.append(
            '<mark style="background-color:#fde68a;padding:0 2px;'
            f'border-radius:3px;" title="{tooltip}">'
            f"{html.escape(text[start:end])}</mark>"
        )
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def render_candidate(item) -> None:
    candidate = item.candidate
    header = f"[{item.candidate_id}] {candidate.type_hint.value.upper()} - '{candidate.span_text}'"
    with st.expander(header, expanded=item.is_genuine):
        st.markdown("**1. Rule-based candidate (Phase 2)**")
        st.write(candidate.explanation)

        st.markdown("**2. Semantic evidence (Phase 3)**")
        ranking = item.sense_ranking
        if ranking is None:
            st.caption("N/A (sense ranking applies to lexical candidates only)")
        elif not ranking.status.is_usable:
            st.caption(f"unavailable: {ranking.status.value}")
        else:
            for sense in ranking.senses[:3]:
                similarity = "n/a" if sense.context_similarity is None else f"{sense.context_similarity:.3f}"
                st.write(f"#{sense.rank} {similarity}  {sense.sense_key} - {sense.definition[:80]}")
            margin = "n/a" if ranking.margin is None else f"{ranking.margin:.3f}"
            st.caption(f"margin over runner-up: {margin} (similarity, not probability)")

        st.markdown("**3. LLM judgement (Phase 4)**")
        judgement = item.judgement
        if judgement is None:
            st.error(f"status: {item.status.value} - {item.error}")
            st.caption("No verdict. A failure is never reported as 'not ambiguous'.")
            return

        st.write(f"Verdict: **{judgement.verdict.value.upper()}**")
        confidence = (
            "not reported" if judgement.confidence is None
            else f"{judgement.confidence:.2f} (self-reported)"
        )
        st.caption(f"Confidence: {confidence}")
        st.write(judgement.explanation)

        heading = "Interpretations" if item.is_genuine else "Readings considered"
        st.markdown(f"*{heading}:*")
        for number, interpretation in enumerate(judgement.interpretations, start=1):
            st.write(f"{number}. {interpretation.meaning}")
            if interpretation.explanation:
                st.caption(interpretation.explanation)

        if item.rewrite_status is RewriteStatus.GENERATED:
            st.markdown("*Suggested rewrites:*")
            for number, rewrite in enumerate(item.rewrites, start=1):
                st.write(f"{number}. {rewrite}")


def main() -> None:
    st.title("AmbiSense")
    st.caption("Hybrid NLP + LLM ambiguity detection — academic prototype")

    try:
        settings = get_settings()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    with st.sidebar:
        st.subheader("LLM server")
        server = check_server(settings.llm)
        if server.ready:
            st.success(f"{settings.llm.provider} / {settings.llm.model} ready")
        elif server.reachable:
            st.warning(f"Model not installed. {ollama_pull_hint(settings.llm.model)}")
        else:
            st.error(f"Server unreachable. {ollama_serve_hint()}")
        st.caption(
            "Local LLM calls take from a few seconds to several minutes per "
            "sentence, depending on the model (README Section 13.16)."
        )

    demo_cases = load_demo_cases(settings)
    demo_labels = ["(custom)"] + [f"{case['id']}: {case['text']}" for case in demo_cases]
    choice = st.selectbox("Demo sentence", demo_labels)
    default_text, default_context = "", ""
    if choice != "(custom)":
        selected = demo_cases[demo_labels.index(choice) - 1]
        default_text = selected["text"]
        default_context = selected.get("context", "") or ""

    text = st.text_area("Sentence", value=default_text, height=80)
    context = st.text_input("Optional context", value=default_context)

    analyze_clicked = st.button("Analyze", disabled=not server.ready, type="primary")
    if not server.ready:
        st.info("Analysis is disabled until the local Ollama server and model are ready.")

    if not analyze_clicked:
        return
    if not text.strip():
        st.warning("Enter a sentence to analyze.")
        return

    with st.spinner("Rule-based detection, semantic ranking, then LLM adjudication..."):
        try:
            analyzer = get_analyzer(settings)
            adjudicator = get_adjudicator(settings)
            report = analyze(
                settings, text, context.strip() or None,
                adjudicator=adjudicator, analyzer=analyzer,
            )
        except InputValidationError as exc:
            st.error(f"Input error: {exc}")
            return
        except (ModelLoadError, SetupError) as exc:
            st.error(f"Setup error: {exc}")
            return

    icon, color, label = VERDICT_STYLE[report.summary.verdict]
    st.markdown(
        f"### {icon} Sentence verdict: <span style='color:{color}'>{label}</span>",
        unsafe_allow_html=True,
    )
    st.write(report.summary.explanation)

    st.markdown("#### Highlighted text")
    st.markdown(
        f"<p style='font-size:1.1rem'>{render_highlighted_text(report.text, report)}</p>",
        unsafe_allow_html=True,
    )
    if not any(item.is_genuine for item in report.adjudications):
        st.caption("No spans are highlighted: no candidate was confirmed as a genuine ambiguity.")

    st.markdown("#### Candidates")
    if not report.adjudications:
        st.info(report.summary.explanation)
    for item in report.adjudications:
        render_candidate(item)

    diagnostics = report.diagnostics
    st.markdown("---")
    st.caption(
        f"LLM: {diagnostics.llm_provider} {diagnostics.llm_model} | "
        f"requests: {diagnostics.llm_attempts} | "
        f"cache hit: {diagnostics.cache_hit} | "
        f"repair: {diagnostics.repair_attempted}"
    )
    st.caption(
        "LLM confidence is self-reported and NOT a calibrated probability. "
        "Verdicts are judgements over the evidence shown above and can be wrong."
    )


if __name__ == "__main__":
    main()
