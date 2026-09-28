"""
Demo web app (Streamlit).

Run with:  uv run streamlit run app/app.py

Left side : pick one or more models and which entity types to hide.
Main area : paste text, then see per model (side by side) the anonymized
            version, the original with the found entities highlighted, and
            a table of what was found.
Second tab: the benchmark report.
"""

import html
from pathlib import Path

import streamlit as st

from anonymizer.anonymize import anonymize, build_pipeline
from anonymizer.extractors.registry import available_models
from anonymizer.spans import TARGET_LABELS, Span

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "benchmark.md"

SAMPLE_TEXT = (
    "John Smith works for Apple Inc. as a software engineer. His email address is "
    "john.smith@apple.com and his phone number is 015 29 58 58. On 2022-12-27 he paid "
    "€50.000 from BE68 5390 0754 7034. He studied at Harvard Law School and lives in London."
)

# One background colour per entity type, for the highlighted view.
COLORS = {
    "PERSON": "#ffd6a5", "ORG": "#caffbf", "JOB": "#fdffb6", "EMAIL_ADDRESS": "#9bf6ff",
    "LOCATION": "#a0c4ff", "AMOUNT": "#bdb2ff", "DATE_TIME": "#ffc6ff", "UNIVERSITY": "#e2f0cb",
    "PHONE_NUMBER": "#ffadad", "URL": "#c9e4de", "IBAN": "#f1c0e8", "SSN": "#fbf8cc",
}


@st.cache_resource(show_spinner="Loading model...")
def get_pipeline(model_key: str):
    """Build the model + regex pipeline once and keep it in memory between clicks."""
    return build_pipeline(model_key)


def highlight(text: str, spans: list[Span]) -> str:
    """Turn the text into HTML where every span is a coloured box with its label."""
    parts, cursor = [], 0
    for span in spans:  # spans are sorted and never overlap
        parts.append(html.escape(text[cursor : span.start]))
        parts.append(
            f'<mark style="background:{COLORS[span.label]}; padding:2px 4px; border-radius:4px">'
            f"{html.escape(span.text(text))}"
            f'<small style="opacity:.6"> {span.label}</small></mark>'
        )
        cursor = span.end
    parts.append(html.escape(text[cursor:]))
    return '<div style="line-height:2">' + "".join(parts) + "</div>"


def differences(text: str, spans_by_model: dict[str, list[Span]]) -> list[dict]:
    """
    One row per entity text on which the models disagree.
    Each model column holds the label that model gave, or "-" if it did not
    find that entity. Entities every model agrees on are left out.
    """
    # entity text -> {model: label}, in reading order
    labels_by_entity: dict[str, dict[str, str]] = {}
    for key, spans in spans_by_model.items():
        for span in spans:
            labels_by_entity.setdefault(span.text(text), {})[key] = span.label

    rows = []
    for entity, labels in labels_by_entity.items():
        row = {"entity": entity} | {key: labels.get(key, "-") for key in spans_by_model}
        if len(set(row[key] for key in spans_by_model)) > 1:  # not everyone agrees
            rows.append(row)
    return rows


st.set_page_config(page_title="Anonymizer", layout="wide")
st.title("Text anonymizer")

# ---- sidebar: settings ----------------------------------------------------
st.sidebar.header("Settings")
model_keys = st.sidebar.multiselect("Models (pick several to compare)", available_models(), default=["gliner"])
st.sidebar.caption("Every model is combined with the same regex layer for emails, phones, IBANs, ... "
                   "The llm model only appears when Ollama is running.")

st.sidebar.subheader("Entity types to hide")
selected = {label for label in TARGET_LABELS if st.sidebar.checkbox(label, value=True)}

# ---- main area ------------------------------------------------------------
tab_demo, tab_report = st.tabs(["Anonymize", "Benchmark"])

with tab_demo:
    text = st.text_area("Text", SAMPLE_TEXT, height=150)
    if not model_keys:
        st.warning("Pick at least one model in the sidebar.")
    elif st.button("Anonymize", type="primary") and text.strip():
        spans_by_model: dict[str, list[Span]] = {}
        # One column per model, so the results can be compared side by side.
        for key, column in zip(model_keys, st.columns(len(model_keys))):
            with column:
                st.subheader(key)
                spans = get_pipeline(key).extract(text)
                shown = [s for s in spans if s.label in selected]
                spans_by_model[key] = shown

                st.caption("Anonymized text")
                st.code(anonymize(text, spans, selected), language=None, wrap_lines=True)

                st.caption("Found entities")
                st.markdown(highlight(text, shown), unsafe_allow_html=True)

                if shown:
                    st.dataframe(
                        [{"entity": s.text(text), "label": s.label, "source": s.source, "score": round(s.score, 2)} for s in shown],
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("Nothing found for the selected entity types.")

        if len(model_keys) > 1:
            st.subheader("Differences between models")
            rows = differences(text, spans_by_model)
            if rows:
                st.caption("Only entities the models disagree on. A dash means the model did not find it.")
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.success("All selected models found exactly the same entities.")

with tab_report:
    if REPORT_PATH.exists():
        st.markdown(REPORT_PATH.read_text())
    else:
        st.warning("No report yet. Run: uv run python scripts/run_benchmark.py")
