"""Shared helpers for the Streamlit app. All model logic stays in src/pipeline.py."""
import json
import re

import pandas as pd
import streamlit as st

from . import config as C

VERDICT_COLORS = {"AUTO-BLOCK": "#dc2626", "AUTO-APPROVE": "#16a34a",
                  "MANUAL REVIEW": "#d97706", "TRANSACTION REJECTED": "#6b7280"}


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline():
    from .pipeline import FraudPipeline
    return FraudPipeline()            # loads DNN, LR, preprocessor, kNN index and SHAP explainer once


@st.cache_data
def read_json(name):
    p = C.REPORTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def read_csv(name):
    p = C.REPORTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def read_test():
    return pd.read_csv(C.TEST_CSV)


def require_artifacts():
    missing = [p.name for p in (C.DNN_PATH, C.LR_PATH, C.PREPROCESSOR_PATH, C.METADATA_PATH, C.REPORTS_DIR / "metrics.json")
               if not p.exists()]
    if missing:
        st.error("Models or reports are missing. Run `bash scripts/run_all.sh` first "
                 f"(missing: {', '.join(missing)}).")
        st.stop()


def badge(verdict):
    c = VERDICT_COLORS.get(verdict, "#6b7280")
    st.markdown(f"<div style='background:{c};color:white;padding:14px 22px;border-radius:10px;"
                f"font-size:1.6rem;font-weight:700;text-align:center'>{verdict}</div>", unsafe_allow_html=True)


def gauge(title, value, threshold):
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"valueformat": ".3f"}, title={"text": title},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#2563eb"},
               "threshold": {"line": {"color": "#dc2626", "width": 4}, "value": threshold}}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def report_pdf(text: str) -> bytes:
    from fpdf import FPDF
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Courier", size=9)
    for line in text.encode("latin-1", "replace").decode("latin-1").split("\n"):
        pdf.multi_cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def presets():
    """Real test-set rows chosen by the saved model outputs."""
    test, pred = read_test(), read_csv("test_predictions.csv")
    thr = read_json("metrics.json")["thresholds"]["dnn_operating"]
    cols = C.INPUT_COLUMNS
    def pick(mask, fallback, key, asc):
        d = pred[mask] if mask.any() else pred[fallback]       # fall back to the label alone if no row matches
        return test.loc[d.sort_values(key, ascending=asc).index[0], cols].to_dict()
    legit, fraud = pred.y_true == 0, pred.y_true == 1
    out = {
        "Clear Legit": pick(legit & (pred.verdict == "AUTO-APPROVE"), legit, "dnn_prob", True),
        "Clear Fraud": pick(fraud & (pred.verdict == "AUTO-BLOCK"), fraud, "dnn_prob", False),
    }
    review = pred.verdict == "MANUAL REVIEW"
    bl = pred[review if review.any() else slice(None)].assign(d=lambda x: (x.dnn_prob - thr).abs()).sort_values("d")
    out["Borderline"] = test.loc[bl.index[0], cols].to_dict()
    inv = dict(out["Clear Legit"]); inv["Amount"] = -250.0
    out["Invalid Input"] = inv
    return out


def render_markdown_with_images(text, base):
    """st.markdown cannot load relative images, so split them out."""
    for part in re.split(r"(!\[[^\]]*\]\([^)]+\))", text):
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", part)
        if m and (base / m.group(2)).exists():
            st.image(str(base / m.group(2)), caption=m.group(1))
        elif part.strip():
            st.markdown(part)
