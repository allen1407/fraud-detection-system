import streamlit as st

from src import config as C
from src.ui import render_markdown_with_images

st.set_page_config(page_title="Documentation", page_icon="📚", layout="wide")
st.title("📚 Documentation")
try:
    docs = sorted(C.DOCS_DIR.glob("*.md"))
    if not docs:
        st.info("No documentation found in docs/.")
    else:
        tabs = st.tabs([p.stem.replace("_", " ") for p in docs] + ["Limitations"])
        for tab, p in zip(tabs, docs):
            with tab:
                render_markdown_with_images(p.read_text(), C.DOCS_DIR)
        with tabs[-1]:
            st.markdown("""
### Limitations
- **No predictive signal:** on this dataset both models score ROC-AUC ≈ 0.5 (chance). Reported metrics are real, but the models cannot detect fraud.
- **Unrealistic prevalence:** ~49% fraud is far above real banking traffic; thresholds and the cost trade-off need recalibration before any real use.
- **Likely synthetic data:** unique account numbers and generated text suggest simulated data; findings may not transfer.
- **Small groups:** fairness estimates rest on 110–300 rows per group.
- **Decision support only:** the system recommends; humans own MANUAL REVIEW decisions.
- **SHAP is approximate** (KernelExplainer on 100 background rows) and explains the DNN only.""")
except Exception:
    import traceback; traceback.print_exc()   # logged to the console only, never shown in the UI
    st.error("The documentation could not be displayed.")
