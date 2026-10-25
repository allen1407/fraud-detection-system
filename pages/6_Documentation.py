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
- **Few positive examples:** only 473 frauds in total (71 in the test set), so recall, precision and subgroup estimates have wide confidence intervals.
- **Anonymised features:** V1–V28 are PCA components; SHAP factors cannot be translated into business reasons for a customer or analyst.
- **Old, short time window:** two days of European card traffic from September 2013; patterns will have drifted since.
- **No customer attributes:** fairness across protected groups cannot be audited; only transaction segments are compared.
- **Random split:** train and test come from the same two days, so the drift baseline and metrics are optimistic about future performance.
- **Decision support only:** the system recommends; humans own MANUAL REVIEW decisions.
- **SHAP is approximate** (KernelExplainer on 100 background rows) and explains the DNN only.""")
except Exception:
    import traceback; traceback.print_exc()   # logged to the console only, never shown in the UI
    st.error("The documentation could not be displayed.")
