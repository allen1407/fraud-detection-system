import pandas as pd
import plotly.express as px
import streamlit as st

from src import config as C
from src.ui import VERDICT_COLORS, get_pipeline, require_artifacts

st.set_page_config(page_title="Batch Scoring", page_icon="📦", layout="wide")
st.title("📦 Batch Scoring")
try:
    require_artifacts()
    pipe = get_pipeline()
    sample = pd.read_csv(C.SAMPLES_DIR / "sample_batch.csv")
    template = sample.head(1)
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV template", template.to_csv(index=False), "batch_template.csv", "text/csv")
    c2.download_button("Download 20-row sample (2 deliberately invalid rows)", sample.to_csv(index=False), "sample_batch.csv", "text/csv")
    st.caption("Required columns: " + ", ".join(C.REQUIRED_FIELDS))
    up = st.file_uploader("Upload a CSV", type="csv")
    explain = st.checkbox("Include SHAP top-5 factors (slower, ~0.1 s per row)")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            st.error("That file could not be read as CSV."); st.stop()
        missing = [c for c in C.REQUIRED_FIELDS if c not in df.columns]
        if missing:
            st.error(f"The file is missing required columns: {', '.join(missing)}"); st.stop()
        if len(df) > 5000:
            st.warning("Only the first 5,000 rows are scored."); df = df.head(5000)
        bar, parts = st.progress(0.0, "Scoring..."), []
        for i in range(0, len(df), 50):
            parts.append(pipe.score_batch(df.iloc[i:i + 50], explain=explain))
            bar.progress(min((i + 50) / len(df), 1.0), f"Scored {min(i + 50, len(df))}/{len(df)} rows")
        out = pd.concat(parts, ignore_index=True)
        bar.empty()
        show = out.drop(columns=[c for c in ["top_factors"] if not explain])
        st.dataframe(show, use_container_width=True)
        vc = out.verdict.value_counts().reset_index(); vc.columns = ["verdict", "rows"]
        st.plotly_chart(px.bar(vc, x="verdict", y="rows", color="verdict", color_discrete_map=VERDICT_COLORS,
                               title="Verdict summary"), use_container_width=True)
        bad = out[out.verdict == "TRANSACTION REJECTED"]
        if len(bad):
            st.warning(f"{len(bad)} row(s) were rejected; see the 'errors' column for reasons.")
        st.download_button("Download results (CSV)", out.to_csv(index=False), "batch_results.csv", "text/csv")
except Exception:
    import traceback; traceback.print_exc()   # logged to the console only, never shown in the UI
    st.error("The batch could not be processed. Please check the file format and try again.")
