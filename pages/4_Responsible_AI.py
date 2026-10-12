import pandas as pd
import plotly.express as px
import streamlit as st

from src import config as C
from src.ui import read_csv, read_json, require_artifacts

st.set_page_config(page_title="Responsible AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Responsible AI")
try:
    require_artifacts()
    st.subheader("Fairness (DNN at the operating threshold, test set)")
    for f, title, fig in [("fairness_gender.csv", "By gender", "fairness_gender.png"), ("fairness_age.csv", "By age band", "fairness_age.png")]:
        df = read_csv(f); st.markdown(f"**{title}**")
        a, b = st.columns([3, 2]); a.image(str(C.FIGURES_DIR / fig)); b.dataframe(df.round(3), hide_index=True)
    st.caption("Group sizes are small (≈110–300 per group), and the model is near chance, so these gaps are mostly sampling noise. "
               "They still exceed the 0.10 audit criterion and would need investigation on real data.")

    st.subheader("Robustness (counterfactual search, 15 cases)")
    rb = read_csv("robustness.csv"); st.dataframe(rb.round(3), hide_index=True)
    st.write(f"DNN class flipped in **{rb.dnn_class_flipped.mean():.0%}** of cases, verdict flipped in **{rb.verdict_flipped.mean():.0%}**; "
             f"median distance to flip **{rb.distance_std.median():.2f}** training standard deviations (numeric inputs).")

    st.subheader("Explainability")
    st.write("**SHAP stability** (1% noise on numeric inputs, 30 samples)")
    s = read_json("shap_stability.json")
    st.metric("Mean Spearman rank correlation", f"{s['mean_spearman']:.3f} ± {s['std_spearman']:.3f}")
    imp = read_csv("shap_global.csv").head(12).iloc[::-1]
    st.plotly_chart(px.bar(imp, x="mean_abs_shap", y="feature", orientation="h", title="Global SHAP feature importance (mean |SHAP|)"), use_container_width=True)
    st.image(str(C.FIGURES_DIR / "shap_summary.png"))

    st.subheader("Human-in-the-loop: why MANUAL REVIEW exists")
    st.markdown("""
The system acts automatically **only** when all of these hold:
1. The DNN and the logistic regression **agree** (both flag, or both clear);
2. The **trust score ≥ 1.0**, i.e. the transaction is as familiar as typical training transactions.

Everything else (model disagreement or an unfamiliar transaction) goes to **MANUAL REVIEW** for a human analyst.
Invalid input is never scored; it is rejected with reasons.""")
    rt = read_json("metrics.json")["routing"]
    st.write("Test-set routing: " + ", ".join(f"**{k}** {v}" for k, v in rt.items()))
except Exception:
    import traceback; traceback.print_exc()   # logged to the console only, never shown in the UI
    st.error("Responsible-AI results could not be displayed. Run `bash scripts/run_all.sh` to regenerate reports.")
