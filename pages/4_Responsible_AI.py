import pandas as pd
import plotly.express as px
import streamlit as st

from src import config as C
from src.ui import read_csv, read_json, require_artifacts

st.set_page_config(page_title="Responsible AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Responsible AI")
try:
    require_artifacts()
    st.subheader("Fairness & subgroup performance (DNN at the operating threshold, test set)")
    st.info("The dataset contains no customer attributes (no gender, age or location), so fairness across protected groups "
            "cannot be audited. As a proxy, performance is compared across transaction segments.")
    n_fraud = []
    for f, title in [("fairness_amount", "By amount band"), ("fairness_time", "By time of day")]:
        df = read_csv(f"{f}.csv"); st.markdown(f"**{title}**"); n_fraud += df.n_fraud.iloc[:-1].tolist()
        a, b = st.columns([3, 2]); a.image(str(C.FIGURES_DIR / f"{f}.png")); b.dataframe(df.round(4), hide_index=True)
    st.caption(f"Each segment contains only {min(n_fraud)}–{max(n_fraud)} frauds, so a recall gap mostly reflects a handful of "
               "transactions. Gaps above the 0.20 audit criterion should be re-checked on more data before deployment.")

    st.subheader("Robustness (counterfactual search, 15 cases: 5 fraud, 10 legitimate)")
    rb, rs = read_csv("robustness.csv"), read_json("robustness_summary.json"); st.dataframe(rb.round(4), hide_index=True)
    st.write(f"DNN class flipped in **{rs['dnn_flip_rate']:.0%}** of cases, verdict flipped in **{rs['verdict_flip_rate']:.0%}**; "
             f"{rs['no_flip_cases']} of {rs['n_cases']} cases never flipped within **{rs['search_max_std']:.1f}** training standard "
             "deviations of random perturbation (V1-V28 and Amount).")

    st.subheader("Explainability")
    st.write("**SHAP stability** (1% noise on V1-V28 and Amount, 30 samples)")
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
